"""The tasting room: click a dish, the critic fly's brain tastes it and rates it 0-10 (7 decimals).

    python flies/critic/tasting.py            # then open http://localhost:8010
    python flies/critic/tasting.py --port 8011 --device cpu

Uses runs/critic/brain.pt (pull it with `python scripts/remote.py pull critic`) and reloads it
automatically when a newer one appears. Without it, an UNTRAINED tiny brain is used and labelled.

Nothing here decides for the fly: the serving goes in through the brain's sensory neurons, and the
rating is sum_i p_i * VALUES[i], where p is the softmax over the 8 verdict groups of real
descending neurons. The lesion toggle zeroes the connectome, and the rating goes flat.
"""
import argparse
import base64
import http.server
import json
import sys
import threading
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
sys.path.insert(0, str(ROOT))

import numpy as np  # noqa: E402
import torch  # noqa: E402

from core.brain import FlyBrain, pick_device  # noqa: E402
from core.registry import load_task, run_dir  # noqa: E402

Task = load_task("critic")
T = Task.reset.__globals__          # the task module's namespace (load_task doesn't register it)
MENU, VALUES, FEATURES, rating, verdict_for = T["MENU"], T["VALUES"], T["FEATURES"], T["rating"], T["verdict_for"]
BRAIN_FILES = {f"/data/{p}.{ext}" for p in ("tiny", "small", "central", "full") for ext in ("points.bin", "meta.json")}


WIRING = {"edge_index", "t_perm", "t_index", "sign", "input_idx"}   # rebuilt from data/<preset>.npz


def load_trained(path, device):
    """Like core.brain.load_brain, but also reads brains saved by newer core versions, which no longer
    store the fixed wiring in the file (it is rebuilt from data/<preset>.npz, checked by graph hash)."""
    ck = torch.load(path, map_location=device, weights_only=False)
    if list(ck["actions"]) != list(Task.actions) or ck["obs_dim"] != Task.obs_dim:
        raise ValueError(f"{path}: actions/obs_dim changed since training; retrain this fly")
    brain = FlyBrain(ck["preset"], ck["obs_dim"], len(ck["actions"]), ck["fly"]).to(device)
    if brain.graph_hash != ck["graph_hash"]:
        raise ValueError(f"{path}: trained on graph {ck['graph_hash']} but data/{ck['preset']}.npz is "
                         f"{brain.graph_hash}; copy that file from the GPU pod")
    own = brain.state_dict()
    sd = {k: v for k, v in ck["state_dict"].items() if k in own}
    missing = set(own) - set(sd)
    if missing - WIRING:
        raise ValueError(f"{path}: missing trained weights {sorted(missing - WIRING)}")
    brain.load_state_dict(sd, strict=False)
    return brain


class Critic:
    def __init__(self, device, untrained_size):
        self.device, self.untrained_size = device, untrained_size
        self.lock = threading.Lock()
        self.rng = np.random.default_rng()
        self.env = Task()
        self.env.reset(self.rng)
        self.ck = run_dir("critic") / "brain.pt"
        self.mtime = None
        self.result = {}
        self.load()

    def load(self):
        mtime = self.ck.stat().st_mtime if self.ck.exists() else None
        if self.mtime is not None and mtime == self.mtime:
            return
        self.brain, self.trained = None, False
        if mtime is not None:
            try:
                self.brain = load_trained(self.ck, self.device)
                self.trained = True
                rj = run_dir("critic") / "result.json"
                self.result = json.loads(rj.read_text()) if rj.exists() else {}
            except ValueError as e:
                print(f"[critic] {e} - using an untrained brain", flush=True)
        if self.brain is None:
            self.brain = FlyBrain(self.untrained_size, Task.obs_dim, len(Task.actions), "critic").to(self.device)
        self.brain.eval()
        self.mtime = mtime
        print(f"[critic] {'trained' if self.trained else 'UNTRAINED'} {self.brain.preset} brain, "
              f"{self.brain.n:,} neurons", flush=True)

    def info(self):
        return dict(trained=self.trained, preset=self.brain.preset, neurons=self.brain.n,
                    actions=list(Task.actions), values=VALUES.tolist(), bands=T["BAND_EDGES"].tolist(),
                    groups=self.brain.groups.tolist(), input_idx=self.brain.input_idx.tolist(),
                    menu=[dict(name=n, group=g, emoji=e) for n, (g, e, _, _) in MENU.items()],
                    result={k: self.result.get(k) for k in
                            ("teacher_agreement", "teacher_agreement_chance", "score_trained", "score_random")})

    def taste(self, food, lesion=False, record=True):
        if food not in MENU:
            raise KeyError(food)
        with self.lock:
            self.load()
            self.env.new_dish(food)
            obs = torch.as_tensor(self.env.obs()[None], device=self.device)
            with torch.no_grad():
                logits, info = self.brain(obs, lesion=lesion, record=record)
            probs = torch.softmax(logits[0], -1).cpu().numpy().astype(np.float64)
            score = rating(probs)
            verdict = verdict_for(score)          # the word is the band the brain's score falls in
            activity = None
            if record:                            # every neuron's firing at each of the brain's steps
                r = info["rates"][:, 0, :].cpu().numpy()
                q = np.clip(np.sqrt(np.clip(r, 0, None) / 2.0) * 255, 0, 255).astype(np.uint8)
                activity = dict(steps=int(q.shape[0]), n=int(q.shape[1]),
                                data=base64.b64encode(q.tobytes()).decode())
            feats = {f: round(float(v), 3) for f, v in zip(FEATURES,self.env.feats)}
            return dict(food=food, emoji=MENU[food][1], group=MENU[food][0],
                        rating=round(score, 7), rating_text=f"{score:.7f}",
                        verdict=Task.actions[verdict], probs=probs.tolist(),
                        group_rate=info["group_rate"][0].cpu().numpy().tolist(),
                        mean_rate=info["mean_rate"], palate=round(self.env.score, 7),
                        serving=feats, lesioned=bool(lesion), trained=self.trained, preset=self.brain.preset,
                        activity=activity)


def make_handler(critic):
    class H(http.server.BaseHTTPRequestHandler):
        def log_message(self, *a):
            pass

        def send(self, code, body, ctype="application/json"):
            data = body if isinstance(body, bytes) else json.dumps(body).encode()
            self.send_response(code)
            self.send_header("Content-Type", ctype)
            self.send_header("Content-Length", str(len(data)))
            self.send_header("Cache-Control", "no-store")
            self.end_headers()
            self.wfile.write(data)

        def do_GET(self):
            if self.path in ("/", "/index.html"):
                self.send(200, (HERE / "tasting.html").read_bytes(), "text/html; charset=utf-8")
            elif self.path == "/api/info":
                with critic.lock:
                    critic.load()
                    self.send(200, critic.info())
            elif self.path in BRAIN_FILES and (ROOT / self.path[1:]).exists():   # neuron positions, read-only
                ctype = "application/json" if self.path.endswith(".json") else "application/octet-stream"
                self.send(200, (ROOT / self.path[1:]).read_bytes(), ctype)
            else:
                self.send(404, {"error": "not found"})

        def do_POST(self):
            if self.path != "/api/taste":
                return self.send(404, {"error": "not found"})
            try:
                n = int(self.headers.get("Content-Length", 0))
                req = json.loads(self.rfile.read(n) or b"{}")
                self.send(200, critic.taste(req.get("food"), bool(req.get("lesion"))))
            except KeyError:
                self.send(400, {"error": "unknown dish"})

    return H


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--host", default="127.0.0.1")
    ap.add_argument("--port", type=int, default=8010)
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--untrained-size", default="tiny")
    a = ap.parse_args()
    critic = Critic(pick_device(a.device), a.untrained_size)
    srv = http.server.ThreadingHTTPServer((a.host, a.port), make_handler(critic))
    print(f"\nTasting room open: http://{'localhost' if a.host in ('127.0.0.1', '0.0.0.0') else a.host}:{a.port}")
    srv.serve_forever()


if __name__ == "__main__":
    main()
