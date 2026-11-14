"""Run every fly live and serve the 3D world.

    python -m server.serve                 # then open http://localhost:8000/web/
    python -m server.serve --only gym inbox
    python -m server.serve --host 0.0.0.0  # let other laptops on the network watch

Each fly uses runs/<fly>/brain.pt if it exists; otherwise an UNTRAINED brain
(clearly labelled) on --untrained-size so the world still works before training.
A decision only happens when the browser says the previous animation finished,
so what you see is always the fly's live choice.
"""
import argparse
import asyncio
import base64
import functools
import http.server
import json
import threading
from pathlib import Path

import numpy as np
import torch
import websockets

from core.brain import FlyBrain, load_brain, pick_device
from core.registry import fly_names, load_task, run_dir

ROOT = Path(__file__).resolve().parents[1]


class Fly:
    def __init__(self, name, device, untrained_size):
        self.name = name
        self.cls = load_task(name)
        ck = run_dir(name) / "brain.pt"
        self.result = {}
        try:
            self.brain = load_brain(ck, self.cls, device) if ck.exists() else None
        except ValueError as e:
            print(f"[{name}] {e} - using an untrained brain")
            self.brain = None
        self.trained = self.brain is not None
        if self.trained and (run_dir(name) / "result.json").exists():
            self.result = json.loads((run_dir(name) / "result.json").read_text())
        if not self.trained:
            self.brain = FlyBrain(untrained_size, self.cls.obs_dim, len(self.cls.actions), name).to(device)
        self.brain.eval()
        self.device = device
        print(f"[{name}] {'trained' if self.trained else 'UNTRAINED'} {self.brain.preset} brain, "
              f"{self.brain.n:,} neurons")

    def info(self):
        return dict(name=self.name, actions=list(self.cls.actions), preset=self.brain.preset,
                    trained=self.trained, synthetic=self.brain.synthetic, mode=self.cls.mode,
                    groups=self.brain.groups.tolist(),
                    scores={k: self.result.get(k) for k in ("score_trained", "score_random", "score_lesioned")})


class Session:
    """One browser connection: its own copy of every fly's world, sharing the brains."""

    def __init__(self, flies):
        self.flies = flies
        self.rng = np.random.default_rng()
        self.env, self.obs, self.lesion = {}, {}, {}
        self.inspect = None
        for n, f in flies.items():
            self.env[n] = f.cls()
            self.obs[n] = np.asarray(self.env[n].reset(self.rng), np.float32)
            self.lesion[n] = False

    def decide(self, name):
        f, env = self.flies[name], self.env[name]
        k = len(f.cls.actions)
        m = env.mask()
        m = np.ones(k, bool) if m is None else np.asarray(m, bool)
        record = self.inspect == name
        with torch.no_grad():
            logits, info = f.brain(torch.as_tensor(self.obs[name][None], device=f.device),
                                   torch.as_tensor(m[None], device=f.device),
                                   lesion=self.lesion[name], record=record)
        probs = torch.softmax(logits[0], -1).cpu().numpy()
        a = int(probs.argmax()) if f.cls.greedy else int(self.rng.choice(k, p=probs / probs.sum()))
        obs, reward, done, _ = env.step(a, self.rng)
        caption = env.caption(a)
        self.obs[name] = np.asarray(obs, np.float32)
        if done:
            self.obs[name] = np.asarray(env.reset(self.rng), np.float32)
        msg = dict(type="decision", fly=name, action=f.cls.actions[a], action_index=a,
                   probs={n: float(p) for n, p in zip(f.cls.actions, probs)},
                   reward=float(reward), episode_done=bool(done), caption=caption,
                   lesioned=self.lesion[name], mean_rate=info["mean_rate"])
        if record:
            r = info["rates"][:, 0, :].cpu().numpy()                 # (STEPS, N)
            q = np.clip(np.sqrt(np.clip(r, 0, None) / 2.0) * 255, 0, 255).astype(np.uint8)
            msg["activity"] = dict(steps=int(q.shape[0]), n=int(q.shape[1]),
                                   data=base64.b64encode(q.tobytes()).decode())
        return msg


async def handler(ws, flies):
    s = Session(flies)
    async for raw in ws:
        m = json.loads(raw)
        t = m.get("type")
        if t == "hello":
            await ws.send(json.dumps(dict(type="init", flies=[f.info() for f in flies.values()])))
        elif t == "ready" and m.get("fly") in flies:
            await ws.send(json.dumps(s.decide(m["fly"])))
        elif t == "inspect":
            s.inspect = m.get("fly")
        elif t == "lesion" and m.get("fly") in flies:
            s.lesion[m["fly"]] = bool(m.get("on"))


def serve_static(host, port):
    h = functools.partial(http.server.SimpleHTTPRequestHandler, directory=str(ROOT))
    h.log_message = lambda *a, **k: None
    http.server.ThreadingHTTPServer((host, port), h).serve_forever()


async def main_async(a):
    device = pick_device(a.device)
    flies = {n: Fly(n, device, a.untrained_size) for n in (a.only or fly_names())}
    threading.Thread(target=serve_static, args=(a.host, a.port), daemon=True).start()
    async with websockets.serve(lambda ws: handler(ws, flies), a.host, a.ws_port, max_size=None):
        shown = "localhost" if a.host in ("127.0.0.1", "0.0.0.0") else a.host
        print(f"\nOpen http://{shown}:{a.port}/web/   (websocket on port {a.ws_port})")
        await asyncio.Future()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--only", nargs="*")
    ap.add_argument("--host", default="127.0.0.1")
    ap.add_argument("--port", type=int, default=8000)
    ap.add_argument("--ws-port", type=int, default=8765)
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--untrained-size", default="tiny")
    asyncio.run(main_async(ap.parse_args()))


if __name__ == "__main__":
    main()
