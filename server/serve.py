"""Run every fly live and serve the 3D world.

    python -m server.serve                 # then open http://localhost:8000/web/
    python -m server.serve --only gym inbox
    python -m server.serve --host 0.0.0.0  # let other laptops on the network watch

Page and live brain data share one port (the websocket lives at /ws).
Each fly uses runs/<fly>/brain.pt if it exists; otherwise an UNTRAINED brain
(clearly labelled) on --untrained-size so the world still works before training.
A decision only happens when the browser says the previous animation finished,
so what you see is always the fly's live choice.
"""
import argparse
import asyncio
import base64
import json
import mimetypes
from http import HTTPStatus
from pathlib import Path

import numpy as np
import torch
from websockets.asyncio.server import serve
from websockets.datastructures import Headers
from websockets.http11 import Response

from core.brain import FlyBrain, load_brain, pick_device
from core.registry import exhibit_names, fly_names, load_task, run_dir

ROOT = Path(__file__).resolve().parents[1]


class Fly:
    def __init__(self, name, device, untrained_size):
        self.name = name
        self.cls = load_task(name)
        # a fly may name the checkpoint the world should run (relative to runs/<fly>/), e.g. its best one
        ck = run_dir(name) / getattr(self.cls, "world_checkpoint", "brain.pt")
        self.result = {}
        try:
            self.brain = load_brain(ck, self.cls, device) if ck.exists() else None
        except ValueError as e:
            print(f"[{name}] {e} - using an untrained brain")
            self.brain = None
        self.trained = self.brain is not None
        if self.trained and ck == run_dir(name) / "brain.pt" and (run_dir(name) / "result.json").exists():
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
                    groups=self.brain.groups.tolist(), menu=getattr(self.cls, "menu", None),
                    scores={k: self.result.get(k) for k in ("score_trained", "score_random", "score_lesioned")})


class Session:
    """One browser connection: its own copy of every fly's world, sharing the brains."""

    def __init__(self, flies):
        self.flies = flies
        self.rng = np.random.default_rng()
        self.env, self.obs, self.lesion = {}, {}, {}
        self.inspect = set()   # flies whose full neuron activity the browser is showing
        self.idle = set()      # menu flies whose animation finished and are waiting to be fed
        self.pending = {}      # menu fly -> item picked while it was still busy (latest wins)
        for n, f in flies.items():
            self.env[n] = f.cls(**getattr(f.cls, "world_args", {}))   # e.g. the level a fly was trained on
            self.obs[n] = np.asarray(self.env[n].reset(self.rng), np.float32)
            self.lesion[n] = False

    def decide(self, name):
        f, env = self.flies[name], self.env[name]
        k = len(f.cls.actions)
        m = env.mask()
        m = np.ones(k, bool) if m is None else np.asarray(m, bool)
        record = name in self.inspect
        with torch.no_grad():
            logits, info = f.brain(torch.as_tensor(self.obs[name][None], device=f.device),
                                   torch.as_tensor(m[None], device=f.device),
                                   lesion=self.lesion[name], record=record)
        probs = torch.softmax(logits[0], -1).cpu().numpy()
        a = int(probs.argmax()) if f.cls.greedy else int(self.rng.choice(k, p=probs / probs.sum()))
        shown = env.view() if hasattr(env, "view") else None   # optional: what the fly was choosing between
        obs, reward, done, step_info = env.step(a, self.rng)
        caption = env.caption(a)
        self.obs[name] = np.asarray(obs, np.float32)
        if done:
            self.obs[name] = np.asarray(env.reset(self.rng), np.float32)
        msg = dict(type="decision", fly=name, action=f.cls.actions[a], action_index=a,
                   probs={n: float(p) for n, p in zip(f.cls.actions, probs)},
                   reward=float(reward), episode_done=bool(done), caption=caption,
                   lesioned=self.lesion[name], mean_rate=info["mean_rate"])
        if shown is not None:
            msg["view"] = shown
        if step_info:                                          # what step() reported, for the station (if small)
            try:
                if len(json.dumps(step_info)) < 50_000:
                    msg["info"] = step_info
            except (TypeError, ValueError):
                pass
        if record:
            r = info["rates"][:, 0, :].cpu().numpy()                 # (STEPS, N)
            q = np.clip(np.sqrt(np.clip(r, 0, None) / 2.0) * 255, 0, 255).astype(np.uint8)
            msg["activity"] = dict(steps=int(q.shape[0]), n=int(q.shape[1]),
                                   data=base64.b64encode(q.tobytes()).decode())
        return msg


async def handler(ws, flies, exhibits=()):
    s = Session(flies)
    async for raw in ws:
        m = json.loads(raw)
        t = m.get("type")
        if t == "hello":
            await ws.send(json.dumps(dict(type="init", flies=[f.info() for f in flies.values()],
                                          exhibits=[dict(name=n) for n in exhibits])))
        elif t == "ready" and m.get("fly") in flies:
            n = m["fly"]
            if not flies[n].cls.menu:                          # normal flies decide again right away
                await ws.send(json.dumps(s.decide(n)))
            elif n in s.pending:                               # menu flies only act when fed
                s.obs[n] = np.asarray(s.env[n].choose(s.pending.pop(n)), np.float32)
                await ws.send(json.dumps(s.decide(n)))
            else:
                s.idle.add(n)
        elif t == "choose" and m.get("fly") in flies and flies[m["fly"]].cls.menu:
            n, item = m["fly"], m.get("item")
            if item not in {x["name"] for x in flies[n].cls.menu}:
                continue
            if n in s.idle:
                s.idle.discard(n)
                s.obs[n] = np.asarray(s.env[n].choose(item), np.float32)
                await ws.send(json.dumps(s.decide(n)))
            else:
                s.pending[n] = item
        elif t == "inspect":                                   # {"fly": name|null} or {"flies": [names]}
            v = m.get("flies", m.get("fly"))
            s.inspect = set(v) if isinstance(v, list) else ({v} if v else set())
        elif t == "lesion" and m.get("fly") in flies:
            s.lesion[m["fly"]] = bool(m.get("on"))


STATIC_ROOTS = ("web", "flies", "data")   # never serve runs/, jobs/, .remote.json, ...


def static_files(connection, request):
    """Serve the page and its files on the same port as the websocket (/ws), so one port is enough
    (needed behind Runpod's HTTP proxy)."""
    path = request.path.split("?", 1)[0]
    if path == "/ws":
        return None                                   # continue with the websocket handshake
    if path in ("/", "", "/web"):
        return Response(HTTPStatus.FOUND.value, "Found", Headers([("Location", "/web/"), ("Content-Length", "0")]), b"")
    if path.endswith("/"):
        path += "index.html"
    target = (ROOT / path.lstrip("/")).resolve()
    ok = target.is_file() and any(target.is_relative_to(ROOT / r) for r in STATIC_ROOTS)
    if not ok:
        return connection.respond(HTTPStatus.NOT_FOUND, "not found\n")
    body = target.read_bytes()
    ctype = mimetypes.guess_type(target.name)[0] or "application/octet-stream"
    if target.suffix == ".js":
        ctype = "text/javascript"
    headers = Headers([("Content-Type", ctype), ("Content-Length", str(len(body))),
                       ("Cache-Control", "no-cache")])
    return Response(HTTPStatus.OK.value, "OK", headers, body)


async def main_async(a):
    device = pick_device(a.device)
    names = a.only or fly_names() + exhibit_names()
    exhibits = [n for n in names if n in exhibit_names()]
    flies = {n: Fly(n, device, a.untrained_size) for n in names if n not in exhibits}
    for n in exhibits:
        print(f"[{n}] browser exhibit (its brains run in the viewer's page)")
    async with serve(lambda ws: handler(ws, flies, exhibits), a.host, a.port, max_size=None,
                     process_request=static_files):
        shown = "localhost" if a.host in ("127.0.0.1", "0.0.0.0") else a.host
        print(f"\nOpen http://{shown}:{a.port}/web/", flush=True)
        await asyncio.Future()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--only", nargs="*")
    ap.add_argument("--host", default="127.0.0.1")
    ap.add_argument("--port", type=int, default=8000)
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--untrained-size", default="tiny")
    asyncio.run(main_async(ap.parse_args()))


if __name__ == "__main__":
    main()
