"""Train one fly.

    python -m core.train gym                    # default preset: small
    python -m core.train gym --size full --minutes 15
    python -m core.train inbox --redo-demos     # teacher mode: recollect teacher demos

Writes runs/<fly>/brain.pt and runs/<fly>/result.json.
"""
import argparse
import json
import time

import numpy as np
import torch

from core.brain import FlyBrain, pick_device, save_brain
from core.registry import load_task, run_dir, FLIES

DEFAULTS = dict(
    updates=300,           # reward mode: policy updates (stops early at the time budget)
    episodes=32,           # reward mode: episodes per update
    lr=1e-3,
    entropy=0.01,
    gamma=0.98,
    chunk=256,             # decisions per gradient chunk (lower if out of memory)
    epochs=8,              # teacher mode: passes over the demos
    teacher_episodes=200,  # teacher mode: episodes of demos to collect
    teacher_epsilon=0.15,  # teacher mode: random detours so the fly sees off-script states
    eval_episodes=32,
)


def settings(task_cls, overrides):
    s = dict(DEFAULTS)
    s.update(getattr(task_cls, "train", {}) or {})
    s.update({k: v for k, v in overrides.items() if v is not None})
    return s


def _mask_array(env, k):
    m = env.mask()
    m = np.ones(k, bool) if m is None else np.asarray(m, bool)
    if m.shape != (k,) or not m.any():
        raise ValueError(f"{env.name}: mask() must return {k} bools with at least one True")
    return m


def rollout(task_cls, n, rng, policy, brain=None, device="cpu", lesion=False, greedy=False, eps=0.15):
    """Play n episodes in parallel. policy: 'brain' | 'random' | 'teacher' | 'teacher_eps'.

    Returns per-episode returns and a list of transitions (obs, mask, action, teacher, reward, ep, t).
    """
    k = len(task_cls.actions)
    envs = [task_cls() for _ in range(n)]
    rngs = [np.random.default_rng(rng.integers(1 << 62)) for _ in range(n)]
    obs = [np.asarray(e.reset(r), np.float32) for e, r in zip(envs, rngs)]
    done = [False] * n
    returns = np.zeros(n)
    trans = []
    for t in range(task_cls.max_steps):
        active = [i for i in range(n) if not done[i]]
        if not active:
            break
        masks = np.stack([_mask_array(envs[i], k) for i in active])
        teach = None
        if policy.startswith("teacher"):
            teach = np.array([int(envs[i].teacher()) for i in active])
        if policy == "brain":
            with torch.no_grad():
                o = torch.as_tensor(np.stack([obs[i] for i in active]), device=device)
                m = torch.as_tensor(masks, device=device)
                logits, _ = brain(o, m, lesion=lesion)
                acts = (logits.argmax(-1) if greedy else
                        torch.distributions.Categorical(logits=logits).sample()).cpu().numpy()
        elif policy == "random" or policy == "teacher_eps":
            acts = np.array([rng.choice(np.flatnonzero(m)) for m in masks])
            if policy == "teacher_eps":
                use = rng.random(len(active)) > eps
                acts = np.where(use, teach, acts)
        else:
            acts = teach
        for j, i in enumerate(active):
            a = int(acts[j])
            o2, r, d, _ = envs[i].step(a, rngs[i])
            trans.append((obs[i], masks[j], a, -1 if teach is None else int(teach[j]), float(r), i, t))
            returns[i] += r
            obs[i] = np.asarray(o2, np.float32)
            done[i] = bool(d)
    return returns, trans


def _logp_batches(brain, trans, chunk, device):
    for s in range(0, len(trans), chunk):
        b = trans[s:s + chunk]
        o = torch.as_tensor(np.stack([x[0] for x in b]), device=device)
        m = torch.as_tensor(np.stack([x[1] for x in b]), device=device)
        a = torch.as_tensor([x[2] for x in b], device=device)
        logits, _ = brain(o, m)
        yield b, a, torch.log_softmax(logits, -1), s


def train_reward(brain, task_cls, cfg, device, deadline, log):
    opt = torch.optim.Adam(brain.parameters(), lr=cfg["lr"])
    rng = np.random.default_rng(0)
    for u in range(cfg["updates"]):
        if time.time() > deadline:
            log(f"time budget reached after {u} updates")
            break
        returns, trans = rollout(task_cls, cfg["episodes"], rng, "brain", brain, device)
        # discounted return-to-go per episode, baseline = mean over episodes at the same step
        G = np.zeros(len(trans))
        by_ep = {}
        for idx, x in enumerate(trans):
            by_ep.setdefault(x[5], []).append(idx)
        for idxs in by_ep.values():
            g = 0.0
            for idx in reversed(idxs):
                g = trans[idx][4] + cfg["gamma"] * g
                G[idx] = g
        steps = np.array([x[6] for x in trans])
        base = np.zeros_like(G)
        for t in np.unique(steps):
            base[steps == t] = G[steps == t].mean()
        adv = G - base
        adv = adv / (adv.std() + 1e-8)
        opt.zero_grad()
        total = len(trans)
        for b, a, logp, s in _logp_batches(brain, trans, cfg["chunk"], device):
            A = torch.as_tensor(adv[s:s + len(b)], device=device, dtype=torch.float32)
            lp = logp.gather(1, a[:, None]).squeeze(1)
            p = logp.exp()
            ent = -(p * logp.clamp(min=-30)).sum(-1)
            loss = -(lp * A).sum() / total - cfg["entropy"] * ent.sum() / total
            loss.backward()
        torch.nn.utils.clip_grad_norm_(brain.parameters(), 1.0)
        opt.step()
        if u % 5 == 0 or u == cfg["updates"] - 1:
            log(f"update {u:4d}  avg return {returns.mean():8.3f}")


def collect_demos(task_cls, cfg, redo, log):
    path = FLIES / task_cls.name / "demos.npz"
    if path.exists() and not redo:
        z = np.load(path)
        log(f"using {len(z['action'])} cached teacher demos from flies/{task_cls.name}/demos.npz")
        return z["obs"], z["mask"], z["action"]
    log(f"collecting teacher demos ({cfg['teacher_episodes']} episodes) ...")
    _, trans = rollout(task_cls, cfg["teacher_episodes"], np.random.default_rng(1), "teacher_eps",
                       eps=cfg["teacher_epsilon"])
    obs = np.stack([x[0] for x in trans]); mask = np.stack([x[1] for x in trans])
    action = np.array([x[3] for x in trans])
    np.savez_compressed(path, obs=obs, mask=mask, action=action)
    log(f"saved {len(action)} demos to flies/{task_cls.name}/demos.npz (commit it so nobody pays for them twice)")
    return obs, mask, action


def train_teacher(brain, task_cls, cfg, device, deadline, log, redo_demos=False):
    obs, mask, action = collect_demos(task_cls, cfg, redo_demos, log)
    n = len(action)
    rng = np.random.default_rng(0)
    perm = rng.permutation(n)
    hold = perm[: max(1, n // 10)]
    tr = perm[max(1, n // 10):]
    opt = torch.optim.Adam(brain.parameters(), lr=cfg["lr"])
    O = torch.as_tensor(obs, device=device); M = torch.as_tensor(mask, device=device)
    Y = torch.as_tensor(action, device=device)
    for ep in range(cfg["epochs"]):
        if time.time() > deadline:
            log(f"time budget reached after {ep} epochs")
            break
        rng.shuffle(tr)
        tot = 0.0
        for s in range(0, len(tr), cfg["chunk"]):
            i = torch.as_tensor(tr[s:s + cfg["chunk"]], device=device)
            logits, _ = brain(O[i], M[i])
            loss = torch.nn.functional.cross_entropy(logits, Y[i])
            opt.zero_grad(); loss.backward()
            torch.nn.utils.clip_grad_norm_(brain.parameters(), 1.0); opt.step()
            tot += loss.item() * len(i)
        log(f"epoch {ep}  loss {tot / len(tr):.4f}  held-out agreement {agreement(brain, O, M, Y, hold, device):.3f}")
    return agreement(brain, O, M, Y, hold, device), (O, M, Y, hold)


def agreement(brain, O, M, Y, idx, device, lesion=False):
    with torch.no_grad():
        i = torch.as_tensor(idx, device=device)
        logits, _ = brain(O[i], M[i], lesion=lesion)
        return (logits.argmax(-1) == Y[i]).float().mean().item()


def evaluate(brain, task_cls, cfg, device):
    rng = np.random.default_rng(123)
    n = cfg["eval_episodes"]
    trained, _ = rollout(task_cls, n, rng, "brain", brain, device, greedy=task_cls.greedy)
    random_, _ = rollout(task_cls, n, rng, "random")
    lesioned, _ = rollout(task_cls, n, rng, "brain", brain, device, lesion=True, greedy=task_cls.greedy)
    return dict(score_trained=float(trained.mean()), score_random=float(random_.mean()),
                score_lesioned=float(lesioned.mean()))


def train_fly(name, size="small", minutes=15.0, device="auto", redo_demos=False, log=print, **overrides):
    task_cls = load_task(name)
    cfg = settings(task_cls, overrides)
    dev = pick_device(device)
    t0 = time.time()
    deadline = t0 + minutes * 60
    brain = FlyBrain(size, task_cls.obs_dim, len(task_cls.actions), name).to(dev)
    if brain.synthetic:
        log("WARNING: synthetic graph - this is NOT a fly brain. Fine for testing, never for the demo.")
    log(f"training '{name}' ({task_cls.mode} mode) on {size} brain: {brain.n:,} neurons, device {dev}")
    result = dict(fly=name, preset=size, synthetic=brain.synthetic, mode=task_cls.mode,
                  actions=list(task_cls.actions), graph_hash=brain.graph_hash)
    if task_cls.mode == "reward":
        train_reward(brain, task_cls, cfg, dev, deadline, log)
    elif task_cls.mode == "teacher":
        acc, (O, M, Y, hold) = train_teacher(brain, task_cls, cfg, dev, deadline, log, redo_demos)
        result.update(teacher_agreement=acc,
                      teacher_agreement_lesioned=agreement(brain, O, M, Y, hold, dev, lesion=True),
                      teacher_agreement_chance=float((1.0 / M[torch.as_tensor(hold, device=dev)]
                                                      .sum(1).float()).mean().item()))
    else:
        raise SystemExit(f"{name}: mode must be 'reward' or 'teacher'")
    result.update(evaluate(brain, task_cls, cfg, dev))
    result.update(train_seconds=round(time.time() - t0, 1),
                  action_groups=brain.describe_groups(task_cls.actions), settings=cfg)
    out = run_dir(name)
    save_brain(brain, out / "brain.pt", task_cls)
    (out / "result.json").write_text(json.dumps(result, indent=2))
    log(f"done in {result['train_seconds']}s: trained {result['score_trained']:.3f} | "
        f"random {result['score_random']:.3f} | lesioned {result['score_lesioned']:.3f}  -> runs/{name}/")
    return result


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("fly")
    ap.add_argument("--size", default="small")
    ap.add_argument("--minutes", type=float, default=15)
    ap.add_argument("--device", default="auto")
    ap.add_argument("--updates", type=int)
    ap.add_argument("--episodes", type=int)
    ap.add_argument("--epochs", type=int)
    ap.add_argument("--chunk", type=int)
    ap.add_argument("--redo-demos", action="store_true")
    a = ap.parse_args()
    train_fly(a.fly, a.size, a.minutes, a.device, a.redo_demos,
              updates=a.updates, episodes=a.episodes, epochs=a.epochs, chunk=a.chunk)


if __name__ == "__main__":
    main()
