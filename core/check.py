"""Smoke test a fly in about a minute (run before every push).

    python -m core.check gym              # one fly, tiny brain, CPU-friendly
    python -m core.check --all            # every fly
    python -m core.check gym --size small

Checks: the task follows the interface, a brain can be built and stepped, a few
training steps run, and the lesioned brain ignores its input (the "it's really the
fly deciding" test). Does not overwrite runs/.
"""
import argparse
import time
import traceback

import numpy as np
import torch

from core.brain import FlyBrain, pick_device
from core.registry import fly_names, load_task
from core.train import rollout, settings, train_reward, _mask_array


def check_interface(cls, say):
    assert cls.name and isinstance(cls.name, str), "name missing"
    assert 2 <= len(cls.actions) <= 8, "need 2-8 actions"
    assert len(set(cls.actions)) == len(cls.actions), "action names must be unique"
    assert isinstance(cls.obs_dim, int) and cls.obs_dim > 0, "obs_dim must be a positive int"
    assert cls.mode in ("reward", "teacher"), "mode must be 'reward' or 'teacher'"
    rng = np.random.default_rng(0)
    env = cls()
    o = np.asarray(env.reset(rng))
    assert o.shape == (cls.obs_dim,), f"reset() returned shape {o.shape}, expected ({cls.obs_dim},)"
    t0 = time.time()
    steps = 0
    for _ in range(3):
        o = env.reset(rng)
        for _ in range(cls.max_steps):
            m = _mask_array(env, len(cls.actions))
            if cls.mode == "teacher":
                a = env.teacher()
                assert isinstance(a, (int, np.integer)) and 0 <= a < len(cls.actions), "teacher() must return an action index"
                assert m[a], "teacher() picked a masked action"
            a = int(rng.choice(np.flatnonzero(m)))
            out = env.step(a, rng)
            assert len(out) == 4, "step() must return (obs, reward, done, info)"
            o, r, d, _ = out
            o = np.asarray(o, np.float32)
            assert o.shape == (cls.obs_dim,) and np.isfinite(o).all(), "bad observation from step()"
            assert np.isfinite(r), "reward must be a finite number"
            assert isinstance(env.caption(a), str), "caption() must return a string"
            steps += 1
            if d:
                break
    per_step = (time.time() - t0) / max(steps, 1) * 1e6
    say(f"interface ok ({steps} random steps, {per_step:.0f} us/step)")
    if per_step > 2000:
        say("  WARNING: step() is slow; training will be too. Remove I/O from step().")


def check_brain(cls, size, device, say):
    dev = pick_device(device)
    brain = FlyBrain(size, cls.obs_dim, len(cls.actions), cls.name).to(dev)
    if brain.synthetic:
        say("  NOTE: synthetic graph (not a fly) - fine for testing only")
    rng = np.random.default_rng(0)
    rets, trans = rollout(cls, 8, rng, "random")
    say(f"random baseline: avg return {rets.mean():.3f}")
    obs = torch.as_tensor(np.stack([t[0] for t in trans[:64]]), device=dev)
    with torch.no_grad():
        logits, info = brain(obs)
        les, _ = brain(obs, lesion=True)
    p, pl = logits.softmax(-1), les.softmax(-1)
    say(f"brain ok: {brain.n:,} neurons, mean firing rate {info['mean_rate']:.3f}")
    if info["mean_rate"] < 1e-4:
        say("  WARNING: brain is almost silent; check your observations aren't all zero")
    sens, sens_l = p.std(0).max().item(), pl.std(0).max().item()
    assert sens_l < 1e-5, "lesioned brain still reacts to input - something bypasses the connectome"
    say(f"lesion test ok: intact brain reacts to input (spread {sens:.4f}), lesioned brain doesn't ({sens_l:.1e})")
    return brain, dev


def quick_train(cls, brain, dev, say):
    if cls.mode == "teacher":
        say("teacher mode: skipping quick train (run `python -m core.train` to collect demos)")
        return
    cfg = settings(cls, dict(updates=2, episodes=4))
    train_reward(brain, cls, cfg, dev, time.time() + 120, lambda m: None)
    say("quick train ok (2 updates)")


def check(name, size, device):
    lines = []
    say = lambda m: (print(f"  {m}"), lines.append(m))
    print(f"[{name}]")
    try:
        cls = load_task(name)
        check_interface(cls, say)
        brain, dev = check_brain(cls, size, device, say)
        quick_train(cls, brain, dev, say)
        print(f"  PASS")
        return True
    except (AssertionError, ValueError, SystemExit) as e:
        print(f"  FAIL: {e}")
    except Exception:
        traceback.print_exc()
        print("  FAIL (crash above)")
    return False


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("fly", nargs="?")
    ap.add_argument("--all", action="store_true")
    ap.add_argument("--size", default="tiny")
    ap.add_argument("--device", default="auto")
    a = ap.parse_args()
    names = fly_names() if a.all or not a.fly else [a.fly]
    results = {n: check(n, a.size, a.device) for n in names}
    print("\n" + "  ".join(f"{n}:{'PASS' if ok else 'FAIL'}" for n, ok in results.items()))
    raise SystemExit(0 if all(results.values()) else 1)


if __name__ == "__main__":
    main()
