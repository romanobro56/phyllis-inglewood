"""Watch the bard fly write, one word per decision, straight from its trained brain.

    python flies/bard/write.py                      # 60 words, 1 s between words, shows each tray
    python flies/bard/write.py --words 200 --cooldown 0 --quiet
    python flies/bard/write.py --lesion             # same, with the connectome zeroed (control)

Each line shows the 8-word tray the fly was dealt, the brain's probability for each slot (the
mean firing of that slot's descending neurons, softmaxed), and the word it picked (sampled from
those probabilities, as in the world).
"""
import argparse
import sys
import time
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from core.brain import load_brain  # noqa: E402
from core.registry import load_task, run_dir  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--words", type=int, default=60)
    ap.add_argument("--cooldown", type=float, default=1.0)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--lesion", action="store_true")
    ap.add_argument("--quiet", action="store_true", help="don't print the trays")
    a = ap.parse_args()

    cls = load_task("bard")
    VOCAB, pretty = cls.reset.__globals__["VOCAB"], cls.reset.__globals__["pretty"]
    cls.free_run = True
    brain = load_brain(run_dir("bard") / "brain.pt", cls).eval()
    rng = np.random.default_rng(a.seed)
    env = cls()
    env.max_steps = a.words
    obs = env.reset(rng)
    print(f"bard fly, {brain.preset} brain ({brain.n:,} neurons){' LESIONED' if a.lesion else ''}\n")
    for t in range(a.words):
        t0 = time.time()
        with torch.no_grad():
            logits, _ = brain(torch.as_tensor(obs)[None], lesion=a.lesion)
        p = logits.softmax(-1)[0].numpy().astype(np.float64)
        act = int(rng.choice(len(p), p=p / p.sum()))
        tray = [VOCAB[i] for i in env.tray]
        obs, _, _, _ = env.step(act, rng)
        if not a.quiet:
            opts = "  ".join(f"{w}:{q:.2f}" for w, q in zip(tray, p))
            print(f"{t + 1:3d}  [{opts}]")
            print(f"     -> {env.word!r:14}  so far: ...{pretty(env.text)[-70:]}")
        time.sleep(max(0.0, a.cooldown - (time.time() - t0)))
    print("\n" + pretty(env.text))


if __name__ == "__main__":
    main()
