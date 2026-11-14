"""Bard fly: a fruit fly at a typewriter, writing Shakespeare one word at a time.

Each decision the fly faces a tray of 8 words drawn from a 29,749-word dictionary (every word
Shakespeare used twice or more, 20k common English words, and , . ? !). It picks one; that word
is written. The tray is drawn without looking at the text so far (mostly common words, plus a
couple of completely random dictionary words), so any sense in the sentence comes from the fly.

What the fly senses (obs, 60 floats):
    8 tray words x [freq, scent(5)]    how Shakespearean each word is + what it "smells" like
    last word     [freq, scent(5)]
    word before   scent(5)
    words since the last full stop
The scent of a word is fixed (see build_words.py); nothing in the observation says which word
fits next. Each action reads one tray slot from its own group of real descending neurons.

Training (teacher mode): walk through real Shakespeare; the tray hides the true next word among
7 decoys and the teacher points at it. The fly learns to tell a likely next word from a decoy.
Writing (free_run = True): all 8 tray words are random draws and the fly's picks become the text.
Sampling the brain's choice over such trays approximates a (very small) learned language model.
"""
from pathlib import Path

import numpy as np

from core.task import FlyTask

TRAY = 8
_Z = np.load(Path(__file__).resolve().parent / "words.npz")
VOCAB = [str(w) for w in _Z["vocab"]]
FREQ = _Z["freq"]
SCENT = _Z["scent"]
CORPUS = _Z["corpus"]
COUNT = _Z["count"].astype(np.float64)
SMELL = np.concatenate([FREQ[:, None], SCENT], 1).astype(np.float32)   # (V, 6)
ENDS = {VOCAB.index(p) for p in (".", "?", "!")}
STOP = VOCAB.index(".")

# tray distribution: 75% common Shakespeare-ish words (flattened frequency), 25% any dictionary word
_q = 0.75 * COUNT ** 0.75 / (COUNT ** 0.75).sum() + 0.25 / len(VOCAB)
TRAY_CDF = np.cumsum(_q / _q.sum())


def pretty(words):
    """Token list -> readable text: punctuation attached, sentences capitalised, 'i' -> 'I'."""
    out, cap = "", True
    for w in words:
        if w in (",", ".", "?", "!"):
            out = out.rstrip() + w + " "
            cap = cap or w != ","
            continue
        w = "I" + w[1:] if w == "i" or w.startswith("i'") else w
        out += (w[:1].upper() + w[1:] if cap else w) + " "
        cap = False
    return out.strip()


class BardFly(FlyTask):
    name = "bard"
    actions = [f"slot_{i}" for i in range(TRAY)]
    obs_dim = TRAY * 6 + 6 + 5 + 1
    mode = "teacher"
    max_steps = 40
    free_run = False      # True: write freely (world/demo). False: walk real Shakespeare (training)
    train = {"teacher_episodes": 1500, "teacher_epsilon": 0.0, "epochs": 12}
    world_args = {"free_run": True, "words": 60}   # the 3D world: write freely, a 60-word page at a time

    def __init__(self, free_run=None, words=None):
        if free_run is not None:
            self.free_run = free_run
        if words is not None:
            self.max_steps = words

    def view(self):
        """For the world's typewriter: the 8 words on the tray and the page so far."""
        return {"tray": [VOCAB[int(i)] for i in self.tray], "page": pretty(self.text) if self.free_run else ""}

    def reset(self, rng):
        self.rng = rng
        self.t = 0
        if self.free_run:
            self.text = []
            self.ctx = [STOP, STOP]
            self.since_stop = 0
        else:
            self.pos = int(rng.integers(32, len(CORPUS) - self.max_steps - 2))
            self.ctx = [int(CORPUS[self.pos - 2]), int(CORPUS[self.pos - 1])]
            back = CORPUS[self.pos - 30:self.pos][::-1]
            hits = [i for i, w in enumerate(back) if int(w) in ENDS]
            self.since_stop = hits[0] if hits else 30
        self.deal()
        return self.obs()

    def deal(self):
        tray = np.searchsorted(TRAY_CDF, self.rng.random(TRAY)).clip(0, len(VOCAB) - 1)
        self.answer = -1
        if not self.free_run:
            true = int(CORPUS[self.pos])
            for i in range(TRAY):
                while tray[i] == true:
                    tray[i] = np.searchsorted(TRAY_CDF, self.rng.random())
            self.answer = int(self.rng.integers(TRAY))
            tray[self.answer] = true
        self.tray = tray

    def obs(self):
        o = np.concatenate([SMELL[self.tray].ravel(), SMELL[self.ctx[1]], SCENT[self.ctx[0]],
                            [min(self.since_stop / 15, 1.0)]])
        return o.astype(np.float32)

    def teacher(self):
        return self.answer

    def step(self, action, rng):
        self.word = VOCAB[int(self.tray[action])]
        reward = float(action == self.answer) if not self.free_run else 0.0
        nxt = int(self.tray[action]) if self.free_run else int(CORPUS[self.pos])
        if self.free_run:
            self.text.append(self.word)
        else:
            self.pos += 1
        self.ctx = [self.ctx[1], nxt]
        self.since_stop = 0 if nxt in ENDS else self.since_stop + 1
        self.t += 1
        self.deal()
        return self.obs(), reward, self.t >= self.max_steps, {}

    def caption(self, action):
        return f'types "{self.word}"'
