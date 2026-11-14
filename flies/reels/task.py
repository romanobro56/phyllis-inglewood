"""Reels fly (Scrolly McScrollface): a fruit fly doomscrolling short videos on a giant phone.

Each reel is a cat video, a cooking video, a dance, a fail compilation, a lecture or an ad. For every reel
the fly can double-tap to like it, open the comments (open, scroll down a set amount, close), or keep
watching for 1, 3, 6 or 10 more seconds and then swipe to the next one. Like and comments are once per reel.

What gets rewarded:
- watching good stuff (and finishing a great reel); every second on junk (lectures, ads) costs a little,
  and rewatching a loop only pays for great reels.
- liking reels it enjoys. That also trains "the algorithm": the feed serves more of what the fly likes.
  Liking junk is cringe and teaches the algorithm the wrong thing.
- opening the comments when they're spicy (lots of comments, fails and dances start the best fights).
  Reading takes 3 s of the reel, and ad comments are all bots.
Boredom builds up on bad content and dulls even good reels, so scrolling past junk fast matters.
"""
import numpy as np

from core.task import FlyTask

CATS = ["cat", "cooking", "dance", "fail", "lecture", "ad"]
TASTE = np.array([0.95, 0.65, 0.5, 0.8, 0.15, 0.02])     # how much a fruit fly enjoys each kind of reel
DRAMA = np.array([0.05, 0.15, 0.25, 0.35, 0.0, -0.3])    # how spicy each kind's comment section gets
FEED = np.array([0.15, 0.17, 0.17, 0.15, 0.18, 0.18])    # what an untuned feed serves
WATCH = [1, 3, 6, 10]                                    # "scroll after N seconds" options
COMMENT_SECONDS = 3


class ReelsFly(FlyTask):
    name = "reels"
    actions = ["like", "comments"] + [f"scroll_{s}s" for s in WATCH]
    obs_dim = 16
    mode = "reward"
    max_steps = 40

    def reset(self, rng):
        self.rng = rng
        self.t = 0
        self.tune = 0.0           # 0 = default feed, 1 = the algorithm knows exactly what the fly likes
        self.boredom = 0.2
        self.seen = 0
        self.last = {}
        self.new_reel()
        return self.obs()

    def new_reel(self):
        r = self.rng
        p = FEED * (1 - self.tune) + self.tune * TASTE / TASTE.sum()
        self.cat = int(r.choice(len(CATS), p=p / p.sum()))
        self.quality = float(np.clip(TASTE[self.cat] + r.normal(0, 0.15), 0, 1))
        self.length = int(r.integers(5, 16))                 # seconds
        self.watched = 0.0
        self.likes = float(np.clip(0.5 * self.quality + r.normal(0.25, 0.15), 0, 1))     # log-ish social proof
        self.n_comments = float(np.clip(r.beta(2, 3) + 0.3 * DRAMA[self.cat], 0, 1))
        self.spicy = float(np.clip(0.7 * self.n_comments + DRAMA[self.cat] + r.normal(0, 0.1), 0, 1))
        self.liked = False
        self.opened = False
        self.seen += 1

    def obs(self):
        seen_q = np.clip(self.quality + self.rng.normal(0, 0.06), 0, 1)                 # what it looks like
        return np.concatenate([np.eye(len(CATS))[self.cat],
                               [seen_q, self.length / 15, min(self.watched / self.length, 1.5) / 1.5,
                                self.likes, self.n_comments, self.liked, self.opened,
                                self.boredom, self.tune, self.t / self.max_steps]]).astype(np.float32)

    def mask(self):
        return [not self.liked, not self.opened] + [True] * len(WATCH)

    def enjoy(self):
        return self.quality * (1 - 0.5 * self.boredom)

    def step(self, action, rng):
        self.t += 1
        name = self.actions[action]
        q = self.enjoy()
        if name == "like":
            self.liked = True
            good = q > 0.55
            reward = 0.6 if good else -0.4
            self.tune = float(np.clip(self.tune + (0.12 if good else -0.12), 0, 1))
            self.last = dict(kind="like", good=good, cat=CATS[self.cat])
        elif name == "comments":
            self.opened = True
            reward = 1.6 * self.spicy - 0.6
            self.watched += COMMENT_SECONDS
            self.last = dict(kind="comments", spicy=self.spicy, cat=CATS[self.cat])
        else:
            s = WATCH[action - 2]
            fresh = min(s, max(0.0, self.length - self.watched))      # first watch
            loop = s - fresh                                          # rewatching the loop
            reward = 0.25 * fresh * (q - 0.4) + 0.25 * loop * (q - 0.75)
            finished = self.watched + s >= self.length
            if finished and self.watched < self.length and q > 0.6:
                reward += 0.8                                         # watched a great reel to the end
            self.boredom = float(np.clip(self.boredom + 0.03 * s * (0.55 - self.quality), 0, 1))
            reward -= 0.05                                            # every swipe costs a little
            prev = dict(cat=CATS[self.cat], len=self.length, watched=self.watched, q=q)
            self.watched += s
            self.new_reel()
            self.last = dict(kind="scroll", s=s, finished=finished, loop=loop > 0, prev=prev,
                             next=CATS[self.cat], next_len=self.length)
        return self.obs(), float(reward), self.t >= self.max_steps, {}

    def caption(self, action):
        L = self.last
        if L["kind"] == "like":
            return f"likes the {L['cat']} reel" if L["good"] else f"hate-likes the {L['cat']} reel"
        if L["kind"] == "comments":
            mood = "drama!" if L["spicy"] > 0.6 else "meh" if L["spicy"] > 0.3 else "all bots"
            return f"reads the comments on the {L['cat']} reel ({mood})"
        p, nxt = L["prev"], f"swipes to a {L['next_len']}s {L['next']} reel"
        if L["loop"] and p["q"] > 0.6:
            return f"rewatches the {p['cat']} reel ({L['s']}s), {nxt}"
        if L["s"] == 1 and p["q"] < 0.3:
            return f"nope: skips the {p['cat']} reel, {nxt}"
        return f"watches {L['s']}s of the {p['cat']} reel, {nxt}"
