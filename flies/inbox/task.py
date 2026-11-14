"""Inbox fly: triages email. Reference fly for teacher mode (copies a rule-based expert)."""
import numpy as np

from core.task import FlyTask

SENDERS = ["boss", "friend", "newsletter", "stranger", "mom"]


class InboxFly(FlyTask):
    name = "inbox"
    actions = ["archive", "reply", "flag", "unsubscribe"]
    obs_dim = 9
    mode = "teacher"
    max_steps = 30

    def reset(self, rng):
        self.rng = rng
        self.handled = 0
        self.new_email()
        return self.obs()

    def new_email(self):
        r = self.rng
        self.sender = int(r.integers(len(SENDERS)))
        self.urgent = float(r.random() < 0.3)
        self.question = float(r.random() < 0.4)
        self.spammy = float(SENDERS[self.sender] in ("newsletter", "stranger") and r.random() < 0.7)

    def obs(self):
        onehot = np.eye(len(SENDERS))[self.sender]
        return np.concatenate([onehot, [self.urgent, self.question, self.spammy, self.handled / 30]]).astype(np.float32)

    def mask(self):
        # can only unsubscribe from newsletters
        return [True, True, True, SENDERS[self.sender] == "newsletter"]

    def teacher(self):
        s = SENDERS[self.sender]
        if s == "newsletter" and self.spammy:
            return 3
        if self.urgent and s == "boss":
            return 2
        if self.question and s in ("boss", "friend", "mom"):
            return 1
        return 0

    def step(self, action, rng):
        reward = 1.0 if action == self.teacher() else -0.5
        self.last = (SENDERS[self.sender], action)
        self.handled += 1
        self.new_email()
        return self.obs(), reward, self.handled >= self.max_steps, {}

    def caption(self, action):
        sender, a = self.last
        verb = ["archives", "replies to", "flags", "unsubscribes from"][a]
        return f"{verb} the email from {sender}"
