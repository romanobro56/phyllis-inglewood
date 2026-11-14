"""Eight paint choices, one fixed raster scan, no policy outside FlyBrain."""
import json
from pathlib import Path

import numpy as np

from core.task import FlyTask


_DATA = json.loads(Path(__file__).with_name("reference.json").read_text())
TARGET = np.asarray(_DATA["target"], dtype=np.float32) / 255
PALETTE = np.asarray(_DATA["palette"], dtype=np.float32) / 255
BLANK = np.asarray(_DATA["blank"], dtype=np.float32) / 255
for _array in (TARGET, PALETTE, BLANK):
    _array.setflags(write=False)


class PaintingFly(FlyTask):
    name = "painting"
    actions = [f"paint_{i}" for i in range(8)]
    obs_dim = 12
    mode = "reward"
    max_steps = 64
    # Immediate color feedback; avoid adding unrelated future-cell rewards.
    train = {"gamma": 0.0, "chunk": 128, "episodes": 16}

    def reset(self, rng):
        self.canvas = np.broadcast_to(BLANK, TARGET.shape).copy()
        self.t = 0
        self.error = float(np.mean((self.canvas - TARGET) ** 2))
        self.last_cell = None
        return self.obs()

    def obs(self):
        y, x = divmod(min(self.t, self.max_steps - 1), 8)
        left = self.canvas[y, x - 1] if x else BLANK
        return np.concatenate((TARGET[y, x], self.canvas[y, x],
                               [x / 7, y / 7, self.t / self.max_steps], left)).astype(np.float32)

    def mask(self):
        # Every color is always legal. Terminal states must be reset before step.
        return [True] * len(self.actions)

    def step(self, action, rng):
        if self.t >= self.max_steps:
            raise RuntimeError("Painting complete; call reset before step")
        if not isinstance(action, (int, np.integer)) or not 0 <= action < 8:
            raise ValueError("Expected a paint action index in 0..7")
        y, x = divmod(self.t, 8)
        old = float(np.mean((self.canvas[y, x] - TARGET[y, x]) ** 2))
        self.canvas[y, x] = PALETTE[action]
        new = float(np.mean((self.canvas[y, x] - TARGET[y, x]) ** 2))
        reward = old - new
        self.error -= reward / self.max_steps
        self.last_cell = (y, x)
        self.t += 1
        done = self.t == self.max_steps
        info = {"similarity": float(np.clip(1 - self.error, 0, 1)),
                "completion": self.t / self.max_steps, "cell": [y, x]}
        return self.obs(), reward, done, info

    def caption(self, action):
        # The existing server transports caption, but discards step info.
        return (f"daubs pigment {action + 1}; cell {self.t}/64; "
                f"similarity {100 * (1 - self.error):.2f}%")
