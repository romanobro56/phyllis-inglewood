"""FlyTask: the one class every fly implements. See flies/_template/task.py."""
import numpy as np


class FlyTask:
    name: str = None            # must match the folder name: flies/<name>/
    actions: list = []          # 2-8 action names; each gets a fixed group of descending neurons
    obs_dim: int = None         # length of the observation vector
    mode: str = "reward"        # "reward" (simulate + score) or "teacher" (copy an expert)
    max_steps: int = 100        # decisions per episode
    greedy: bool = False        # demo: always take the top choice instead of sampling
    train: dict = {}            # optional overrides, e.g. {"lr": 3e-4, "episodes": 64}

    def reset(self, rng: np.random.Generator) -> np.ndarray:
        """Start a new episode, return the first observation (float32, shape (obs_dim,))."""
        raise NotImplementedError

    def step(self, action: int, rng: np.random.Generator):
        """Apply the action. Return (obs, reward, done, info)."""
        raise NotImplementedError

    def mask(self):
        """Optional: list of bools, False = action not allowed right now. None = all allowed."""
        return None

    def teacher(self) -> int:
        """Teacher mode only: the action an expert would take in the current state."""
        raise NotImplementedError

    def caption(self, action: int) -> str:
        """Called right after step(); one short line shown in the 3D world."""
        return self.actions[action].replace("_", " ")
