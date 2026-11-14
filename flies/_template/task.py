"""TEMPLATE - copy with:  python scripts/new_fly.py <your_fly_name>

Fill in the five TODOs. Keep step() fast (pure numpy, no network calls):
it runs tens of thousands of times during training.
"""
import numpy as np

from core.task import FlyTask


class TemplateFly(FlyTask):
    name = "TEMPLATE_NAME"                 # must equal the folder name
    actions = ["option_a", "option_b"]     # TODO 1: 2-8 small, always-sensible choices
    obs_dim = 3                            # TODO 2: how many numbers the fly sees
    mode = "reward"                        # TODO 3: "reward" if you can score it, "teacher" if you can show it
    max_steps = 50                         # decisions per episode

    def reset(self, rng):
        # TODO 4: set up a fresh episode
        self.t = 0
        self.energy = 1.0
        return self.obs()

    def obs(self):
        # keep values roughly in 0..1
        return np.array([self.energy, self.t / self.max_steps, 1.0], dtype=np.float32)

    def step(self, action, rng):
        # TODO 5: apply the action, return (obs, reward, done, info)
        self.t += 1
        if self.actions[action] == "option_a":
            reward = self.energy
            self.energy = max(0.0, self.energy - 0.2)
        else:
            reward = 0.0
            self.energy = min(1.0, self.energy + 0.3)
        return self.obs(), reward, self.t >= self.max_steps, {}

    # Optional ---------------------------------------------------------------
    # def mask(self):           # False = not allowed right now
    #     return [True, self.energy > 0]
    #
    # def teacher(self):        # required for mode = "teacher"
    #     return 0 if self.energy > 0.5 else 1

    def caption(self, action):
        # one short line for the 3D world, called right after step()
        return f"picks {self.actions[action]} (energy {self.energy:.1f})"
