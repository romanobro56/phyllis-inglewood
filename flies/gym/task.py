"""Gym fly: alternates between bench press and squats. Reference fly for reward mode."""
import numpy as np

from core.task import FlyTask


class GymFly(FlyTask):
    name = "gym"
    actions = ["do_rep", "switch_station"]
    obs_dim = 6
    mode = "reward"
    max_steps = 60

    STATIONS = ["bench", "squat rack"]

    def reset(self, rng):
        self.station = 0
        self.reps = 0
        self.total_reps = 0
        self.t = 0
        self.fatigue = np.zeros(2)
        return self.obs()

    def obs(self):
        return np.array([self.station == 0, self.station == 1, self.reps / 15,
                         self.fatigue[0], self.fatigue[1], self.t / self.max_steps], dtype=np.float32)

    def step(self, action, rng):
        self.t += 1
        self.failed = False
        if self.actions[action] == "do_rep":
            f = self.fatigue[self.station]
            if rng.random() < f ** 3:                  # exhausted: failed rep
                reward = -2.0
                self.failed = True
            else:
                reward = 1.0 - f                       # gains shrink as the muscle tires
                self.reps += 1
                self.total_reps += 1
            self.fatigue[self.station] = min(1.0, f + 0.1)
        else:                                          # walk to the other station
            self.station, self.reps = 1 - self.station, 0
            reward = -0.3                              # walking costs time
        self.fatigue = np.maximum(0.0, self.fatigue - 0.03)   # recovery
        return self.obs(), reward, self.t >= self.max_steps, {}

    def caption(self, action):
        if action == 0:
            return "fails the rep!" if self.failed else f"rep #{self.reps} on the {self.STATIONS[self.station]}"
        return f"heads to the {self.STATIONS[self.station]}"
