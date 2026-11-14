"""Gym fly (Marlo Inglewood): bench press and squats, with rest between sets. Reference fly for reward mode.

Each decision: do a rep, walk to the other station, or rest 1-5 minutes (doing another rep straight away is
the "0 minutes" option). Resting lets the muscles recover; the world plays rests 20x faster than real time.
What gets rewarded most: sets of 5-10 reps, and 2-4 minute rests between them. A rest only pays after a
real set (5+ reps); resting after a token rep costs reward, so "1 rep, nap, 1 rep, nap" doesn't pay.
"""
import numpy as np

from core.task import FlyTask

REST_MINUTES = [1, 2, 3, 4, 5]


class GymFly(FlyTask):
    name = "gym"
    actions = ["do_rep", "switch_station"] + [f"rest_{m}min" for m in REST_MINUTES]
    obs_dim = 9
    mode = "reward"
    max_steps = 60

    STATIONS = ["bench", "squat rack"]
    FATIGUE_PER_REP = 0.07     # fresh muscle: ~8 reps before failure becomes likely
    RECOVERY_MINUTES = 1.6     # fatigue decays as exp(-minutes / this) while resting

    def reset(self, rng):
        self.station = 0
        self.reps = 0              # reps in the current set
        self.total_reps = 0
        self.sets = 0
        self.t = 0
        self.fatigue = np.zeros(2)
        self.last_rest = 0         # minutes of the last rest (0 = none since the set started)
        self.failed = False
        self.last_set = 0
        return self.obs()

    def obs(self):
        return np.array([self.station == 0, self.station == 1, min(self.reps, 12) / 12,
                         self.fatigue[0], self.fatigue[1], self.t / self.max_steps,
                         self.reps > 0, self.last_rest / 5, self.failed], dtype=np.float32)

    def mask(self):
        # resting only makes sense after a set; there is always something else to do
        rest_ok = self.reps > 0
        return [True, True] + [rest_ok] * len(REST_MINUTES)

    @staticmethod
    def set_bonus(reps):
        if 5 <= reps <= 10:
            return 0.6 * reps                          # the sweet spot: 3.0 for 5 reps ... 6.0 for 10
        if 3 <= reps <= 4:
            return 0.3
        if reps > 10:
            return 0.5                                 # junk volume
        return 0.0                                     # 1-2 reps isn't a set

    @staticmethod
    def rest_bonus(minutes, reps):
        if reps < 5:
            return -1.0                                # a rest you didn't earn (no 1-rep-then-nap loop)
        if 2 <= minutes <= 4:
            return 1.5                                 # the sweet spot
        return 0.4 if minutes == 5 else 0.2            # too long / too short

    def step(self, action, rng):
        self.t += 1
        self.failed = False
        name = self.actions[action]
        if name == "do_rep":
            f = self.fatigue[self.station]
            if rng.random() < np.clip((f - 0.55) * 2.2, 0, 1):   # exhausted: failed rep
                reward = -2.0
                self.failed = True
            else:
                self.reps += 1
                self.total_reps += 1
                reward = 0.3 if self.reps <= 10 else -0.4        # past 10 reps it stops paying off
            self.fatigue[self.station] = min(1.0, f + self.FATIGUE_PER_REP)
            self.last_rest = 0
        elif name == "switch_station":                           # ends the set with no rest (a superset)
            reward = self.set_bonus(self.reps) - 0.3              # walking costs time
            self.end_set()
            self.station = 1 - self.station
            self.last_rest = 0
        else:                                                     # rest_<m>min, always right after a set
            m = REST_MINUTES[action - 2]
            reward = self.set_bonus(self.reps) + self.rest_bonus(m, self.reps)
            self.end_set()
            self.fatigue *= np.exp(-m / self.RECOVERY_MINUTES)
            self.last_rest = m
        self.fatigue = np.maximum(0.0, self.fatigue - 0.01)        # a little recovery every decision
        return self.obs(), reward, self.t >= self.max_steps, {}

    def end_set(self):
        self.last_set = self.reps
        if self.reps:
            self.sets += 1
        self.reps = 0

    def caption(self, action):
        name = self.actions[action]
        if name == "do_rep":
            return "fails the rep!" if self.failed else f"rep #{self.reps} on the {self.STATIONS[self.station]}"
        if name == "switch_station":
            return f"heads to the {self.STATIONS[self.station]}"
        return f"rests {REST_MINUTES[action - 2]} min after a {self.last_set}-rep set"
