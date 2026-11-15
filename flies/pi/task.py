"""Pi Fly: two masked descending-neuron decisions per completed digit guess."""
import numpy as np

from core.task import FlyTask
from flies.pi.reference import PI_DIGITS


class PiFly(FlyTask):
    name = "pi"
    actions = [f"choose_{i}" for i in range(8)]
    obs_dim = 40
    mode = "reward"
    greedy = False
    max_steps = 128             # 64 completed guesses; never cut a pair in half
    max_guesses = 64
    max_position_attempts = 20
    prefix_length = 32          # finite training curriculum, not all of pi
    start_index = 0             # evaluation can supply a known historical prefix
    correct_reward = 1.0
    streak_bonus = 0.1
    streak_cap = 10
    incorrect_reward = -0.2
    train = {"gamma": 0.995, "episodes": 16, "chunk": 64, "lr": 0.0003}

    def reset(self, rng):
        if not (0 <= self.start_index < self.prefix_length <= len(PI_DIGITS)):
            raise ValueError("invalid finite prefix interval")
        if self.max_steps < 2 or self.max_steps % 2 or min(self.max_guesses, self.max_position_attempts) < 1:
            raise ValueError("limits must permit complete two-stage guesses")
        if not (0 < self.correct_reward <= 2 and 0 <= self.streak_bonus <= .1
                and 0 <= self.streak_cap <= 10 and -1 <= self.incorrect_reward < 0):
            raise ValueError("reward settings must be finite and bounded")
        self.episode = getattr(self, "episode", 0) + 1
        self.index = self.start_index
        self.accepted = PI_DIGITS[:self.start_index]
        self.stage = 0
        self.selected_group = None
        self.selected_digit = None
        self.steps = self.guesses = self.correct = self.incorrect = 0
        self.streak = self.best_streak = self.attempts = 0
        self.total_reward = 0.0
        self.previous_guess = None
        self.previous_feedback = 0
        self.history = []
        self.first_attempts = self.first_attempt_correct = 0
        self.terminated = self.truncated = False
        self.last_event = None
        return self.obs()

    def mask(self):
        return [i < (2 if self.stage == 0 else 5) for i in range(8)]

    def obs(self):
        """Only public history/state. Never read PI_DIGITS here (README schema)."""
        scalar = [1., self.index / 1024, (self.index % 32) / 31,
                  (self.index // 32) / 32, float(self.stage),
                  0. if self.selected_group is None else (self.selected_group + 1) / 2,
                  self.attempts / self.max_position_attempts,
                  self.guesses / self.max_guesses, min(self.streak, 10) / 10,
                  min(self.best_streak, 10) / 10,
                  0. if self.previous_guess is None else (self.previous_guess + 1) / 10,
                  float(self.previous_feedback), self.correct / self.max_guesses,
                  self.incorrect / self.max_guesses, self.total_reward / (3 * self.max_guesses),
                  self.steps / self.max_steps]
        past = [0.] * (16 - min(len(self.accepted), 16))
        past += [(int(d) + 1) / 10 for d in self.accepted[-16:]]
        recent = [0.] * (2 * (4 - min(len(self.history), 4)))
        for digit, feedback in self.history[-4:]:
            recent.extend(((digit + 1) / 10, float(feedback)))
        return np.asarray(scalar + past + recent, dtype=np.float32)

    def step(self, action, rng):
        if self.terminated or self.truncated:
            raise RuntimeError("episode finished; reset before another decision")
        if not isinstance(action, (int, np.integer)) or isinstance(action, (bool, np.bool_)):
            raise ValueError("action must be an integer")
        if not 0 <= action < 8 or not self.mask()[action]:
            raise ValueError("masked or out-of-range action")
        self.steps += 1
        reward = 0.0
        if self.stage == 0:
            self.selected_group = int(action)
            self.selected_digit = None
            self.stage = 1
            kind = "group"
        else:
            self.selected_digit = self.selected_group * 5 + int(action)
            self.guesses += 1
            self.attempts += 1
            # The only current-target read: grade the completed neural choice.
            success = self.selected_digit == int(PI_DIGITS[self.index])
            self.previous_guess = self.selected_digit
            self.previous_feedback = 1 if success else -1
            self.history.append((self.selected_digit, self.previous_feedback))
            if self.attempts == 1:
                self.first_attempts += 1
                self.first_attempt_correct += int(success)
            if success:
                reward = self.correct_reward + self.streak_bonus * min(self.streak, self.streak_cap)
                self.accepted += str(self.selected_digit)
                self.index += 1
                self.correct += 1
                self.streak += 1
                self.best_streak = max(self.best_streak, self.streak)
                self.attempts = 0
            else:
                reward = self.incorrect_reward
                self.incorrect += 1
                self.streak = 0
            kind = "correct" if success else "incorrect"
            self.stage = 0
            self.terminated = self.index == self.prefix_length
            self.truncated = not self.terminated and (
                self.steps >= self.max_steps or self.guesses >= self.max_guesses
                or self.attempts >= self.max_position_attempts)
        self.total_reward += reward
        self.last_event = dict(version=1, episode=self.episode, transition=self.steps,
                               kind=kind, group=self.selected_group, digit=self.selected_digit,
                               index=self.index, accepted=self.accepted, reward=reward,
                               guesses=self.guesses, correct=self.correct, incorrect=self.incorrect,
                               streak=self.streak, best_streak=self.best_streak,
                               attempts=self.attempts, total_reward=self.total_reward,
                               terminated=self.terminated, truncated=self.truncated)
        return self.obs(), float(reward), self.terminated or self.truncated, dict(self.last_event)

    def caption(self, action):
        # Native transport is string-only: readable, bounded, versioned protocol.
        e = self.last_event
        if e is None:
            return "Pi Fly · waiting"
        choice = f"group {e['group']}" if e["kind"] == "group" else f"{e['digit']} {'✓' if e['kind'] == 'correct' else '✗'}"
        status = "complete" if e["terminated"] else "limit" if e["truncated"] else "live"
        return (f"π1 {e['episode']}:{e['transition']} | {choice} | "
                f"P{e['index']}:{e['accepted'][-6:] or '_'} | "
                f"S{e['streak']} W{e['incorrect']} R{e['total_reward']:.2f} | {status}")
