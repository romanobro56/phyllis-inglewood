"""Native demo: FlyBrain White versus uniformly random legal Black.

Genuine two-FlyBrain matches live in selfplay.py, never in step().
"""

from core.task import FlyTask
from flies.chess.chess_env import ACTIONS, OBS_DIM, STAGES, ChessEnv


class ChessFly(FlyTask):
    name = "chess"
    actions = ACTIONS
    obs_dim = OBS_DIM
    mode = "reward"
    max_steps = 80  # neural decisions, including incomplete coordinate selections
    train = {"chunk": 128}

    def reset(self, rng):
        self.env = ChessEnv(max_plies=40)
        self.t = 0
        self.last = dict(fen=self.env.board.fen(), events=[], mode="fly-vs-random-demo")
        return self.env.observe()

    def mask(self):
        return self.env.mask()

    def step(self, action, rng):
        # Engineered reward shaping only; never used to select or rank moves.
        values = (0, 1, 3, 3, 5, 9, 0)
        def material():
            return sum(values[p.piece_type] * (1 if p.color else -1)
                       for p in self.env.board.piece_map().values())
        before_material = material()
        event = self.env.select(action)
        self.t += 1
        events = []
        if event:
            event["controller"] = "FlyBrain"
            events.append(event)
            if not self.env.done:
                moves = tuple(self.env.board.legal_moves)
                reply = self.env.push(moves[int(rng.integers(len(moves)))])
                reply["controller"] = "uniform-random-demo-opponent"
                events.append(reply)
        if not self.env.done and self.t >= self.max_steps:
            self.env.truncated = True
        outcome = self.env.outcome()
        terminal_reward = (1.0 if outcome["winner"] == "white" else -1.0 if outcome["winner"] == "black" else 0.0)
        shaping = max(-.1, min(.1, (material() - before_material) / 100)) if event else 0.0
        reward = terminal_reward if self.env.done and not self.env.truncated else shaping
        self.last = dict(fen=self.env.board.fen(), events=events, stage=STAGES[self.env.stage],
                         mode="fly-vs-random-demo", outcome=outcome, decisions=self.t,
                         partial=self.env.choices.copy(), material_shaping=shaping,
                         terminal_reward=terminal_reward)
        return self.env.observe(), reward, self.env.done, self.last.copy()

    def caption(self, action):
        moves = " / ".join(e["san"] for e in self.last["events"])
        text = moves or f"chooses {self.actions[action]}"
        return f"Chess vs random: {text}"   # the board state reaches the station as step info (ctx.info)
