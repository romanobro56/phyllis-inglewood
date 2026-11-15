import json
import unittest
import chess
import numpy as np
from core.registry import load_task
from flies.chess.chess_env import ChessEnv
from flies.chess.task import ChessFly


def move(env, uci):
    m = chess.Move.from_uci(uci)
    coords = env.coordinates(m)
    event = None
    for a in coords:
        assert env.mask().shape == (8,) and env.mask().any()
        event = env.select(a)
    assert event is not None
    return event


class EnvironmentTests(unittest.TestCase):
    def test_contract(self):
        self.assertEqual(load_task("chess").__name__, "ChessFly")
        rng = np.random.default_rng(42)
        task = ChessFly()
        obs = task.reset(rng)
        for _ in range(task.max_steps):
            self.assertEqual(obs.shape, (64,))
            self.assertEqual(obs.dtype, np.float32)
            self.assertTrue(np.isfinite(obs).all())
            action = int(rng.choice(np.flatnonzero(task.mask())))
            obs, reward, done, info = task.step(action, rng)
            data = info                                  # board state travels as step info (ctx.info in the world)
            self.assertNotIn("|CHESS|", task.caption(action))
            self.assertEqual(data["fen"], task.env.board.fen())
            self.assertTrue(np.isfinite(reward))
            if done:
                break
        self.assertTrue(done)
        self.assertLessEqual(task.t, task.max_steps)
        if info["outcome"]["truncated"]:
            self.assertEqual(info["outcome"]["result"], "*")

    def test_all_start_moves_and_masks_exact(self):
        for m in chess.Board().legal_moves:
            env = ChessEnv()
            expected = chess.Board()
            event = move(env, m.uci())
            expected.push(m)
            self.assertEqual(event["fen"], expected.fen())
        rng = np.random.default_rng(9)
        env = ChessEnv(max_plies=300, claim_draw=False)
        while not env.done:
            expected = np.zeros(8, bool)
            for m in env.candidates():
                expected[env.coordinates(m)[env.stage]] = True
            np.testing.assert_array_equal(env.mask(), expected)
            env.select(int(rng.choice(np.flatnonzero(expected))))
        self.assertLessEqual(env.plies, 300)
        self.assertEqual(env.illegal_attempts, 0)

    def test_special_moves(self):
        cases = [
            ("r3k2r/8/8/8/8/8/8/R3K2R w KQkq - 0 1", "e1g1"),
            ("r3k2r/8/8/8/8/8/8/R3K2R w KQkq - 0 1", "e1c1"),
            ("r3k2r/8/8/8/8/8/8/R3K2R b KQkq - 0 1", "e8g8"),
            ("r3k2r/8/8/8/8/8/8/R3K2R b KQkq - 0 1", "e8c8"),
            ("7k/8/8/3pP3/8/8/8/7K w - d6 0 1", "e5d6"),
            ("7k/8/8/8/3Pp3/8/8/7K b - d3 0 1", "e4d3"),
        ]
        for piece in "qrbn":
            cases.extend([("7k/P7/8/8/8/8/8/7K w - - 0 1", "a7a8" + piece),
                          ("7k/8/8/8/8/8/p7/7K b - - 0 1", "a2a1" + piece)])
        for fen, uci in cases:
            with self.subTest(uci=uci):
                env, expected = ChessEnv(fen), chess.Board(fen)
                expected.push_uci(uci)
                self.assertEqual(move(env, uci)["fen"], expected.fen())

    def test_checks_pins_and_single_move(self):
        for fen in ["4r2k/8/8/8/8/8/4R3/4K3 w - - 0 1",  # pinned rook
                    "8/8/8/8/8/6k1/7r/7K w - - 0 1",  # check, only Kg1
                    "4r2k/8/8/8/8/8/8/4K3 w - - 0 1"]:
            env = ChessEnv(fen)
            legal = list(env.board.legal_moves)
            self.assertTrue(legal)
            for m in legal:
                self.assertEqual(move(ChessEnv(fen), m.uci())["uci"], m.uci())
            if "7r/7K" in fen:
                self.assertEqual(len(legal), 1)
        env = ChessEnv("4r2k/8/8/8/8/8/4R3/4K3 w - - 0 1")
        with self.assertRaises(ValueError):
            env.push(chess.Move.from_uci("e2f2"))
        self.assertEqual(env.illegal_attempts, 1)

    def test_terminal_and_timeout(self):
        for fen, reason in [("7k/6Q1/5K2/8/8/8/8/8 b - - 0 1", "CHECKMATE"),
                            ("7k/5K2/6Q1/8/8/8/8/8 b - - 0 1", "STALEMATE"),
                            ("7k/8/8/8/8/8/8/K7 w - - 0 1", "INSUFFICIENT_MATERIAL")]:
            env = ChessEnv(fen)
            self.assertTrue(env.done)
            self.assertFalse(env.mask().any())
            self.assertEqual(env.outcome()["termination"], reason)
            with self.assertRaises(RuntimeError):
                env.select(0)
        env = ChessEnv(max_plies=1)
        move(env, "e2e4")
        self.assertEqual(env.outcome()["result"], "*")
        self.assertTrue(env.truncated)

    def test_draw_claim_policies(self):
        cycle = ["g1f3", "g8f6", "f3g1", "f6g8"]
        env = ChessEnv(claim_draw=False)
        for uci in cycle * 4:
            move(env, uci)
        self.assertEqual(env.outcome()["termination"], "FIVEFOLD_REPETITION")
        env = ChessEnv(claim_draw=True)
        for uci in cycle * 2:
            if env.done:
                break
            move(env, uci)
        self.assertEqual(env.outcome()["termination"], "THREEFOLD_REPETITION")
        for clock, claim, terminal, reason in [(100, True, True, "FIFTY_MOVES"),
                                              (100, False, False, None),
                                              (150, False, True, "SEVENTYFIVE_MOVES")]:
            env = ChessEnv(f"7k/8/8/8/8/8/R7/K7 w - - {clock} 80", claim_draw=claim)
            self.assertEqual(env.done, terminal)
            self.assertEqual(env.outcome()["termination"], reason)

    def test_rejects_invalid_actions(self):
        env = ChessEnv()
        for action in (-1, 8, 1.5):
            with self.assertRaises(ValueError):
                env.select(action)
        self.assertEqual(env.choices, [])
        env.select(0)
        with self.assertRaises(ValueError):
            env.select(7)  # a8 is not a White legal source at the initial position
        self.assertEqual(env.choices, [0])
