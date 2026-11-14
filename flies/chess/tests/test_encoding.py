import unittest
import chess
import numpy as np
from flies.chess.chess_env import ChessEnv, canonical_square, decode_planes
from flies.chess.tests.test_env import move


class EncodingTests(unittest.TestCase):
    def test_roundtrip_planes(self):
        rng = np.random.default_rng(6)
        env = ChessEnv(claim_draw=False)
        for _ in range(40):
            obs = env.observe()
            planes = decode_planes(obs)
            self.assertEqual(planes.shape, (12, 64))
            self.assertEqual(planes.sum(), len(env.board.piece_map()))
            self.assertTrue(((0 <= obs) & (obs <= 1)).all())
            for sq, p in env.board.piece_map().items():
                channel = p.piece_type - 1 + (0 if p.color == env.board.turn else 6)
                self.assertEqual(planes[channel, canonical_square(sq, env.board.turn)], 1)
            if env.done:
                break
            moves = tuple(env.board.legal_moves)
            move(env, moves[int(rng.integers(len(moves)))].uci())

    def test_black_inverse_and_mirror(self):
        for side in (chess.WHITE, chess.BLACK):
            for sq in chess.SQUARES:
                self.assertEqual(canonical_square(canonical_square(sq, side), side), sq)
        w = ChessEnv("r3k2r/pp6/8/8/8/8/PP6/R3K2R w KQkq - 0 1")
        b = ChessEnv(w.board.mirror().fen())
        np.testing.assert_array_equal(w.observe()[:48], b.observe()[:48])
        np.testing.assert_array_equal(w.mask(), b.mask())
        self.assertEqual(w.coordinates(chess.Move.from_uci("a2a4")),
                         b.coordinates(chess.Move.from_uci("a7a5")))
        for action in w.coordinates(chess.Move.from_uci("a2a4"))[:3]:
            w.select(action); b.select(action)
            np.testing.assert_array_equal(w.mask(), b.mask())
            np.testing.assert_array_equal(w.observe()[32:48], b.observe()[32:48])

    def test_metadata_order(self):
        env = ChessEnv("r3k2r/8/8/3pP3/8/8/8/R3K2R w KQkq d6 7 20")
        obs = env.observe()
        np.testing.assert_array_equal(obs[32:37], [1, 0, 0, 0, 0])
        np.testing.assert_array_equal(obs[41:45], [1, 1, 1, 1])
        self.assertAlmostEqual(float(obs[45]), (chess.D6 + 1) / 64)
        self.assertAlmostEqual(float(obs[47]), 7 / 150)
        env.select(4)
        self.assertEqual(env.observe()[37], 5 / 8)
