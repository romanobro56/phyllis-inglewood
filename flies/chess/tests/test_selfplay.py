import copy
import io
import os
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import chess
import chess.pgn
import numpy as np
import torch
from core.brain import FlyBrain
from flies.chess.chess_env import ChessEnv, OBS_DIM
from flies.chess.selfplay import play_game, to_pgn, choose_stage, assert_independent
from flies.chess.storage import (ROOT, atomic_write, brain_payload, identity,
                                 load_policy, validate_and_load, state_hash)
from flies.chess.train_selfplay import policy_update, train
from flies.chess.tests.support import BrainTest


class SelfplayTests(BrainTest):
    def test_two_separate_forward_calls_and_replay(self):
        white, black = self.brain(), self.brain()
        with patch.object(white, "forward", wraps=white.forward) as w, patch.object(black, "forward", wraps=black.forward) as b:
            result, trace = play_game(white, black, seed=4, max_plies=6, activity=True)
            self.assertEqual(w.call_count, 12)
            self.assertEqual(b.call_count, 12)
        self.assertEqual(len(trace["white"]), 3)
        self.assertEqual(len(trace["black"]), 3)
        self.assertEqual(result["illegal_attempts"], 0)
        self.assertTrue(result["outcome"]["truncated"])
        replay = chess.pgn.read_game(io.StringIO(to_pgn(result)))
        self.assertEqual(replay.end().board().fen(), result["final_fen"])
        again, _ = play_game(white, black, seed=4, max_plies=6)
        self.assertEqual([m["uci"] for m in result["moves"]], [m["uci"] for m in again["moves"]])
        self.assertTrue(all(m["controller"] == "FlyBrain" for m in result["moves"]))
        self.assertTrue(all(len(m["stages"]) == 4 for m in result["moves"]))

    def test_gradient_isolation_and_clone(self):
        white, black = self.brain(), self.brain()
        validate_and_load(black, brain_payload(white))
        assert_independent(white, black)
        before = state_hash(black)
        game, trace = play_game(white, black, seed=2, max_plies=4)
        opt = torch.optim.Adam(white.parameters(), lr=.001)
        metric, _ = policy_update(white, opt, [(trace["white"], 1.0)], baseline=0)
        self.assertGreater(metric["grad_norm"], 0)
        self.assertTrue(white.log_w.grad.abs().sum() > 0)
        self.assertNotEqual(state_hash(white), before)
        self.assertEqual(state_hash(black), before)
        self.assertTrue(all(p.grad is None for p in black.parameters()))
        for key in ("edge_index", "sign", "groups", "input_idx"):
            self.assertTrue(torch.equal(getattr(white, key), getattr(black, key)))
        with self.assertRaises(ValueError):
            play_game(white, white, max_plies=2)

    def test_checkpoint_roundtrip_and_compatibility(self):
        brain = self.brain()
        with tempfile.TemporaryDirectory(dir=ROOT) as folder:
            path = Path(folder) / "brain.pt"
            atomic_write(path, brain_payload(brain), binary=True)
            other = load_policy(path, size="test")
            self.assertEqual(state_hash(brain), state_hash(other))
            self.assertEqual(identity(other, path)["label"], "TRAINED")
            for field, value in [("graph_hash", "bad"), ("actions", ["bad"]), ("encoding", "bad"), ("obs_dim", 63)]:
                ck = copy.deepcopy(brain_payload(brain)); ck[field] = value
                with self.assertRaises(ValueError):
                    validate_and_load(other, ck)
            ck = copy.deepcopy(brain_payload(brain)); ck["state_dict"]["sign"] = other.sign.clone()
            ck["state_dict"]["sign"][0] *= -1          # a checkpoint that carries tampered wiring is still refused
            with self.assertRaisesRegex(ValueError, "fixed biological"):
                validate_and_load(other, ck)

    def test_atomic_interrupt_preserves_checkpoint(self):
        with tempfile.TemporaryDirectory(dir=ROOT) as folder:
            path = Path(folder) / "checkpoint.pt"
            atomic_write(path, {"value": 1}, binary=True)
            with patch("torch.save", side_effect=KeyboardInterrupt):
                with self.assertRaises(KeyboardInterrupt):
                    atomic_write(path, {"value": 2}, binary=True)
            self.assertEqual(torch.load(path, weights_only=False)["value"], 1)
            self.assertEqual([p.name for p in Path(folder).iterdir()], ["checkpoint.pt"])

    def test_resume_exact_continuation(self):
        with tempfile.TemporaryDirectory(dir=ROOT) as folder:
            full, resumed = Path(folder) / "full", Path(folder) / "resumed"
            kwargs = dict(size="test", seed=7, max_plies=4, games_per_update=2, snapshot_every=1)
            brains, _ = train(full, updates=2, **kwargs)
            train(resumed, updates=1, **kwargs)
            continued, _ = train(resumed, updates=2, resume=resumed / "resume.pt", **kwargs)
            for a, b in zip(brains, continued):
                self.assertEqual(state_hash(a), state_hash(b))
            ck = torch.load(resumed / "resume.pt", weights_only=False)
            self.assertEqual(ck["update"], 2)
            self.assertEqual(len(ck["optimizers"]), 2)
            old = copy.deepcopy(ck)
            old["config"].pop("algorithm")
            atomic_write(resumed / "old_algorithm.pt", old, binary=True)
            with self.assertRaisesRegex(ValueError, "learning algorithm"):
                train(resumed, updates=3, resume=resumed / "old_algorithm.pt", **kwargs)
            with self.assertRaisesRegex(ValueError, "configuration"):
                train(resumed, updates=3, resume=resumed / "resume.pt", **dict(kwargs, max_plies=5))

    def test_lesion_is_actual_forward_and_input_independent(self):
        brain = self.brain()
        env = ChessEnv()
        with patch.object(brain, "forward", wraps=brain.forward) as forward:
            _, record, _ = choose_stage(brain, env, np.random.default_rng(3), lesion=True)
            self.assertEqual(forward.call_count, 1)
        p = np.asarray(record["probabilities"])
        np.testing.assert_allclose(p[p > 0], np.ones(8) / 8, atol=1e-7)
        with torch.no_grad():
            logits, _ = brain(torch.stack([torch.zeros(OBS_DIM), torch.ones(OBS_DIM)]), lesion=True)
        self.assertTrue(torch.equal(logits[0], logits[1]))

    def test_terminal_makes_no_forward_call(self):
        white, black = self.brain(), self.brain()
        with patch.object(white, "forward", wraps=white.forward) as w, patch.object(black, "forward", wraps=black.forward) as b:
            game, _ = play_game(white, black, fen="7k/6Q1/5K2/8/8/8/8/8 b - - 0 1")
            self.assertEqual(w.call_count + b.call_count, 0)
            self.assertEqual(game["outcome"]["termination"], "CHECKMATE")

    def test_bounded_long_match_and_live_stream(self):
        white, black = self.brain(), self.brain()
        snapshots = []
        def observe(position):
            snapshots.append((len(position["moves"]), position["final_fen"]))
        game, _ = play_game(white, black, seed=81, max_plies=160, claim_draw=False, on_position=observe)
        self.assertLessEqual(game["plies"], 160)
        self.assertEqual(game["illegal_attempts"], 0)
        self.assertEqual(len(snapshots), game["plies"] + 1)
        self.assertEqual(snapshots[-1][1], game["final_fen"])
        for move in game["moves"]:
            for stage in move["stages"]:
                p = np.asarray(stage["probabilities"])
                self.assertAlmostEqual(float(p.sum()), 1)
                self.assertTrue((p[~np.asarray(stage["mask"]) ] == 0).all())


@unittest.skipUnless(os.environ.get("CHESS_REAL_GRAPH_TESTS") == "1", "optional actual FlyWire tiny integration")
class RealGraphTests(unittest.TestCase):
    def test_real_tiny_two_brain_backward(self):
        torch.set_num_threads(1)
        torch.manual_seed(3)
        white, black = load_policy(size="tiny"), load_policy(size="tiny")
        self.assertIsInstance(white, FlyBrain)
        self.assertFalse(white.synthetic)
        with patch.object(white, "forward", wraps=white.forward) as w, patch.object(black, "forward", wraps=black.forward) as b:
            game, trace = play_game(white, black, max_plies=4)
            self.assertGreater(w.call_count, 0)
            self.assertGreater(b.call_count, 0)
        self.assertEqual(game["illegal_attempts"], 0)
        before = state_hash(black)
        metric, _ = policy_update(white, torch.optim.Adam(white.parameters()), [(trace["white"], 1.0)], 0)
        self.assertGreater(metric["grad_norm"], 0)
        self.assertEqual(state_hash(black), before)
