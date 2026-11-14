"""No downloads: tiny test-only synthetic graph, still executing core FlyBrain."""
import unittest
from unittest.mock import patch
import numpy as np
import torch
from core.brain import FlyBrain
from flies.chess.chess_env import OBS_DIM


def test_graph():
    rng = np.random.default_rng(13)
    n = 176
    pairs = np.unique(rng.integers(0, n, (3000, 2)), axis=0)
    return dict(n=n, graph_hash="TEST-SYNTHETIC-ONLY", synthetic=True,
                root_id=np.arange(n), pre=pairs[:, 0], post=pairs[:, 1],
                sign=np.where(pairs[:, 0] % 4, 1, -1), count=np.ones(len(pairs)) * 5,
                input_idx=np.arange(32), output_idx=np.arange(48, n))


class BrainTest(unittest.TestCase):
    def setUp(self):
        torch.set_num_threads(1)
        torch.manual_seed(123)
        self.patcher = patch("core.brain.load_graph", return_value=test_graph())
        self.patcher.start()
        self.addCleanup(self.patcher.stop)

    def brain(self):
        return FlyBrain("test", OBS_DIM, 8, "chess")
