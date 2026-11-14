"""Run: python -m unittest discover -s flies/painting -p 'test_*.py' -v.

Reference-based choices below are test fixtures only, never an inference policy.
"""
import builtins
import unittest
from unittest.mock import patch

import numpy as np

from core.registry import load_task, fly_names


class EnvironmentTests(unittest.TestCase):
    def setUp(self):
        self.cls = load_task('painting')
        self.env = self.cls()
        self.rng = np.random.default_rng(7)
        self.env.reset(self.rng)

    def test_registry_and_interface(self):
        self.assertIn('painting', fly_names())
        self.assertEqual(self.cls.mode, 'reward')
        self.assertEqual(self.cls.actions, [f'paint_{i}' for i in range(8)])
        for t in range(64):
            obs = self.env.obs()
            self.assertEqual(obs.shape, (12,))
            self.assertEqual(obs.dtype, np.float32)
            self.assertTrue(np.isfinite(obs).all())
            self.assertTrue(((obs >= 0) & (obs <= 1)).all())
            self.assertEqual(self.env.mask(), [True]*8)
            _, _, done, info = self.env.step(t % 8, self.rng)
            self.assertEqual(done, t == 63)
            self.assertEqual(info['completion'], (t+1)/64)
        with self.assertRaises(RuntimeError):
            self.env.step(0, self.rng)
        self.env.reset(self.rng)
        self.assertEqual(self.env.t, 0)
        self.assertTrue((self.env.canvas == 128/255).all())

    def test_exact_pigment_single_cell_and_rewards(self):
        module = __import__('flies.painting.task', fromlist=['PALETTE'])
        # Registry dynamically loads the class; asset constants are identical.
        palette, target = module.PALETTE, module.TARGET
        positive = negative = False
        for y in range(8):
            for x in range(8):
                for action in range(8):
                    self.env.reset(self.rng)
                    self.env.t = y*8+x
                    before = self.env.canvas.copy()
                    old_error = float(np.mean((before-target)**2))
                    _, reward, _, info = self.env.step(action, self.rng)
                    expected = before.copy(); expected[y,x] = palette[action]
                    np.testing.assert_array_equal(self.env.canvas, expected)
                    new_error = float(np.mean((expected-target)**2))
                    self.assertAlmostEqual(reward/64, old_error-new_error, places=7)
                    self.assertAlmostEqual(info['similarity'], 1-new_error, places=7)
                    positive |= reward > 0
                    negative |= reward < 0
        self.assertTrue(positive)
        self.assertTrue(negative)

    def test_deterministic_offline_and_telescoping_return(self):
        other = self.cls(); other.reset(np.random.default_rng(99))
        start = 1-self.env.error
        total = 0
        # After module load, reset/obs/step/caption have no file/network needs.
        with (patch.object(builtins, 'open', side_effect=AssertionError('I/O')),
              patch('socket.socket', side_effect=AssertionError('network'))):
            for t in range(64):
                action = t % 8
                obs, reward, done, info = self.env.step(action, self.rng)
                obs2, reward2, done2, info2 = other.step(action, self.rng)
                np.testing.assert_array_equal(obs, obs2)
                self.assertEqual((reward,done,info),(reward2,done2,info2))
                self.assertIn(f'cell {t+1}/64', self.env.caption(action))
                total += reward
            self.env.reset(self.rng)
        self.assertAlmostEqual(total/64, info['similarity']-start, places=7)

    def test_invalid_action_is_not_a_transition(self):
        for action in [-1,8,1.5,'paint_0']:
            with self.assertRaises(ValueError):
                self.env.step(action, self.rng)
            self.assertEqual(self.env.t, 0)


if __name__ == '__main__':
    unittest.main()
