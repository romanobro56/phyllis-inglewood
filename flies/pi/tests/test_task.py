"""Run: PYTHONDONTWRITEBYTECODE=1 .venv/bin/python -m unittest discover -s flies/pi/tests -v"""
from decimal import Decimal, localcontext
import hashlib
import inspect
from pathlib import Path
import subprocess
import time
import unittest
from unittest.mock import patch

import numpy as np
import torch

from core.brain import FlyBrain
from core.registry import load_task
from core.train import rollout, settings, train_reward
from flies.pi.reference import PI_DIGITS, machin_digits
from flies.pi.task import PiFly


class TaskTests(unittest.TestCase):
    def setUp(self):
        self.rng = np.random.default_rng(7)
        self.env = PiFly()
        self.env.reset(self.rng)

    def guess(self, digit, env=None):
        e = env or self.env
        first = e.step(digit // 5, self.rng)
        self.assertEqual(first[1], 0)
        self.assertFalse(first[2])
        return e.step(digit % 5, self.rng)

    def test_registration_and_eight_outputs(self):
        cls = load_task('pi')
        self.assertEqual(cls.name, 'pi')
        self.assertEqual(cls.mode, 'reward')
        self.assertEqual(cls.actions, [f'choose_{i}' for i in range(8)])
        self.assertIs(cls.teacher, PiFly.__bases__[0].teacher)

    def test_masks_and_all_ten_digits(self):
        for digit in range(10):
            self.env.reset(self.rng)
            self.assertEqual(self.env.mask(), [True] * 2 + [False] * 6)
            self.env.step(digit // 5, self.rng)
            self.assertEqual(self.env.mask(), [True] * 5 + [False] * 3)
            _, _, _, event = self.env.step(digit % 5, self.rng)
            self.assertEqual(event['digit'], digit)
            self.assertEqual(event['kind'], 'correct' if digit == 3 else 'incorrect')

    def test_invalid_actions_are_atomic(self):
        for action in [-1, 2, 5, 8, 1.5, True]:
            with self.assertRaises(ValueError):
                self.env.step(action, self.rng)
            self.assertEqual(self.env.steps, 0)
        self.env.step(1, self.rng)
        with self.assertRaises(ValueError):
            self.env.step(5, self.rng)
        self.assertEqual(self.env.steps, 1)

    def test_reference_exact_and_independent_algorithms(self):
        self.assertTrue(PI_DIGITS.startswith('31415926535897932384626433832795028841971693993751058209749445923078164062862089986280348253421170679'))
        self.assertEqual(machin_digits(), PI_DIGITS)
        with localcontext() as c:
            c.prec = 1060
            M, L, X, K = 1, 13591409, 1, 6
            S = Decimal(L)
            for k in range(1, 80):
                M = M * (K**3 - 16*K) // k**3
                L += 545140134
                X *= -262537412640768000
                S += Decimal(M * L) / X
                K += 12
            digits = str(426880 * Decimal(10005).sqrt() / S).replace('.', '')[:1024]
        self.assertEqual(digits, PI_DIGITS)
        path = Path('flies/pi/pi_digits.txt')
        self.assertEqual(hashlib.sha256(path.read_bytes()).hexdigest(), 'cc362dbd5fd9161a2ed268b6f2f8a86cb07c4fc2a68a7cc65bf442e12b8a6a78')

    def test_no_answer_leakage_or_policy_override(self):
        # Perturb all unseen digits without changing public state. Both stages
        # must have identical observations, masks and group transitions.
        for stage in [0, 1]:
            self.env.reset(self.rng)
            if stage:
                self.env.step(1, self.rng)
            before = self.env.obs().copy(), self.env.mask()
            with patch('flies.pi.task.PI_DIGITS', '9' * 1024):
                np.testing.assert_array_equal(self.env.obs(), before[0])
                self.assertEqual(self.env.mask(), before[1])
        source = inspect.getsource(PiFly.obs) + inspect.getsource(PiFly.mask)
        self.assertNotIn('PI_DIGITS[', source)
        self.assertNotIn('FlyBrain', inspect.getsource(PiFly.step))
        # A target change cannot alter which group a requested action selects.
        self.env.reset(self.rng)
        with patch('flies.pi.task.PI_DIGITS', '9' * 1024):
            self.env.step(0, self.rng)
        self.assertEqual(self.env.selected_group, 0)

    def test_correct_progress_and_streak_reward(self):
        rewards = [self.guess(int(d))[1] for d in PI_DIGITS[:12]]
        np.testing.assert_allclose(rewards, [1 + .1 * min(i, 10) for i in range(12)])
        self.assertEqual(self.env.accepted, PI_DIGITS[:12])
        self.assertEqual(self.env.index, 12)
        self.assertEqual(self.env.correct, 12)
        self.assertEqual(self.env.best_streak, 12)
        self.assertAlmostEqual(self.env.total_reward, sum(rewards))
        self.assertEqual(self.env.first_attempt_correct, 12)

    def test_repeated_wrong_same_digit_then_recovery(self):
        self.guess(3)
        for _ in range(4):
            _, r, done, e = self.guess(9)
            self.assertEqual((e['index'], e['accepted'], r, done), (1, '3', -.2, False))
        self.assertEqual(self.env.streak, 0)
        self.assertEqual(self.env.attempts, 4)
        self.guess(1)
        self.assertEqual(self.env.accepted, '31')
        self.assertEqual(self.env.streak, 1)
        self.assertEqual(self.env.attempts, 0)
        self.assertEqual(self.env.first_attempts, 2)
        self.assertEqual(self.env.first_attempt_correct, 1)
        self.assertAlmostEqual(self.env.total_reward, 1.2)

    def test_success_terminal_not_truncation(self):
        self.env.prefix_length = 2
        self.guess(3)
        out = self.guess(1)
        self.assertTrue(out[2])
        self.assertTrue(self.env.terminated)
        self.assertFalse(self.env.truncated)
        self.assertIn('complete', self.env.caption(1))
        before = self.env.total_reward
        for _ in range(2):
            with self.assertRaises(RuntimeError):
                self.env.step(0, self.rng)
        self.assertEqual(self.env.total_reward, before)

    def test_each_limit_truncates(self):
        for attr, value in [('max_steps', 4), ('max_guesses', 2), ('max_position_attempts', 2)]:
            e = PiFly()
            setattr(e, attr, value)
            e.reset(self.rng)
            self.guess(0, e)
            self.assertTrue(self.guess(0, e)[2])
            self.assertTrue(e.truncated)
            self.assertFalse(e.terminated)
            self.assertEqual(e.index, 0)
            self.assertIn('limit', e.caption(0))

    def test_deterministic_reset_and_finite_observation(self):
        initial = self.env.obs().copy()
        for _ in range(5):
            self.guess(2)
        np.testing.assert_array_equal(initial, self.env.reset(np.random.default_rng(999)))
        for _ in range(self.env.max_steps):
            action = int(self.rng.choice(np.flatnonzero(self.env.mask())))
            obs, reward, done, _ = self.env.step(action, self.rng)
            self.assertEqual(obs.shape, (40,))
            self.assertEqual(obs.dtype, np.float32)
            self.assertTrue(np.isfinite(obs).all())
            self.assertLessEqual(np.abs(obs).max(), 1)
            self.assertTrue(np.isfinite(reward))
            if done:
                break
        self.assertTrue(done)

    def test_caption_is_read_only_and_unique(self):
        self.guess(3)
        state = self.env.obs().copy()
        caption = self.env.caption(3)
        self.assertEqual(caption, self.env.caption(3))
        np.testing.assert_array_equal(state, self.env.obs())
        self.assertIn('3 ✓ | P1:3', caption)
        self.env.reset(self.rng)
        self.guess(3)
        self.assertNotEqual(caption, self.env.caption(3))

    def test_only_pi_sources_changed(self):
        tracked = subprocess.check_output(['git', 'diff', '--name-only', 'HEAD'], text=True).splitlines()
        new = subprocess.check_output(['git', 'ls-files', '--others', '--exclude-standard'], text=True).splitlines()
        self.assertTrue(all(p.startswith('flies/pi/') for p in tracked + new), tracked + new)


class BrainTests(unittest.TestCase):
    def test_native_server_activity_and_terminal_caption(self):
        import base64
        from types import SimpleNamespace
        from server.serve import Session
        torch.set_num_threads(1)
        brain = FlyBrain('tiny', 40, 8, 'pi').eval()
        before = {n: p.clone() for n, p in brain.named_parameters()}
        session = Session({'pi': SimpleNamespace(brain=brain, cls=PiFly, device='cpu')})
        session.inspect = 'pi'
        session.env['pi'].max_steps = 2
        first = session.decide('pi')
        second = session.decide('pi')
        self.assertIn(first['action_index'], [0, 1])
        self.assertIn(second['action_index'], range(5))
        self.assertEqual(first['reward'], 0)
        self.assertFalse(first['episode_done'])
        self.assertTrue(second['episode_done'])
        self.assertIn('limit', second['caption'])
        self.assertIsNone(session.env['pi'].last_event)  # reset after saved caption
        self.assertEqual(second['activity']['n'], 5000)
        self.assertEqual(len(base64.b64decode(second['activity']['data'])), 8 * 5000)
        self.assertTrue(all(torch.equal(before[n], p) for n, p in brain.named_parameters()))

    def test_real_flywire_backward_training_and_inference(self):
        torch.set_num_threads(1)
        torch.manual_seed(123)
        brain = FlyBrain('tiny', PiFly.obs_dim, 8, 'pi')
        self.assertFalse(brain.synthetic)
        self.assertEqual(brain.groups.shape, (8, 16))
        self.assertEqual(len(brain.groups.unique()), 128)
        graph = __import__('core.brain', fromlist=['load_graph']).load_graph('tiny')
        self.assertTrue(np.all(graph['super_class'][brain.groups.numpy()] == 'descending'))
        topology = {n: b.clone() for n, b in brain.named_buffers()}
        e = PiFly(); rng = np.random.default_rng(3)
        o = e.reset(rng); m = e.mask()
        logits, record = brain(torch.tensor(o[None]), torch.tensor([m]), record=True)
        self.assertEqual(record['rates'].shape, (8, 1, 5000))
        self.assertTrue(torch.all(logits[0, 2:] == -1e9))
        (-logits.log_softmax(-1)[0, 0]).backward()
        for name in ['log_w', 'w_in', 'bias', 'leak_logit']:
            g = getattr(brain, name).grad
            self.assertIsNotNone(g)
            self.assertTrue(torch.isfinite(g).all())
        self.assertGreater(brain.log_w.grad.abs().sum().item(), 0)
        self.assertGreater(brain.w_in.grad.abs().sum().item(), 0)
        before = {n: p.clone() for n, p in brain.named_parameters()}
        cfg = settings(PiFly, dict(updates=1, episodes=2, chunk=32))
        train_reward(brain, PiFly, cfg, 'cpu', time.time() + 60, lambda _: None)
        self.assertTrue(any(not torch.equal(before[n], p) for n, p in brain.named_parameters()))
        for n, b in brain.named_buffers():
            self.assertTrue(torch.equal(topology[n], b), n)
        before = {n: p.clone() for n, p in brain.named_parameters()}
        brain.eval()
        for lesion in [False, True]:
            returns, trans = rollout(PiFly, 2, rng, 'brain', brain, lesion=lesion)
            self.assertTrue(np.isfinite(returns).all())
            self.assertTrue(all(t[1][t[2]] for t in trans))
        for n, p in brain.named_parameters():
            self.assertTrue(torch.equal(before[n], p), n)


if __name__ == '__main__':
    unittest.main()
