"""Tests use an oracle only to arrange validator/reward fixtures, never as a policy."""
import copy
import json
from pathlib import Path
import tempfile
import unittest
import numpy as np
from flies.jobs.legacy_task import JobsFly
from flies.jobs.history import History
from flies.jobs.forms import encode
from flies.jobs.jobs import load_jobs, job_id


class JobsTests(unittest.TestCase):
    def setUp(self):
        self.rng = np.random.default_rng(3)
        self.env = JobsFly(trace=True)
        self.env.reset(self.rng)

    def first_text(self):
        env = JobsFly(difficulty=.1, trace=True)
        env.reset(self.rng)
        return env

    def oracle_choice_for_test_only(self, env):
        return next(i for i, c in enumerate(env.candidates()) if c['value'] == env._validator.expected[env.current['id']])

    def test_oracle_cannot_change_observation_view_mask_or_candidates(self):
        env = self.env
        observation, mask, view, choices = env.obs().copy(), env.mask(), env.public_view(), env.candidates()
        env._validator.expected = {f['id']: 'An entirely different answer' for f in env.fields}
        np.testing.assert_array_equal(observation, env.obs())
        self.assertEqual(mask, env.mask())
        self.assertEqual(view, env.public_view())
        self.assertEqual(choices, env.candidates())

    def test_oracle_feedback_does_not_change_focus_or_next_observation(self):
        env = self.first_text()
        other = copy.deepcopy(env)
        other._validator.expected = {f['id']: 'unreachable' for f in other.fields}
        action = self.oracle_choice_for_test_only(env)
        focus = env.focus
        observation, reward, _, _ = env.step(action, self.rng)
        second_obs, second_reward, _, _ = other.step(action, self.rng)
        self.assertNotEqual(reward, second_reward)
        self.assertEqual(env.focus, focus)
        self.assertEqual(other.focus, focus)
        np.testing.assert_array_equal(observation, second_obs)

    def test_independent_validation_and_explicit_submission(self):
        env = self.first_text()
        self.assertEqual(env.step(6, self.rng)[3]['event'], 'validation_rejected')
        env.step(self.oracle_choice_for_test_only(env), self.rng)
        self.assertFalse(env.done)
        self.assertEqual(env.completed, 0)
        _, reward, done, info = env.step(6, self.rng)
        self.assertTrue(done)
        self.assertEqual(reward, 15.)
        self.assertEqual(info['event'], 'application_completed')
        with self.assertRaises(ValueError):
            env.step(6, self.rng)

    def test_correction_reward_cannot_be_farmed(self):
        env = self.first_text()
        right = self.oracle_choice_for_test_only(env)
        wrong = (right + 1) % len(env.candidates())
        self.assertEqual(env.step(wrong, self.rng)[1], -1.)
        self.assertEqual(env.step(right, self.rng)[1], 2.)
        self.assertLess(env.step(right, self.rng)[1], 0)
        env.step(wrong, self.rng)
        self.assertLessEqual(env.step(right, self.rng)[1], 0)

    def test_repeated_wrong_entries_are_not_a_cheap_escape(self):
        env = self.first_text()
        wrong = (self.oracle_choice_for_test_only(env) + 1) % len(env.candidates())
        first = env.step(wrong, self.rng)[1]
        repeated = env.step(wrong, self.rng)[1]
        self.assertLess(repeated, first)
        self.assertLessEqual(repeated, -1.)

    def test_dropdown_requires_brain_to_select_input_mode(self):
        for seed in range(100):
            self.env.reset(np.random.default_rng(seed))
            if self.env.current['kind'] == 'select':
                break
        action = self.oracle_choice_for_test_only(self.env)
        self.assertEqual(self.env.step(action, self.rng)[3]['event'], 'invalid_action')
        self.assertEqual(self.env.values, {})
        self.env.step(7, self.rng)
        self.assertEqual(self.env.step(action, self.rng)[3]['event'], 'field_correct')

    def test_generation_is_diverse_and_has_no_answer_keys_in_view(self):
        counts, kinds, pages, layouts, labels, required = set(), set(), set(), set(), set(), set()
        for seed in range(80):
            self.env.reset(np.random.default_rng(seed))
            counts.add(len(self.env.fields))
            kinds.update(f['kind'] for f in self.env.fields)
            pages.add(self.env.layout['pages'])
            layouts.add(self.env.layout['layout'])
            labels.update(f['label'] for f in self.env.fields)
            required.update(f['required'] for f in self.env.fields)
            self.assertNotIn('expected', self.env.public_view())
            self.assertEqual(self.env.obs().shape, (64,))
            self.assertTrue(np.isfinite(self.env.obs()).all())
        self.assertEqual(counts, {3, 4, 5})
        self.assertEqual(kinds, {'text', 'textarea', 'select'})
        self.assertEqual(pages, {1, 2, 3})
        self.assertEqual(layouts, {'stack', 'grid', 'cards'})
        self.assertEqual(required, {True, False})
        self.assertGreater(len(labels), 20)

    def test_invalid_actions_penalized_and_navigation_does_not_repair(self):
        env = self.first_text()
        self.assertEqual(env.step(99, self.rng)[1], -1)
        self.assertEqual(env.values, {})
        self.assertFalse(env.mask()[5])
        self.assertEqual(env.step(5, self.rng)[1], -1)

    def test_history_and_weights_are_separate_and_resume_pending_form(self):
        env = self.env
        env.step(0, self.rng)
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / 'history.sqlite3'
            history = History(path, env.profile['facts']['email'])
            history.save(env, event={**env.traces[-1], 'policy': 'untrained', 'checkpoint': 'fresh'})
            history.close()
            history = History(path, env.profile['facts']['email'])
            self.assertEqual(job_id(history.next_job([])), job_id(env.job))
            other = JobsFly(jobs=[env.job])
            other.reset(np.random.default_rng(15))
            history.restore(other)
            np.testing.assert_array_equal(env.obs(), other.obs())
            self.assertEqual(env.values, other.values)
            self.assertEqual(env.cumulative_reward, other.cumulative_reward)
            self.assertEqual(len(history.details()[0]['events'][0]['observation']), 64)
            other.report_error('already_applied')
            history.save(other)
            self.assertEqual(history.get(env.job)['status'], 'blocked')
            self.assertNotEqual(job_id(history.next_job(env.jobs)), job_id(env.job))
            history.close()
            actual = History(path, env.profile['facts']['email'], mode='live')
            self.assertIsNone(actual.get(env.job))
            with self.assertRaises(ValueError):
                actual.save(env)
            actual.close()

    def test_legacy_snapshot_rejected(self):
        with self.assertRaises(ValueError):
            self.env.restore({'version': 1})

    def test_text_file_deduplicates_without_inventing_skills(self):
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / 'jobs.txt'
            path.write_text('https://example.com/job?id=1&utm_source=a\nhttps://example.com/job?id=1\n')
            self.assertEqual(len(load_jobs(path)), 1)
            self.assertEqual(load_jobs(path)[0]['skills'], [])

    def test_resume_slots_are_shuffled_and_values_reach_sensors(self):
        orders = set()
        for seed in range(20):
            env = JobsFly(difficulty=.1)
            env.reset(np.random.default_rng(seed))
            orders.add(tuple(c['label'] for c in env.candidates()))
        self.assertGreater(len(orders), 10)
        before = env.obs().copy()
        env.resume_choices[0]['value'] = 'changed literal resume fact'
        self.assertFalse(np.array_equal(before, env.obs()))

    def test_public_encoder_and_environment_are_identical(self):
        np.testing.assert_array_equal(self.env.obs(), JobsFly.encode_view(self.env.public_view()))

    def test_encoder_uses_only_literal_input_string(self):
        np.testing.assert_array_equal(encode('email'), encode('email'))
        self.assertFalse(np.array_equal(encode('email'), encode('education')))


if __name__ == '__main__':
    unittest.main()
