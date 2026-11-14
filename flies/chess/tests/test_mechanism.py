import copy
from fractions import Fraction
from itertools import product
import json
from pathlib import Path
import tempfile

import chess
import numpy as np
import torch
from flies.chess.chess_env import ChessEnv, SPATIAL_ENCODING
from flies.chess.mechanism_train import leave_one_out, grouped_specs, sample_baselines, train
from flies.chess.reward_validation import rational_potential, rational_rewards, audit_games
from flies.chess.research_train import potential, own_turn_rewards
from flies.chess.research_positions import make_matched
from flies.chess.storage import ROOT, atomic_write, brain_payload, state_hash
from flies.chess.tests.support import BrainTest


class MechanismTests(BrainTest):
    def test_control_identity_does_not_mislabel_initial_checkpoint(self):
        from flies.chess.mechanism_eval import policy_identity
        model = self.brain()
        with tempfile.TemporaryDirectory(dir=ROOT) as folder:
            path = Path(folder)/'initial_0.pt'; atomic_write(path,brain_payload(model),binary=True)
            self.assertEqual(policy_identity('untrained',model,path)['label'],'UNTRAINED')
            self.assertEqual(policy_identity('frozen_opponent',model,path)['label'],'FROZEN-UNTRAINED')
            self.assertEqual(policy_identity('parent',model,path)['label'],'FROZEN-PARENT')
            self.assertFalse(policy_identity('random_legal',model)['brain_used_for_candidate'])

    def test_gradient_cosine_precision_and_zero_signal(self):
        from flies.chess.mechanism_gradient import gradient_cosines
        values = torch.logspace(-9, 3, 223441)
        result = gradient_cosines(dict(a=values, b=-values, zero=torch.zeros_like(values)))
        self.assertAlmostEqual(result['a/a'], 1., places=12)
        self.assertAlmostEqual(result['a/b'], -1., places=12)
        self.assertIsNone(result['a/zero'])
        self.assertTrue(all(x is None or -1<=x<=1 for x in result.values()))

    def corpus(self):
        pairs = make_matched(train=2, validation=1, test=1)['train']
        row = dict(fen='5k2/8/4K3/8/8/3Q4/8/8 w - - 0 1')
        return dict(sha256='test-curriculum', matched_train=pairs, easy_train=pairs[:1],
                    suffix_train=[row], mate2_train=[row])

    def game(self, fen, moves):
        env = ChessEnv(fen, max_plies=len(moves))
        events = [env.push(chess.Move.from_uci(m)) for m in moves]
        return dict(moves=events, final_fen=env.board.fen(), outcome=env.outcome())

    def test_integer_reward_rational_oracle_special_moves(self):
        cases = [
            ('r6k/8/8/8/8/8/Q7/7K w - - 0 1', ['h1g1', 'a8a2']),
            ('7k/P7/8/8/8/8/8/7K w - - 0 1', ['a7a8q']),
            ('7k/8/8/8/8/8/p7/7K b - - 0 1', ['a2a1q']),
            ('7k/8/8/3pP3/8/8/8/7K w - d6 0 1', ['e5d6']),
            ('r3k2r/8/8/8/8/8/8/R3K2R w KQkq - 0 1', ['e1g1', 'e8c8']),
            (chess.STARTING_FEN, ['g1f3', 'g8f6', 'f3g1', 'f6g8']),
            ('7k/8/5KQ1/8/8/8/8/8 w - - 0 1', ['g6g7']),
        ]
        games = [self.game(fen, moves) for fen, moves in cases]
        result = audit_games(games)
        self.assertEqual(result['integer_false_nonzeros'], 0)
        for game in games:
            for side in ('white', 'black'):
                np.testing.assert_allclose(own_turn_rewards(game, side, True), rational_rewards(game, side), atol=1e-15)
                self.assertAlmostEqual(float(rational_potential(game['final_fen'], side)),
                                       potential(chess.Board(game['final_fen']), side))
        self.assertEqual(rational_potential(chess.STARTING_FEN, 'white'), Fraction(0))

    def test_loo_unbiased_and_no_own_reward_or_context_leak(self):
        values = np.array([1., 0., -1., 0.])
        base = leave_one_out(values); values[1] = 999.
        self.assertEqual(base[1], leave_one_out(values)[1])
        # Exhaustively verify the score-function gradient of a Bernoulli
        # terminal reward. Including one's own outcome would shrink it by 1/2.
        p = .23; expected = 0.
        for a, b in product((0, 1), repeat=2):
            probability = (p if a else 1-p)*(p if b else 1-p)
            expected += probability*(a-leave_one_out([a, b])[0])*(a-p)
        self.assertAlmostEqual(expected, p*(1-p))
        with self.assertRaises(ValueError):leave_one_out([1.])
        specs, tiers, groups = grouped_specs(self.corpus(), np.random.default_rng(9), 1, games=64)
        records = [dict(candidate='white' if chess.Board(s['fen']).turn else 'black',
                        outcome=dict(winner=None)) for s in specs]
        bases, prior, informative = sample_baselines(records, specs, tiers, groups, 0, 'ema', {})
        self.assertEqual(bases, [0.]*64); self.assertEqual(informative, 0)
        corrupt = copy.deepcopy(specs); corrupt[1]['fen'] = chess.STARTING_FEN
        with self.assertRaises(ValueError):sample_baselines(records, corrupt, tiers, groups, 0, 'loo', {})
        corrupt = copy.deepcopy(specs); corrupt[1]['seed'] = corrupt[0]['seed']
        with self.assertRaises(ValueError):sample_baselines(records, corrupt, tiers, groups, 0, 'loo', {})
        records[0]['outcome']['winner'] = records[0]['candidate']
        bases, next_prior, _ = sample_baselines(records, specs, tiers, groups, 0, 'ema', {})
        self.assertEqual(bases, [0.]*64)  # Current rewards affect only the next update.
        self.assertTrue(any(x != 0 for x in next_prior.values()))

    def test_group_budget_pair_colors_and_tier_progression(self):
        corpus = self.corpus()
        for update in (0, 8, 31, 32, 47):
            a = grouped_specs(corpus, np.random.default_rng(5), update)
            b = grouped_specs(corpus, np.random.default_rng(5), update)
            self.assertEqual(a, b)
            specs, tiers, groups = a
            self.assertEqual(len(specs), 128)
            for group in set(groups):
                rows = [s for s, g in zip(specs, groups) if g == group]
                self.assertEqual(len(rows), 8)
                self.assertEqual(len({s['fen'] for s in rows}), 1)
                self.assertEqual(len({s['seed'] for s in rows}), 8)
                self.assertNotEqual(rows[0]['players'][0], rows[0]['players'][1])
            expected = {'matched': 128} if update < 32 else {'matched': 64, 'suffix': 32, 'mate2': 32}
            self.assertEqual({t: tiers.count(t) for t in set(tiers)}, expected)

    def test_suffix_curation_and_freeze_guard(self):
        from flies.chess.mechanism_data import make_curriculum
        from flies.chess.mechanism_eval import generate_fresh
        from flies.chess.research_positions import own_family, mate_one
        pairs = make_matched(train=2, validation=1, test=1)
        board = chess.Board('5k2/8/4K3/8/8/3Q4/8/8 w - - 0 1')
        row = dict(fen=board.fen(), family=own_family(board), tier='mate2', answers=['unused'])
        extended = dict(sha256='fixture', train=[row], validation=[], test=[])
        data = make_curriculum(pairs, extended)
        self.assertEqual(data, make_curriculum(pairs, extended))
        self.assertTrue(data['suffix_train'])
        known = {r['family'] for s in ('train','validation','test') for r in pairs[s]} | {row['family']}
        for suffix in data['suffix_train']:
            self.assertNotIn(suffix['family'], known)
            self.assertEqual(suffix['parent_family'], row['family'])
            self.assertTrue(mate_one(chess.Board(suffix['fen'])))
        with self.assertRaises(ValueError):generate_fresh(data, dict(frozen=False), ROOT/'artifacts/forbidden-test.json')

    def test_grouped_resume_and_labels_no_effect(self):
        corpus = self.corpus()
        with tempfile.TemporaryDirectory(dir=ROOT) as folder:
            root = Path(folder); sources = []
            for i in range(2):
                model = self.brain(); model.encoding = SPATIAL_ENCODING
                path = root/f'parent-{i}.pt'; atomic_write(path, brain_payload(model), binary=True); sources.append(path)
            kwargs = dict(seed=7, games=64, group_size=8, matched_updates=1, easy_updates=1, size='test')
            a, _ = train(corpus, sources, root/'full', updates=2, **kwargs)
            train(corpus, sources, root/'split', updates=1, **kwargs)
            b, _ = train(corpus, sources, root/'split', updates=2, resume=root/'split/resume.pt', **kwargs)
            corrupt = copy.deepcopy(corpus)
            for key in ('matched_train', 'easy_train', 'suffix_train', 'mate2_train'):
                for row in corrupt[key]:row['answers'] = ['INVALID-NOT-POLICY-INPUT']
            c, _ = train(corrupt, sources, root/'labels', updates=2, **kwargs)
            for x, y, z in zip(a, b, c):
                self.assertEqual(state_hash(x), state_hash(y)); self.assertEqual(state_hash(x), state_hash(z))
            zero, history = train(corpus, sources, root/'zero', updates=1, method='zero', entropy=0., **kwargs)
            self.assertGreater(history[0]['outcomes']['matched']['wins'], 0)
            for i, model in enumerate(zero):
                original = torch.load(sources[i], weights_only=False)['state_dict']
                self.assertTrue(all(torch.equal(value, original[key]) for key, value in model.state_dict().items()))
            with self.assertRaises(ValueError):
                train(corpus, sources, root/'bad', updates=2, method='ema', resume=root/'split/resume.pt', **kwargs)
