import copy
import json
from pathlib import Path
import tempfile
from unittest.mock import patch

import chess
import numpy as np
import torch
from flies.chess.research_positions import make_matched,mate_one,legal_signature,own_family,forced_mate_moves,SearchLimit
from flies.chess.matched_eval import exact_distribution,evaluate_pairs,prepared_tree,swapped_report
from flies.chess.research_train import own_turn_rewards,train,clone_source,potential
from flies.chess.train_selfplay import discounted_returns
from flies.chess.chess_env import SPATIAL_ENCODING,STAGE_ONLY_ENCODING,ChessEnv
from flies.chess.storage import ROOT,brain_payload,atomic_write,state_hash
from flies.chess.tests.support import BrainTest
from flies.chess.benchmark import graph_storage


class ResearchTests(BrainTest):
    def pairs(self):return make_matched(train=2,validation=1,test=2)

    def test_pair_full_legal_tree_masks_and_distinct_answers(self):
        c=self.pairs();keys=[]
        for split in ('train','validation','test'):
            for row in c[split]:
                a,b=[chess.Board(f) for f in row['positions']]
                self.assertEqual(legal_signature(a),legal_signature(b))
                self.assertFalse(set(row['answers'][0])&set(row['answers'][1]))
                self.assertEqual(row['answers'],[mate_one(a),mate_one(b)])
                ta,tb=[prepared_tree(f,SPATIAL_ENCODING) for f in row['positions']]
                self.assertEqual(set(ta),set(tb))
                for prefix in ta:np.testing.assert_array_equal(ta[prefix]['mask'],tb[prefix]['mask'])
                self.assertEqual(own_family(a),own_family(b));keys.append(row['family'])
        self.assertEqual(len(keys),len(set(keys)))
        self.assertEqual(c,self.pairs())
        excluded=make_matched(train=1,validation=1,test=1,exclude=keys)
        self.assertTrue(set(keys).isdisjoint(r['family'] for s in ('train','validation','test') for r in excluded[s]))

    def test_exact_probabilities_and_restricted_pair_null(self):
        row=self.pairs()['test'][0];brain=self.brain();brain.encoding=SPATIAL_ENCODING
        p,_=exact_distribution(brain,row['positions'][0])
        self.assertAlmostEqual(sum(p.values()),1.,places=6)
        self.assertEqual(set(p),set(row['legal_moves']))
        for mode in ('interface','stage','own_coordinates'):
            r=evaluate_pairs(brain,[row],mode)
            self.assertAlmostEqual(r['mean_adaptation'],0.,places=6)
            self.assertAlmostEqual(r['mean_total_variation'],0.,places=6)
            self.assertEqual(r['both_argmax_correct'],0.)
        self.assertAlmostEqual(evaluate_pairs(brain,[row],lesion=True)['mean_adaptation'],0.,places=6)
        a=evaluate_pairs(brain,[row]);b=evaluate_pairs(brain,[row],swap=True)
        self.assertAlmostEqual(a['mean_adaptation'],-b['mean_adaptation'],places=6)
        self.assertAlmostEqual(a['rows'][0]['cross_target_mass'],b['mean_correct_mass'],places=6)
        self.assertEqual(swapped_report(a)['both_argmax_correct'],b['both_argmax_correct'])

    def test_unknown_proof_never_becomes_negative_label(self):
        board=chess.Board('7k/8/5KQ1/8/8/8/8/8 w - - 0 1');before=board.fen()
        answers,_=forced_mate_moves(board,1)
        self.assertEqual(set(answers),set(mate_one(board)))
        with self.assertRaises(SearchLimit):forced_mate_moves(board,2,max_nodes=1)
        self.assertEqual(board.fen(),before)

    def test_forced_mate_two_checks_every_defender_reply(self):
        b=chess.Board('5k2/8/4K3/8/8/3Q4/8/8 w - - 0 1')
        self.assertFalse(mate_one(b))
        answers,_=forced_mate_moves(b,2)
        self.assertEqual(set(answers),{'d3h7','d3g3'})
        for uci in answers:
            board=b.copy();board.push_uci(uci)
            for reply in list(board.legal_moves):
                board.push(reply);self.assertTrue(mate_one(board));board.pop()

    def test_extended_corpus_has_no_family_leak(self):
        from flies.chess.extended_curriculum import make_extended
        matched=self.pairs();extended=make_extended(matched,train=1,validation=1,test=1)
        all_keys={r['family'] for s in ('train','validation','test') for r in matched[s]}
        for split in ('train','validation','test'):
            self.assertEqual({r['tier'] for r in extended[split]},{'mate2','endgame','complex'})
            for row in extended[split]:
                self.assertNotIn(row['family'],all_keys);all_keys.add(row['family'])
                self.assertTrue(chess.Board(row['fen']).is_valid())
        with self.assertRaises(ValueError):train({}, {}, ROOT/'artifacts/invalid',[],updates=41,games=8)

    def test_vector_rewards_discount_and_opponent_reply_credit(self):
        np.testing.assert_allclose(discounted_returns(1.,3,.9)[0],[.81,.9,1.])
        np.testing.assert_allclose(discounted_returns([.1,-.2,1.],3,.9)[0],[.73,.7,1.])
        with self.assertRaises(ValueError):discounted_returns([1.,2.],3,.9)
        with self.assertRaises(ValueError):discounted_returns([np.nan],1,.9)
        env=ChessEnv('r6k/8/8/8/8/8/Q7/7K w - - 0 1',max_plies=2)
        events=[env.push(chess.Move.from_uci('h1g1')),env.push(chess.Move.from_uci('a8a2'))]
        game=dict(moves=events,final_fen=env.board.fen(),outcome=env.outcome())
        self.assertEqual(game['outcome']['result'],'*')
        self.assertAlmostEqual(own_turn_rewards(game,'white',True)[0],-.225)
        self.assertAlmostEqual(own_turn_rewards(game,'black',True)[0],.225)
        self.assertEqual(own_turn_rewards(game,'white',False),[0.])

    def test_cache_has_no_stale_policy_probabilities(self):
        row=self.pairs()['test'][0];brain=self.brain();brain.encoding=SPATIAL_ENCODING
        a,_=exact_distribution(brain,row['positions'][0])
        with torch.no_grad():brain.bias[brain.groups[0]]+=.05
        b,_=exact_distribution(brain,row['positions'][0])
        self.assertGreater(sum(abs(a[k]-b[k]) for k in a),1e-5)

    def test_projection_monotonic_and_explicit_lower_bound(self):
        small=graph_storage(5000,50000,1000,1);large=graph_storage(5000,50000,1000,32)
        self.assertGreater(large['two_adam_plus_frozen_and_rate_history_lower_bound_bytes'],small['two_adam_plus_frozen_and_rate_history_lower_bound_bytes'])
        self.assertIn('Lower bound',large['note'])

    def test_zero_wins_and_one_opening_do_not_imply_zero_uncertainty(self):
        from flies.chess.generalization import bounded_outcome_interval
        self.assertGreater(bounded_outcome_interval([0.]*24)['ci95'][1],0.)
        self.assertIsNone(bounded_outcome_interval([0.])['ci95'])

    def test_material_arithmetic_ignores_piece_iteration_order(self):
        a=chess.Board('4k3/pp6/8/8/8/8/PP6/R3K2R w KQ - 0 1')
        b=a.copy();b.remove_piece_at(chess.A1);b.set_piece_at(chess.B1,chess.Piece(chess.ROOK,True))
        self.assertEqual(potential(a,'white'),potential(b,'white'))
        self.assertAlmostEqual(potential(a,'white'),potential(a,'white','float-v1'))

    def test_extended_phase_does_not_read_answer_labels(self):
        from flies.chess.extended_curriculum import make_extended
        matched=self.pairs();extended=make_extended(matched,train=1,validation=1,test=1)
        with tempfile.TemporaryDirectory(dir=ROOT) as folder:
            root=Path(folder);sources=[]
            for i in range(2):
                model=self.brain();model.encoding=SPATIAL_ENCODING
                path=root/f'parent_{i}.pt';atomic_write(path,brain_payload(model),binary=True);sources.append(path)
            kwargs=dict(sources=sources,seed=9,games=32,size='test')
            train(matched,extended,root/'seed',updates=1,**kwargs)
            ck=torch.load(root/'seed/resume.pt',weights_only=False)
            ck['update']=40;ck['history']=[]  # Synthetic branch fixture, not a reported 40-update experiment.
            path=root/'phase40.pt';atomic_write(path,ck,binary=True)
            a,_=train(matched,extended,root/'a',updates=41,resume=path,**kwargs)
            corrupt=copy.deepcopy(extended)
            for row in corrupt['train']:row['answers']=['forbidden-oracle-label']
            b,_=train(matched,corrupt,root/'b',updates=41,resume=path,**kwargs)
            for x,y in zip(a,b):self.assertEqual(state_hash(x),state_hash(y))

    def test_research_resume_labels_and_architecture(self):
        matched=self.pairs();extended=dict(sha256='test-unused-in-matched-phase',train=[])
        with tempfile.TemporaryDirectory(dir=ROOT) as folder:
            root=Path(folder);sources=[]
            for i in range(2):
                model=self.brain();model.encoding=SPATIAL_ENCODING
                path=root/f'parent_{i}.pt';atomic_write(path,brain_payload(model),binary=True);sources.append(path)
            # Test fixture uses preset "test", matching its actual synthetic graph.
            kwargs=dict(sources=sources,seed=9,games=8,size='test')
            a,_=train(matched,extended,root/'full',updates=2,**kwargs)
            train(matched,extended,root/'split',updates=1,**kwargs)
            b,_=train(matched,extended,root/'split',updates=2,resume=root/'split/resume.pt',**kwargs)
            for x,y in zip(a,b):self.assertEqual(state_hash(x),state_hash(y))
            changed=copy.deepcopy(matched)
            for row in changed['train']:row['answers']=[['not-a-move'],['not-a-move']]
            z,_=train(changed,extended,root/'corrupt-labels',updates=2,**kwargs)
            for x,y in zip(a,z):self.assertEqual(state_hash(x),state_hash(y))
            restricted=clone_source(sources[0],STAGE_ONLY_ENCODING,size='test')
            self.assertEqual(state_hash(restricted),state_hash(clone_source(sources[0],SPATIAL_ENCODING,size='test')))
