import copy
from pathlib import Path
import tempfile
from unittest.mock import patch
import chess
import numpy as np
import torch

from flies.chess.batched import play_batch
from flies.chess.chess_env import ChessEnv, ENCODING, SPATIAL_ENCODING, decode_planes
from flies.chess.curriculum import make_corpus, family_key
from flies.chess.learning import cluster_interval, paired_difference, specs_for, run_train
from flies.chess.selfplay import play_game, choose_stage
from flies.chess.storage import ROOT, brain_payload, load_policy, validate_and_load, state_hash
from flies.chess.train_selfplay import policy_update
from flies.chess.evaluate import summarize
from flies.chess.tests.support import BrainTest


class LearningTests(BrainTest):
    def test_batch_matches_serial_both_brains(self):
        models=[self.brain(),self.brain()]
        specs=[dict(fen=chess.STARTING_FEN,seed=s,players=(0,1)) for s in range(4)]
        with patch.object(models[0],'forward',wraps=models[0].forward) as white, patch.object(models[1],'forward',wraps=models[1].forward) as black:
            batch,_=play_batch(models,specs,max_plies=4)
            self.assertGreater(white.call_count,0);self.assertGreater(black.call_count,0)
        for game,spec in zip(batch,specs):
            serial,_=play_game(*models,seed=spec['seed'],max_plies=4)
            self.assertEqual([m['uci'] for m in game['moves']],[m['uci'] for m in serial['moves']])
            self.assertEqual(game['final_fen'],serial['final_fen'])
            self.assertEqual(game['illegal_attempts'],0)
        with self.assertRaises(ValueError):
            play_batch(models,[dict(fen=chess.STARTING_FEN,seed=0,players=(0,0))])
        with self.assertRaises(ValueError):
            play_batch(models,[dict(fen=chess.STARTING_FEN,seed=0,players=(0,.5))])

    def test_reward_sign_and_mate_not_opponents_credit(self):
        # Real FlyBrain sampling still chooses the move. Find an actual mate outcome
        # in a finite test seed sweep; no oracle action or substitute move is used.
        models=[self.brain(),self.brain()]
        fen='7k/R7/5KQ1/8/8/8/8/8 w - - 0 1'
        specs=[dict(fen=fen,seed=s,players=(0,1)) for s in range(128)]
        games,traces=play_batch(models,specs,4)
        winners=[(g,t) for g,t in zip(games,traces) if g['outcome']['winner'] is not None]
        self.assertTrue(winners)
        game,trace=winners[0]
        winner=game['outcome']['winner'];loser='black' if winner=='white' else 'white'
        self.assertTrue(trace[winner])
        # Outcome attribution function is shared with training (not test-only arithmetic).
        from flies.chess.learning import outcome_reward
        self.assertEqual(outcome_reward(game,winner),1)
        self.assertEqual(outcome_reward(game,loser),-1)
        self.assertEqual(outcome_reward(dict(outcome=dict(winner=None,truncated=True)),'white'),0)

    def test_episode_normalization_forced_actions_and_empty_gradients(self):
        brain=self.brain()
        env=ChessEnv();_,_,transition=choose_stage(brain,env,np.random.default_rng(5))
        forced=dict(obs=transition['obs'],mask=np.array([True]+[False]*7),action=0)
        opt=torch.optim.SGD(brain.parameters(),lr=1e-5)
        obs=torch.tensor(transition['obs'][None]);mask=torch.tensor(transition['mask'][None])
        logits,_=brain(obs,mask)
        expected_loss=-.01*torch.log_softmax(logits,-1)[0,transition['action']]/2
        expected=torch.autograd.grad(expected_loss,brain.w_in)[0]
        metric,_=policy_update(brain,opt,[([[transition]],.01),([[forced],[forced]],0.),([], -1.)],0.,entropy=0)
        torch.testing.assert_close(brain.w_in.grad,expected)
        self.assertEqual(metric['episodes'],2)
        self.assertEqual(metric['forced_stages'],2)
        self.assertAlmostEqual(metric['reward'],.005)
        before=state_hash(brain)
        policy_update(brain,opt,[([],1.)],0.)
        self.assertTrue(all(p.grad is None for p in brain.parameters()))
        self.assertEqual(state_hash(brain),before)

    def test_positive_reward_increases_selected_joint_probability(self):
        brain=self.brain();env=ChessEnv();rng=np.random.default_rng(5)
        action,_,transition=choose_stage(brain,env,rng)
        def logp():
            with torch.no_grad():
                logits,_=brain(torch.tensor(transition['obs'][None]),torch.tensor(transition['mask'][None]))
                return float(logits.log_softmax(-1)[0,action])
        old=logp()
        metric,_=policy_update(brain,torch.optim.SGD(brain.parameters(),lr=.0001),[([[transition]],1.)],0.,entropy=0)
        self.assertGreater(logp(),old)
        for name in ('log_w','w_in','bias','leak_logit'):
            self.assertGreater(metric['gradient_norms'][name],0)
            self.assertGreater(metric['parameter_deltas'][name],0)

    def test_spatial_encoding_preserves_board_and_black_transform(self):
        w=ChessEnv('r3k2r/8/8/3pP3/8/8/8/R3K2R w KQkq d6 7 20')
        b=ChessEnv(w.board.mirror().fen())
        for env in (w,b):
            obs=env.observe(SPATIAL_ENCODING)
            self.assertEqual(obs.shape,(64,));self.assertTrue(np.isfinite(obs).all())
            np.testing.assert_array_equal(decode_planes(obs),decode_planes(env.observe()))
            self.assertTrue(((0<=obs)&(obs<=1)).all())
        np.testing.assert_array_equal(w.observe(SPATIAL_ENCODING)[53:],b.observe(SPATIAL_ENCODING)[53:])
        queen=ChessEnv('7k/R7/5KQ1/8/8/8/8/8 w - - 0 1')
        mirrored=ChessEnv(queen.board.mirror().fen())
        for env in (queen,mirrored):
            self.assertEqual(float(env.observe(SPATIAL_ENCODING)[57]),7/8)
            self.assertEqual(float(env.observe(SPATIAL_ENCODING)[58]),6/8)
            env.select(6);env.select(5)
            self.assertAlmostEqual(float(env.observe(SPATIAL_ENCODING)[61]),5/6)
            self.assertAlmostEqual(float(env.observe(SPATIAL_ENCODING)[62]),8/14)
            self.assertAlmostEqual(float(env.observe(SPATIAL_ENCODING)[63]),9/14)
        w.select(0);w.select(0);b.select(0);b.select(0)
        np.testing.assert_array_equal(w.observe(SPATIAL_ENCODING)[53:],b.observe(SPATIAL_ENCODING)[53:])
        brain=self.brain();brain.encoding=SPATIAL_ENCODING;brain.policy_temperature=2.
        with tempfile.TemporaryDirectory(dir=ROOT) as folder:
            from flies.chess.storage import atomic_write
            path=Path(folder)/'brain.pt';atomic_write(path,brain_payload(brain),binary=True)
            loaded=load_policy(path,size='test')
            self.assertEqual(loaded.encoding,SPATIAL_ENCODING)
            self.assertEqual(loaded.policy_temperature,2.)
            with self.assertRaises(ValueError):load_policy(path,size='test',encoding=ENCODING)

    def test_stage_only_ablation_contains_no_board_or_partial_choice(self):
        from flies.chess.chess_env import STAGE_ONLY_ENCODING
        a=ChessEnv();b=ChessEnv('7k/8/5KQ1/8/8/8/8/8 w - - 0 1')
        np.testing.assert_array_equal(a.observe(STAGE_ONLY_ENCODING),b.observe(STAGE_ONLY_ENCODING))
        self.assertEqual(a.observe(STAGE_ONLY_ENCODING).sum(),1.)
        a.select(0);b.select(5)
        np.testing.assert_array_equal(a.observe(STAGE_ONLY_ENCODING),b.observe(STAGE_ONLY_ENCODING))

    def test_curriculum_no_symmetry_leak_or_terminal_starts(self):
        corpus=make_corpus(train=16,validation=8,test=8)
        self.assertEqual(corpus,make_corpus(train=16,validation=8,test=8))
        keys=[]
        for split in ('train','validation','test'):
            for row in corpus[split]:
                board=chess.Board(row['fen']);self.assertIsNone(board.outcome())
                self.assertTrue(row['mate_moves'])
                self.assertLessEqual(row['stalemate_moves']/row['legal_moves'],.15)
                self.assertEqual(family_key(board),family_key(board.mirror()))
                self.assertEqual(family_key(board),family_key(board.transform(chess.flip_horizontal)))
                keys.append(row['family'])
        self.assertEqual(len(keys),len(set(keys)))
        specs=specs_for(corpus['test'],4)
        self.assertEqual(sum(chess.Board(s['fen']).turn for s in specs),len(specs)//2)
        for s in specs:self.assertEqual(s['players'][0 if chess.Board(s['fen']).turn else 1],0)

    def test_candidate_only_stage_metrics_and_clustered_statistics(self):
        models=[self.brain(),self.brain()]
        game,_=play_game(*models,max_plies=4)
        report=summarize([game],['white'])
        self.assertEqual(sum(s['decisions'] for s in report['per_stage'].values()),8)
        with self.assertRaises(ValueError):summarize([game],[])
        interval=cluster_interval([0.,1.],draws=1000)
        self.assertEqual(interval['clusters'],2);self.assertEqual(interval['mean'],.5)
        self.assertEqual(paired_difference(dict(position_mate_rates=[.5,.5]),dict(position_mate_rates=[.5,.5]))['ci95'],[0,0])

    def test_new_trainer_resume_exact(self):
        corpus=make_corpus(train=8,validation=4,test=4)
        with tempfile.TemporaryDirectory(dir=ROOT) as folder:
            full=Path(folder)/'full';split=Path(folder)/'split'
            args=dict(games=8,seed=2,eval_every=1)
            a,_=run_train(corpus,full,updates=2,**args)
            run_train(corpus,split,updates=1,**args)
            b,_=run_train(corpus,split,updates=2,resume=split/'resume.pt',**args)
            for m,n in zip(a,b):self.assertEqual(state_hash(m),state_hash(n))
            import json
            history=json.loads((full/'history.json').read_text())['history']
            opponents=[r['validation_opponents'] for r in history if 'validation' in r]
            self.assertTrue(all(p==opponents[0] for p in opponents))

    def test_study_statistics_and_bounds(self):
        from flies.chess.study_report import two_way_interval
        interval=two_way_interval(np.full((3,8),.25),alpha=.0125,draws=200)
        self.assertEqual(interval['interval'],[.25,.25])
        self.assertEqual(interval['training_seeds'],3)
        self.assertEqual(interval['position_families'],8)
        self.assertEqual(interval['confidence'],.9875)
        with self.assertRaises(ValueError):run_train({},ROOT/'artifacts/test-invalid',games=7)

    def test_validation_uses_actual_initial_opponent_not_changing_pool(self):
        from flies.chess.train_selfplay import cpu_snapshot
        corpus=make_corpus(train=2,validation=2,test=2)
        with tempfile.TemporaryDirectory(dir=ROOT) as folder:
            folder=Path(folder)
            calls=[]
            def different_pool(brain):
                ck=cpu_snapshot(brain)
                ck['state_dict']['bias']+=.25
                return ck
            def inspect_opponent(candidate,opponent,*args,**kwargs):
                learner=len(calls)
                expected=load_policy(folder/f'initial_{1-learner}.pt',size='tiny')
                self.assertEqual(state_hash(opponent),state_hash(expected))
                calls.append(learner)
                return dict(first_move_mate=dict(mean=0.)),{}
            with patch('flies.chess.learning.cpu_snapshot',side_effect=different_pool), patch('flies.chess.learning.evaluate_models',side_effect=inspect_opponent):
                run_train(corpus,folder,updates=0,games=8)
            self.assertEqual(calls,[0,1])

    def test_shuffle_deranges_boards_despite_adjacent_color_repeats(self):
        corpus=make_corpus(train=2,validation=2,test=2)
        specs=specs_for(corpus['test'],repeats=4)
        games,_=play_batch([self.brain(),self.brain()],specs,max_plies=2,shuffle_observations=True)
        first=[s for g in games for s in g['moves'][0]['stages']]
        self.assertEqual(len(first),32)
        self.assertTrue(all(not s['same_board_donor'] for s in first))

    def test_training_never_reads_mate_labels(self):
        corpus=make_corpus(train=8,validation=4,test=4)
        changed=copy.deepcopy(corpus)
        for row in changed['train']:row['mate_moves']=['deliberately-invalid-oracle-label']
        with tempfile.TemporaryDirectory(dir=ROOT) as folder:
            a,_=run_train(corpus,Path(folder)/'a',updates=1,games=8,seed=31,eval_every=1)
            b,_=run_train(changed,Path(folder)/'b',updates=1,games=8,seed=31,eval_every=1)
            for m,n in zip(a,b):self.assertEqual(state_hash(m),state_hash(n))
