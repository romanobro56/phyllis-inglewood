import io
import copy
from pathlib import Path
import tempfile
from unittest.mock import patch
import chess
import chess.pgn
import torch
from flies.chess.selfplay import play_game,run_matches,to_pgn
from flies.chess.storage import ROOT,state_hash,file_hash
from flies.chess.train_selfplay import train
from flies.chess.tests.support import BrainTest


class OperationsTests(BrainTest):
    def test_cli_untrained_control_matches_encoding_and_temperature(self):
        from flies.chess.chess_env import SPATIAL_ENCODING
        from flies.chess.storage import atomic_write,brain_payload,load_policy,identity
        from flies.chess import evaluate
        model=self.brain();model.encoding=SPATIAL_ENCODING;model.policy_temperature=1.7
        with tempfile.TemporaryDirectory(dir=ROOT) as folder:
            root=Path(folder);path=root/'trained.pt';atomic_write(path,brain_payload(model),binary=True)
            seen=[];original=evaluate.compare
            def compare(candidate,*args,**kwargs):
                seen.append((candidate.encoding,candidate.policy_temperature))
                return original(candidate,*args,**kwargs)
            with patch('sys.argv',['evaluate','--checkpoint',str(path),'--size','test','--games','2',
                                   '--max-plies','2','--output',str(root/'eval')]),patch.object(evaluate,'compare',side_effect=compare):
                evaluate.main()
            self.assertEqual(seen[1],(SPATIAL_ENCODING,1.7))
            probe=root/'probe.pt';atomic_write(probe,dict(brain_payload(model),training_status='engineering-probe-only'),binary=True)
            self.assertEqual(identity(load_policy(probe,size='test'),probe)['label'],'ENGINEERING-PROBE')

    def test_export_rejects_fabricated_final_position_and_result(self):
        result,_=play_game(self.brain(),self.brain(),max_plies=2)
        invalid=copy.deepcopy(result);invalid['final_fen']=chess.STARTING_FEN
        with self.assertRaisesRegex(ValueError,'final FEN'):to_pgn(invalid)
        invalid=copy.deepcopy(result);invalid['outcome']['result']='1-0';invalid['outcome']['winner']='white'
        with self.assertRaisesRegex(ValueError,'outcome'):to_pgn(invalid)

    def test_cancel_at_move_boundary_and_canonical_live_fen(self):
        white,black=self.brain(),self.brain();snapshots=[]
        original=chess.STARTING_FEN.replace(' w ','  w ')
        game,_=play_game(white,black,fen=original,max_plies=20,
                         on_position=lambda p:snapshots.append((len(p['moves']),p['final_fen'])),
                         should_stop=lambda:len(snapshots)>=3)
        self.assertEqual(game['initial_fen'],chess.STARTING_FEN)
        self.assertEqual(game['plies'],2)
        self.assertTrue(game['outcome']['truncated']);self.assertEqual(game['outcome']['result'],'*')
        self.assertEqual(game['outcome']['truncation_reason'],'user_stop')
        self.assertEqual(chess.pgn.read_game(io.StringIO(to_pgn(game))).end().board().fen(),game['final_fen'])
        self.assertTrue(all(m['controller']=='FlyBrain' for m in game['moves']))

    def test_run_guard_preserves_existing_recordings_and_checkpoints(self):
        with tempfile.TemporaryDirectory(dir=ROOT) as folder:
            output=Path(folder)/'match'
            run_matches(output,size='test',games=1,max_plies=2)
            before=file_hash(output/'game_0000.json')
            with self.assertRaises(FileExistsError):run_matches(output,size='test',games=1,max_plies=2)
            self.assertEqual(file_hash(output/'game_0000.json'),before)
            training=Path(folder)/'training'
            train(training,updates=1,size='test',games_per_update=2,max_plies=2)
            before=file_hash(training/'resume.pt')
            with self.assertRaises(FileExistsError):train(training,updates=1,size='test')
            self.assertEqual(file_hash(training/'resume.pt'),before)

    def test_first_rollout_interrupt_and_noop_resume_repair(self):
        with tempfile.TemporaryDirectory(dir=ROOT) as folder:
            root=Path(folder);kwargs=dict(size='test',seed=71,games_per_update=2,max_plies=2)
            with patch('flies.chess.train_selfplay.play_game',side_effect=KeyboardInterrupt):
                with self.assertRaises(KeyboardInterrupt):train(root/'interrupted',updates=1,**kwargs)
            ck=torch.load(root/'interrupted/resume.pt',weights_only=False)
            self.assertEqual(ck['update'],0)
            restored,_=train(root/'interrupted',updates=1,resume=root/'interrupted/resume.pt',**kwargs)
            expected,_=train(root/'fresh',updates=1,**kwargs)
            self.assertEqual([state_hash(m) for m in restored],[state_hash(m) for m in expected])
            # Reproduce the post-commit/pre-export crash window without deleting
            # any research artifacts: only this isolated temporary test fixture.
            (root/'interrupted/brain_0.pt').unlink()
            (root/'interrupted/training.json').unlink()
            with patch('flies.chess.train_selfplay.play_game',side_effect=AssertionError('No rollout on no-op resume')):
                repaired,_=train(root/'interrupted',updates=1,resume=root/'interrupted/resume.pt',**kwargs)
            self.assertTrue((root/'interrupted/brain_0.pt').exists())
            self.assertTrue((root/'interrupted/training.json').exists())
            self.assertEqual([state_hash(m) for m in repaired],[state_hash(m) for m in expected])
