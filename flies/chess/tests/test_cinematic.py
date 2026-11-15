"""Live protocol regression tests, executing the real FlyBrain implementation."""
import base64
import os
import tempfile
import threading
import time
import unittest
from pathlib import Path
from unittest import skipUnless
from unittest.mock import patch

import chess
import numpy as np
import torch

from flies.chess.live_match import LiveMatch, ProtocolError, visual_move
from flies.chess.tests.support import BrainTest
from flies.chess.storage import ROOT, state_hash, load_policy
from flies.chess.selfplay import choose_stage


class LiveTests(BrainTest):
    def make(self, **kwargs):
        temp = tempfile.TemporaryDirectory(dir=ROOT / 'artifacts'); self.addCleanup(temp.cleanup)
        controller = LiveMatch(output=Path(temp.name), loader=lambda *args: self.brain(), **kwargs)
        self.addCleanup(controller.close)
        state = controller.new(size='test', max_plies=4)
        return controller, self.wait(controller, 'ready')

    def wait(self, controller, phase):
        end = time.monotonic() + 15
        while time.monotonic() < end:
            state = controller.state()
            if state['phase'] == phase: return state
            if state['phase'] == 'error': self.fail(state['error'])
            time.sleep(.01)
        self.fail('Timeout: ' + str(state))

    def auth(self, state):
        return {k: state[k] for k in ('match_id', 'token', 'epoch')}

    def ack(self, state):
        p = state['pending']
        return dict(self.auth(state), sequence=p['sequence'], rendered_fen=p['fen'],
                    rendered_placement=chess.Board(p['fen']).board_fen())

    def test_animation_gate_and_activity_provenance(self):
        c, s = self.make(); before = [state_hash(b) for b in c.match['brains']]
        c.next(self.auth(s)); s = self.wait(c, 'animating')
        self.assertEqual(s['inference_counts'], dict(white=4, black=0))
        with self.assertRaises(ProtocolError): c.next(self.auth(s))
        time.sleep(.1); self.assertEqual(c.state()['inference_counts']['black'], 0)
        packet = c.telemetry('white')
        self.assertEqual(packet['match_id'], s['match_id']); self.assertEqual(packet['side'], 'white')
        self.assertEqual(len(base64.b64decode(packet['activity']['data'])), 8*176)
        self.assertEqual(len(packet['observation']), 64); self.assertIsNone(c.telemetry('black'))
        with torch.no_grad():
            _, info=c.match['brains'][0](torch.tensor([packet['observation']]),
                torch.tensor([packet['mask']]), record=True)
        rates=info['rates'][:,0,:].cpu().numpy()
        expected=np.clip(np.sqrt(np.clip(rates,0,None)/2)*255,0,255).astype(np.uint8)
        self.assertEqual(base64.b64decode(packet['activity']['data']),expected.tobytes())
        bad = self.ack(s); bad['sequence'] += 1
        with self.assertRaises(ProtocolError): c.ack(bad)
        bad = self.ack(s); bad['rendered_placement'] = chess.Board().board_fen()
        with self.assertRaises(ProtocolError): c.ack(bad)
        c.ack(self.ack(s))
        with self.assertRaises(ProtocolError): c.ack(self.ack(s))
        c.next(self.auth(s)); s = self.wait(c, 'animating')
        self.assertEqual(s['inference_counts']['black'], 4)
        previous=c.match['moves'][0]
        self.assertLess(previous['inference_completed'],previous['animation_acknowledged'])
        self.assertLess(previous['animation_acknowledged'],s['pending']['inference_started'])
        self.assertEqual(c.telemetry('black')['side'], 'black')
        self.assertEqual(before, [state_hash(b) for b in c.match['brains']])
        self.assertNotEqual(c.match['brains'][0].log_w.data_ptr(), c.match['brains'][1].log_w.data_ptr())
        c.ack(self.ack(s))

    def test_pause_recovery_epoch_restart_and_timeout(self):
        c, s = self.make(ack_timeout=.12)
        c.pause(dict(self.auth(s), paused=True))
        with self.assertRaises(ProtocolError): c.next(self.auth(s))
        c.pause(dict(self.auth(s), paused=False)); c.next(self.auth(s)); s = self.wait(c, 'animating')
        old = self.ack(s); recovered = c.claim(s['match_id'])
        with self.assertRaises(ProtocolError): c.ack(old)
        c.pause(dict(self.auth(recovered), paused=False))
        end = time.monotonic()+2
        while c.state()['phase'] != 'error' and time.monotonic() < end: time.sleep(.02)
        self.assertEqual(c.state()['phase'], 'error'); self.assertEqual(c.state()['sequence'], 0)
        self.assertEqual(c.state()['inference_counts']['black'], 0)
        c.ack(self.ack(c.state()))
        new = c.new(size='test', max_plies=1); self.wait(c,'ready')
        with self.assertRaises(ProtocolError): c.ack(old)
        c.next(self.auth(new)); s = self.wait(c,'animating'); s = c.ack(self.ack(s))
        self.assertEqual(s['phase'], 'finished'); self.assertTrue(s['outcome']['truncated'])
        self.assertEqual(s['outcome']['result'], '*')
        c.cancel(self.auth(s))
        import json
        recorded=json.loads((c.output/s['match_id']/'game_0000.json').read_text())
        self.assertNotIn('truncation_reason',recorded['outcome'])

    def test_rules_visual_metadata(self):
        fixtures = [
            ('r3k2r/8/8/8/8/8/8/R3K2R w KQkq - 0 1','e1g1','rook',['h1','f1']),
            ('4k3/8/8/3pP3/8/8/8/4K3 w - d6 0 1','e5d6','capture','d5'),
            ('4k3/P7/8/8/8/8/8/4K3 w - - 0 1','a7a8n','promotion','n')]
        for fen, uci, key, expected in fixtures:
            b=chess.Board(fen); before=b.fen();b.push_uci(uci)
            self.assertEqual(visual_move(dict(before=before,fen=b.fen(),uci=uci))[key], expected)

    def test_restart_during_inference_cannot_publish_stale_move(self):
        c,s=self.make();entered=threading.Event();release=threading.Event()
        def blocked(*args,**kwargs):
            entered.set();release.wait(3);return choose_stage(*args,**kwargs)
        with patch('flies.chess.live_match.choose_stage',side_effect=blocked):
            c.next(self.auth(s));self.assertTrue(entered.wait(2))
            new=c.new(size='test',max_plies=2);release.set();new=self.wait(c,'ready')
        self.assertNotEqual(new['match_id'],s['match_id'])
        self.assertEqual(new['inference_counts'],dict(white=0,black=0))
        self.assertEqual(new['fen'],chess.STARTING_FEN);self.assertIsNone(new['pending'])

    def test_disconnection_pauses_without_acknowledging(self):
        c,s=self.make(lease_timeout=.18)
        c.next(self.auth(s));s=self.wait(c,'animating')
        time.sleep(.4);s=c.state()
        self.assertTrue(s['paused']);self.assertEqual(s['sequence'],0)
        self.assertIsNotNone(s['pending']);self.assertEqual(s['inference_counts']['black'],0)
        self.assertIn('connection_lost',[e['kind'] for e in s['events']])

    def test_invalid_new_settings_preserve_active_match(self):
        c,s=self.make()
        with self.assertRaises(ProtocolError):c.new(size='test',seed=-1)
        self.assertEqual(c.state()['match_id'],s['match_id'])
        self.assertEqual(c.state()['phase'],'ready')

    @skipUnless(os.environ.get('CHESS_REAL_GRAPH_TESTS')=='1','opt-in actual FlyWire graph')
    def test_actual_tiny_live_gate(self):
        # Bypass this class's unit graph patch only for this integration.
        self.patcher.stop()
        temp = tempfile.TemporaryDirectory(dir=ROOT/'artifacts');self.addCleanup(temp.cleanup)
        c = LiveMatch(output=Path(temp.name));self.addCleanup(c.close)
        s=c.new(max_plies=2);s=self.wait(c,'ready')
        self.assertFalse(s['players']['white']['synthetic']);self.assertEqual(s['players']['white']['neurons'],5000)
        for side in ('white','black'):
            c.next(self.auth(s));s=self.wait(c,'animating')
            self.assertEqual(s['pending']['side'],side)
            packet=c.telemetry(side);self.assertEqual(packet['activity']['n'],5000)
            s=c.ack(self.ack(s))
        self.assertEqual(s['phase'],'finished');self.assertEqual(s['sequence'],2)


class TransportTests(unittest.TestCase):
    def test_static_read_only_routes_and_checkpoint_scope(self):
        import io
        from flies.chess.cinematic import handler_for, checkpoint_path
        Handler=handler_for(None,{})
        h=Handler.__new__(Handler)
        h.wfile=io.BytesIO();h.send_response=lambda value:setattr(h,'status',value)
        h.send_header=lambda *args:None;h.end_headers=lambda:None
        for path,status in [('/web/lib/fly.js',200),('/vendor/matrix.js',200),('/vendor/matrix_host.html',200),
                            ('/data/tiny.meta.json',200),('/data/tiny.npz',404),
                            ('/../AGENTS.md',404),('/artifacts/none.pt',404),('/',200)]:
            h.path=path;h.do_GET();self.assertEqual(h.status,status,path)
        with self.assertRaises(ProtocolError):checkpoint_path('../../core/brain.py')
        with self.assertRaises(ProtocolError):checkpoint_path('artifacts/none.pt')

    def test_cross_origin_commands_rejected(self):
        from flies.chess.cinematic import handler_for
        Handler=handler_for(None,{})
        h=Handler.__new__(Handler);h.headers={'Origin':'https://elsewhere.invalid','Host':'127.0.0.1:8770'}
        responses=[];h.reply=lambda value,status=200:responses.append((value,status))
        h.do_POST();self.assertEqual(responses[0][1],409)
        self.assertIn('Cross-origin',responses[0][0]['error'])
