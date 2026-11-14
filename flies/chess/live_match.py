"""Animation-gated live inference. No worker may schedule its own next turn."""
import copy
import threading
import time
import uuid
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone

import chess
import numpy as np
import torch

from flies.chess.chess_env import ChessEnv, ACTIONS
from flies.chess.selfplay import assert_independent, choose_stage, save_game
from flies.chess.storage import ROOT, identity, load_policy, write_json


def stamp():
    return datetime.now(timezone.utc).isoformat()


class ProtocolError(ValueError):
    pass


def visual_move(event):
    """Rules metadata only; never a strategic decision."""
    board = chess.Board(event['before']); move = chess.Move.from_uci(event['uci'])
    if move not in board.legal_moves:
        raise ValueError('Illegal visual move')
    capture = move.to_square if board.is_capture(move) else None
    if board.is_en_passant(move):
        capture = move.to_square + (-8 if board.turn else 8)
    rook = None
    if board.is_castling(move):
        rank = chess.square_rank(move.from_square)
        kingside = chess.square_file(move.to_square) > chess.square_file(move.from_square)
        rook = [chess.square_name(chess.square(7 if kingside else 0, rank)),
                chess.square_name(chess.square(5 if kingside else 3, rank))]
    board.push(move)
    if board.fen() != event['fen']:
        raise ValueError('Visual FEN mismatch')
    return dict(source=chess.square_name(move.from_square), destination=chess.square_name(move.to_square),
                capture=chess.square_name(capture) if capture is not None else None,
                rook=rook, promotion=chess.piece_symbol(move.promotion) if move.promotion else None,
                check=board.is_check(), checkmate=board.is_checkmate(), stalemate=board.is_stalemate())


class LiveMatch:
    """Single serial worker; lock protects board, lease, epoch and event sequence.

    HTTP polling is observational. Only explicit next() starts inference, only
    a validated ack() leaves ANIMATING. Timeout/disconnect never count as ACK.
    """
    def __init__(self, output=ROOT / 'artifacts/cinematic', loader=load_policy,
                 ack_timeout=120., lease_timeout=20.):
        self.lock = threading.Condition(threading.RLock())
        self.worker = ThreadPoolExecutor(max_workers=1, thread_name_prefix='chess-brain')
        self.output = output
        self.loader = loader
        self.ack_timeout = ack_timeout
        self.lease_timeout = lease_timeout
        self.match = None
        self.closed = False
        self.watchdog = threading.Thread(target=self._watch, daemon=True)
        self.watchdog.start()

    def _event(self, kind, **data):
        m = self.match
        m['event_id'] += 1
        m['events'].append(dict(event_id=m['event_id'], kind=kind, timestamp=stamp(), **data))
        m['events'] = m['events'][-64:]

    def new(self, size='tiny', white=None, black=None, seed=42, max_plies=160,
            fen=chess.STARTING_FEN, greedy=False):
        seed, max_plies = int(seed), int(max_plies)
        if (size not in ('tiny', 'small', 'central', 'full', 'test')
                or not 1 <= max_plies <= 1024 or not 0 <= seed < 2**63):
            raise ProtocolError('Invalid preset, seed or move limit')
        env = ChessEnv(fen, max_plies)
        with self.lock:
            self._cancel_locked()
            key = uuid.uuid4().hex
            self.match = dict(id=key, token=uuid.uuid4().hex, epoch=0, phase='loading', paused=False,
                              error=None, event_id=0, events=[], env=env, pending=None, acked=0,
                              moves=[], telemetry={}, brains=None, players={}, seed=int(seed), greedy=bool(greedy),
                              heartbeat=time.monotonic(), pending_elapsed=0., tick=time.monotonic(),
                              created=stamp(), inference_counts={'white': 0, 'black': 0})
            self._event('loading')
            self.worker.submit(self._load, key, size, white, black)
            return self._snapshot()

    def _load(self, key, size, white, black):
        try:
            # This is the only worker. Initialization and all inference are serial.
            torch.set_num_threads(1)
            with self.lock:
                if not self._current(key): return
                seed = self.match['seed']
            brains = []
            for i, path in enumerate((white, black)):
                with torch.random.fork_rng():
                    torch.manual_seed(seed + i)
                    brain = self.loader(path, size, 'cpu')
                brain.eval(); brain.requires_grad_(False); brains.append(brain)
            assert_independent(*brains)
            players = {side: identity(brain, path) for side, brain, path in
                       zip(('white', 'black'), brains, (white, black))}
            infos = {side: dict(name=side, preset=b.preset, groups=b.groups.cpu().tolist(),
                               actions=ACTIONS, trained=bool(path), synthetic=b.synthetic)
                     for side, b, path in zip(('white', 'black'), brains, (white, black))}
            with self.lock:
                if not self._current(key): return
                m = self.match; m.update(brains=brains, players=players, infos=infos,
                                         rng=np.random.default_rng(seed),
                                         phase='finished' if m['env'].done else 'ready')
                self._event('ready'); self._persist()
        except Exception as exc:
            self._fail(key, exc)

    def _current(self, key):
        return self.match is not None and self.match['id'] == key and self.match['phase'] != 'cancelled'

    def _auth(self, request):
        m = self.match
        if not m or request.get('match_id') != m['id'] or request.get('token') != m['token']:
            raise ProtocolError('Stale match or invalid token')
        if request.get('epoch') != m['epoch']:
            raise ProtocolError('Stale connection epoch; claim the match again')
        m['heartbeat'] = time.monotonic()
        return m

    def claim(self, match_id):
        with self.lock:
            if not self.match or self.match['id'] != match_id:
                raise ProtocolError('Unknown match')
            self.match['epoch'] += 1
            self.match['paused'] = True  # Recovery requires deliberate resume.
            self.match['heartbeat'] = time.monotonic()
            self._event('claimed', epoch=self.match['epoch'])
            self.lock.notify_all()
            return self._snapshot()

    def next(self, request):
        with self.lock:
            m = self._auth(request)
            if m['phase'] != 'ready' or m['paused'] or m['pending']:
                raise ProtocolError('Next inference requires READY, resumed, acknowledged board')
            m['phase'] = 'thinking'
            side = 'white' if m['env'].board.turn else 'black'
            self._event('inference_started', side=side, sequence=m['acked'] + 1)
            self.worker.submit(self._infer, m['id'])
            return self._snapshot()

    def _infer(self, key):
        try:
            stages = []
            while True:
                with self.lock:
                    while self._current(key) and self.match['paused'] and not self.closed:
                        self.lock.wait(.2)
                    if not self._current(key) or self.closed: return
                    m = self.match; env = m['env']; side = 'white' if env.board.turn else 'black'
                    brain = m['brains'][0 if side == 'white' else 1]
                    rng = m['rng']; seq = m['acked'] + 1
                started = stamp()
                action, record, transition = choose_stage(brain, env, rng, greedy=m['greedy'],
                                                          activity=True, record_rates=True)
                completed = stamp()
                with self.lock:
                    if not self._current(key): return
                    record.update(started=started, timestamp=completed)
                    packet = dict(match_id=key, side=side, sequence=seq, stage=record['stage'],
                                  decision_id=f'{key}:{seq}:{len(stages)}', started=started,
                                  timestamp=completed, observation=transition['obs'].tolist(),
                                  observation_encoding=getattr(brain, 'encoding', None),
                                  mask=record['mask'], output_groups=record['activity']['group_rate'],
                                  mean_rate=record['activity']['mean_rate'], action=ACTIONS[action],
                                  probs=dict(zip(ACTIONS, record['probabilities'])), activity=record.pop('rates'),
                                  caption=f"{side.title()} · {record['stage']} → {ACTIONS[action]}")
                    m['telemetry'][side] = packet
                    m['inference_counts'][side] += 1
                    stages.append(record)
                    self._event('inference_stage', side=side, sequence=seq, stage=record['stage'],
                                decision_id=packet['decision_id'])
                    event = env.select(action)
                    if event is not None:
                        event.update(sequence=seq, match_id=key, controller='FlyBrain', stages=stages)
                        event.update(inference_started=stages[0]['started'], inference_completed=completed,
                                     animation_acknowledged=None)
                        event['visual'] = visual_move(event)
                        m['moves'].append(event); m['pending'] = event
                        m['phase'] = 'animating'; m['pending_elapsed'] = 0.; m['tick'] = time.monotonic()
                        self._event('inference_completed', side=side, sequence=seq)
                        self._event('animation_required', sequence=seq)
                        self._persist()
                        return  # NEVER schedule the other player's inference here.
        except Exception as exc:
            self._fail(key, exc)

    def ack(self, request):
        with self.lock:
            m = self._auth(request); p = m['pending']
            if not p or m['phase'] not in ('animating', 'error'):
                raise ProtocolError('No pending animation; duplicate acknowledgment rejected')
            if request.get('sequence') != p['sequence'] or request.get('rendered_fen') != p['fen']:
                raise ProtocolError('Out-of-order sequence or rendered FEN mismatch')
            expected = chess.Board(p['fen']).board_fen()
            if request.get('rendered_placement') != expected:
                raise ProtocolError('Rendered piece placement disagrees with authoritative board')
            m['acked'] = p['sequence']; m['pending'] = None; m['error'] = None
            p['animation_acknowledged'] = stamp()
            m['phase'] = 'finished' if m['env'].done else 'ready'
            self._event('animation_acknowledged', sequence=m['acked'])
            self._persist()
            return self._snapshot()

    def pause(self, request):
        with self.lock:
            m = self._auth(request); m['paused'] = bool(request.get('paused', True))
            self._event('paused' if m['paused'] else 'resumed')
            self.lock.notify_all()
            return self._snapshot()

    def cancel(self, request):
        with self.lock:
            self._auth(request); self._cancel_locked()
            return self._snapshot()

    def _cancel_locked(self):
        if self.match and self.match['phase'] not in ('cancelled', 'finished'):
            m = self.match
            m['phase'] = 'cancelled'; m['paused'] = True
            m['user_stop'] = not m['env'].done
            if not m['env'].done:
                m['env'].truncated = True
            self._event('cancelled')
            if m['brains']: self._persist()
            self.lock.notify_all()

    def _fail(self, key, exc):
        with self.lock:
            if self._current(key):
                self.match.update(phase='error', error=str(exc), paused=True)
                self._event('error', message=str(exc))

    def _watch(self):
        while not self.closed:
            with self.lock:
                m = self.match
                if m:
                    now = time.monotonic(); elapsed = now - m['tick']; m['tick'] = now
                    if not m['paused'] and m['phase'] in ('thinking', 'animating', 'ready'):
                        if now - m['heartbeat'] > self.lease_timeout:
                            m['paused'] = True; self._event('connection_lost')
                        elif m['phase'] == 'animating':
                            m['pending_elapsed'] += elapsed
                            if m['pending_elapsed'] > self.ack_timeout:
                                m.update(phase='error', paused=True, error='Animation acknowledgment timeout; recover pending move or restart')
                                self._event('animation_timeout')
            time.sleep(.1)

    def state(self, request=None):
        with self.lock:
            if request: self._auth(request)
            return self._snapshot()

    def telemetry(self, side):
        if side not in ('white', 'black'): raise ProtocolError('Unknown side')
        with self.lock:
            if not self.match: return None
            return copy.deepcopy(self.match['telemetry'].get(side))

    def _snapshot(self):
        m = self.match
        if not m: return dict(phase='empty')
        env = m['env']
        return dict(match_id=m['id'], token=m['token'], epoch=m['epoch'], phase=m['phase'],
                    paused=m['paused'], error=m['error'], sequence=m['acked'], event_id=m['event_id'],
                    events=copy.deepcopy(m['events']), pending=copy.deepcopy(m['pending']),
                    initial_fen=env.initial_fen, fen=env.board.fen(), side='white' if env.board.turn else 'black',
                    check=env.board.is_check(), outcome=env.outcome(),
                    history=[dict(sequence=e['sequence'], san=e['san'], uci=e['uci'], side=e['side']) for e in m['moves']],
                    players=m['players'], infos=m.get('infos', {}), inference_counts=dict(m['inference_counts']),
                    activity_ids={s: p['decision_id'] for s, p in m['telemetry'].items()},
                    recording=f"/artifacts/cinematic/{m['id']}/game_0000.json",
                    pgn=f"/artifacts/cinematic/{m['id']}/game_0000.pgn")

    def _persist(self):
        m = self.match; env = m['env']
        record = dict(format_version=1, mode='live-two-FlyBrain-animation-gated', seed=m['seed'],
                      initial_fen=env.initial_fen, final_fen=env.board.fen(), outcome=env.outcome(),
                      moves=m['moves'], players=m['players'], plies=env.plies,
                      visually_acknowledged_through=m['acked'], match_id=m['id'], created=m['created'])
        if m.get('user_stop') and record['outcome']['truncated']:
            record['outcome']['truncation_reason'] = 'user_stop'
        directory = self.output / m['id']
        save_game(record, directory, 0)
        write_json(directory / 'protocol.json', dict(events=m['events'], phase=m['phase'], acked=m['acked'],
                                                     inference_counts=m['inference_counts']))

    def close(self):
        with self.lock:
            self.closed = True; self._cancel_locked(); self.lock.notify_all()
        self.worker.shutdown(wait=True)
