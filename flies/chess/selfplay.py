"""Genuine two-FlyBrain chess; each stage calls the current side's own brain."""
import argparse
import signal
import threading
import time
from pathlib import Path

import chess
import chess.pgn
import numpy as np
import torch

from core.brain import pick_device
from flies.chess.chess_env import ChessEnv, STAGES, ENCODING
from flies.chess.storage import ROOT, atomic_write, identity, load_policy, write_json, prepare_run


def choose_stage(brain, env, rng, lesion=False, greedy=False, activity=False, record_rates=False):
    if env.done:
        raise RuntimeError("Attempted inference on terminal board")
    device = next(brain.parameters()).device
    obs, mask = env.observe(getattr(brain, "encoding", ENCODING)), env.mask()
    with torch.no_grad():
        kwargs = dict(lesion=lesion)
        if record_rates:
            kwargs['record'] = True
        logits, info = brain(torch.as_tensor(obs[None], device=device),
                             torch.as_tensor(mask[None], device=device), **kwargs)
        if not torch.isfinite(logits).all():
            raise ValueError("FlyBrain returned nonfinite logits")
        probs = (logits[0] / getattr(brain, "policy_temperature", 1.0)).softmax(-1).cpu().numpy().astype(float)
    probs[~mask] = 0
    probs /= probs.sum()
    action = int(np.argmax(probs)) if greedy else int(rng.choice(8, p=probs))
    if not mask[action]:
        raise RuntimeError("FlyBrain chose a masked action; no fallback is allowed")
    record = dict(stage=STAGES[env.stage], action=action, mask=mask.tolist(),
                  probabilities=probs.tolist(), selected_probability=float(probs[action]),
                  side="white" if env.board.turn else "black", fen=env.board.fen())
    if activity:
        record["activity"] = dict(mean_rate=info["mean_rate"], group_rate=info["group_rate"][0].tolist())
    if record_rates:
        import base64
        rates = info['rates'][:, 0, :].detach().cpu().numpy()
        quantized = np.clip(np.sqrt(np.clip(rates, 0, None) / 2) * 255, 0, 255).astype(np.uint8)
        record['rates'] = dict(steps=len(rates), n=brain.n,
                               data=base64.b64encode(quantized.tobytes()).decode('ascii'),
                               encoding='uint8-sqrt-rate-over-2', units='simulated neuron rates')
    # Only bounded CPU data retained, never a whole game's autograd graph.
    transition = dict(obs=obs, mask=mask, action=action)
    return action, record, transition


def assert_independent(white, black):
    if white is black:
        raise ValueError("Self-play requires two distinct FlyBrain models")
    if (white.graph_hash != black.graph_hash or white.preset != black.preset
            or white.fly_name != "chess" or black.fly_name != "chess"):
        raise ValueError("Both brains must use the same chess graph architecture")
    for a, b in zip(white.parameters(), black.parameters()):
        if a.data_ptr() == b.data_ptr():
            raise ValueError("Self-play parameters share storage")


def play_game(white, black, seed=0, max_plies=160, fen=chess.STARTING_FEN,
              greedy=False, lesions=(False, False), activity=False, claim_draw=True,
              metadata=None, random_side=None, on_position=None, should_stop=None):
    """random_side is exclusively an explicitly labeled evaluation baseline."""
    assert_independent(white, black)
    if random_side not in (None, "white", "black"):
        raise ValueError("Invalid random baseline side")
    rng = np.random.default_rng(seed)
    env = ChessEnv(fen, max_plies, claim_draw)
    fen = env.initial_fen  # Canonical FEN also makes live/replay continuity exact.
    t0 = time.monotonic()
    moves, trajectories = [], {"white": [], "black": []}
    user_stopped = False
    if on_position:
        on_position(dict(initial_fen=fen, final_fen=fen, moves=[], players=metadata or {},
                         seed=seed, outcome=env.outcome(), mode="random-legal-baseline" if random_side else "two-FlyBrain-selfplay"))
    while not env.done:
        if should_stop and should_stop():
            env.truncated = True  # A user stop is unfinished, never a chess draw.
            user_stopped = True
            break
        side = "white" if env.board.turn else "black"
        brain = white if env.board.turn else black
        stages, transitions = [], []
        if side == random_side:
            legal = tuple(env.board.legal_moves)
            event = env.push(legal[int(rng.integers(len(legal)))])
            event["controller"] = "uniform-random-legal-baseline"
        else:
            event = None
            while event is None:
                action, record, transition = choose_stage(brain, env, rng,
                    lesions[0 if side == "white" else 1], greedy, activity)
                stages.append(record)
                transitions.append(transition)
                event = env.select(action)
            event["controller"] = "FlyBrain"
            trajectories[side].append(transitions)
        event["stages"] = stages
        moves.append(event)
        if on_position:
            on_position(dict(initial_fen=fen, final_fen=env.board.fen(), moves=moves,
                             players=metadata or {}, seed=seed, outcome=env.outcome(),
                             mode="random-legal-baseline" if random_side else "two-FlyBrain-selfplay"))
    result = dict(format_version=1, mode="random-legal-baseline" if random_side else "two-FlyBrain-selfplay",
                  seed=seed, numpy_bit_generator=type(rng.bit_generator).__name__, greedy=greedy,
                  max_plies=max_plies, initial_fen=fen, final_fen=env.board.fen(),
                  outcome=env.outcome(), plies=env.plies, illegal_attempts=env.illegal_attempts,
                  elapsed_seconds=time.monotonic() - t0, moves=moves, players=metadata or {},
                  random_side=random_side)
    if user_stopped:result['outcome']['truncation_reason']='user_stop'
    return result, trajectories


def run_matches(output, white_checkpoint=None, black_checkpoint=None, games=2, seed=42,
                size='tiny', device='cpu', max_plies=160, fen=chess.STARTING_FEN,
                greedy=False, activity=False, live=False, delay=0., no_claim_draw=False,
                should_stop=None):
    if games < 1 or max_plies<1 or not np.isfinite(delay) or delay < 0:
        raise ValueError('Games/plies must be positive and delay finite/nonnegative')
    output = prepare_run(output)
    torch.set_num_threads(1); torch.manual_seed(seed)
    dev = pick_device(device)
    white = load_policy(white_checkpoint, size, dev); black = load_policy(black_checkpoint, size, dev)
    players = dict(white=identity(white, white_checkpoint), black=identity(black, black_checkpoint))
    print({side:{k:p[k] for k in ('label','preset','synthetic','graph_hash')} for side,p in players.items()},flush=True)
    results = []
    def publish(position):
        write_json(output/'live.json', position)
        if delay:
            until=time.monotonic()+delay
            while time.monotonic()<until and not (should_stop and should_stop()):
                time.sleep(min(.1,max(0.,until-time.monotonic())))
    for index in range(games):
        if should_stop and should_stop():break
        result, _ = play_game(white, black, seed+index, max_plies, fen, greedy,
                             activity=activity, claim_draw=not no_claim_draw, metadata=players,
                             on_position=publish if live else None, should_stop=should_stop)
        result['initialization_seed'] = seed; result['device'] = str(dev)
        save_game(result, output, index)
        if live:write_json(output/'live.json',result)
        results.append({k:result[k] for k in ('seed','outcome','plies','elapsed_seconds','illegal_attempts')})
        write_json(output/'summary.json',dict(players=players,games=results))
        print(results[-1],flush=True)
    return dict(players=players,games=results)


def to_pgn(result):
    game = chess.pgn.Game()
    board = chess.Board(result["initial_fen"])
    game.setup(board)
    game.headers.update(Event=result["mode"], Result=result["outcome"]["result"],
                        White=result.get("players", {}).get("white", {}).get("label", "FlyBrain"),
                        Black=result.get("players", {}).get("black", {}).get("label", "FlyBrain"))
    node = game
    for item in result["moves"]:
        move = chess.Move.from_uci(item["uci"])
        if item["before"] != board.fen() or move not in board.legal_moves:
            raise ValueError("Recording failed replay verification")
        board.push(move)
        if item["fen"] != board.fen():
            raise ValueError("Recording FEN mismatch")
        node = node.add_variation(move)
    if board.fen()!=result['final_fen']:
        raise ValueError('Recording final FEN mismatch')
    declared=result['outcome'];actual=board.outcome(claim_draw=declared.get('claim_draw',True))
    expected_result=actual.result() if actual else '*'
    expected_winner=('white' if actual.winner else 'black') if actual and actual.winner is not None else None
    if (declared['result']!=expected_result or declared.get('winner')!=expected_winner
            or declared.get('termination')!=(actual.termination.name if actual else None)
            or (declared['truncated'] and actual is not None)):
        raise ValueError('Recording outcome disagrees with chess rules')
    node.comment = ("Stopped by user at a move boundary; no chess result adjudicated."
                    if result['outcome'].get('truncation_reason')=='user_stop' else
                    "Truncated at explicit compute cap; no chess result adjudicated."
                    if result["outcome"]["truncated"] else str(result["outcome"]["termination"]))
    return str(game) + "\n"


def save_game(result, directory, index):
    directory = Path(directory)
    # PGN verifies the move/FEN stream before either artifact is published.
    pgn = to_pgn(result)
    atomic_write(directory / f"game_{index:04d}.pgn", pgn)
    write_json(directory / f"game_{index:04d}.json", result)


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--white-checkpoint")
    ap.add_argument("--black-checkpoint")
    ap.add_argument("--games", type=int, default=2)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--size", default="tiny")
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--max-plies", type=int, default=160)
    ap.add_argument("--output", default=str(ROOT / "artifacts/matches"))
    ap.add_argument("--fen", default=chess.STARTING_FEN)
    ap.add_argument("--greedy", action="store_true")
    ap.add_argument("--activity", action="store_true")
    ap.add_argument("--live", action="store_true", help="publish atomic live.json after each model move")
    ap.add_argument("--delay", type=float, default=0, help="optional live display pacing, seconds per completed move")
    ap.add_argument("--no-claim-draw", action="store_true")
    a = ap.parse_args()
    if a.games < 1 or a.delay < 0:
        ap.error("--games must be positive and --delay nonnegative")
    stop=threading.Event();previous=signal.getsignal(signal.SIGINT)
    def request_stop(signum,frame):
        if stop.is_set():raise KeyboardInterrupt
        stop.set()
    signal.signal(signal.SIGINT,request_stop)
    try:run_matches(**vars(a),should_stop=stop.is_set)
    finally:signal.signal(signal.SIGINT,previous)


if __name__ == "__main__":
    main()
