"""Independent rational audit of integer-v2 rewards on validation trajectories."""
import argparse
from fractions import Fraction
import json
from pathlib import Path

import chess
import numpy as np
import torch
from flies.chess.batched import play_batch
from flies.chess.learning import specs_for
from flies.chess.research_train import own_turn_rewards, potential
from flies.chess.storage import ROOT, load_policy, state_hash, write_json
from flies.chess.train_selfplay import policy_update


def rational_potential(fen, side):
    """Independent implementation: material + pawn advance in exact fractions."""
    board = chess.Board(fen)
    values = {chess.PAWN: 1, chess.KNIGHT: 3, chess.BISHOP: 3,
              chess.ROOK: 5, chess.QUEEN: 9, chess.KING: 0}
    total = Fraction(0)
    for square in chess.SQUARES:
        piece = board.piece_at(square)
        if piece is None:
            continue
        value = Fraction(values[piece.piece_type], 10)
        if piece.piece_type == chess.PAWN:
            rank = chess.square_rank(square)
            value += Fraction(rank - 1 if piece.color else 6 - rank, 200)
        total += value if piece.color == (side == 'white') else -value
    return total


def rational_rewards(game, side, auxiliary=True):
    events = [m for m in game['moves'] if m['side'] == side]
    rewards = []
    for i, event in enumerate(events):
        end = events[i+1]['before'] if i+1 < len(events) else game['final_fen']
        delta = rational_potential(end, side) - rational_potential(event['before'], side)
        rewards.append(float(max(Fraction(-1), min(Fraction(1), delta)) / 4) if auxiliary else 0.)
    if rewards:
        winner = game['outcome']['winner']
        rewards[-1] += 0. if winner is None else 1. if winner == side else -1.
    return rewards


def audit_games(games):
    max_error = 0.; legacy_error = 0.; zeros = 0; legacy_false_nonzero = 0; moves = 0
    for game in games:
        for side in ('white', 'black'):
            reference = np.asarray(rational_rewards(game, side))
            actual = np.asarray(own_turn_rewards(game, side, True, 'integer-v2'))
            legacy = np.asarray(own_turn_rewards(game, side, True, 'float-v1'))
            np.testing.assert_allclose(actual, reference, atol=1e-15, rtol=0)
            zero = reference == 0
            if np.any(actual[zero] != 0):
                raise AssertionError('A rational zero received a nonzero integer-v2 reward')
            zeros += int(zero.sum()); moves += len(actual)
            legacy_false_nonzero += int(np.count_nonzero(legacy[zero]))
            max_error = max(max_error, float(np.max(abs(actual-reference), initial=0)))
            legacy_error = max(legacy_error, float(np.max(abs(legacy-reference), initial=0)))
    return dict(games=len(games), own_turn_rewards=moves, exact_rational_zeros=zeros,
                integer_false_nonzeros=0, legacy_false_nonzeros=legacy_false_nonzero,
                integer_max_absolute_error=max_error, legacy_max_absolute_error=legacy_error)


def run(extended, sources, output):
    torch.set_num_threads(1)
    models = [load_policy(p) for p in sources]
    hashes = [state_hash(m) for m in models]
    # Validation only; no historical/fresh test positions are opened here.
    rows = extended['validation']
    games, traces = play_batch(models, specs_for(rows, 2, seed=410001), max_plies=8)
    audit = audit_games(games)
    gradients = {}
    for version in ('float-v1', 'integer-v2'):
        brain = load_policy(sources[0]); samples = []
        for game, trace in zip(games, traces):
            side = game['candidate']
            samples.append((trace[side], own_turn_rewards(game, side, True, version)))
        metric, _ = policy_update(brain, torch.optim.SGD(brain.parameters(), lr=0.), samples,
                                  0., gamma=.98, entropy=0.)
        assert state_hash(brain) == hashes[0]
        gradients[version] = metric
    assert [state_hash(m) for m in models] == hashes
    result = dict(audit=audit, reward_only_gradients=gradients, sources=list(map(str, sources)),
                  corpus_sha256=extended['sha256'], models_unchanged=True,
                  note='Actual independent FlyBrain validation games; rational oracle scores only; lr=0, entropy=0. No training or test evaluation.')
    write_json(output, result)
    print(json.dumps(audit, indent=2), flush=True)
    return result


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--extended', default=str(ROOT/'research_data/extended.json'))
    parser.add_argument('--sources', nargs=2, required=True)
    parser.add_argument('--output', default=str(ROOT/'artifacts/mechanism/reward_validation.json'))
    args = parser.parse_args()
    run(json.loads(Path(args.extended).read_text()), args.sources, args.output)
