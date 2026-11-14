"""Passive deterministic replay audit; never change a fly's observations/actions."""
import argparse
import json
from pathlib import Path

import chess
import torch
import flies.chess.mechanism_train as training
from flies.chess.chess_env import SPATIAL_ENCODING
from flies.chess.research_positions import own_family
from flies.chess.storage import ROOT, state_hash, write_json


def same(a, b):
    if torch.is_tensor(a):
        return torch.equal(a, b)
    if isinstance(a, dict):
        return a.keys() == b.keys() and all(same(v, b[k]) for k, v in a.items())
    if isinstance(a, (list, tuple)):
        return len(a) == len(b) and all(same(x, y) for x, y in zip(a, b))
    return a == b


def run(corpus, seed):
    root = ROOT/'artifacts/mechanism'; visited = [set(), set()]; actual = training.play_batch
    def observe(models, specs, **kwargs):
        games, traces = actual(models, specs, **kwargs)
        for game, spec in zip(games, specs):
            for move in game['moves']:
                board = chess.Board(move['before'])
                learner = spec['players'][0 if board.turn else 1]
                visited[learner].add(own_family(board))
        return games, traces
    # This wrapper records completed actual games. It does not intercept neural
    # calls, change RNG states, provide rewards or supply any policy inputs.
    training.play_batch = observe
    output = root/f'audit-loo-{seed}'
    try:
        models, _ = training.train(corpus, [root/f'replication-{seed}/checkpoint_0056_{i}.pt' for i in (0,1)],
                                   output, seed=seed, encoding=SPATIAL_ENCODING)
    finally:
        training.play_batch = actual
    original = torch.load(root/f'loo-{seed}/resume.pt', map_location='cpu', weights_only=False)
    replay = torch.load(output/'resume.pt', map_location='cpu', weights_only=False)
    assert same(original['brains'], replay['brains']) and same(original['optimizers'], replay['optimizers'])
    checkpoint_equality = {}
    for update in (16,32,48):
        checkpoint_equality[str(update)] = []
        for i in (0,1):
            a = torch.load(root/f'loo-{seed}/checkpoint_{update:04d}_{i}.pt', weights_only=False)
            b = torch.load(output/f'checkpoint_{update:04d}_{i}.pt', weights_only=False)
            checkpoint_equality[str(update)].append(same(a, b))
        assert all(checkpoint_equality[str(update)])
    result = dict(seed=seed, visited_decision_families=[sorted(v) for v in visited],
                  models_identical=True, optimizers_identical=True, checkpoint_equality=checkpoint_equality,
                  final_state_hashes=[state_hash(m) for m in models],
                  note='Passive replay of the same fixed training experiment, not independent evidence or additional learning. Historical warm-start visits cannot be completely reconstructed.')
    write_json(root/f'visits-{seed}.json', result)
    print('PASS passive replay', seed, [len(v) for v in visited], flush=True)
    return result


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__); parser.add_argument('--seed', type=int, required=True)
    parser.add_argument('--corpus', default=str(ROOT/'artifacts/mechanism/curriculum.json'))
    args = parser.parse_args(); run(json.loads(Path(args.corpus).read_text()), args.seed)
