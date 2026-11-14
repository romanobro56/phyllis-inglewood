"""Training-only opportunity strata and mate-one suffix curriculum curation."""
import argparse
import hashlib
import json
from pathlib import Path

import chess
import numpy as np
from flies.chess.research_positions import mate_one, own_family
from flies.chess.storage import ROOT, write_json


def digest(data):
    return hashlib.sha256(json.dumps(data, sort_keys=True).encode()).hexdigest()


def suffixes(rows, forbidden, rng, limit=4):
    """Enumerate rule-legal construction prefixes, never play these for a fly.

    Retain at most four distinct mate-one suffix families per training root.
    The oracle curates *starting positions* only. Training never reads its moves.
    """
    result = []; seen = set(forbidden)
    for row in rows:
        root = chess.Board(row['fen']); first = list(root.legal_moves); rng.shuffle(first)
        retained = 0
        for move in first:
            after = root.copy(); after.push(move)
            replies = list(after.legal_moves); rng.shuffle(replies)
            for reply in replies:
                board = after.copy(); board.push(reply)
                family = own_family(board)
                if family in seen or not board.is_valid() or board.outcome() is not None:
                    continue
                if not mate_one(board):
                    continue
                result.append(dict(fen=board.fen(), family=family, parent_family=row['family'],
                                   construction_prefix=[move.uci(), reply.uci()],
                                   curation='mate-one suffix of legal two-ply construction; not a played fly prefix'))
                seen.add(family); retained += 1
                if retained == limit:
                    break
            if retained == limit:
                break
    return result


def make_curriculum(matched, extended, previous=None, seed=1229):
    rng = np.random.default_rng(seed)
    forbidden = {r['family'] for d in (matched, extended) for s in ('train', 'validation', 'test') for r in d[s]}
    if previous:
        forbidden.update(own_family(chess.Board(r['fen'])) for s in ('train', 'validation', 'test') for r in previous[s])
    training_roots = [r for r in extended['train'] if r['tier'] == 'mate2']
    validation_roots = [r for r in extended['validation'] if r['tier'] == 'mate2']
    train = suffixes(training_roots, forbidden, rng)
    forbidden.update(r['family'] for r in train)
    validation = suffixes(validation_roots, forbidden, rng)
    forbidden.update(r['family'] for r in validation)
    ordered = sorted(matched['train'], key=lambda r: min(map(len, r['answers']))/len(r['legal_moves']), reverse=True)
    easy = ordered[:max(1, len(ordered)//4)]
    data = dict(version='outcome-grouped-curriculum-v1', seed=seed,
                matched_sha256=matched['sha256'], extended_sha256=extended['sha256'],
                matched_train=matched['train'], easy_train=easy,
                matched_validation=matched['validation'], suffix_train=train, suffix_validation=validation,
                mate2_train=training_roots, mate2_validation=validation_roots,
                forbidden_evaluation_families=sorted(forbidden),
                reward='actual terminal outcome only; no answer-based progress reward')
    data['sha256'] = digest(data)
    return data


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', default=str(ROOT/'artifacts/mechanism/curriculum.json'))
    args = parser.parse_args()
    old = ROOT/'artifacts/learning/corpus.json'
    data = make_curriculum(json.loads((ROOT/'research_data/matched.json').read_text()),
                           json.loads((ROOT/'research_data/extended.json').read_text()),
                           json.loads(old.read_text()) if old.exists() else None)
    if not data['suffix_train'] or not data['suffix_validation']:
        raise RuntimeError('Insufficient suffix curriculum coverage')
    write_json(args.output, data)
    print(data['sha256'], 'suffixes', len(data['suffix_train']), len(data['suffix_validation']), flush=True)
