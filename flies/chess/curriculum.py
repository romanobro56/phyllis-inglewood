"""Rules-curated tactical starts, split by board-symmetry family.

Mate labels curate and audit positions only. Policies never receive these
labels, rankings, or an engine evaluation; rewards follow actual board outcomes.
"""
import hashlib
import json
import chess
import numpy as np
from flies.chess.chess_env import ChessEnv

VERSION = 'kqk-kqrk-diverse-v2'


def family_key(board):
    b = board if board.turn else board.mirror()
    keys = []
    for reflect in (False, True):
        for rotation in range(4):
            items = []
            for square, p in b.piece_map().items():
                x, y = chess.square_file(square), chess.square_rank(square)
                if reflect:
                    x = 7-x
                for _ in range(rotation):
                    x, y = 7-y, x
                items.append((int(p.color), p.piece_type, x, y))
            keys.append(tuple(sorted(items)))
    return repr(min(keys))


def make_corpus(seed=731, train=256, validation=64, test=128):
    rng = np.random.default_rng(seed)
    positions = {}
    attempts = 0
    boundary = [s for s in chess.SQUARES if chess.square_file(s) in (0,7) or chess.square_rank(s) in (0,7)]
    while len(positions) < train + validation + test:
        attempts += 1
        if attempts > 500000:
            raise RuntimeError(f'Constructed only {len(positions)} disjoint tactical families')
        bk = int(rng.choice(boundary))
        wk, q = [int(s) for s in rng.choice(64, 2, replace=False)]
        if bk in (wk,q) or chess.square_distance(wk,bk) not in (2,3):
            continue
        board = chess.Board(None)
        for square, p in ((bk,chess.Piece(chess.KING,False)),(wk,chess.Piece(chess.KING,True)),(q,chess.Piece(chess.QUEEN,True))):
            board.set_piece_at(square,p)
        if rng.random() < .5:
            rook = int(rng.integers(64))
            if rook in (bk,wk,q):
                continue
            board.set_piece_at(rook,chess.Piece(chess.ROOK,True))
        if not board.is_valid() or board.outcome() is not None:
            continue
        key = family_key(board)
        if key in positions:
            continue
        legal = tuple(board.legal_moves)
        mates, stale = [], 0
        for move in legal:
            board.push(move)
            if board.is_checkmate():
                mates.append(move.uci())
            stale += board.is_stalemate()
            board.pop()
        if not mates or stale / len(legal) > .15:
            continue
        env = ChessEnv(board.fen())
        flat_mate_probability = sum(np.prod([1 / len({env.coordinates(m)[stage] for m in legal
                    if env.coordinates(m)[:stage] == env.coordinates(chess.Move.from_uci(uci))[:stage]})
                    for stage in range(4)]) for uci in mates)
        positions[key] = dict(fen=board.fen(), family=hashlib.sha256(key.encode()).hexdigest(),
                              legal_moves=len(legal), mate_moves=mates, stalemate_moves=stale,
                              random_legal_mate_probability=len(mates)/len(legal),
                              uniform_hierarchy_mate_probability=float(flat_mate_probability))
    rows = list(positions.values())
    rng.shuffle(rows)
    corpus = dict(version=VERSION, seed=seed, attempts=attempts,
                  train=rows[:train], validation=rows[train:train+validation], test=rows[train+validation:])
    corpus['sha256'] = hashlib.sha256(json.dumps(corpus,sort_keys=True).encode()).hexdigest()
    return corpus


def main():
    import argparse
    from flies.chess.storage import ROOT, write_json
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--seed',type=int,default=731)
    ap.add_argument('--output',default=str(ROOT/'artifacts/learning/corpus.json'))
    args=ap.parse_args()
    corpus=make_corpus(args.seed)
    write_json(args.output,corpus)
    print(corpus['sha256'],{k:len(corpus[k]) for k in ('train','validation','test')})


if __name__=='__main__':main()
