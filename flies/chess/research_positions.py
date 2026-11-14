"""Rules-only research datasets. Labels never enter policies or choose their moves."""
import argparse
import hashlib
import json
from collections import defaultdict

import chess
import numpy as np
from flies.chess.curriculum import family_key
from flies.chess.storage import ROOT, write_json


def mate_one(board):
    result=[]
    for move in list(board.legal_moves):
        board.push(move)
        if board.is_checkmate(): result.append(move.uci())
        board.pop()
    return result


def legal_signature(board):
    return tuple(sorted(m.uci() for m in board.legal_moves))


def own_family(board):
    """Split all enemy placements for one own-piece arrangement together."""
    b=board.copy(stack=False)
    for sq,p in list(b.piece_map().items()):
        if p.color!=b.turn:b.remove_piece_at(sq)
    return hashlib.sha256(family_key(b).encode()).hexdigest()


def make_matched(seed=1907,train=96,validation=24,test=48,exclude=()):
    rng=np.random.default_rng(seed);pairs=[];seen=set(exclude);attempts=0
    boundary=[s for s in chess.SQUARES if chess.square_file(s) in (0,7) or chess.square_rank(s) in (0,7)]
    while len(pairs)<train+validation+test:
        attempts+=1
        if attempts>200000:raise RuntimeError(f'Only {len(pairs)} matched families found')
        wk,q,r=[int(x) for x in rng.choice(64,3,replace=False)]
        root=chess.Board(None)
        for square,pt in ((wk,chess.KING),(q,chess.QUEEN),(r,chess.ROOK)):
            root.set_piece_at(square,chess.Piece(pt,chess.WHITE))
        family=own_family(root)
        if family in seen:continue
        groups=defaultdict(list)
        for bk in boundary:
            if bk in (wk,q,r) or chess.square_distance(wk,bk)<3:continue
            board=root.copy(stack=False);board.set_piece_at(bk,chess.Piece(chess.KING,False))
            if not board.is_valid() or board.outcome() is not None:continue
            answers=mate_one(board)
            if answers:groups[legal_signature(board)].append((board.fen(),answers))
        found=None
        for signature,rows in groups.items():
            for i,a in enumerate(rows):
                for b in rows[i+1:]:
                    if set(a[1]).isdisjoint(b[1]):
                        found=dict(family=family,positions=[a[0],b[0]],answers=[a[1],b[1]],
                                   legal_moves=list(signature),match='exact-full-legal-UCI-set',tier='mate1')
                        break
                if found:break
            if found:break
        if found:pairs.append(found);seen.add(family)
    rng.shuffle(pairs)
    result=dict(version='matched-legal-trees-v1',seed=seed,attempts=attempts,excluded_families=len(set(exclude)),
                train=pairs[:train],validation=pairs[train:train+validation],test=pairs[train+validation:])
    result['sha256']=hashlib.sha256(json.dumps(result,sort_keys=True).encode()).hexdigest()
    return result


class SearchLimit(RuntimeError):pass


def forced_mate_moves(board,attacker_moves=2,max_nodes=100000):
    """Finite AND/OR proof using legality/outcomes only, never a playing policy.

    Returns exact winning first moves within the specified horizon. If bounded
    proof work is exhausted, raise; unknown must never be labeled a failure.
    """
    if attacker_moves<1:raise ValueError('Need at least one attacking move')
    attacker=board.turn;nodes=0;cache={}
    def prove(b,plies):
        nonlocal nodes
        nodes+=1
        if nodes>max_nodes:raise SearchLimit('Tactical label search cap reached')
        outcome=b.outcome(claim_draw=False)
        if outcome is not None:return outcome.winner==attacker
        if plies==0:return False
        key=(b.fen(),plies)
        if key in cache:return cache[key]
        vals=[]
        for m in list(b.legal_moves):
            b.push(m);value=prove(b,plies-1);b.pop();vals.append(value)
            if b.turn==attacker and value:break
            if b.turn!=attacker and not value:break
        result=any(vals) if b.turn==attacker else all(vals)
        cache[key]=result;return result
    winners=[]
    for move in list(board.legal_moves):
        b=board.copy(stack=False);b.push(move)
        if prove(b,2*attacker_moves-2):winners.append(move.uci())
    return winners,nodes


def main():
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--output',default=str(ROOT/'artifacts/research/matched.json'))
    ap.add_argument('--seed',type=int,default=1907)
    ap.add_argument('--train',type=int,default=96);ap.add_argument('--validation',type=int,default=24);ap.add_argument('--test',type=int,default=48)
    a=ap.parse_args()
    old=ROOT/'artifacts/learning/corpus.json';exclude=[]
    if old.exists():
        previous=json.loads(old.read_text())
        exclude=[own_family(chess.Board(row['fen'])) for split in ('train','validation','test') for row in previous[split]]
    corpus=make_matched(a.seed,a.train,a.validation,a.test,exclude);write_json(a.output,corpus)
    print(corpus['sha256'],'attempts',corpus['attempts'],flush=True)


if __name__=='__main__':main()
