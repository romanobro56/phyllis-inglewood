"""Multi-ply/endgame/general legal-position curriculum; no playing oracle."""
import argparse
import hashlib
import json
import chess
import numpy as np
from flies.chess.research_positions import own_family,mate_one,forced_mate_moves,SearchLimit
from flies.chess.storage import ROOT,write_json


def make_extended(matched,seed=2907,train=48,validation=12,test=24,exclude=()):
    rng=np.random.default_rng(seed)
    seen={r['family'] for split in ('train','validation','test') for r in matched[split]}|set(exclude)
    buckets={tier:[] for tier in ('mate2','endgame','complex')};attempts=0;unknown=0
    needed=train+validation+test
    while len(buckets['mate2'])<needed:
        attempts+=1
        if attempts>50000:raise RuntimeError('Mate-two corpus generation exhausted')
        bk=int(rng.choice([s for s in chess.SQUARES if chess.square_file(s) in (0,7) or chess.square_rank(s) in (0,7)]))
        wk,q=[int(s) for s in rng.choice(64,2,replace=False)]
        if bk in (wk,q) or chess.square_distance(wk,bk) not in (2,3):continue
        b=chess.Board(None)
        for square,pt,color in ((bk,chess.KING,False),(wk,chess.KING,True),(q,chess.QUEEN,True)):
            b.set_piece_at(square,chess.Piece(pt,color))
        family=own_family(b)
        if family in seen or not b.is_valid() or b.outcome() or mate_one(b):continue
        try:answers,nodes=forced_mate_moves(b,2,max_nodes=12000)
        except SearchLimit:unknown+=1;continue
        if answers:
            buckets['mate2'].append(dict(fen=b.fen(),family=family,tier='mate2',answers=answers,proof_nodes=nodes));seen.add(family)
    for tier in ('endgame','complex'):
        while len(buckets[tier])<needed:
            if tier=='endgame':
                wk,bk,p=[int(s) for s in rng.choice(64,3,replace=False)];b=chess.Board(None)
                if chess.square_distance(wk,bk)<2:continue
                if rng.random()<.5:
                    if not 1<=chess.square_rank(p)<=6:continue
                    piece=chess.PAWN
                else:piece=chess.ROOK
                for sq,pt,c in ((wk,chess.KING,True),(bk,chess.KING,False),(p,piece,True)):
                    b.set_piece_at(sq,chess.Piece(pt,c))
            else:
                b=chess.Board()
                # Dataset construction only, never a reported fly/self-play move.
                for _ in range(int(rng.integers(12,29))):
                    if b.is_game_over():break
                    legal=tuple(b.legal_moves);b.push(legal[int(rng.integers(len(legal)))])
                if not b.turn:b=b.mirror()
            family=own_family(b)
            if family in seen or not b.is_valid() or b.outcome():continue
            buckets[tier].append(dict(fen=b.fen(),family=family,tier=tier,answers=None));seen.add(family)
    result=dict(version='extended-chess-v1',seed=seed,mate2_attempts=attempts,unknown_searches=unknown,
                matched_sha256=matched['sha256'])
    for split in ('train','validation','test'):result[split]=[]
    for tier,rows in buckets.items():
        rng.shuffle(rows)
        for split,start,end in (('train',0,train),('validation',train,train+validation),('test',train+validation,needed)):
            result[split].extend(rows[start:end])
    result['sha256']=hashlib.sha256(json.dumps(result,sort_keys=True).encode()).hexdigest()
    return result


def main():
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--matched',default=str(ROOT/'artifacts/research/matched.json'))
    ap.add_argument('--output',default=str(ROOT/'artifacts/research/extended.json'))
    a=ap.parse_args();old=ROOT/'artifacts/learning/corpus.json';exclude=[]
    if old.exists():
        previous=json.load(open(old));exclude=[own_family(chess.Board(r['fen'])) for s in ('train','validation','test') for r in previous[s]]
    corpus=make_extended(json.load(open(a.matched)),exclude=exclude);write_json(a.output,corpus)
    print(corpus['sha256'],corpus['mate2_attempts'],corpus['unknown_searches'],flush=True)


if __name__=='__main__':main()
