"""Exact FlyBrain move probabilities on legal-tree-matched counterfactuals."""
import argparse
from functools import lru_cache
import json
from pathlib import Path

import chess
import numpy as np
import torch
from flies.chess.chess_env import ChessEnv,ENCODING,STAGE_ONLY_ENCODING
from flies.chess.storage import ROOT,load_policy,identity,write_json
from flies.chess.study_report import two_way_interval


def restricted(obs,mode):
    x=obs.copy()
    if mode=='interface':x[:32]=0;x[41:]=0
    elif mode=='stage':x[:32]=0;x[37:]=0
    elif mode=='own_coordinates':x[:32]=0;x[41:53]=0;x[55:57]=0;x[62:64]=0
    elif mode=='packed':x[53:]=0
    elif mode=='coordinates':x[:32]=0
    elif mode!='intact':raise ValueError('Unknown observation control')
    return x


@lru_cache(maxsize=768)
def prepared_tree(fen,encoding):
    """Cache rules/observations only; never cache model probabilities or labels."""
    env=ChessEnv(fen)
    if env.done:raise ValueError('Probe must be nonterminal')
    trie={}
    for move in env.board.legal_moves:
        path=env.coordinates(move)
        for stage in range(len(path)):
            prefix=tuple(path[:stage]);entry=trie.setdefault(prefix,dict(actions=set(),leaves={}))
            entry['actions'].add(path[stage])
            if stage==len(path)-1:entry['leaves'][path[stage]]=move.uci()
    for prefix,entry in trie.items():
        env.stage=len(prefix);env.choices=list(prefix)
        entry['obs']=env.observe(encoding)
        entry['mask']=np.array([i in entry['actions'] for i in range(8)],bool)
        entry['actions']=tuple(sorted(entry['actions']))
    return trie


def exact_distributions(brain,fens,mode='intact',donor_fens=None,lesion=False,batch_size=128):
    encoding=getattr(brain,'encoding',ENCODING)
    trees=[prepared_tree(f,encoding) for f in fens]
    donors=[prepared_tree(f,encoding) for f in donor_fens] if donor_fens else trees
    if any({p:tuple(e['actions']) for p,e in a.items()}!={p:tuple(e['actions']) for p,e in b.items()}
           for a,b in zip(trees,donors)):raise ValueError('Counterfactual requires identical complete legal trees')
    frontier=[(i,(),1.) for i in range(len(fens))];results=[{} for _ in fens]
    device=next(brain.parameters()).device
    while frontier:
        next_nodes=[]
        for start in range(0,len(frontier),batch_size):
            nodes=frontier[start:start+batch_size]
            obs=np.stack([restricted(donors[i][prefix]['obs'],mode) for i,prefix,_ in nodes])
            masks=np.stack([trees[i][prefix]['mask'] for i,prefix,_ in nodes])
            with torch.no_grad():
                logits,_=brain(torch.as_tensor(obs,device=device),torch.as_tensor(masks,device=device),lesion=lesion)
                probs=(logits/getattr(brain,'policy_temperature',1.)).softmax(-1).cpu().numpy().astype(float)
            if not np.isfinite(probs).all():raise ValueError('Nonfinite FlyBrain probabilities')
            for (i,prefix,parent),mask,p in zip(nodes,masks,probs):
                p[~mask]=0;p/=p.sum();entry=trees[i][prefix]
                for action in entry['actions']:
                    prob=parent*float(p[action])
                    if action in entry['leaves']:results[i][entry['leaves'][action]]=prob
                    else:next_nodes.append((i,prefix+(action,),prob))
        frontier=next_nodes
    if any(abs(sum(r.values())-1)>1e-6 for r in results):raise RuntimeError('Move probability mass lost')
    return results


def exact_distribution(brain,fen,mode='intact',donor_fen=None,lesion=False,batch_size=128):
    p=exact_distributions(brain,[fen],mode,[donor_fen] if donor_fen else None,lesion,batch_size)[0]
    return p,dict(decisions=len(prepared_tree(fen,getattr(brain,'encoding',ENCODING))))


def mirror_move(uci):
    m=chess.Move.from_uci(uci)
    return chess.Move(m.from_square^56,m.to_square^56,promotion=m.promotion).uci()


def evaluate_pairs(brain,pairs,mode='intact',swap=False,lesion=False):
    fens=[];targets=[]
    for pair in pairs:
        for black in (False,True):
            fens.extend([chess.Board(f).mirror().fen() if black else f for f in pair['positions']])
            targets.append([[mirror_move(m) if black else m for m in a] for a in pair['answers']])
    donor=[fens[i^1] for i in range(len(fens))] if swap else None
    distributions=exact_distributions(brain,fens,mode,donor,lesion)
    rows=[]
    for index,pair in enumerate(pairs):
        colors=[]
        for color in range(2):
            base=index*4+color*2;p,q=distributions[base:base+2];a,b=targets[index*2+color]
            correct=.5*(sum(p[m] for m in a)+sum(q[m] for m in b))
            cross=.5*(sum(p[m] for m in b)+sum(q[m] for m in a))
            colors.append(dict(correct_mass=correct,cross_target_mass=cross,adaptation=correct-cross,
                               total_variation=.5*sum(abs(p[m]-q[m]) for m in p),
                               both_argmax_correct=float(max(p,key=p.get) in a and max(q,key=q.get) in b),
                               both_cross_argmax_correct=float(max(p,key=p.get) in b and max(q,key=q.get) in a)))
        rows.append(dict(family=pair['family'],**{k:float(np.mean([r[k] for r in colors])) for k in colors[0]}))
    return dict(mean_correct_mass=float(np.mean([r['correct_mass'] for r in rows])),
                mean_adaptation=float(np.mean([r['adaptation'] for r in rows])),
                mean_total_variation=float(np.mean([r['total_variation'] for r in rows])),
                both_argmax_correct=float(np.mean([r['both_argmax_correct'] for r in rows])),rows=rows)


def evaluate_checkpoint(path,pairs,size='tiny',identity_label=None):
    brain=load_policy(path,size=size);conditions={}
    for label,mode,lesion in [('intact','intact',False),('stage_partial','interface',False),
                             ('own_coordinates','own_coordinates',False),('coordinates_only','coordinates',False),
                             ('packed_only','packed',False),('stage_only','stage',False),('lesioned','intact',True)]:
        if brain.encoding==STAGE_ONLY_ENCODING and not lesion and mode!='intact':
            conditions[label]=conditions['intact']  # Exactly identical inputs by construction.
        else:conditions[label]=evaluate_pairs(brain,pairs,mode,lesion=lesion)
        print(label,conditions[label]['mean_correct_mass'],conditions[label]['mean_adaptation'],flush=True)
    # Exact intervention exchanges the two entire input states under the same
    # legal tree; its score is the intact model's cross-target probability mass.
    conditions['paired_swap']=swapped_report(conditions['intact'])
    who=identity(brain,path)
    if identity_label:who['label']=identity_label
    return dict(identity=who,conditions=conditions)


def swapped_report(report):
    rows=[dict(r,correct_mass=r['cross_target_mass'],cross_target_mass=r['correct_mass'],
               adaptation=-r['adaptation'],both_argmax_correct=r['both_cross_argmax_correct'],
               both_cross_argmax_correct=r['both_argmax_correct']) for r in report['rows']]
    return dict(mean_correct_mass=float(np.mean([r['correct_mass'] for r in rows])),
                mean_adaptation=-report['mean_adaptation'],
                both_argmax_correct=float(np.mean([r['both_argmax_correct'] for r in rows])),rows=rows)


def pool_reports(reports):
    pooled={label:{} for label in reports[0]['conditions']}
    for label in pooled:
        for metric in ('correct_mass','adaptation'):
            matrix=[[r[metric] for r in report['conditions'][label]['rows']] for report in reports]
            pooled[label][metric]=two_way_interval(matrix,alpha=.05/4,draws=5000)
    return pooled


def main():
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--corpus',default=str(ROOT/'artifacts/research/matched.json'))
    ap.add_argument('--checkpoints',nargs='+',required=True);ap.add_argument('--output',required=True)
    ap.add_argument('--split',choices=('train','validation','test'),default='test')
    ap.add_argument('--label',default=None)
    a=ap.parse_args();torch.set_num_threads(1);corpus=json.loads(Path(a.corpus).read_text());reports=[]
    for ck in a.checkpoints:
        reports.append(evaluate_checkpoint(ck,corpus[a.split],identity_label=a.label))
        write_json(a.output,dict(corpus_sha256=corpus['sha256'],split=a.split,reports=reports))
    write_json(a.output,dict(corpus_sha256=corpus['sha256'],split=a.split,reports=reports,pooled=pool_reports(reports)))


if __name__=='__main__':main()
