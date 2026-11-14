"""Separate tactical proof accuracy, bounded game outcomes and longer play."""
import argparse
from collections import Counter
import json
from pathlib import Path
import chess
import numpy as np
import torch
from flies.chess.batched import play_batch
from flies.chess.learning import specs_for,cluster_interval
from flies.chess.matched_eval import exact_distributions,mirror_move
from flies.chess.research_train import own_turn_rewards
from flies.chess.evaluate import summarize
from flies.chess.storage import ROOT,load_policy,identity,write_json
from flies.chess.selfplay import save_game


def bounded_outcome_interval(values):
    values=np.asarray(values,float)
    if len(values)<2:
        return dict(mean=float(values.mean()),ci95=None,clusters=len(values),note='Too few position families for an uncertainty interval')
    if np.all(values==0) or np.all(values==1):
        bound=.025**(1/len(values));zero=bool(np.all(values==0))
        return dict(mean=float(values.mean()),ci95=[0.,1-bound] if zero else [bound,1.],clusters=len(values),
                    note='Conservative position-cluster boundary bound; ordinary bootstrap would give misleading zero width')
    return cluster_interval(values)


def evaluate(candidate,opponent,rows,repeats=4,cap=24,random=False,lesion=False):
    specs=specs_for(rows,repeats=repeats,seed=810001)
    games=[]
    for start in range(0,len(specs),64):
        batch,_=play_batch([candidate,opponent],specs[start:start+64],cap,random_candidate=random,lesion=lesion)
        games.extend(batch)
    sides=[g['candidate'] for g in games];summary=summarize(games,sides)
    wins=np.array([g['outcome']['winner']==s for g,s in zip(games,sides)],float)
    summary['win_within_cap']=bounded_outcome_interval(wins.reshape(len(rows),repeats).mean(1))
    summary['per_position_win']=wins.reshape(len(rows),repeats).mean(1).tolist()
    summary['official_completion_rate']=float(np.mean([not g['outcome']['truncated'] for g in games]))
    summary['auxiliary_return']=float(np.mean([sum(own_turn_rewards(g,s,True))-sum(own_turn_rewards(g,s,False)) for g,s in zip(games,sides)]))
    summary['cap']=cap
    summary['terminations']=dict(Counter(g['outcome']['termination'] or 'UNFINISHED' for g in games))
    return summary,games[0]


def tactical(brain,rows,lesion=False,random=False):
    fens=[];targets=[]
    for row in rows:
        for black in (False,True):
            fens.append(chess.Board(row['fen']).mirror().fen() if black else row['fen'])
            targets.append([mirror_move(m) if black else m for m in row['answers']])
    if random:
        probs=[{m.uci():1/chess.Board(f).legal_moves.count() for m in chess.Board(f).legal_moves} for f in fens]
    else:probs=exact_distributions(brain,fens,lesion=lesion)
    mass=np.array([sum(p[m] for m in labels) for p,labels in zip(probs,targets)]).reshape(len(rows),2).mean(1)
    return dict(probability_mass=cluster_interval(mass),per_position_mass=mass.tolist(),
                note='First-move mass on rules-proven forced mate-two moves, not completion of the tactic')


def run(extended,run_dir,parent_dir,stage_dir,output,update=56,repeats=4,long_games=True):
    run_dir,parent_dir,stage_dir,output=map(Path,(run_dir,parent_dir,stage_dir,output));report={}
    candidate=load_policy(run_dir/f'checkpoint_{update:04d}_0.pt')
    opponent=load_policy(parent_dir/'initial_1.pt')
    prior=load_policy(run_dir/'checkpoint_0000_0.pt')
    matched_model=load_policy(run_dir/'checkpoint_0040_0.pt')
    untrained=load_policy(parent_dir/'initial_0.pt')
    restricted_path=stage_dir/f'checkpoint_{update:04d}_0.pt'
    restricted=load_policy(restricted_path)
    modes={'trained':(candidate,False,False),'matched40':(matched_model,False,False),'parent_pretrained':(prior,False,False),
           'untrained':(untrained,False,False),'random':(untrained,True,False),
           'lesioned':(candidate,False,True),'restricted_trained':(restricted,False,False)}
    paths={'trained':run_dir/f'checkpoint_{update:04d}_0.pt','matched40':run_dir/'checkpoint_0040_0.pt',
           'parent_pretrained':run_dir/'checkpoint_0000_0.pt','untrained':parent_dir/'initial_0.pt',
           'lesioned':run_dir/f'checkpoint_{update:04d}_0.pt','restricted_trained':restricted_path}
    opponent_id=identity(opponent,parent_dir/'initial_1.pt');opponent_id['label']='FROZEN-UNTRAINED'
    def annotate(summary,sample,label,model,random,lesion):
        who=identity(model,paths[label],lesion) if not random else dict(label='RANDOM-LEGAL')
        if label=='untrained':who['label']='UNTRAINED'
        summary['identity']=who;summary['opponent']=opponent_id
        side=sample['candidate'];sample['players']={side:who,('black' if side=='white' else 'white'):opponent_id}
    for tier,cap in [('mate2',12),('endgame',48),('complex',24)]:
        rows=[r for r in extended['test'] if r['tier']==tier];reports={}
        for label,(model,random,lesion) in modes.items():
            summary,sample=evaluate(model,opponent,rows,repeats,cap,random,lesion)
            annotate(summary,sample,label,model,random,lesion)
            if tier=='mate2':summary['tactical']=tactical(model,rows,lesion,random)
            reports[label]=summary;save_game(sample,output/tier/label,0)
            print(tier,label,summary['wins'],summary['draws'],summary['truncated'],flush=True)
        report[tier]=reports
        write_json(output/'generalization.json',dict(corpus_sha256=extended['sha256'],update=update,repeats=repeats,reports=report))
    if long_games:
        rows=[r for r in extended['test'] if r['tier']=='endgame'][:12]
        long={}
        for label,(model,random,lesion) in modes.items():
            summary,sample=evaluate(model,opponent,rows,2,160,random,lesion)
            annotate(summary,sample,label,model,random,lesion)
            long[label]=summary;save_game(sample,output/'long_endgames'/label,0)
        openings=[dict(fen=chess.STARTING_FEN)]
        opening,sample=evaluate(candidate,opponent,openings,8,160)
        save_game(sample,output/'opening_start',0)
        report['long_endgames']=long;report['opening_start']=opening
    write_json(output/'generalization.json',dict(corpus_sha256=extended['sha256'],update=update,repeats=repeats,
              identity=identity(candidate,run_dir/f'checkpoint_{update:04d}_0.pt'),reports=report,
              note='Truncations are not draws. No Elo estimate; all controls use a fixed independent initial opponent.'))
    return report


def tactics_only(extended,run_dir,parent_dir,stage_dir,output,update=56):
    rows=[r for r in extended['test'] if r['tier']=='mate2'];run_dir,parent_dir,stage_dir=map(Path,(run_dir,parent_dir,stage_dir))
    paths={'trained':run_dir/f'checkpoint_{update:04d}_0.pt','matched40':run_dir/'checkpoint_0040_0.pt',
           'parent_pretrained':run_dir/'checkpoint_0000_0.pt','untrained':parent_dir/'initial_0.pt',
           'restricted_trained':stage_dir/f'checkpoint_{update:04d}_0.pt'}
    results={}
    for label,path in paths.items():results[label]=tactical(load_policy(path),rows)
    brain=load_policy(paths['trained']);results['lesioned']=tactical(brain,rows,lesion=True)
    results['random']=tactical(brain,rows,random=True)
    write_json(output,dict(corpus_sha256=extended['sha256'],reports=results))
    return results


def main():
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--extended',default=str(ROOT/'artifacts/research/extended.json'));ap.add_argument('--run',required=True)
    ap.add_argument('--parent',required=True);ap.add_argument('--stage-run',required=True);ap.add_argument('--output',required=True)
    ap.add_argument('--update',type=int,default=56);ap.add_argument('--repeats',type=int,default=4);ap.add_argument('--no-long',action='store_true')
    ap.add_argument('--tactical-only',action='store_true')
    a=ap.parse_args();torch.set_num_threads(1)
    if a.tactical_only:tactics_only(json.load(open(a.extended)),a.run,a.parent,a.stage_run,a.output,a.update)
    else:run(json.load(open(a.extended)),a.run,a.parent,a.stage_run,a.output,a.update,a.repeats,not a.no_long)


if __name__=='__main__':main()
