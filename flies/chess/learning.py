"""Reproducible tiny-graph learning study. No external engine or imitation.

Primary endpoint: first-move mate rate on symmetry-disjoint tactical starts.
Continuation W/L/D/truncation is recorded separately; this is not general chess.
"""
import argparse
import json
from pathlib import Path
import time

import chess
import numpy as np
import torch

from flies.chess.batched import play_batch
from flies.chess.chess_env import ENCODING, SPATIAL_ENCODING, STAGE_ONLY_ENCODING
from flies.chess.evaluate import summarize
from flies.chess.storage import (ROOT, atomic_write, brain_payload, identity, load_policy,
                                 validate_and_load, write_json, file_hash, prepare_run)
from flies.chess.train_selfplay import cpu_snapshot, policy_update, rng_state, restore_rng
from flies.chess.selfplay import save_game


def outcome_reward(game, side):
    winner=game['outcome']['winner']
    return 0. if winner is None else 1. if winner==side else -1.


def specs_for(rows, repeats=4, seed=900001, players=(0,1)):
    specs=[]
    for row_index,row in enumerate(rows):
        board=chess.Board(row['fen'])
        for repeat in range(repeats):
            b=board if repeat%2==0 else board.mirror()
            specs.append(dict(fen=b.fen(), seed=seed+row_index*100+repeat,
                              players=players if b.turn else tuple(reversed(players))))
    return specs


def cluster_interval(values, seed=582, draws=5000):
    values=np.asarray(values,float)
    if values.ndim!=1 or not len(values):
        raise ValueError('Need nonempty position-level observations')
    rng=np.random.default_rng(seed)
    estimates=values[rng.integers(len(values),size=(draws,len(values)))].mean(1)
    return dict(mean=float(values.mean()), ci95=[float(x) for x in np.quantile(estimates,[.025,.975])],
                clusters=len(values), bootstrap_draws=draws)


def evaluate_models(candidate, opponent, rows, repeats=4, seed=900001, max_plies=4,
                    random_candidate=False, lesion=False, shuffle=False, batch_size=128):
    specs=specs_for(rows,repeats,seed)
    games=[]
    for start in range(0,len(specs),batch_size):
        batch,_=play_batch([candidate,opponent],specs[start:start+batch_size], max_plies,
                          random_candidate=random_candidate,lesion=lesion,shuffle_observations=shuffle)
        games.extend(batch)
    mates=np.array([int(bool(g['moves']) and g['moves'][0]['outcome']['termination']=='CHECKMATE') for g in games])
    per_position=mates.reshape(len(rows),repeats).mean(1)
    report=summarize(games,[g['candidate'] for g in games])
    report.update(first_move_mates=int(mates.sum()),first_move_mate=cluster_interval(per_position),
                  position_mate_rates=per_position.tolist(), per_game_mate=mates.tolist(),
                  random_legal_expected=float(np.mean([r['random_legal_mate_probability'] for r in rows])),
                  uniform_hierarchy_expected=float(np.mean([r['uniform_hierarchy_mate_probability'] for r in rows])))
    if shuffle:
        stages=[s for g in games for s in g['moves'][0]['stages']]
        report['first_move_same_board_donors']=sum(s.get('same_board_donor',True) for s in stages)
        report['first_move_shuffled_stages']=len(stages)
    # One replay-verified sample is enough for provenance; aggregate raw binary
    # outcomes remain in the report rather than writing thousands of huge records.
    return report,games[0]


def paired_difference(trained, control):
    x=np.asarray(trained['position_mate_rates'])-np.asarray(control['position_mate_rates'])
    return cluster_interval(x)


def run_train(corpus, output, updates=60, games=128, seed=11, encoding=SPATIAL_ENCODING,
              lr=.003, entropy=.003, max_plies=4, eval_every=10, resume=None):
    if updates < 0 or games < 8 or games % 8 or max_plies < 2 or eval_every < 1 or lr <= 0 or entropy < 0:
        raise ValueError('Study needs nonnegative updates, games divisible by 8, at least 2 plies, and valid optimizer bounds')
    output=prepare_run(output,resume)
    torch.set_num_threads(1); torch.manual_seed(seed)
    rng=np.random.default_rng(seed)
    models=[load_policy(size='tiny',encoding=encoding) for _ in range(2)]
    frozen=load_policy(size='tiny',encoding=encoding)
    for p in frozen.parameters():p.requires_grad_(False)
    optimizers=[torch.optim.Adam(m.parameters(),lr=lr) for m in models]
    config=dict(corpus_sha256=corpus['sha256'],seed=seed,encoding=encoding,lr=lr,entropy=entropy,
                max_plies=max_plies,games=games,eval_every=eval_every,algorithm='episode-reinforce-v2')
    pool=[cpu_snapshot(m) for m in models]
    baseline=[{'attack':0.,'defend':0.} for _ in models]
    history=[];start=0
    if resume:
        ck=torch.load(resume,map_location='cpu',weights_only=False)
        if ck['config']!=config:raise ValueError('Study resume configuration differs')
        for i in range(2):
            validate_and_load(models[i],ck['brains'][i]);optimizers[i].load_state_dict(ck['optimizers'][i])
        baseline,pool,history,start=ck['baseline'],ck['pool'],ck['history'],ck['update']
        restore_rng(ck['rng'],rng)
    else:
        for i,m in enumerate(models):atomic_write(output/f'initial_{i}.pt',brain_payload(m),binary=True)
        atomic_write(output/'resume.pt',dict(config=config,update=0,brains=[cpu_snapshot(m) for m in models],
                     optimizers=[o.state_dict() for o in optimizers],pool=pool,baseline=baseline,history=history,
                     rng=rng_state(rng)),binary=True)
    if start>updates:raise ValueError('Requested updates precede resume checkpoint')
    t0=time.monotonic()
    for update in range(start,updates+1):
        if update==0 or update%eval_every==0 or update==updates:
            validation=[]
            for i,m in enumerate(models):
                # A changing pool opponent confounds checkpoint W/L/D curves.
                fixed=torch.load(output/f'initial_{1-i}.pt',map_location='cpu',weights_only=False)
                validate_and_load(frozen,fixed)
                path=output/f'checkpoint_{update:04d}_{i}.pt'
                atomic_write(path,brain_payload(m),binary=True)
                report,sample=evaluate_models(m,frozen,corpus['validation'],repeats=2,seed=700001,max_plies=max_plies)
                validation.append(report)
            entry=dict(update=update,validation=validation,seconds=time.monotonic()-t0,
                       validation_opponents=[file_hash(output/f'initial_{1-i}.pt') for i in range(2)])
            history.append(entry);write_json(output/'history.json',dict(config=config,history=history))
            print('validation',update,[round(v['first_move_mate']['mean'],4) for v in validation],flush=True)
        if update==updates:break
        # Equal assignments to attacking learner, actual color and historical/live opponent.
        specs=[]
        for j in range(games):
            row=corpus['train'][int(rng.integers(len(corpus['train'])))]
            board=chess.Board(row['fen'])
            if (j//2+update)%2:board=board.mirror()
            attacker=(j+update)%2
            defender=1-attacker
            # Half live-vs-live, half independent frozen opponents; roles balanced.
            if (j//4)%2:defender=2
            players=(attacker,defender) if board.turn else (defender,attacker)
            specs.append(dict(fen=board.fen(),seed=int(rng.integers(2**31)),players=players))
        validate_and_load(frozen,pool[int(rng.integers(len(pool)))])
        records,traces=play_batch(models+[frozen],specs,max_plies)
        metrics=[]
        for learner in range(2):
            samples=[];bases=[];roles=[]
            for game,trace,spec in zip(records,traces,specs):
                for side_index,side in enumerate(('white','black')):
                    if spec['players'][side_index]!=learner:continue
                    role='attack' if side==game['candidate'] else 'defend'
                    reward=outcome_reward(game,side)
                    samples.append((trace[side],reward));bases.append(baseline[learner][role]);roles.append(role)
            metric,_=policy_update(models[learner],optimizers[learner],samples,0.,gamma=1.,
                                   entropy=entropy,chunk=64,sample_baselines=bases)
            for role in ('attack','defend'):
                rewards=[reward for (moves,reward),r in zip(samples,roles) if r==role and moves]
                if rewards:baseline[learner][role]=.9*baseline[learner][role]+.1*float(np.mean(rewards))
            metrics.append(metric)
        history.append(dict(update=update+1,training=metrics,
                            outcomes=summarize(records,[g['candidate'] for g in records])))
        if (update+1)%10==0:
            pool.extend(cpu_snapshot(m) for m in models);pool=pool[-4:]
        checkpoint=dict(config=config,update=update+1,brains=[cpu_snapshot(m) for m in models],
                        optimizers=[o.state_dict() for o in optimizers],pool=pool,baseline=baseline,
                        history=history,rng=rng_state(rng))
        atomic_write(output/'resume.pt',checkpoint,binary=True)
        save_game(records[0],output/'samples',update)
        print('update',update+1,'mates',sum(g['outcome']['termination']=='CHECKMATE' for g in records),
              'grad',[round(m['grad_norm'],4) for m in metrics],flush=True)
    return models,history


def final_evaluation(corpus,run,output,update=60,repeats=4,split='test'):
    if repeats < 2 or repeats % 2:
        raise ValueError('Evaluation requires an even number of color-balanced repeats')
    run,output=Path(run),Path(output)
    reports={}
    for learner in range(2):
        candidate=load_policy(run/f'checkpoint_{update:04d}_{learner}.pt',size='tiny')
        initial=load_policy(run/f'initial_{learner}.pt',size='tiny')
        opponent=load_policy(run/f'initial_{1-learner}.pt',size='tiny')
        conditions={'trained':(candidate,False,False,False),'initial':(initial,False,False,False),
                    'random_legal':(initial,True,False,False),'lesioned':(candidate,False,True,False),
                    'shuffled_board':(candidate,False,False,True)}
        if candidate.encoding==STAGE_ONLY_ENCODING:
            conditions.pop('shuffled_board')  # No board exists in this diagnostic input.
        rows={}
        for label,(model,random,lesion,shuffle) in conditions.items():
            report,sample=evaluate_models(model,opponent,corpus[split],repeats=repeats,
                random_candidate=random,lesion=lesion,shuffle=shuffle)
            report['identity']=identity(model,run/(f'checkpoint_{update:04d}_{learner}.pt' if model is candidate else f'initial_{learner}.pt'),lesion)
            if random:
                report['identity']={'label':'RANDOM-LEGAL', 'brain_used_for_candidate':False}
            elif model is initial:
                report['identity']['label']='UNTRAINED-INITIALIZATION'
            elif shuffle:
                report['identity']['label']='TRAINED-BOARD-SHUFFLED'
            sample['players']={sample['candidate']:report['identity'],
                ('black' if sample['candidate']=='white' else 'white'):identity(opponent,run/f'initial_{1-learner}.pt')}
            sample['players']['black' if sample['candidate']=='white' else 'white']['label']='FROZEN-INITIALIZATION'
            rows[label]=report;save_game(sample,output/f'learner_{learner}'/label,0)
            print('final',learner,label,report['first_move_mate'],flush=True)
        rows['differences']={label:paired_difference(rows['trained'],rows[label]) for label in conditions if label!='trained'}
        reports[str(learner)]=rows
        write_json(output/'evaluation.json',dict(corpus_sha256=corpus['sha256'],split=split,update=update,
            repeats=repeats,reports=reports,note='95% intervals cluster by symmetry-disjoint position. Exploratory controls; no Elo or general chess claim.'))
    return reports


def main():
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument('mode',choices=('train','evaluate'))
    ap.add_argument('--corpus',default=str(ROOT/'artifacts/learning/corpus.json'))
    ap.add_argument('--output',required=True)
    ap.add_argument('--run')
    ap.add_argument('--updates',type=int,default=60)
    ap.add_argument('--games',type=int,default=128)
    ap.add_argument('--seed',type=int,default=11)
    ap.add_argument('--encoding',choices=('packed','spatial','stage-only-control'),default='spatial')
    ap.add_argument('--lr',type=float,default=.003)
    ap.add_argument('--entropy',type=float,default=.003)
    ap.add_argument('--max-plies',type=int,default=4)
    ap.add_argument('--eval-every',type=int,default=10)
    ap.add_argument('--repeats',type=int,default=4)
    ap.add_argument('--split',choices=('validation','test'),default='test')
    ap.add_argument('--resume')
    args=ap.parse_args();torch.set_num_threads(1)
    corpus=json.loads(Path(args.corpus).read_text())
    encoding={'packed':ENCODING,'spatial':SPATIAL_ENCODING,'stage-only-control':STAGE_ONLY_ENCODING}[args.encoding]
    if args.mode=='train':run_train(corpus,args.output,args.updates,args.games,args.seed,
        encoding,args.lr,args.entropy,args.max_plies,args.eval_every,args.resume)
    else:
        if not args.run:ap.error('--run is required for evaluation')
        final_evaluation(corpus,args.run,args.output,args.updates,args.repeats,args.split)


if __name__=='__main__':main()
