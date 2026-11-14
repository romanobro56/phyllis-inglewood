"""Counterfactual curriculum and multi-ply two-FlyBrain reinforcement learning."""
import argparse
import json
from pathlib import Path
import time

import chess
import numpy as np
import torch
from flies.chess.batched import play_batch
from flies.chess.chess_env import SPATIAL_ENCODING,STAGE_ONLY_ENCODING
from flies.chess.storage import ROOT,load_policy,brain_payload,validate_and_load,atomic_write,write_json,file_hash,prepare_run
from flies.chess.train_selfplay import cpu_snapshot,policy_update,rng_state,restore_rng
from flies.chess.learning import outcome_reward
from flies.chess.selfplay import save_game


def potential(board,side,version='integer-v2'):
    if version not in ('integer-v2','float-v1'):raise ValueError('Unknown reward arithmetic')
    color=side=='white';value=(0,1,3,3,5,9,0);total=0 if version=='integer-v2' else 0.
    for square,p in board.piece_map().items():
        progress=(chess.square_rank(square)-1 if p.color else 6-chess.square_rank(square)) if p.piece_type==chess.PAWN else 0
        units=20*value[p.piece_type]+progress if version=='integer-v2' else value[p.piece_type]+.05*progress
        total+=(1 if p.color==color else -1)*units
    return total/(200 if version=='integer-v2' else 10)


def own_turn_rewards(game,side,auxiliary=False,reward_version='integer-v2'):
    events=[m for m in game['moves'] if m['side']==side]
    rewards=[]
    for i,event in enumerate(events):
        before=chess.Board(event['before'])
        after=chess.Board(events[i+1]['before'] if i+1<len(events) else game['final_fen'])
        rewards.append(.25*float(np.clip(potential(after,side,reward_version)-potential(before,side,reward_version),-1,1)) if auxiliary else 0.)
    if rewards:rewards[-1]+=outcome_reward(game,side)
    return rewards


def clone_source(source,encoding,size='tiny',device='cpu'):
    brain=load_policy(size=size,device=device,encoding=encoding)
    ck=torch.load(source,map_location='cpu',weights_only=False)
    # Deliberate observation restriction intervention, not a different architecture.
    converted=dict(ck,encoding=encoding)
    validate_and_load(brain,converted)
    return brain


def train(matched,extended,output,sources,seed=11,updates=56,games=64,encoding=SPATIAL_ENCODING,
          size='tiny',device='cpu',resume=None,chunk=64,reward_version='integer-v2',lr=.003,entropy=.003):
    if games<8 or games%8 or not 0<=updates<=56:raise ValueError('Use 0..56 updates and games divisible by 8')
    if updates>40 and games<32:raise ValueError('Extended phases need at least 32 games to cover all tiers')
    if chunk<1:raise ValueError('Gradient chunk must be positive')
    if reward_version not in ('integer-v2','float-v1'):raise ValueError('Unknown reward arithmetic')
    if not np.isfinite(lr) or lr<=0 or not np.isfinite(entropy) or entropy<0:raise ValueError('Invalid optimizer settings')
    output=prepare_run(output,resume);torch.set_num_threads(1);torch.manual_seed(seed);rng=np.random.default_rng(seed)
    models=[clone_source(p,encoding,size,device) for p in sources]
    frozen=clone_source(sources[0],encoding,size,device)
    for p in frozen.parameters():p.requires_grad_(False)
    opts=[torch.optim.Adam(m.parameters(),lr=lr) for m in models]
    config=dict(algorithm='matched-plus-progress-v1' if reward_version=='float-v1' else 'matched-plus-progress-v2',
                reward_version=reward_version,seed=seed,games=games,encoding=encoding,size=size,device=str(device),
                matched=matched['sha256'],extended=extended['sha256'],source_hashes=[file_hash(s) for s in sources],
                matched_updates=40,extended_updates=16,gamma=.98,auxiliary_clip=.25,chunk=chunk,lr=lr,entropy=entropy)
    pool=[cpu_snapshot(m) for m in models];baselines=[{},{}];history=[];start=0
    if resume:
        ck=torch.load(resume,map_location='cpu',weights_only=False)
        ck['config'].setdefault('chunk',64)  # The original research runs used this fixed chunk.
        ck['config'].setdefault('reward_version','float-v1')
        ck['config'].setdefault('lr',.003);ck['config'].setdefault('entropy',.003)
        if ck['config']!=config:raise ValueError('Research resume configuration mismatch')
        for i in range(2):validate_and_load(models[i],ck['brains'][i]);opts[i].load_state_dict(ck['optimizers'][i])
        start,pool,baselines,history=ck['update'],ck['pool'],ck['baselines'],ck['history'];restore_rng(ck['rng'],rng)
    else:
        for i,m in enumerate(models):atomic_write(output/f'checkpoint_0000_{i}.pt',brain_payload(m),binary=True)
        atomic_write(output/'resume.pt',dict(config=config,update=0,brains=[cpu_snapshot(m) for m in models],
                     optimizers=[o.state_dict() for o in opts],pool=pool,baselines=baselines,history=history,
                     rng=rng_state(rng)),binary=True)
    if start>updates:raise ValueError('Requested updates precede resume checkpoint')
    for i,m in enumerate(models):
        if not (output/f'checkpoint_{start:04d}_{i}.pt').exists():
            atomic_write(output/f'checkpoint_{start:04d}_{i}.pt',brain_payload(m),binary=True)
    write_json(output/'history.json',dict(config=config,history=history))
    t0=time.monotonic()
    for update in range(start,updates):
        specs=[];tiers=[];phase='matched' if update<40 else 'extended'
        # Independent RNG seeds for the two members: no correlated paired-outcome baseline.
        for j in range(games//2 if phase=='matched' else games//4):
            if phase=='matched':
                pair=matched['train'][int(rng.integers(len(matched['train'])))];fens=pair['positions'];tier='mate1'
            elif j%4==0:
                pair=matched['train'][int(rng.integers(len(matched['train'])))];fens=pair['positions'];tier='mate1'
            else:
                tier=('mate2','endgame','complex')[(j-1)%3]
                rows=[r for r in extended['train'] if r['tier']==tier];row=rows[int(rng.integers(len(rows)))];fens=[row['fen'],row['fen']]
            attacker=(j+update)%2;defender=1-attacker if (j//4)%2==0 else 2
            for fen in fens:
                board=chess.Board(fen)
                if (j//2+update)%2:board=board.mirror()
                specs.append(dict(fen=board.fen(),seed=int(rng.integers(2**31)),players=(attacker,defender) if board.turn else (defender,attacker)))
                tiers.append(tier)
        validate_and_load(frozen,pool[int(rng.integers(len(pool)))])
        records=[None]*len(specs);traces=[None]*len(specs)
        for tier,cap in [('mate1',2 if phase=='matched' else 8),('mate2',12),('endgame',24),('complex',16)]:
            indices=[i for i,t in enumerate(tiers) if t==tier]
            if not indices:continue
            batch,trace=play_batch(models+[frozen],[specs[i] for i in indices],max_plies=cap)
            for i,g,t in zip(indices,batch,trace):records[i]=g;traces[i]=t
        metrics=[]
        for learner in range(2):
            samples=[];bases=[];keys=[]
            for game,trace,spec,tier in zip(records,traces,specs,tiers):
                for index,side in enumerate(('white','black')):
                    if spec['players'][index]!=learner:continue
                    rewards=own_turn_rewards(game,side,auxiliary=tier in ('endgame','complex'),reward_version=reward_version)
                    key=tier+('attack' if side==game['candidate'] else 'defend')
                    samples.append((trace[side],rewards));bases.append(baselines[learner].get(key,0.));keys.append(key)
            metric,_=policy_update(models[learner],opts[learner],samples,0.,gamma=.98,entropy=entropy,chunk=chunk,sample_baselines=bases)
            for key in set(keys):
                returns=[sum(rewards) for (moves,rewards),k in zip(samples,keys) if k==key and moves]
                if returns:baselines[learner][key]=.9*baselines[learner].get(key,0.)+.1*float(np.mean(returns))
            metrics.append(metric)
        outcomes={tier:dict(games=sum(t==tier for t in tiers),wins=sum(g['outcome']['winner']==g['candidate'] for g,t in zip(records,tiers) if t==tier),
                    draws=sum(g['outcome']['result']=='1/2-1/2' for g,t in zip(records,tiers) if t==tier),
                    truncated=sum(g['outcome']['truncated'] for g,t in zip(records,tiers) if t==tier)) for tier in set(tiers)}
        history.append(dict(update=update+1,phase=phase,metrics=metrics,outcomes=outcomes,elapsed=time.monotonic()-t0))
        if (update+1)%10==0:pool=(pool+[cpu_snapshot(m) for m in models])[-4:]
        ck=dict(config=config,update=update+1,brains=[cpu_snapshot(m) for m in models],optimizers=[o.state_dict() for o in opts],
                pool=pool,baselines=baselines,history=history,rng=rng_state(rng))
        atomic_write(output/'resume.pt',ck,binary=True)
        if update+1 in (20,40,48,56) or update+1==updates:
            for i,m in enumerate(models):atomic_write(output/f'checkpoint_{update+1:04d}_{i}.pt',brain_payload(m),binary=True)
        save_game(records[0],output/'samples',update)
        write_json(output/'history.json',dict(config=config,history=history))
        print(update+1,phase,outcomes,'grad',[round(m['grad_norm'],4) for m in metrics],flush=True)
    return models,history


def main():
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--matched',default=str(ROOT/'artifacts/research/matched.json'));ap.add_argument('--extended',default=str(ROOT/'artifacts/research/extended.json'))
    ap.add_argument('--sources',nargs=2,required=True);ap.add_argument('--output',required=True)
    ap.add_argument('--seed',type=int,default=11);ap.add_argument('--updates',type=int,default=56);ap.add_argument('--games',type=int,default=64)
    ap.add_argument('--stage-only',action='store_true');ap.add_argument('--size',default='tiny');ap.add_argument('--device',default='cpu');ap.add_argument('--resume')
    ap.add_argument('--chunk',type=int,default=64)
    ap.add_argument('--reward-version',choices=('integer-v2','float-v1'),default='integer-v2')
    ap.add_argument('--lr',type=float,default=.003);ap.add_argument('--entropy',type=float,default=.003)
    a=ap.parse_args();train(json.load(open(a.matched)),json.load(open(a.extended)),a.output,a.sources,a.seed,a.updates,a.games,
                          STAGE_ONLY_ENCODING if a.stage_only else SPATIAL_ENCODING,a.size,a.device,a.resume,a.chunk,a.reward_version,a.lr,a.entropy)


if __name__=='__main__':main()
