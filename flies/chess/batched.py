"""Batch independent legal chess games through the existing FlyBrain.forward.

This changes throughput only: identical per-game RNG streams and masks reproduce
serial selfplay.play_game choices, with normal floating-point batching tolerance.
"""
import time
import numpy as np
import torch
from flies.chess.chess_env import ChessEnv, STAGES, ENCODING
from flies.chess.selfplay import assert_independent


def play_batch(models, specs, max_plies=4, random_candidate=False, lesion=False, shuffle_observations=False):
    """Specs contain fen/seed/players=(white model index, black model index).

    Candidate means the initial side to move. Random candidate is an explicitly
    labeled evaluation control only. Training calls this with every flag false.
    """
    for i in range(len(models)):
        for j in range(i):
            assert_independent(models[i],models[j])
    for spec in specs:
        players=spec['players']
        if len(players)!=2 or players[0]==players[1] or any(not isinstance(i,(int,np.integer)) or not 0<=i<len(models) for i in players):
            raise ValueError('Every game requires two distinct valid model indices')
    envs = [ChessEnv(s['fen'], max_plies=max_plies) for s in specs]
    rngs = [np.random.default_rng(s['seed']) for s in specs]
    games = [dict(initial_fen=e.board.fen(), moves=[], candidate='white' if e.board.turn else 'black',
                  seed=s['seed'], mode='random-legal-baseline' if random_candidate else 'two-FlyBrain-selfplay')
             for e,s in zip(envs,specs)]
    traces = [{'white':[], 'black':[]} for _ in specs]
    pending = [[] for _ in specs]
    stage_records = [[] for _ in specs]
    t0=time.monotonic()
    while any(not e.done for e in envs):
        for model_index, model in enumerate(models):
            indices = [i for i,e in enumerate(envs) if not e.done
                       and specs[i]['players'][0 if e.board.turn else 1] == model_index]
            if not indices:
                continue
            neural=[]
            for i in indices:
                env=envs[i]; side='white' if env.board.turn else 'black'
                if random_candidate and side==games[i]['candidate']:
                    legal=tuple(env.board.legal_moves)
                    event=env.push(legal[int(rngs[i].integers(len(legal)))])
                    event.update(stages=[], controller='uniform-random-legal-baseline')
                    games[i]['moves'].append(event)
                else:
                    neural.append(i)
            if not neural:
                continue
            obs=np.stack([envs[i].observe(getattr(model,'encoding',ENCODING)) for i in neural])
            masks=np.stack([envs[i].mask() for i in neural])
            candidate_flags=[('white' if envs[i].board.turn else 'black')==games[i]['candidate'] for i in neural]
            same_board_donor=list(candidate_flags)
            if shuffle_observations:
                # Evaluation-only deterministic scrambling within current turn/stage.
                # Preserve actual masks; never replace a chosen action or full move.
                for stage in range(5):
                    rows=[j for j,x in enumerate(candidate_flags) if x and envs[neural[j]].stage==stage]
                    if len(rows)>1:
                        # Repeats are adjacent: a one-row shift mostly preserves
                        # the same board! Derange board identities, not row IDs.
                        shifts=[max(1,len(rows)//2)]+list(range(1,len(rows)))
                        best=None
                        for shift in shifts:
                            donor=np.roll(rows,shift)
                            same=np.all(obs[rows,:32]==obs[donor,:32],axis=1)
                            if best is None or same.sum()<best[0]:
                                best=(int(same.sum()),donor,same)
                            if not same.any():break
                        _,donor,same=best
                        for row,flag in zip(rows,same):same_board_donor[row]=bool(flag)
                        obs[rows, :32]=obs[donor, :32]
                        obs[rows, 53:64]=obs[donor, 53:64]
            device=next(model.parameters()).device
            with torch.no_grad():
                # Lesion the candidate only. Mixed batches require two real forward calls.
                probs=np.zeros((len(neural),8),dtype=float)
                for flag in (False,True):
                    rows=[j for j,x in enumerate(candidate_flags) if x==flag]
                    if not rows:
                        continue
                    logits,_=model(torch.as_tensor(obs[rows],device=device),
                        torch.as_tensor(masks[rows],device=device),lesion=lesion and flag)
                    if not torch.isfinite(logits).all():
                        raise ValueError('Nonfinite FlyBrain logits')
                    probs[rows]=(logits/getattr(model,'policy_temperature',1.0)).softmax(-1).cpu().numpy()
            for j,i in enumerate(neural):
                env=envs[i]; side='white' if env.board.turn else 'black'
                p=probs[j]; p[~masks[j]]=0; p/=p.sum()
                a=int(rngs[i].choice(8,p=p))
                record=dict(stage=STAGES[env.stage],action=a,mask=masks[j].tolist(),
                            probabilities=p.tolist(),selected_probability=float(p[a]),side=side,fen=env.board.fen())
                if shuffle_observations and candidate_flags[j]:
                    record['same_board_donor']=same_board_donor[j] or len(neural)==1
                stage_records[i].append(record)
                pending[i].append(dict(obs=obs[j].copy(),mask=masks[j].copy(),action=a))
                event=env.select(a)
                if event:
                    event.update(stages=stage_records[i],controller='FlyBrain')
                    games[i]['moves'].append(event)
                    traces[i][side].append(pending[i])
                    pending[i],stage_records[i]=[],[]
    elapsed=time.monotonic()-t0
    for game,env in zip(games,envs):
        game.update(final_fen=env.board.fen(),outcome=env.outcome(),plies=env.plies,
                    illegal_attempts=env.illegal_attempts,elapsed_seconds=elapsed/max(1,len(games)))
    return games,traces
