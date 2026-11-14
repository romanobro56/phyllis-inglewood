"""Actual full FlyWire CPU compatibility probe; not a chess-learning experiment."""
import argparse
import hashlib
import platform
import resource
import time
from pathlib import Path
import torch
import numpy as np
from flies.chess.chess_env import ChessEnv,SPATIAL_ENCODING
from flies.chess.selfplay import play_game,save_game,assert_independent,choose_stage
from flies.chess.storage import ROOT,FIXED,load_policy,identity,state_hash,brain_payload,atomic_write,write_json,prepare_run
from flies.chess.train_selfplay import policy_update


def fixed_hash(model):
    h=hashlib.sha256()
    for key in FIXED:
        h.update(key.encode());h.update(model.state_dict()[key].cpu().contiguous().numpy().tobytes())
    return h.hexdigest()


def run(output,size='full'):
    output=prepare_run(output);torch.set_num_threads(1);torch.manual_seed(381)
    start=time.monotonic();models=[load_policy(size=size) for _ in range(2)]
    assert_independent(*models)
    assert not any(m.synthetic for m in models)
    fixed=[fixed_hash(m) for m in models];counts=[0,0]
    for i,model in enumerate(models):
        original=model.forward
        def forward(*args,_i=i,_original=original,**kwargs):
            counts[_i]+=1;return _original(*args,**kwargs)
        model.forward=forward
    players={side:dict(identity(model),label='UNTRAINED-FULL-CPU-PROBE') for side,model in zip(('white','black'),models)}
    game,traces=play_game(*models,seed=381,max_plies=2,activity=True,metadata=players)
    calls=list(counts);assert min(calls)>=4 and game['illegal_attempts']==0
    save_game(game,output,0)
    opts=[torch.optim.Adam(m.parameters(),lr=.0001) for m in models];metrics=[]
    for i,side in enumerate(('white','black')):
        before_other=state_hash(models[1-i])
        # From the normal start these two plies are unfinished. Entropy alone
        # exercises full sparse backward/Adam; do not call this learned chess.
        metric,_=policy_update(models[i],opts[i],[(traces[side],0.)],0.,entropy=.001,chunk=1)
        assert metric['grad_norm']>0 and before_other==state_hash(models[1-i])
        metrics.append(metric)
        atomic_write(output/f'brain_{i}.pt',dict(brain_payload(models[i]),training_status='engineering-probe-only'),binary=True)
    restored=load_policy(output/'brain_0.pt',size=size)
    assert state_hash(restored)==state_hash(models[0])
    # Both supported 64-float board encodings use the same full sensory mapping.
    # This extra forward is a compatibility probe, never a trained-policy claim.
    restored.encoding=SPATIAL_ENCODING
    _,spatial,_=choose_stage(restored,ChessEnv(),np.random.default_rng(381),activity=True)
    assert len(spatial['activity']['group_rate'])==8
    assert [fixed_hash(m) for m in models]==fixed
    peak=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    result=dict(status='PASS',size=size,neurons=models[0].n,edges=models[0].log_w.numel(),
                graph_hash=models[0].graph_hash,synthetic=False,device='cpu',game_forward_calls=calls,
                illegal_attempts=game['illegal_attempts'],optimizer_metrics=metrics,
                fixed_architecture_unchanged=True,independent_parameter_and_optimizer_state=True,
                checkpoint_roundtrip=True,elapsed_seconds=time.monotonic()-start,
                spatial_encoding_forward=True,
                peak_process_bytes=int(peak if platform.system()=='Darwin' else peak*1024),
                note='Two full CPU brains, two actual legal plies, group activity, independent entropy-only backward/Adam and checkpoint round-trip. No GPU, full training, performance or full-game claim.')
    write_json(output/'verification.json',result);print(result,flush=True);return result


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--output',default=str(ROOT/'artifacts/full-cpu-verification'))
    a=p.parse_args();run(a.output)
