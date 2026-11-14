"""CPU measurements on actual tiny; transparent full-graph storage projections."""
import argparse
import json
import platform
import resource
import time
from pathlib import Path
import numpy as np
import torch
from flies.chess.chess_env import ChessEnv,SPATIAL_ENCODING
from flies.chess.storage import ROOT,load_policy,write_json


def peak_bytes():
    n=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    return int(n if platform.system()=='Darwin' else n*1024)


def graph_storage(n,e,inputs,batch):
    parameters=e+2*n+64*inputs
    buffers=44*e+8*inputs+8*128
    model=4*parameters+buffers
    # Two independent Adam learners (parameters, grads, first/second moments),
    # one frozen model, and at least eight saved dense neuron-rate states.
    lower=2*(16*parameters+buffers)+model+8*n*batch*4
    return dict(parameters=parameters,fixed_buffer_bytes=buffers,model_bytes=model,
                two_adam_plus_frozen_and_rate_history_lower_bound_bytes=lower,
                note='Lower bound excludes backend scratch, other autograd intermediates, allocator overhead and host snapshot pool')


def run(output):
    torch.set_num_threads(1);torch.manual_seed(1);before=peak_bytes()
    brain=load_policy(size='tiny',encoding=SPATIAL_ENCODING)
    env=ChessEnv();base=env.observe(SPATIAL_ENCODING);mask=env.mask();rows=[]
    for batch in (1,8,32):
        obs=torch.tensor(np.repeat(base[None],batch,axis=0));m=torch.tensor(np.repeat(mask[None],batch,axis=0))
        opt=torch.optim.Adam(brain.parameters(),lr=0.)  # Allocate/update Adam state without changing rates between repetitions.
        for _ in range(2):
            with torch.no_grad():brain(obs,m)
        forward=[];backward=[]
        for _ in range(5):
            s=time.perf_counter()
            with torch.no_grad():brain(obs,m)
            forward.append(time.perf_counter()-s)
            s=time.perf_counter();opt.zero_grad(set_to_none=True);logits,_=brain(obs,m)
            (-logits.log_softmax(-1)[:,0].mean()).backward();opt.step();backward.append(time.perf_counter()-s)
        rows.append(dict(batch=batch,forward_seconds_median=float(np.median(forward)),
                         forward_backward_adam_seconds_median=float(np.median(backward)),peak_process_bytes=peak_bytes()))
    graphs={}
    for size in ('tiny','full'):
        path=ROOT.parents[1]/'data'/f'{size}.npz'
        if path.exists():
            with np.load(path,allow_pickle=False) as z:
                n,e,inputs=int(z['n']),len(z['pre']),len(z['input_idx'])
                graphs[size]=dict(n=n,edges=e,inputs=inputs,graph_hash=str(z['graph_hash']),
                                  projections={str(b):graph_storage(n,e,inputs,b) for b in (1,8,32)})
    result=dict(device='cpu',synthetic=brain.synthetic,neurons=brain.n,threads=1,measurements=rows,
                initial_process_peak_bytes=before,final_process_peak_bytes=peak_bytes(),graphs=graphs,
                note='Full graph was read for metadata only, never instantiated or trained. No CUDA performance measurement.')
    write_json(output,result);print(json.dumps(result,indent=2))
    return result


if __name__=='__main__':
    ap=argparse.ArgumentParser(description=__doc__);ap.add_argument('--output',default=str(ROOT/'artifacts/research/benchmark.json'))
    run(ap.parse_args().output)
