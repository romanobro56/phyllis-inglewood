"""Check graph/device, public sensors, neural forward/backward and memory. Not training evidence."""
import argparse
import json
import numpy as np
import torch
from core.brain import FlyBrain,pick_device
from flies.jobs.task import JobsFly
from flies.jobs.policy import digest


def main():
    parser=argparse.ArgumentParser(__doc__)
    parser.add_argument('--size',choices=['tiny','small','central','full'],default='full')
    parser.add_argument('--device',default='cuda')
    parser.add_argument('--batch-size',type=int,default=64)
    args=parser.parse_args()
    if args.batch_size<2:parser.error('batch-size must be >= 2')
    device=pick_device(args.device)
    if device.type=='cuda' and not torch.cuda.is_available():
        parser.error('CUDA unavailable; install CUDA-enabled PyTorch on the 4090 host')
    torch.set_num_threads(2);torch.manual_seed(2026)
    brain=FlyBrain(args.size,64,8,'jobs').to(device)
    if brain.synthetic:raise ValueError('real FlyWire graph required')
    env=JobsFly(level=1);rng=np.random.default_rng(0)
    obs=[];masks=[]
    for _ in range(args.batch_size):
        obs.append(env.reset(rng));masks.append(env.mask())
    before=digest(brain)
    opt=torch.optim.Adam(brain.parameters(),lr=.0005)
    if device.type=='cuda':torch.cuda.reset_peak_memory_stats(device)
    logits,_=brain(torch.as_tensor(np.stack(obs),device=device),torch.as_tensor(np.stack(masks),device=device))
    distribution=torch.distributions.Categorical(logits=logits)
    loss=-distribution.log_prob(distribution.sample()).mean()
    loss.backward();opt.step()
    if before==digest(brain):raise RuntimeError('neural parameter update failed')
    print(json.dumps({'status':'PASS','purpose':'hardware and core API preflight, not proficiency training',
                      'preset':brain.preset,'neurons':brain.n,'graph_hash':brain.graph_hash,'synthetic':brain.synthetic,
                      'device':str(device),'batch_size':args.batch_size,
                      'peak_cuda_memory_bytes':torch.cuda.max_memory_allocated(device) if device.type=='cuda' else None},indent=2))


if __name__=='__main__':main()
