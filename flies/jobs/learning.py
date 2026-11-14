"""Resumable reward-only training, validation-gated curriculum and paired controls.

Uses core.train.rollout and _logp_batches. No teacher, imitation or scripted policy.
Test evaluation is a separate explicit command and never advances the curriculum.
"""
import argparse
import csv
import gzip
import hashlib
import json
from pathlib import Path
import time
import signal
import math
import numpy as np
import torch
from core.brain import FlyBrain, pick_device
from core.train import rollout, settings, _logp_batches
from flies.jobs.contract import SCHEMA
from flies.jobs.policy import digest, save_model, load_model, probabilities, checkpoint_path
from flies.jobs.sim_state import SimulationState, SecretVault, atomic_json, private_vault_path
from flies.jobs.task import JobsFly
from flies.jobs.workday_forms import fingerprint, SPLITS


class Curriculum:
    def __init__(self, level=1, streak=0, history=None):
        self.level, self.streak, self.history = level, streak, history or []

    def consider(self, metrics, episodes, update):
        # No advance by time/updates, and no promotion from safe-review shortcuts.
        passed = metrics['completed'] >= .75 and metrics['field_accuracy'] >= .85 and metrics['episodes'] >= 32
        self.streak = self.streak + 1 if passed else 0
        self.history.append({'update': update, 'episodes': episodes, 'level': self.level, 'passed': passed, 'completion': metrics['completed']})
        if self.streak >= 2 and self.level < 10:
            self.level += 1
            self.streak = 0

    def sample(self, rng):
        # 30% rehearsal of previous levels; level 10 mixes all earlier workflows.
        if self.level > 1 and rng.random() < .3:
            return int(rng.integers(1, self.level))
        return self.level

    def snapshot(self):
        return dict(level=self.level, streak=self.streak, history=self.history)


def reinforce(brain, optimizer, transitions, cfg, device='cpu'):
    returns = np.zeros(len(transitions))
    by_episode = {}
    for i, t in enumerate(transitions):
        by_episode.setdefault(t[5], []).append(i)
    for indices in by_episode.values():
        value = 0.
        for i in reversed(indices):
            value = transitions[i][4] + cfg['gamma'] * value
            returns[i] = value
    steps = np.asarray([t[6] for t in transitions])
    advantage = returns.copy()
    for t in np.unique(steps):
        advantage[steps == t] -= returns[steps == t].mean()
    advantage /= advantage.std()+1e-8
    optimizer.zero_grad()
    for batch, actions, logp, offset in _logp_batches(brain, transitions, cfg['chunk'], device):
        adv = torch.as_tensor(advantage[offset:offset+len(batch)], dtype=torch.float32, device=device)
        selected = logp.gather(1, actions[:, None]).squeeze(1)
        entropy = -(logp.exp() * logp.clamp(min=-30)).sum(-1)
        loss = -(selected*adv).sum()/len(transitions) - cfg['entropy']*entropy.sum()/len(transitions)
        loss.backward()
    torch.nn.utils.clip_grad_norm_(brain.parameters(), 1.)
    optimizer.step()


def aggregate(envs):
    episodes = [e.metrics() for e in envs]
    result = {k: float(np.mean([e[k] for e in episodes])) for k in episodes[0]}
    for numerator,denominator in (('account_creation_success','account_creation_opportunity'),('login_success','login_opportunity'),('resume_upload_success','resume_upload_opportunity')):
        attempts=sum(e[denominator] for e in episodes)
        result[numerator+'_rate']=sum(e[numerator] for e in episodes)/attempts if attempts else None
    result.update(episodes=len(episodes), per_episode=episodes)
    return result


def evaluate(brain, policy, seeds, level, split='validation', log=None, checkpoint='', device='cpu', browser=None):
    envs, rngs, mirrors = [], [], []
    for seed in seeds:
        # Same generated forms, seeded accounts, vault and action random variates.
        form_rng = np.random.default_rng(SPLITS[split]+seed)
        vault = SecretVault(hashlib.sha256(f'simulation-only:{split}:{seed}'.encode()).digest())
        env = JobsFly(split=split, level=level, vault=vault, trace=True)
        env.reset(form_rng)
        envs.append(env)
        rngs.append(np.random.default_rng(seed+500000))
        if browser:
            from flies.jobs.browser import BrowserMirror
            mirror = BrowserMirror(browser.new_page())
            mirror.open(env)
            mirrors.append(mirror)
    forms_hash = hashlib.sha256(''.join(fingerprint(e.scenario) for e in envs).encode()).hexdigest()
    parameter_hash = digest(brain) if brain is not None and policy!='random' else None
    try:
        for _ in range(JobsFly.max_steps):
            active = [i for i,e in enumerate(envs) if not e.done]
            if not active:
                break
            obs = [(mirrors[i].observation() if mirrors else envs[i].obs()) for i in active]
            p = probabilities(brain, obs, [envs[i].mask() for i in active], policy, device)
            for row, i in enumerate(active):
                env = envs[i]
                action = int(rngs[i].choice(8, p=p[row]/p[row].sum()))
                if mirrors:
                    mirrors[i].execute(env, action)
                _, _, _, info = env.step(action, rngs[i])
                if mirrors:
                    mirrors[i].sync(env, info, policy, checkpoint)
                    mirrors[i].check(env)
        if log:
            for seed, env in zip(seeds, envs):
                for event in env.traces:
                    log.write(json.dumps({**event, 'seed': seed, 'split': split, 'level': level,
                                          'policy': policy, 'checkpoint': 'none:uniform_random' if policy=='random' else str(checkpoint), 'parameter_hash': parameter_hash})+'\n')
        result = aggregate(envs)
        result.update(forms_hash=forms_hash, policy=policy, split=split, level=level, parameter_hash=parameter_hash)
        return result
    finally:
        for m in mirrors:
            m.page.close()


def comparison(a, b):
    rng = np.random.default_rng(123)
    result = {}
    for metric in ('completed','field_accuracy','reward'):
        delta = np.asarray([x[metric]-y[metric] for x,y in zip(a['per_episode'],b['per_episode'])])
        boot = delta[rng.integers(len(delta), size=(2000,len(delta)))].mean(1)
        result[metric] = {'difference': float(delta.mean()), 'paired_95_percent_interval': np.quantile(boot,[.025,.975]).tolist()}
    return result


def report(out, progress):
    atomic_json(out/'progress.json', {'schema': SCHEMA, 'progress': progress})
    columns = ['updates','training_episodes','level','policy','reward','completed','field_accuracy','account_creation_success','login_success','resume_upload_success','account_creation_success_rate','login_success_rate','resume_upload_success_rate','actions','validation_error_rate','failed_application']
    rows = []
    for record in progress:
        for evaluation in record['evaluations']:
            rows.append({k: ({**record, **evaluation})[k] for k in columns})
    with (out/'progress.csv').open('w',newline='') as f:
        writer = csv.DictWriter(f,fieldnames=columns)
        writer.writeheader(); writer.writerows(rows)
    lines = ['# Workday-inspired learning experiment', '', 'Local simulation only. No employer applications. No competence claim without paired full-workflow evidence.', '',
             '| Updates | Episodes | Level | Policy | Reward | Completion | Field accuracy |', '|---:|---:|---:|---|---:|---:|---:|']
    for r in rows:
        lines.append(f"| {r['updates']} | {r['training_episodes']} | {r['level']} | {r['policy']} | {r['reward']:.2f} | {r['completed']:.1%} | {r['field_accuracy']:.1%} |")
    (out/'REPORT.md').write_text('\n'.join(lines)+'\n')
    # Standalone SVG learning curve via standard plotting library, if installed.
    try:
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
        fig, axes = plt.subplots(1,3,figsize=(12,3.5))
        for policy in ('random','untrained','trained','lesioned'):
            selected = [r for r in rows if r['policy']==policy]
            for ax, metric in zip(axes,('reward','completed','field_accuracy')):
                if selected:
                    ax.plot([r['training_episodes'] for r in selected],[r[metric] for r in selected],marker='.',label=policy)
                ax.set_title(metric.replace('_',' ')); ax.set_xlabel('training episodes')
        axes[-1].legend();fig.suptitle('Local simulation; validation levels may change with curriculum')
        fig.tight_layout(); fig.savefig(out/'learning_curve.svg');plt.close(fig)
    except ImportError:
        (out/'learning_curve_unavailable.txt').write_text('Install matplotlib and rerun evaluation/report to create the learning curve. CSV is available.\n')


def save_training(out, brain, optimizer, world, rng, curriculum, update, episodes, progress, cfg):
    # Immutable generations + atomic pointer prevent partially replaced model/state pairs.
    relative = Path('checkpoints') / f'update_{update:06d}'
    directory = out/relative
    retry=0
    while directory.exists():
        retry+=1
        relative=Path('checkpoints')/f'update_{update:06d}_retry_{retry}'
        directory=out/relative
    directory.mkdir(parents=True,exist_ok=False)
    save_model(brain,directory/'brain.pt',JobsFly)
    torch.save({'schema':SCHEMA,'optimizer':optimizer.state_dict(),'numpy_rng':rng.bit_generator.state,
                'torch_rng':torch.get_rng_state(), 'cuda_rng':torch.cuda.get_rng_state_all() if torch.cuda.is_available() else [],
                'curriculum':curriculum.snapshot(),'updates':update,'episodes':episodes,'progress':progress,'settings':cfg,
                'model_hash':digest(brain)},directory/'training.pt')
    (directory/'training.pt').chmod(0o600)
    world.save(directory/'simulation_state.json')
    atomic_json(out/'latest.json',{'schema':SCHEMA,'directory':str(relative),'model_hash':digest(brain)})


def main():
    parser=argparse.ArgumentParser(__doc__)
    parser.add_argument('--out',type=Path,default=Path('flies/jobs/artifacts/workday'))
    parser.add_argument('--resume',action='store_true')
    parser.add_argument('--vault-file',type=Path,help='Private 0600 simulation credential master; default outside mounted repository')
    parser.add_argument('--size',default=None,choices=['tiny','small','central','full'])
    parser.add_argument('--device',default='auto')
    parser.add_argument('--updates',type=int,default=100)
    parser.add_argument('--episodes',type=int,default=None)
    parser.add_argument('--chunk',type=int,default=None,help='Gradient batch size; lower this if GPU memory is limited')
    parser.add_argument('--lr',type=float,default=None)
    parser.add_argument('--gamma',type=float,default=None)
    parser.add_argument('--entropy',type=float,default=None)
    parser.add_argument('--checkpoint-every',type=int,default=10)
    parser.add_argument('--eval-every',type=int,default=10)
    parser.add_argument('--eval-episodes',type=int,default=32)
    parser.add_argument('--minutes',type=float,default=15)
    parser.add_argument('--seed',type=int,default=2026)
    args=parser.parse_args()
    if min(args.updates,args.episodes or 16,args.eval_every,args.eval_episodes)<1 or (args.episodes is not None and args.episodes<2) or (args.chunk is not None and args.chunk<1) or (args.lr is not None and (not math.isfinite(args.lr) or args.lr<=0)) or not math.isfinite(args.minutes) or args.minutes<=0 or args.checkpoint_every<1 or (args.gamma is not None and not 0<=args.gamma<=1) or (args.entropy is not None and (not math.isfinite(args.entropy) or args.entropy<0)):
        parser.error('counts/time must be positive and episodes >= 2')
    torch.set_num_threads(2)
    device=pick_device(args.device)
    if device.type=='cuda' and not torch.cuda.is_available():
        parser.error('CUDA GPU unavailable; use CUDA-enabled PyTorch on the RTX 4090 host')
    args.out.mkdir(parents=True,exist_ok=True)
    vault_file=args.vault_file or private_vault_path(args.out)
    if args.resume and not vault_file.exists():
        raise ValueError('Simulation vault missing; restore the same private vault before resuming')
    vault=SecretVault.open(vault_file)
    cfg=settings(JobsFly,{'episodes':args.episodes,'chunk':args.chunk,'lr':args.lr,'gamma':args.gamma,'entropy':args.entropy})
    if args.resume:
        path=checkpoint_path(args.out)
        brain=load_model(path,JobsFly,device)
        saved=torch.load(path.parent/'training.pt',map_location='cpu',weights_only=False)
        if saved['schema']!=SCHEMA or saved['model_hash']!=digest(brain):
            raise ValueError('checkpoint generation mismatch')
        cfg=saved['settings']
        cfg.update({k:v for k,v in {'episodes':args.episodes,'chunk':args.chunk,'lr':args.lr,'gamma':args.gamma,'entropy':args.entropy}.items() if v is not None})
        if args.size and args.size!=brain.preset:
            raise ValueError('Cannot resume with a different connectome size; start a new full-brain experiment')
        optimizer=torch.optim.Adam(brain.parameters(),lr=cfg['lr'])
        optimizer.load_state_dict(saved['optimizer'])
        for group in optimizer.param_groups:group['lr']=cfg['lr']
        rng=np.random.default_rng();rng.bit_generator.state=saved['numpy_rng']
        torch.set_rng_state(saved['torch_rng'])
        if saved['cuda_rng'] and torch.cuda.is_available():
            torch.cuda.set_rng_state_all(saved['cuda_rng'])
        curriculum=Curriculum(**saved['curriculum'])
        update,episodes,progress=saved['updates'],saved['episodes'],saved['progress']
        world=SimulationState.load(path.parent/'simulation_state.json',required=True)
        world.bind_vault(vault)
        print(f'RESUMED update={update} episodes={episodes} hash={digest(brain)}',flush=True)
    else:
        if (args.out/'untrained.pt').exists() or (args.out/'latest.json').exists():
            parser.error('experiment exists; use --resume or a fresh --out')
        torch.manual_seed(args.seed)
        brain=FlyBrain(args.size or 'tiny',64,8,'jobs').to(device)
        if brain.synthetic:
            raise ValueError('real FlyWire graph required')
        optimizer=torch.optim.Adam(brain.parameters(),lr=cfg['lr'])
        rng=np.random.default_rng(args.seed)
        world=SimulationState();curriculum=Curriculum()
        update=episodes=0;progress=[]
        save_model(brain,args.out/'untrained.pt',JobsFly)
        atomic_json(args.out/'manifest.json',{'schema':SCHEMA,'size':brain.preset,'seed':args.seed,'settings':cfg,'graph_hash':brain.graph_hash})
    validation_seeds=list(range(args.eval_episodes))
    with gzip.open(args.out/'actions.jsonl.gz','at',encoding='utf-8') as log:
        if not progress:
            evaluations=[evaluate(brain,p,validation_seeds,1,log=log,checkpoint=args.out/'untrained.pt',device=device) for p in ('random','untrained','lesioned')]
            progress.append({'updates':0,'training_episodes':0,'evaluations':evaluations})
            report(args.out,progress)
            print('BASELINE '+json.dumps([{k:v for k,v in m.items() if k!='per_episode'} for m in evaluations]),flush=True)
        stopping={'requested':False}
        def request_stop(signum,frame):
            stopping['requested']=True
            print('Stop requested: finishing the current update and saving a checkpoint.',flush=True)
        previous_handlers={sig:signal.signal(sig,request_stop) for sig in (signal.SIGINT,signal.SIGTERM)}
        deadline=time.time()+args.minutes*60
        start_update=update
        target=update+args.updates
        while update<target and time.time()<deadline and not stopping['requested']:
            envs=[]
            # Shared core rollout is the sole action selector during training.
            class Environment(JobsFly):
                def __init__(self):
                    super().__init__(level=curriculum.sample(rng),state=world,vault=vault,trace=True)
                    envs.append(self)
            parameter_hash=digest(brain)
            _,transitions=rollout(Environment,cfg['episodes'],rng,'brain',brain,device)
            for i,env in enumerate(envs):
                for event in env.traces:
                    log.write(json.dumps({**event,'policy':'trained','split':'train','training_episode':episodes+i,'level':env.level,
                                          'parameter_hash':parameter_hash,'checkpoint':str(args.out),'revision':update})+'\n')
            reinforce(brain,optimizer,transitions,cfg,device)
            update+=1;episodes+=cfg['episodes']
            training_metrics=aggregate(envs)
            print(f'update={update} episodes={episodes} level={curriculum.level} reward={training_metrics["reward"]:.3f} completion={training_metrics["completed"]:.3f}',flush=True)
            validation_due=update%args.eval_every==0 or update==target or time.time()>=deadline
            if validation_due and not stopping['requested']:
                before_level=curriculum.level
                evaluations=[evaluate(brain,p,validation_seeds,before_level,log=log,checkpoint=f'update_{update}',device=device) for p in ('trained','lesioned','random')]
                curriculum.consider(evaluations[0],episodes,update)
                progress.append({'updates':update,'training_episodes':episodes,'evaluations':evaluations,'training_metrics':{k:v for k,v in training_metrics.items() if k!='per_episode'}})
                report(args.out,progress)
                print('VALIDATION '+json.dumps([{k:v for k,v in m.items() if k!='per_episode'} for m in evaluations]),flush=True)
            if update==start_update+1 or update%args.checkpoint_every==0 or validation_due or stopping['requested']:
                log.flush()
                save_training(args.out,brain,optimizer,world,rng,curriculum,update,episodes,progress,cfg)
        for sig,handler in previous_handlers.items():signal.signal(sig,handler)
        print(f'Saved {episodes} episodes. Curriculum level {curriculum.level}. Full-workflow learning is NOT established by this training command.',flush=True)


if __name__=='__main__':
    main()
