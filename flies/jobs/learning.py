"""Resumable REINFORCE using core FlyBrain, rollout and gradient batches.

No imitation, teacher, answer-based action selector, or learned output head.
All action selection in rollout is the existing core/train.py implementation.
"""
import argparse
import gzip
import hashlib
import json
from pathlib import Path
import time
import numpy as np
import torch
from core.brain import FlyBrain, save_brain, load_brain
from core.train import rollout, settings, _logp_batches
from flies.jobs.task import JobsFly
from flies.jobs.forms import VERSION


def digest(brain):
    h = hashlib.sha256()
    for name, parameter in brain.named_parameters():
        h.update(name.encode())
        h.update(parameter.detach().cpu().numpy().tobytes())
    return h.hexdigest()


def env_factory(split='train', difficulty=1., trace=True):
    envs = []
    class Environment(JobsFly):
        def __init__(self):
            super().__init__(split=split, difficulty=difficulty, trace=trace)
            envs.append(self)
    return Environment, envs


def evaluate(brain, policy, seeds, checkpoint, log=None, browser=None):
    """Paired forms and action RNG across ALL policies. No state carries between forms."""
    parameter_hash = digest(brain) if brain is not None else None
    envs = [JobsFly(split='test', trace=True) for _ in seeds]
    obs = [e.reset(np.random.default_rng(seed)) for e, seed in zip(envs, seeds)]
    mirrors = []
    if browser:
        from flies.jobs.browser import BrowserMirror
        for env in envs:
            mirror = BrowserMirror(browser.new_page())
            mirror.open(env)
            mirrors.append(mirror)
        obs = [m.observation(e) for m, e in zip(mirrors, envs)]
    action_rngs = [np.random.default_rng(seed + 500000) for seed in seeds]
    forms_hash = hashlib.sha256(json.dumps([e.fields for e in envs], sort_keys=True).encode()).hexdigest()
    for _ in range(JobsFly.max_steps):
        active = [i for i, env in enumerate(envs) if not env.done]
        if not active:
            break
        masks = np.asarray([envs[i].mask() for i in active])
        if policy == 'random':
            probabilities = masks / masks.sum(1, keepdims=True)
        else:
            with torch.no_grad():
                logits, _ = brain(torch.tensor(np.stack([obs[i] for i in active])),
                                  torch.tensor(masks), lesion=policy == 'lesioned')
                probabilities = logits.softmax(-1).cpu().numpy().astype(float)
        for row, i in enumerate(active):
            p = probabilities[row] / probabilities[row].sum()
            action = int(action_rngs[i].choice(len(JobsFly.actions), p=p))
            if mirrors:
                mirrors[i].execute(envs[i], action)
            obs[i], _, _, _ = envs[i].step(action, action_rngs[i])
            if mirrors:
                mirrors[i].check(envs[i])
                obs[i] = mirrors[i].observation(envs[i])
    for mirror in mirrors:
        mirror.page.close()
    episodes = [e.metrics() for e in envs]
    if log:
        for seed, env in zip(seeds, envs):
            for detail in env.traces:
                write_event(log, {**detail, 'seed': seed, 'policy': policy,
                                  'checkpoint': str(checkpoint), 'parameter_hash': parameter_hash, 'split': 'heldout'})
    metrics = {key: float(np.mean([e[key] for e in episodes])) for key in episodes[0]}
    metrics.update(episodes=len(seeds), forms_hash=forms_hash, policy=policy,
                   checkpoint=str(checkpoint), parameter_hash=parameter_hash, per_episode=episodes)
    return metrics


def write_event(log, detail):
    log.write(json.dumps(detail, separators=(',', ':')) + '\n')


def reinforce(brain, optimizer, transitions, cfg):
    """Same return-to-go REINFORCE loss as core.train.train_reward, with saved optimizer."""
    returns = np.zeros(len(transitions))
    by_episode = {}
    for index, transition in enumerate(transitions):
        by_episode.setdefault(transition[5], []).append(index)
    for indices in by_episode.values():
        value = 0.
        for index in reversed(indices):
            value = transitions[index][4] + cfg['gamma'] * value
            returns[index] = value
    steps = np.asarray([t[6] for t in transitions])
    baseline = np.zeros_like(returns)
    for step in np.unique(steps):
        baseline[steps == step] = returns[steps == step].mean()
    advantage = returns - baseline
    advantage /= advantage.std() + 1e-8
    optimizer.zero_grad()
    for batch, actions, logp, offset in _logp_batches(brain, transitions, cfg['chunk'], 'cpu'):
        adv = torch.tensor(advantage[offset:offset + len(batch)], dtype=torch.float32)
        selected = logp.gather(1, actions[:, None]).squeeze(1)
        entropy = -(logp.exp() * logp.clamp(min=-30)).sum(-1)
        loss = -(selected * adv).sum() / len(transitions) - cfg['entropy'] * entropy.sum() / len(transitions)
        loss.backward()
    torch.nn.utils.clip_grad_norm_(brain.parameters(), 1.)
    optimizer.step()


def comparison(trained, baseline):
    """Paired bootstrap CI. Evidence is positive only if its lower bound is > 0."""
    rng = np.random.default_rng(123)
    results = {}
    for metric in ('completed', 'field_accuracy', 'reward'):
        delta = np.asarray([a[metric] - b[metric] for a, b in zip(trained['per_episode'], baseline['per_episode'])])
        boot = delta[rng.integers(len(delta), size=(2000, len(delta)))].mean(1)
        results[metric] = {'difference': float(delta.mean()),
                           'paired_95_percent_interval': np.quantile(boot, [.025, .975]).tolist()}
    return results


def report(directory, progress, final=None):
    payload = {'environment_version': VERSION, 'progress': progress, 'final': final,
               'limitations': ['Tiny real FlyWire subgraph; not the full brain.',
                              'Finite lexical sensors and fixed resume; no general language understanding.',
                              'Synthetic forms only. No employer application was submitted.']}
    (directory / 'progress.json').write_text(json.dumps(payload, indent=2))
    lines = ['# Jobs fly learning evidence', '', 'All results are local simulation, not real applications.', '',
             '| Updates | Training episodes | Policy | Mean reward | Completion | Field accuracy | Actions | Error rate |',
             '|---:|---:|---|---:|---:|---:|---:|---:|']
    for row in progress:
        for m in row['evaluations']:
            lines.append(f"| {row['updates']} | {row['training_episodes']} | {m['policy']} | {m['reward']:.2f} | {m['completed']:.1%} | {m['field_accuracy']:.1%} | {m['actions']:.1f} | {m['error_rate']:.1%} |")
    if final:
        lines += ['', 'Final untouched test-set comparisons and paired confidence intervals are in progress.json.',
                  f"Acceptance criteria met: **{final['acceptance_passed']}**."]
    (directory / 'REPORT.md').write_text('\n'.join(lines) + '\n')


def main():
    parser = argparse.ArgumentParser(__doc__)
    parser.add_argument('--out', type=Path, default=Path('flies/jobs/artifacts/experiment'))
    parser.add_argument('--updates', type=int, default=100)
    parser.add_argument('--episodes', type=int, default=32)
    parser.add_argument('--eval-every', type=int, default=10)
    parser.add_argument('--eval-episodes', type=int, default=64)
    parser.add_argument('--minutes', type=float, default=15)
    parser.add_argument('--seed', type=int, default=2026)
    parser.add_argument('--resume', action='store_true')
    parser.add_argument('--curriculum-stage-updates', type=int, default=40)
    args = parser.parse_args()
    if min(args.updates, args.episodes, args.eval_every, args.eval_episodes, args.curriculum_stage_updates) < 1:
        parser.error('counts must be positive')
    torch.set_num_threads(2)
    args.out.mkdir(parents=True, exist_ok=True)
    config_path = args.out / 'training_config.json'
    if args.resume and config_path.exists():
        args.curriculum_stage_updates = json.loads(config_path.read_text())['curriculum_stage_updates']
    cfg = settings(JobsFly, {'episodes': args.episodes})
    config_path.write_text(json.dumps({'curriculum_stage_updates': args.curriculum_stage_updates,
                                       'settings': cfg, 'seed': args.seed}, indent=2))
    progress = []
    if args.resume:
        brain = load_brain(args.out / 'latest.pt', JobsFly)
        state = torch.load(args.out / 'optimizer.pt', weights_only=False)
        if state['version'] != VERSION or state['model_hash'] != digest(brain):
            raise ValueError('checkpoint/optimizer mismatch or old environment')
        optimizer = torch.optim.Adam(brain.parameters(), lr=cfg['lr'])
        optimizer.load_state_dict(state['optimizer'])
        rng = np.random.default_rng()
        rng.bit_generator.state = state['numpy_rng']
        torch.set_rng_state(state['torch_rng'])
        update, episodes = state['updates'], state['episodes']
        progress = json.loads((args.out / 'progress.json').read_text())['progress']
    else:
        if (args.out / 'latest.pt').exists():
            parser.error('checkpoint exists: use --resume or choose another --out')
        torch.manual_seed(args.seed)
        brain = FlyBrain('tiny', JobsFly.obs_dim, len(JobsFly.actions), JobsFly.name)
        if brain.synthetic:
            raise ValueError('a real FlyWire graph is required')
        optimizer = torch.optim.Adam(brain.parameters(), lr=cfg['lr'])
        rng = np.random.default_rng(args.seed)
        update = episodes = 0
        save_brain(brain, args.out / 'untrained.pt', JobsFly)
    seeds = list(range(10000, 10000 + args.eval_episodes))
    with gzip.open(args.out / 'actions.jsonl.gz', 'at', encoding='utf-8') as log:
        if not progress:
            baseline = [evaluate(brain, p, seeds, 'untrained.pt', log) for p in ('random', 'untrained', 'lesioned')]
            progress.append({'updates': 0, 'training_episodes': 0, 'evaluations': baseline})
            report(args.out, progress)
            print('BASELINE ' + json.dumps([{k: v for k, v in m.items() if k != 'per_episode'} for m in baseline]), flush=True)
        deadline = time.time() + args.minutes * 60
        start_update = update
        while update < start_update + args.updates and time.time() < deadline:
            # Curriculum changes form complexity only; never supplies actions/answers.
            stage = update // args.curriculum_stage_updates
            difficulty = (.1, .5, .8, 1.)[min(stage, 3)]
            task, envs = env_factory(difficulty=difficulty)
            before_hash = digest(brain)
            rets, transitions = rollout(task, cfg['episodes'], rng, 'brain', brain, 'cpu')
            for index, env in enumerate(envs):
                for detail in env.traces:
                    write_event(log, {**detail, 'policy': 'fly_brain', 'checkpoint': str(args.out / 'latest.pt'),
                                      'parameter_hash': before_hash, 'revision': update,
                                      'training_episode': episodes + index, 'split': 'train'})
            reinforce(brain, optimizer, transitions, cfg)
            update += 1
            episodes += cfg['episodes']
            save_brain(brain, args.out / 'latest.pt', JobsFly)
            torch.save({'version': VERSION, 'model_hash': digest(brain), 'optimizer': optimizer.state_dict(),
                        'updates': update, 'episodes': episodes, 'numpy_rng': rng.bit_generator.state,
                        'torch_rng': torch.get_rng_state()}, args.out / 'optimizer.pt')
            print(f'update={update} episodes={episodes} training_return={rets.mean():.3f}', flush=True)
            if update % args.eval_every == 0 or update == start_update + args.updates or time.time() >= deadline:
                path = args.out / f'update_{update:05d}.pt'
                save_brain(brain, path, JobsFly)
                evaluations = [evaluate(brain, p, seeds, path, log) for p in ('trained', 'lesioned')]
                progress.append({'updates': update, 'training_episodes': episodes, 'evaluations': evaluations})
                report(args.out, progress)
                print('EVALUATION ' + json.dumps([{k: v for k, v in m.items() if k != 'per_episode'} for m in evaluations]), flush=True)
        # Untouched final test seeds: never used for curriculum or optimizer updates.
        test_seeds = list(range(20000, 20000 + args.eval_episodes))
        untrained = load_brain(args.out / 'untrained.pt', JobsFly)
        controls = {p: evaluate(untrained if p in ('random', 'untrained') else brain, p, test_seeds,
                                args.out / ('untrained.pt' if p in ('random', 'untrained') else 'latest.pt'), log)
                    for p in ('random', 'untrained', 'trained', 'lesioned')}
        comparisons = {p: comparison(controls['trained'], controls[p]) for p in ('random', 'untrained', 'lesioned')}
        passed = all(comparisons[p]['completed']['paired_95_percent_interval'][0] > 0 and
                     comparisons[p]['field_accuracy']['paired_95_percent_interval'][0] > 0 for p in comparisons)
        final = {'policies': controls, 'comparisons': comparisons, 'acceptance_passed': passed}
        report(args.out, progress, final)
        print('FINAL acceptance_passed=' + str(passed), flush=True)


if __name__ == '__main__':
    main()
