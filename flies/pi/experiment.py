"""CPU-only repeatable experiment, using shared REINFORCE and fixed FlyBrain.

All generated files stay in flies/pi/artifacts. No teacher or replacement model.
"""
import argparse
import hashlib
import json
from pathlib import Path
import time

import numpy as np
import torch

from core.brain import FlyBrain, load_brain, save_brain
from core.train import settings, train_reward
from flies.pi.task import PiFly

HERE = Path(__file__).resolve().parent
ARTIFACTS = HERE / 'artifacts'


def scenario(start, end):
    return type('Scenario', (PiFly,), dict(start_index=start, prefix_length=end))


def evaluate(task_cls, brain=None, episodes=16, seed=123, lesion=False):
    """Batched sampled inference; random samples each hierarchical legal mask."""
    rng = np.random.default_rng(seed)
    torch.manual_seed(seed)
    envs = [task_cls() for _ in range(episodes)]
    obs = [e.reset(rng) for e in envs]
    done = [False] * episodes
    for _ in range(task_cls.max_steps):
        active = [i for i in range(episodes) if not done[i]]
        if not active:
            break
        masks = np.asarray([envs[i].mask() for i in active])
        if brain is None:
            actions = [int(rng.choice(np.flatnonzero(m))) for m in masks]
        else:
            with torch.no_grad():
                logits, _ = brain(torch.tensor(np.stack([obs[i] for i in active])),
                                  torch.tensor(masks), lesion=lesion)
                actions = torch.distributions.Categorical(logits=logits).sample().tolist()
        for i, action in zip(active, actions):
            obs[i], _, done[i], _ = envs[i].step(action, rng)
    rows = [dict(first_attempts=e.first_attempts, first_attempt_correct=e.first_attempt_correct,
                 incorrect_guesses=e.incorrect, completed_guesses=e.guesses,
                 accepted_digits=e.correct, verified_prefix_length=e.index - e.start_index,
                 final_absolute_index=e.index, best_streak=e.best_streak,
                 cumulative_reward=e.total_reward, terminated=e.terminated, truncated=e.truncated)
            for e in envs]
    totals = {k: sum(r[k] for r in rows) for k in rows[0]}
    accepted = totals['accepted_digits']
    return dict(episodes=episodes, seed=seed,
                first_attempt_accuracy=totals['first_attempt_correct'] / max(1, totals['first_attempts']),
                first_attempt_trials=totals['first_attempts'],
                incorrect_guesses=totals['incorrect_guesses'],
                attempts_per_accepted_digit=totals['completed_guesses'] / accepted if accepted else None,
                mean_verified_prefix_length=accepted / episodes,
                longest_correct_streak=max(e.best_streak for e in envs),
                mean_cumulative_reward=totals['cumulative_reward'] / episodes,
                rows=rows)


def run(args):
    torch.set_num_threads(1)
    torch.manual_seed(args.seed)
    ARTIFACTS.mkdir(exist_ok=True)
    train_task = scenario(0, args.prefix)
    checkpoint = ARTIFACTS / 'brain.pt'
    if args.evaluate_only:
        trained = load_brain(checkpoint, PiFly)
    else:
        trained = FlyBrain('tiny', PiFly.obs_dim, len(PiFly.actions), 'pi')
    if trained.synthetic:
        raise ValueError('Pi Fly experiments require real FlyWire data')
    untrained = FlyBrain('tiny', PiFly.obs_dim, len(PiFly.actions), 'pi')
    # Matched initialization is essential for the pre/post comparison.
    if not args.evaluate_only:
        untrained.load_state_dict(trained.state_dict())
    else:
        torch.manual_seed(args.seed)
        untrained = FlyBrain('tiny', PiFly.obs_dim, len(PiFly.actions), 'pi')
    before_buffers = {k: v.clone() for k, v in trained.named_buffers()}
    before_params = {k: v.clone() for k, v in trained.named_parameters()}
    report = dict(graph=dict(preset=trained.preset, hash=trained.graph_hash,
                             synthetic=trained.synthetic, neurons=trained.n,
                             edges=trained.edge_index.shape[1]),
                  seed=args.seed, training_interval=[0, args.prefix],
                  held_out_interval=[args.prefix, args.prefix * 2],
                  protocol='sampled two-stage masked inference; held-out starts with a known historical prefix',
                  controls={}, training_curve=[])
    log_lines = []

    def log(line):
        print(line, flush=True)
        log_lines.append(line)

    start = time.time()
    if not args.evaluate_only:
        cfg = settings(train_task, dict(updates=args.updates, episodes=args.episodes, chunk=64))
        report['settings'] = cfg

        def training_log(line):
            log(line)
            if line.startswith('update'):
                score = evaluate(train_task, trained, min(8, args.eval_episodes), 901)
                report['training_curve'].append(dict(elapsed_seconds=time.time() - start,
                    update=int(line.split()[1]), correct_predictions=sum(r['accepted_digits'] for r in score['rows']),
                    first_attempt_accuracy=score['first_attempt_accuracy'],
                    verified_prefix=score['mean_verified_prefix_length']))
        train_reward(trained, train_task, cfg, 'cpu', start + args.minutes * 60, training_log)
        if any(not torch.equal(v, before_buffers[k]) for k, v in trained.named_buffers()):
            raise AssertionError('fixed topology/signs/groups changed')
        report['parameters_changed'] = [k for k, v in trained.named_parameters()
                                        if not torch.equal(v, before_params[k])]
        save_brain(trained, checkpoint, PiFly)
        (ARTIFACTS / 'training.log').write_text('\n'.join(log_lines) + '\n')
    trained.eval(); untrained.eval()
    for label, cls in [('training_prefix', train_task), ('held_out', scenario(args.prefix, args.prefix * 2))]:
        report['controls'][label] = {}
        for name, brain, lesion in [('untrained', untrained, False), ('trained', trained, False),
                                    ('random', None, False), ('lesioned', trained, True)]:
            result = evaluate(cls, brain, args.eval_episodes, args.eval_seed, lesion)
            report['controls'][label][name] = result
            log(f"{label} {name}: first-attempt {result['first_attempt_accuracy']:.3f}; "
                f"prefix {result['mean_verified_prefix_length']:.2f}; reward {result['mean_cumulative_reward']:.2f}")
    report['checkpoint'] = dict(path='flies/pi/artifacts/brain.pt', sha256=hashlib.sha256(checkpoint.read_bytes()).hexdigest())
    report['elapsed_seconds'] = round(time.time() - start, 2)
    report['interpretation'] = ('Bounded CPU smoke experiment; first-attempt accuracy is the prediction metric. '
                                'Eventual acceptance includes retries. No claim of improvement or arbitrary unseen-digit prediction.')
    target = ARTIFACTS / ('evaluation.json' if args.evaluate_only else 'experiment.json')
    target.write_text(json.dumps(report, indent=2) + '\n')
    log(f'Saved {target.relative_to(HERE.parent.parent)}')


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--updates', type=int, default=3)
    p.add_argument('--episodes', type=int, default=4)
    p.add_argument('--eval-episodes', type=int, default=16)
    p.add_argument('--prefix', type=int, default=16)
    p.add_argument('--minutes', type=float, default=2)
    p.add_argument('--seed', type=int, default=7)
    p.add_argument('--eval-seed', type=int, default=123)
    p.add_argument('--evaluate-only', action='store_true')
    a = p.parse_args()
    if min(a.updates, a.episodes, a.eval_episodes, a.minutes) <= 0 or not 1 <= a.prefix <= 512:
        p.error('positive budgets and prefix 1..512 required')
    run(a)


if __name__ == '__main__':
    main()
