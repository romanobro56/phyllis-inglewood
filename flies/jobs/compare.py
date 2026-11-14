"""Paired held-out comparison of several checkpoints and controls on identical simulated forms.

Every policy sees the same generated applications (same split, level and seeds) and the same
action random variates, so per-episode differences are paired. Actions come only from
flies.jobs.learning.evaluate -> FlyBrain probabilities (or uniform random for the random control).
Never changes training, curriculum or checkpoint state.

    python -m flies.jobs.compare --out runs/jobs/eval_x --device cuda --episodes 256 --levels 1 \
        --checkpoint original=runs/jobs/workday_full --checkpoint phase1=runs/jobs/workday_v2_p1
"""
import argparse
import collections
import gzip
import json
from pathlib import Path
import time
import torch
from core.brain import pick_device
from flies.jobs.learning import evaluate, comparison
from flies.jobs.policy import load_model, checkpoint_path, digest
from flies.jobs.task import JobsFly
from flies.jobs.sim_state import atomic_json

FAILED_EVENTS = ('invalid_action', 'field_incorrect', 'validation_error', 'manual_review_unnecessary',
                 'submission_rejected', 'upload_rejected', 'duplicate_submission')


class EventLog:
    """File-like sink for evaluate(): keeps compact per-step records and per-policy event counts."""
    def __init__(self, handle):
        self.handle, self.label = handle, ''
        self.events = collections.defaultdict(collections.Counter)

    def write(self, line):
        e = json.loads(line)
        self.events[self.label][e['event']] += 1
        a = e.get('selected_action') or {}
        self.handle.write(json.dumps({'label': self.label, 'seed': e['seed'], 'level': e['level'], 'split': e['split'],
                                      'stage': e['public_observation']['stage'], 'mode': e['public_observation']['mode'],
                                      'action': {k: a.get(k) for k in ('kind', 'label')}, 'event': e['event'],
                                      'reward': e['reward'], 'status': e['status'],
                                      'parameter_hash': e['parameter_hash']}) + '\n')


def summary(result, events, episodes):
    keep = ('reward', 'completed', 'task_success', 'field_accuracy', 'error_rate', 'validation_error_rate', 'actions', 'failed_application')
    s = {k: result[k] for k in keep}
    s['successful_applications'] = int(round(result['completed'] * episodes))
    s['failed_actions_per_episode'] = sum(events[k] for k in FAILED_EVENTS) / episodes
    s['events_per_episode'] = {k: v / episodes for k, v in events.most_common()}
    done = [e for e in result['per_episode'] if e['completed']]
    s['mean_actions_when_completed'] = sum(e['actions'] for e in done) / len(done) if done else None
    s['parameter_hash'] = result['parameter_hash']
    return s


def main():
    parser = argparse.ArgumentParser(__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--checkpoint', action='append', required=True, help='name=experiment_dir_or_brain.pt (repeatable)')
    parser.add_argument('--out', type=Path, required=True)
    parser.add_argument('--device', default='auto')
    parser.add_argument('--episodes', type=int, default=256)
    parser.add_argument('--levels', type=int, nargs='+', default=[1])
    parser.add_argument('--split', choices=['validation', 'test'], default='test')
    args = parser.parse_args()
    if args.episodes < 2 or any(not 1 <= l <= 10 for l in args.levels):
        parser.error('episodes >= 2 and levels 1..10 required')
    if args.out.exists():
        parser.error('output exists; choose a fresh --out')
    torch.set_num_threads(2)
    device = pick_device(args.device)
    if device.type == 'cuda' and not torch.cuda.is_available():
        parser.error('CUDA is unavailable')
    trained = {}
    for spec in args.checkpoint:
        name, _, path = spec.partition('=')
        path = Path(path)
        model = checkpoint_path(path)
        trained[name] = (model, load_model(path, JobsFly, device))
    first_dir = Path(args.checkpoint[0].partition('=')[2])
    first_dir = first_dir if first_dir.is_dir() else trained[next(iter(trained))][0].parent.parent.parent
    untrained = load_model(first_dir / 'untrained.pt', JobsFly, device)
    untrained_hashes = {}
    for name, (model, _) in trained.items():
        experiment = model.parent.parent.parent
        if (experiment / 'untrained.pt').exists():
            untrained_hashes[name] = digest(load_model(experiment / 'untrained.pt', JobsFly, 'cpu'))
    args.out.mkdir(parents=True)
    seeds = list(range(args.episodes))
    results, started = {}, time.time()
    with gzip.open(args.out / 'actions.jsonl.gz', 'wt', encoding='utf-8') as handle:
        log = EventLog(handle)
        for level in args.levels:
            runs = {}
            plan = [('random', None, 'random', 'none'), ('untrained', untrained, 'untrained', str(first_dir / 'untrained.pt'))]
            for name, (model, brain) in trained.items():
                plan += [(name, brain, 'trained', str(model)), (name + ':lesioned', brain, 'lesioned', str(model))]
            for label, brain, policy, source in plan:
                log.label = f'L{level}:{label}'
                runs[label] = evaluate(brain, policy, seeds, level, args.split, log=log, checkpoint=source, device=device)
                print(f'level {level} {label:22s} completed {runs[label]["completed"]:.3f} field_accuracy '
                      f'{runs[label]["field_accuracy"]:.3f} reward {runs[label]["reward"]:.2f}', flush=True)
            hashes = {r['forms_hash'] for r in runs.values()}
            assert len(hashes) == 1, 'policies did not see identical forms'
            pairs = {}
            for name in trained:
                for other in ['random', 'untrained', name + ':lesioned'] + [n for n in trained if n != name]:
                    pairs[f'{name} - {other}'] = comparison(runs[name], runs[other])
            results[str(level)] = {'forms_hash': hashes.pop(),
                                   'policies': {k: summary(v, log.events[f'L{level}:{k}'], args.episodes) for k, v in runs.items()},
                                   'paired_differences': pairs}
    gpu = torch.cuda.max_memory_allocated(device) if device.type == 'cuda' else None
    atomic_json(args.out / 'comparison.json', {'split': args.split, 'episodes_per_policy': args.episodes, 'levels': args.levels,
                                               'checkpoints': {n: str(m) for n, (m, _) in trained.items()},
                                               'untrained_parameter_hashes': untrained_hashes,
                                               'seconds': time.time() - started, 'peak_cuda_memory_bytes': gpu,
                                               'results': results}, private=False)
    print(json.dumps({'status': 'done', 'seconds': round(time.time() - started, 1), 'peak_cuda_memory_bytes': gpu}), flush=True)


if __name__ == '__main__':
    main()
