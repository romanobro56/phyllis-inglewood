"""Sequential non-submitting practice with explicit untrained/trained/lesioned policies."""
import argparse
from contextlib import ExitStack
import json
from pathlib import Path
import numpy as np
from flies.jobs.legacy_task import JobsFly
from flies.jobs.history import History
from flies.jobs.errors import BLOCKING, RETRYABLE


def main():
    parser = argparse.ArgumentParser(__doc__)
    parser.add_argument('--policy', choices=['untrained', 'brain', 'random', 'lesioned'], default='untrained')
    parser.add_argument('--checkpoint', type=Path, default=Path('flies/jobs/artifacts/experiment/latest.pt'))
    parser.add_argument('--episodes', type=int, default=5, help='maximum decision chunks; unfinished jobs stay first')
    parser.add_argument('--browser', action='store_true')
    parser.add_argument('--headed', action='store_true')
    parser.add_argument('--status', action='store_true')
    parser.add_argument('--details', action='store_true')
    parser.add_argument('--verbose', action='store_true')
    parser.add_argument('--history', type=Path, default=Path('flies/jobs/practice.sqlite3'))
    parser.add_argument('--legacy-history', action='store_true', help='read earlier version-1 practice records only')
    parser.add_argument('--fresh-evaluation', action='store_true')
    parser.add_argument('--report-error', choices=sorted(BLOCKING | RETRYABLE))
    args = parser.parse_args()
    if args.episodes < 1:
        parser.error('episodes must be positive')
    if args.legacy_history and not (args.status or args.details):
        parser.error('legacy history is read-only; use --status or --details')
    source = JobsFly()
    with ExitStack() as stack:
        history = History(':memory:' if args.fresh_evaluation else args.history, source.profile['facts']['email'],
                          mode='practice' if args.legacy_history else 'practice_v6')
        stack.callback(history.close)
        if args.status or args.details:
            print(json.dumps(history.details() if args.details else history.records(), indent=2))
            return
        rng = np.random.default_rng(41)
        brain = None
        checkpoint = 'none (uniform random)'
        if args.policy != 'random':
            import torch
            from core.brain import FlyBrain, load_brain
            torch.set_num_threads(2)
            if args.policy == 'untrained':
                torch.manual_seed(2026)
                brain = FlyBrain('tiny', JobsFly.obs_dim, len(JobsFly.actions), JobsFly.name)
                checkpoint = 'fresh initialization, seed 2026'
            else:
                from flies.jobs.forms import VERSION
                manifest = json.loads((args.checkpoint.parent / 'progress.json').read_text())
                if manifest['environment_version'] != VERSION:
                    raise ValueError('checkpoint uses an incompatible observation schema; retrain')
                brain = load_brain(args.checkpoint, JobsFly)
                checkpoint = str(args.checkpoint)
            brain.eval()
        mirror = None
        if args.browser:
            from playwright.sync_api import sync_playwright
            from flies.jobs.legacy_browser import BrowserMirror
            driver = stack.enter_context(sync_playwright())
            browser = driver.chromium.launch(headless=not args.headed)
            stack.callback(browser.close)
            mirror = BrowserMirror(browser.new_page())
        print('LOCAL SIMULATION ONLY. No employer applications will be submitted.', flush=True)
        for _ in range(args.episodes):
            job = history.next_job(source.jobs)
            if job is None:
                print('Queue exhausted.')
                break
            env = JobsFly(jobs=[job], split='test', trace=True)
            env.reset(rng)
            history.restore(env)
            if args.report_error:
                env.report_error(args.report_error)
                args.report_error = None
                history.save(env, event={'event': 'adapter_error', 'error': env.last_error,
                                         'policy': args.policy, 'checkpoint': checkpoint})
                if env.blocking_error:
                    print(f'[blocked] {job["url"]}: {env.blocking_error}')
                    continue
            history.save(env)
            if mirror:
                mirror.open(env)
            for _ in range(env.max_steps):
                obs = mirror.observation(env) if mirror else env.obs()
                mask = np.asarray(env.mask())
                if brain is None:
                    probabilities = mask / mask.sum()
                else:
                    with torch.no_grad():
                        logits, _ = brain(torch.tensor(obs[None]), torch.tensor(mask[None]), lesion=args.policy == 'lesioned')
                        probabilities = logits.softmax(-1)[0].numpy().astype(float)
                        probabilities /= probabilities.sum()
                action = int(rng.choice(8, p=probabilities))
                try:
                    if mirror:
                        mirror.execute(env, action)
                    _, _, done, info = env.step(action, rng)
                    if mirror:
                        actual = mirror.check(env)
                        if env.completed and not env._validator.accepted(env.fields, actual):
                            raise RuntimeError('Independent DOM validation failed')
                except Exception:
                    env.completed = 0
                    env.report_error('browser_error')
                    history.save(env, event={'event': 'execution_error', 'policy': args.policy,
                                             'checkpoint': checkpoint, 'observation': obs.tolist(), 'action': action})
                    raise
                detail = {**env.traces[-1], 'policy': args.policy, 'checkpoint': checkpoint,
                          'url': job['url'], 'mode': 'local_non_submitting'}
                history.save(env, event=detail)
                if args.verbose:
                    print(json.dumps(detail), flush=True)
                if done:
                    print(json.dumps({'url': job['url'], **env.metrics(), 'paused': info['paused'],
                                      'policy': args.policy}), flush=True)
                    break


if __name__ == '__main__':
    main()
