"""Paired unseen local DOM fixtures. Never navigates to or submits to employers."""
import argparse
import gzip
import json
from pathlib import Path
import torch
from core.brain import load_brain
from flies.jobs.forms import VERSION
from flies.jobs.task import JobsFly
from flies.jobs.learning import evaluate, comparison


def main():
    parser = argparse.ArgumentParser(__doc__)
    parser.add_argument('--out', type=Path, required=True)
    parser.add_argument('--episodes', type=int, default=16)
    args = parser.parse_args()
    if args.episodes < 1:
        parser.error('episodes must be positive')
    manifest = json.loads((args.out / 'progress.json').read_text())
    if manifest['environment_version'] != VERSION:
        raise ValueError('incompatible checkpoint observation schema')
    from playwright.sync_api import sync_playwright
    torch.set_num_threads(2)
    trained = load_brain(args.out / 'latest.pt', JobsFly)
    untrained = load_brain(args.out / 'untrained.pt', JobsFly)
    seeds = list(range(30000, 30000 + args.episodes))
    with sync_playwright() as driver, gzip.open(args.out / 'browser_actions.jsonl.gz', 'wt') as log:
        browser = driver.chromium.launch(headless=True)
        try:
            results = {p: evaluate(untrained if p in ('random', 'untrained') else trained,
                                   p, seeds, args.out / ('untrained.pt' if p in ('random', 'untrained') else 'latest.pt'),
                                   log, browser=browser)
                       for p in ('random', 'untrained', 'trained', 'lesioned')}
        finally:
            browser.close()
    payload = {'mode': 'local_DOM_non_submitting', 'policies': results,
               'comparisons': {p: comparison(results['trained'], results[p])
                               for p in ('random', 'untrained', 'lesioned')}}
    (args.out / 'browser_report.json').write_text(json.dumps(payload, indent=2))
    print(json.dumps({p: {k: v for k, v in r.items() if k != 'per_episode'} for p, r in results.items()}, indent=2))


if __name__ == '__main__':
    main()
