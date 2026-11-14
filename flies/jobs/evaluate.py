"""Explicit paired held-out evaluation; never changes training/curriculum state."""
import argparse
import gzip
import json
from pathlib import Path
import torch
from core.brain import pick_device
from flies.jobs.learning import evaluate, comparison, report
from flies.jobs.policy import load_model, checkpoint_path
from flies.jobs.task import JobsFly
from flies.jobs.sim_state import atomic_json


def main():
    parser=argparse.ArgumentParser(__doc__)
    parser.add_argument('--checkpoint',type=Path,required=True,help='Experiment directory or immutable brain.pt')
    parser.add_argument('--out',type=Path,required=True)
    parser.add_argument('--device',default='auto')
    parser.add_argument('--episodes',type=int,default=64)
    parser.add_argument('--levels',type=int,nargs='+',default=[1,4,5,6,9,10])
    parser.add_argument('--split',choices=['validation','test'],default='test')
    parser.add_argument('--browser',action='store_true')
    parser.add_argument('--headed',action='store_true')
    args=parser.parse_args()
    if args.episodes<1 or any(not 1<=l<=10 for l in args.levels):
        parser.error('positive episodes and levels 1..10 required')
    torch.set_num_threads(2)
    device=pick_device(args.device)
    if device.type=='cuda' and not torch.cuda.is_available():parser.error('CUDA is unavailable')
    model_path=checkpoint_path(args.checkpoint)
    experiment=args.checkpoint if args.checkpoint.is_dir() else model_path.parent.parent.parent
    trained=load_model(model_path,JobsFly,device)
    untrained=load_model(experiment/'untrained.pt',JobsFly,device)
    args.out.mkdir(parents=True,exist_ok=True)
    browser=None;driver=None
    if args.browser:
        from playwright.sync_api import sync_playwright
        driver=sync_playwright().start()
        browser=driver.chromium.launch(headless=not args.headed)
    results={}
    try:
        with gzip.open(args.out/'actions.jsonl.gz','wt',encoding='utf-8') as log:
            for level in args.levels:
                controls={p:evaluate(None if p=='random' else untrained if p=='untrained' else trained,p,list(range(args.episodes)),level,
                                     args.split,log,experiment/'untrained.pt' if p=='untrained' else model_path,device=device,browser=browser) for p in ('random','untrained','trained','lesioned')}
                comparisons={p:comparison(controls['trained'],controls[p]) for p in ('random','untrained','lesioned')}
                passed=controls['trained']['completed']>=.25 and all(
                    comparisons[p]['completed']['paired_95_percent_interval'][0]>0 and
                    comparisons[p]['field_accuracy']['paired_95_percent_interval'][0]>0 for p in comparisons)
                results[str(level)]={'policies':controls,'comparisons':comparisons,'learning_passed':passed}
                print(json.dumps({'level':level,'learning_passed':passed,'policies':{p:{k:v for k,v in c.items() if k!='per_episode'} for p,c in controls.items()}}),flush=True)
    finally:
        if browser: browser.close()
        if driver: driver.stop()
    full_passed=all(str(level) in results and results[str(level)]['learning_passed'] for level in (4,5,6,9,10))
    atomic_json(args.out/'evaluation.json',{'split':args.split,'local_browser':args.browser,'results':results,'full_workflow_learning_passed':full_passed})
    rows=[{'updates':0,'training_episodes':0,'evaluations':list(r['policies'].values())} for r in results.values()]
    report(args.out,rows)
    (args.out/'REPORT.md').write_text((args.out/'REPORT.md').read_text()+f'\nFull-workflow learning criteria passed: **{full_passed}**.\n')


if __name__=='__main__':
    main()
