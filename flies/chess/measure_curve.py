"""Measure saved checkpoints against fixed initial opponents on validation only."""
import argparse
import json
from pathlib import Path
import torch
from flies.chess.learning import evaluate_models
from flies.chess.storage import load_policy, file_hash, write_json


def measure(corpus,run,updates=(0,10,20,30,40,50,60)):
    run=Path(run);rows=[]
    for update in updates:
        reports=[]
        for learner in range(2):
            candidate=load_policy(run/f'checkpoint_{update:04d}_{learner}.pt',size='tiny')
            opponent=load_policy(run/f'initial_{1-learner}.pt',size='tiny')
            report,_=evaluate_models(candidate,opponent,corpus['validation'],repeats=2,seed=700001)
            report['opponent_sha256']=file_hash(run/f'initial_{1-learner}.pt')
            reports.append(report)
        rows.append(dict(update=update,learners=reports))
        print(update,[r['first_move_mate']['mean'] for r in reports],flush=True)
    result=dict(corpus_sha256=corpus['sha256'],split='validation',checkpoints=rows)
    write_json(run/'fixed_opponent_curve.json',result)
    return result


def main():
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--corpus',required=True);ap.add_argument('--run',required=True)
    a=ap.parse_args();torch.set_num_threads(1)
    measure(json.loads(Path(a.corpus).read_text()),a.run)


if __name__=='__main__':main()
