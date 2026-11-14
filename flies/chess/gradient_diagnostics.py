"""Reward-only gradient checks on harder actual FlyBrain trajectories, no learning."""
import argparse
import json
from pathlib import Path
import numpy as np
import torch
from flies.chess.storage import ROOT,load_policy,state_hash,write_json
from flies.chess.batched import play_batch
from flies.chess.learning import specs_for
from flies.chess.research_train import own_turn_rewards
from flies.chess.train_selfplay import policy_update


def run(extended,checkpoint,opponent_path,output):
    torch.set_num_threads(1);brain=load_policy(checkpoint);opponent=load_policy(opponent_path)
    before=state_hash(brain);other=state_hash(opponent);reports={}
    for tier,cap in [('mate2',12),('endgame',24),('complex',16)]:
        rows=[r for r in extended['validation'] if r['tier']==tier]
        games,traces=play_batch([brain,opponent],specs_for(rows,2,seed=310001),cap)
        samples=[(trace[g['candidate']],own_turn_rewards(g,g['candidate'],tier!='mate2')) for g,trace in zip(games,traces)]
        metrics,_=policy_update(brain,torch.optim.SGD(brain.parameters(),lr=0.),samples,0.,gamma=.98,entropy=0.)
        assert before==state_hash(brain) and other==state_hash(opponent)
        reports[tier]=dict(metrics=metrics,nonzero_move_rewards=sum(sum(abs(r)>1e-12 for r in rewards) for _,rewards in samples),
                           parameters_unchanged=True,opponent_unchanged=True)
    write_json(output,dict(reports=reports,checkpoint=str(checkpoint),note='Entropy disabled, SGD lr=0; real observed rewards, no parameter updates'))
    print(json.dumps(reports,indent=2))
    return reports


if __name__=='__main__':
    ap=argparse.ArgumentParser(description=__doc__);ap.add_argument('--extended',default=str(ROOT/'artifacts/research/extended.json'))
    ap.add_argument('--checkpoint',required=True);ap.add_argument('--opponent',required=True);ap.add_argument('--output',required=True)
    a=ap.parse_args();run(json.load(open(a.extended)),a.checkpoint,a.opponent,a.output)
