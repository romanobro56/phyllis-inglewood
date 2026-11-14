"""Reward-only backward diagnostics for the outcome-only mechanism curriculum."""
import argparse
import json
from pathlib import Path
import torch
from flies.chess.batched import play_batch
from flies.chess.learning import specs_for
from flies.chess.research_train import own_turn_rewards
from flies.chess.storage import ROOT, load_policy, state_hash, write_json
from flies.chess.train_selfplay import policy_update


def gradient_cosines(gradients):
    # Float32 dot/norm reductions on long gradients gave self-cosines >1.
    # Accumulate this diagnostic in float64; never change training tensors.
    vectors = {k:v.double() for k,v in gradients.items()}
    result = {}
    for a, x in vectors.items():
        for b, y in vectors.items():
            denom = float(x.norm()*y.norm())
            result[a+'/'+b] = max(-1.,min(1.,float(x.dot(y))/denom)) if denom else None
    return result


def run(corpus, checkpoint, opponent_path, output):
    torch.set_num_threads(1)
    model, opponent = load_policy(checkpoint), load_policy(opponent_path)
    before = [state_hash(m) for m in (model, opponent)]; reports = {}; gradients = {}
    matched = [dict(fen=fen,family=row['family']) for row in corpus['matched_validation'] for fen in row['positions']]
    for tier, rows, repeats, cap in [('matched',matched,2,2),('suffix',corpus['suffix_validation'],8,2),
                                     ('mate2',corpus['mate2_validation'],8,4)]:
        games, traces = play_batch([model,opponent], specs_for(rows,repeats,seed=610001), max_plies=cap)
        samples = [(trace[g['candidate']], own_turn_rewards(g,g['candidate'],False,'integer-v2'))
                   for g,trace in zip(games,traces)]
        metrics, _ = policy_update(model, torch.optim.SGD(model.parameters(),lr=0.),samples,0.,gamma=1.,entropy=0.)
        gradients[tier] = torch.cat([(p.grad.detach() if p.grad is not None else torch.zeros_like(p)).flatten().cpu()
                                     for p in model.parameters()])
        reports[tier] = dict(games=len(games), starting_positions=len(rows), families=len({r['family'] for r in rows}),
                             wins=sum(g['outcome']['winner']==g['candidate'] for g in games),
                             metrics=metrics, parameters_unchanged=[state_hash(m) for m in (model,opponent)]==before)
        assert reports[tier]['parameters_unchanged']
    cosines = gradient_cosines(gradients)
    result = dict(checkpoint=str(checkpoint), reports=reports, reward_gradient_cosines=cosines,
                  note='Existing validation only; terminal outcomes only; zero baseline, entropy0, SGD lr0. Neither model updated. One checkpoint/seed and few reward-bearing games; cosines are exploratory, not evidence of causal interference.')
    write_json(output,result); print(json.dumps(result,indent=2)); return result


if __name__ == '__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--corpus',default=str(ROOT/'artifacts/mechanism/curriculum.json'))
    p.add_argument('--checkpoint',required=True);p.add_argument('--opponent',required=True);p.add_argument('--output',required=True)
    a=p.parse_args();run(json.loads(Path(a.corpus).read_text()),a.checkpoint,a.opponent,a.output)
