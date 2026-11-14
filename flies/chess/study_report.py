"""Aggregate fixed-checkpoint study results, accounting for position and seed clustering."""
import argparse
import json
from pathlib import Path
import numpy as np
from flies.chess.storage import write_json


def two_way_interval(matrix, alpha=.05, draws=10000, seed=2718):
    matrix=np.asarray(matrix,float)
    if matrix.ndim!=2 or not matrix.size:
        raise ValueError('Need training-seed by position observations')
    rng=np.random.default_rng(seed)
    estimates=[]
    for _ in range(draws):
        models=rng.integers(matrix.shape[0],size=matrix.shape[0])
        positions=rng.integers(matrix.shape[1],size=matrix.shape[1])
        estimates.append(matrix[np.ix_(models,positions)].mean())
    return dict(mean=float(matrix.mean()), interval=[float(x) for x in np.quantile(estimates,[alpha/2,1-alpha/2])],
                confidence=1-alpha, training_seeds=matrix.shape[0], position_families=matrix.shape[1],bootstrap_draws=draws)


def aggregate(paths):
    if len({str(Path(p).resolve()) for p in paths})!=len(paths):
        raise ValueError('Duplicate report paths are not independent training runs')
    data=[json.loads(Path(p).read_text()) for p in paths]
    if len({d['corpus_sha256'] for d in data})!=1 or any(d['split']!='test' for d in data):
        raise ValueError('Test reports must use identical held-out corpus')
    states=[tuple(d['reports'][str(i)]['trained']['identity']['state_sha256'] for i in range(2)) for d in data]
    if len(set(states))!=len(states):
        raise ValueError('Repeated trained states are not independent seeds')
    conditions=('trained','initial','random_legal','lesioned','shuffled_board')
    matrices={}
    for condition in conditions:
        matrices[condition]=np.array([np.mean([d['reports'][str(i)][condition]['position_mate_rates']
                                  for i in range(2)],axis=0) for d in data])
    differences={c:two_way_interval(matrices['trained']-matrices[c],alpha=.05/4)
                 for c in conditions if c!='trained'}
    rates={c:two_way_interval(matrices[c]) for c in conditions}
    return dict(rates=rates,simultaneous_differences=differences,
                per_seed={c:[float(x) for x in matrix.mean(1)] for c,matrix in matrices.items()},
                tactical_improvement_supported=all(differences[c]['interval'][0]>0 for c in ('initial','random_legal')),
                state_dependence_supported=all(differences[c]['interval'][0]>0 for c in ('lesioned','shuffled_board')),
                games_per_condition=sum(d['reports']['0']['trained']['games']+d['reports']['1']['trained']['games'] for d in data),
                corpus_sha256=data[0]['corpus_sha256'],input_reports=[str(p) for p in paths])


def main():
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument('reports',nargs='+');ap.add_argument('--output',required=True)
    args=ap.parse_args();result=aggregate(args.reports);write_json(args.output,result)
    print(json.dumps(result,indent=2))


if __name__=='__main__':main()
