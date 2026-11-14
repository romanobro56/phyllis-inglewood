"""Aggregate counterfactual, progression and actual-outcome evidence without Elo."""
import argparse
import json
from pathlib import Path
import numpy as np
from flies.chess.storage import ROOT,write_json
from flies.chess.study_report import two_way_interval
from flies.chess.generalization import bounded_outcome_interval


def summary(matrix,alpha=.0125):
    x=np.asarray(matrix,float);out=two_way_interval(x,alpha=alpha,draws=5000)
    if np.all(x==0) or np.all(x==1):
        out['interval']=None
        out['note']='Boundary/degenerate bootstrap: no uncertainty claim about unobserved outcomes. See per-run conservative position bounds.'
    return out


def matched_matrix(data,condition='intact',metric='adaptation'):
    return np.array([[r[metric] for r in report['conditions'][condition]['rows']] for report in data['reports']])


def run(root,evidence=None):
    root=Path(root)
    archived=json.loads(Path(evidence).read_text()) if evidence else None
    def read(name):
        if archived is None:return json.loads((root/name).read_text())
        if name.endswith('_probe.json'):return archived['probes'][name[:-11]]
        if name=='matched.json':return json.loads((ROOT/'research_data/matched.json').read_text())
        if name.startswith('tactics-'):return archived['tactics'][name[8:-5]]
        if name.startswith('generalization-'):
            # The compact archive groups every condition consistently, including
            # the single trained opening-start condition.
            data=archived['games'][name.split('/')[0][15:]]
            data['reports']['opening_start']=data['reports']['opening_start']['trained']
            return data
        raise ValueError('Unsupported archived evidence key: '+name)
    labels=('untrained','pretrained','matched40','matched56','restricted40','restricted56')
    probes={name:read(name+'_probe.json') for name in labels}
    expected=probes['pretrained']['corpus_sha256']
    if any(d['corpus_sha256']!=expected for d in probes.values()):raise ValueError('Probe corpora differ')
    matched={}
    for name,d in probes.items():
        matched[name]={}
        for metric in ('correct_mass','adaptation','total_variation','both_argmax_correct'):
            x=matched_matrix(d,metric=metric)
            # The zero adaptation of blinded conditions is a known algebraic
            # identity, not a zero-event statistical estimate.
            stats=two_way_interval(x,alpha=.0125,draws=5000)
            if metric=='both_argmax_correct' and np.all(x==0) and not name.startswith('restricted'):
                stats['interval']=None;stats['note']='No successes observed; zero-width bootstrap is not an uncertainty bound'
            matched[name][metric]=stats
        matched[name]['per_seed_adaptation']=matched_matrix(d).mean(1).tolist()
    corpus=read('matched.json')
    chance=[.5*sum(len(a) for a in row['answers'])/len(row['legal_moves']) for row in corpus['test']]
    analytical=two_way_interval([chance],alpha=.0125,draws=5000)
    analytical.pop('training_seeds');analytical['note']='Analytic uniform-legal policy, uncertainty only across position families'
    matched['random_legal']=dict(correct_mass=analytical,adaptation=dict(mean=0.,interval=[0.,0.],note='Matched-tree algebraic null'))
    changes={label:two_way_interval(matched_matrix(probes[a])-matched_matrix(probes[b]),alpha=.0125,draws=5000)
             for label,a,b in [('matched_training','matched40','pretrained'),('extended_training','matched56','matched40'),
                                ('all_training','matched56','pretrained')]}
    mass_changes={label:two_way_interval(matched_matrix(probes[a],metric='correct_mass')-
                                         matched_matrix(probes[b],metric='correct_mass'),alpha=.0125,draws=5000)
                  for label,a,b in [('matched_training','matched40','pretrained'),('extended_training','matched56','matched40'),
                                   ('all_training','matched56','pretrained')]}
    restricted_contrast={metric:two_way_interval(matched_matrix(probes['matched56'],metric=metric)-
                                                matched_matrix(probes['restricted56'],metric=metric),alpha=.0125,draws=5000)
                         for metric in ('correct_mass','adaptation')}
    decomposition={condition:{metric:two_way_interval(matched_matrix(probes['matched56'],condition,metric),alpha=.0125,draws=5000)
                              for metric in ('correct_mass','adaptation')}
                   for condition in probes['matched56']['reports'][0]['conditions']}
    comparisons={condition:two_way_interval(matched_matrix(probes['matched56'],metric='correct_mass')-
                     matched_matrix(probes['matched56'],condition,metric='correct_mass'),alpha=.0125,draws=5000)
                 for condition in ('paired_swap','stage_only','own_coordinates','lesioned')}
    games=[read(f'generalization-{seed}/generalization.json') for seed in (11,23,37)]
    outcomes={}
    for tier in ('mate2','endgame','complex','long_endgames'):
        outcomes[tier]={}
        for condition in games[0]['reports'][tier]:
            reports=[g['reports'][tier][condition] for g in games]
            totals={k:sum(r[k] for r in reports) for k in ('wins','losses','draws','truncated','games','illegal_attempts')}
            totals['win_within_cap']=summary([r['per_position_win'] for r in reports])
            totals['mean_auxiliary_return']=float(np.mean([r['auxiliary_return'] for r in reports]))
            totals['auxiliary_used_in_training']=tier in ('endgame','complex')
            totals['terminations']={}
            for r in reports:
                for termination,count in r['terminations'].items():totals['terminations'][termination]=totals['terminations'].get(termination,0)+count
            outcomes[tier][condition]=totals
    tactics=[read(f'tactics-{seed}.json')['reports'] for seed in (11,23,37)]
    tactical={condition:summary([d[condition]['per_position_mass'] for d in tactics]) for condition in tactics[0]}
    tactical_changes={label:summary(np.array([d[a]['per_position_mass'] for d in tactics])-
                                   np.array([d[b]['per_position_mass'] for d in tactics]))
                      for label,a,b in [('after_matched','matched40','parent_pretrained'),('after_extended','trained','matched40')]}
    opening={k:sum(g['reports']['opening_start'][k] for g in games) for k in ('wins','losses','draws','truncated','games','illegal_attempts')}
    # Correct any earlier single-position/zero-event bootstrap presentation;
    # counts, trajectories, probabilities and experimental checkpoints are unchanged.
    for seed,g in zip((11,23,37),games):
        for tier,conditions in g['reports'].items():
            rows=[conditions] if tier=='opening_start' else list(conditions.values())
            for r in rows:
                if 'per_position_win' in r:r['win_within_cap']=bounded_outcome_interval(r['per_position_win'])
        if archived is None:write_json(root/f'generalization-{seed}/generalization.json',g)
    adaptation=matched['matched56']['adaptation'];per_seed=matched['matched56']['per_seed_adaptation']
    multi=tactical_changes['after_extended']
    gates=dict(board_adaptation=adaptation['interval'][0]>0 and adaptation['mean']>=.02 and min(per_seed)>0,
               improves_over_parent=changes['all_training']['interval'][0]>0,
               heldout_mate2_improvement=multi['interval'] is not None and multi['interval'][0]>0,
               playing_strength_evaluation_ready=opening['games']>=200 and opening['wins']+opening['losses']>=100,
               recommendation='Do not scale to the shared GPU until counterfactual adaptation and multi-ply outcome gates pass.')
    result=dict(matched=matched,matched_changes=changes,matched_correct_mass_changes=mass_changes,
                independently_trained_restriction_contrast=restricted_contrast,
                final_decomposition=decomposition,control_contrasts=comparisons,
                mate2_tactical=tactical,mate2_changes=tactical_changes,outcomes=outcomes,opening_start=opening,gates=gates,
                matched_sha256=expected,extended_sha256=games[0]['corpus_sha256'],
                uncertainty='Nominal 98.75% two-way bootstrap by three seeds and held-out family; zero-event/single-opening uncertainty is not treated as exact.')
    write_json(root/'summary.json',result)
    return result


if __name__=='__main__':
    ap=argparse.ArgumentParser(description=__doc__);ap.add_argument('--root',default=str(ROOT/'artifacts/research'))
    ap.add_argument('--evidence',help='Reanalyze saved per-family evidence without checkpoints or original artifact reports')
    a=ap.parse_args();r=run(a.root,a.evidence);print(json.dumps({k:r[k] for k in ('matched','matched_changes','gates')},indent=2))
