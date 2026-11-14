"""Aggregate frozen mechanism experiments, with explicit negative-result gates."""
import argparse
import json
from pathlib import Path
import numpy as np
from flies.chess.mechanism_data import digest
from flies.chess.storage import ROOT, write_json
from flies.chess.study_report import two_way_interval


def interval(matrix, structural=False):
    x = np.asarray(matrix, float)
    result = two_way_interval(x, alpha=.0125, draws=5000)
    if structural:
        if not np.all(abs(x) < 1e-10):
            raise AssertionError('Matched-tree blind null violated')
        result['algebraic_null'] = True
    elif np.all(x == 0) or np.all(x == 1):
        result['interval'] = None
        result['note'] = 'Boundary bootstrap is not an uncertainty bound; no zero-width certainty claim'
    return result


def training_summary(run):
    history = run['history']; phases = {}
    for label, items in (('matched', history[:32]), ('mixed', history[32:])):
        tiers = {}
        for item in items:
            for tier, counts in item['outcomes'].items():
                current = tiers.setdefault(tier, {k: 0 for k in counts})
                for key, value in counts.items():current[key] += value
        phases[label] = dict(tiers=tiers,
                             total_neural_stages=sum(m['stages'] for h in items for m in h['metrics']),
                             informative_groups=sum(m['informative_groups'] for h in items for m in h['metrics'])/2,
                             mean_gradient_norms={k:float(np.mean([m['gradient_norms'][k] for h in items for m in h['metrics']]))
                                                  for k in ('log_w','bias','leak_logit','w_in')} if items else {})
    return dict(config=run['config'], phases=phases, updates=history[-1]['update'],
                fixed_structure_unchanged=run['fixed_structure_unchanged'], elapsed_seconds=history[-1]['elapsed'])


def collect(root):
    root = Path(root); read = lambda name: json.loads((root/name).read_text())
    return dict(manifest=json.loads((ROOT/'mechanism_manifest.json').read_text()),
                test=read('fresh_test.json'), corpus=read('curriculum.json'),
                seed_reports=[read(f'test-{s}/evaluation.json') for s in (11,23,37)],
                visits=[read(f'visits-{s}.json') for s in (11,23,37)],
                training={f'{method}-{s}':training_summary(read(f'{method}-{s}/history.json'))
                          for s in (11,23,37) for method in ('loo','ema','stage','zero')},
                reward_validation=read('reward_validation.json'), replication=read('replication_integrity.json'),
                pilot=read('pilot_summary.json'))


def aggregate(data):
    manifest = data['manifest']; reports = data['seed_reports']; test = data['test']
    if any(r['protocol_sha256'] != digest(manifest) or r['test_sha256'] != test['sha256'] for r in reports):
        raise ValueError('Evaluation identities do not match')
    conditions = ('parent','untrained','loo16','loo32','loo48','ema48','restricted48','entropy48','lesioned','random_legal')
    if any(set(r['reports']) != set(conditions) for r in reports):
        raise ValueError('Incomplete condition reports')
    if [r['seed'] for r in reports] != manifest['training_seeds']:
        raise ValueError('Wrong seed order')
    if len({r['reports']['loo48']['identity']['state_sha256'] for r in reports}) != len(reports):
        raise ValueError('Repeated final trained state is not an independent seed')
    def matrix(condition, kind):
        return np.asarray([[row[kind] for row in r['reports'][condition]['matched']['rows']] for r in reports])
    def wins(condition):
        return np.array([r['reports'][condition]['games']['per_position_win'] for r in reports])
    matched = {}; tactical = {}; outcomes = {}
    for condition in conditions:
        expected = [row['family'] for row in test['matched']]
        if any([row['family'] for row in r['reports'][condition]['matched']['rows']] != expected for r in reports):
            raise ValueError('Unpaired family order')
        matched[condition] = {key: interval(matrix(condition, key), structural=key=='adaptation' and condition in ('restricted48','lesioned','random_legal'))
                              for key in ('correct_mass','adaptation')}
        matched[condition]['per_seed_adaptation'] = matrix(condition, 'adaptation').mean(1).tolist()
        tactical[condition] = interval([r['reports'][condition]['mate2']['per_position_mass'] for r in reports])
        rows = [r['reports'][condition]['games'] for r in reports]
        counts = {k:sum(r[k] for r in rows) for k in ('wins','losses','draws','truncated','games','illegal_attempts')}
        counts['win_probability'] = interval(wins(condition))
        counts['terminations'] = {}
        for row in rows:
            for key, value in row['terminations'].items():counts['terminations'][key] = counts['terminations'].get(key,0)+value
        outcomes[condition] = counts
    changes = {label:interval(matrix(a,'adaptation')-matrix(b,'adaptation'))
               for label,a,b in [('loo16_parent','loo16','parent'),('loo32_parent','loo32','parent'),
                                ('loo48_parent','loo48','parent'),('loo48_loo32','loo48','loo32'),
                                ('loo48_ema48','loo48','ema48'),('loo48_entropy48','loo48','entropy48')]}
    win_change = interval(wins('loo48')-wins('parent'))
    tactical_change = interval(np.array([r['reports']['loo48']['mate2']['per_position_mass'] for r in reports])-
                               np.array([r['reports']['loo32']['mate2']['per_position_mass'] for r in reports]))
    primary = dict(adaptation=matched['loo48']['adaptation'], adaptation_gain=changes['loo48_parent'],
                   mate2_win_gain=win_change, mechanism_gain=changes['loo48_ema48'])
    def positive(stat):return stat['interval'] is not None and stat['interval'][0] > 0
    # New-visit audit is descriptive and never replaces the frozen primary test.
    visits = [set(v['visited_decision_families'][0]) for v in data['visits']]
    unseen = {}; overlaps = {}
    for tier, rows in (('matched', test['matched']), ('mate2', test['mate2'])):
        families = [r['family'] for r in rows]
        overlaps[tier] = [len(set(families) & v) for v in visits]
        keep = [i for i,f in enumerate(families) if all(f not in v for v in visits)]
        unseen[tier] = dict(families=len(keep), total=len(families))
        if keep:
            if tier=='matched':
                unseen[tier]['loo48_adaptation'] = interval(matrix('loo48','adaptation')[:,keep])
                unseen[tier]['gain_over_parent'] = interval((matrix('loo48','adaptation')-matrix('parent','adaptation'))[:,keep])
            else:unseen[tier]['win_gain_over_parent'] = interval((wins('loo48')-wins('parent'))[:,keep])
    gates = dict(board_adaptation=positive(primary['adaptation']) and primary['adaptation']['mean']>=.02 and min(matched['loo48']['per_seed_adaptation'])>0,
                 adaptation_learning=positive(primary['adaptation_gain']), mate2_progression=positive(win_change),
                 loo_superiority=positive(primary['mechanism_gain']), meaningful_full_games=False)
    gates['scale_justified'] = all(gates[k] for k in ('board_adaptation','adaptation_learning','mate2_progression','meaningful_full_games'))
    return dict(protocol_sha256=digest(manifest), test_sha256=test['sha256'], matched=matched,
                primary_contrasts=primary, adaptation_changes=changes, mate2_root_mass=tactical,
                mate2_mixed_phase_root_mass_change=tactical_change, actual_mate2_outcomes=outcomes,
                sampled_evaluation_games=sum(r['games'] for r in outcomes.values()),
                new_training_visit_overlap_by_seed=overlaps, new_training_unvisited_subset=unseen,
                gates=gates, training=data['training'], reward_validation=data['reward_validation'],
                replication=data['replication'], pilot=data['pilot'],
                limitations=['Three warm-start seeds; intervals are nominal bootstrap approximations.',
                             'Known curated-start families are excluded; all historical warm-start visits were not recorded.',
                             'Only one validation suffix; curated suffix reward opportunities are not an isolated causal intervention.',
                             'Short curated tactical games against fixed untrained opponents do not measure general chess strength.',
                             'Repeated replay games verify logging only and are not independent evidence.'])


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', default=str(ROOT/'artifacts/mechanism'))
    parser.add_argument('--evidence'); parser.add_argument('--archive'); parser.add_argument('--output', required=True)
    args = parser.parse_args()
    data = json.loads(Path(args.evidence).read_text()) if args.evidence else collect(args.root)
    if args.archive:write_json(args.archive, data)
    result = aggregate(data); write_json(args.output, result)
    print(json.dumps(dict(primary_contrasts=result['primary_contrasts'], gates=result['gates']),indent=2))
