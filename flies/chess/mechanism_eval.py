"""Validation-only pilots and frozen fresh-family learning-mechanism evaluation."""
import argparse
import json
from pathlib import Path

import chess
import numpy as np
import torch
from flies.chess.extended_curriculum import make_extended
from flies.chess.generalization import evaluate, tactical
from flies.chess.matched_eval import evaluate_pairs, swapped_report
from flies.chess.mechanism_data import digest
from flies.chess.research_positions import make_matched
from flies.chess.storage import ROOT, load_policy, identity, write_json


def policy_identity(condition, model, path=None, lesion=False):
    if condition == 'random_legal':
        return dict(label='RANDOM-LEGAL', brain_used_for_candidate=False)
    result = identity(model, path, lesion)
    labels = dict(untrained='UNTRAINED', parent='FROZEN-PARENT', frozen_opponent='FROZEN-UNTRAINED',
                  restricted48='TRAINED-RESTRICTED', entropy48='ENTROPY-ONLY')
    if condition in labels:result['label'] = labels[condition]
    return result


def validation(corpus, output):
    torch.set_num_threads(1); root = ROOT/'artifacts/mechanism'; reports = {}
    paths = {'parent': root/'replication-11/checkpoint_0056_0.pt'}
    for method in ('ema', 'loo', 'zero'):
        for update in (8, 16):
            paths[f'{method}{update}'] = root/f'pilot-{method}/checkpoint_{update:04d}_0.pt'
    for label, path in paths.items():
        model = load_policy(path)
        reports[label] = dict(identity=identity(model, path),
                              matched=evaluate_pairs(model, corpus['matched_validation']),
                              mate2=tactical(model, corpus['mate2_validation']))
        write_json(output, dict(split='existing-validation-only', reports=reports))
        print(label, 'adaptation', reports[label]['matched']['mean_adaptation'],
              'correct', reports[label]['matched']['mean_correct_mass'], flush=True)
    return reports


def generate_fresh(corpus, manifest, output):
    if not manifest.get('frozen') or manifest['training_corpus_sha256'] != corpus['sha256']:
        raise ValueError('Freeze a matching protocol before generating fresh test data')
    if Path(output).exists():
        raise ValueError('Fresh test already exists; do not silently replace a locked evaluation')
    exclude = corpus['forbidden_evaluation_families']
    matched = make_matched(seed=manifest['test_matched_seed'], train=0, validation=0,
                           test=manifest['test_pairs'], exclude=exclude)
    extended = make_extended(matched, seed=manifest['test_mate2_seed'], train=0, validation=0,
                             test=manifest['test_mate2_positions'], exclude=exclude)
    rows = [r for r in extended['test'] if r['tier'] == 'mate2']
    families = [r['family'] for r in matched['test']] + [r['family'] for r in rows]
    assert len(set(families)) == len(families) and set(families).isdisjoint(exclude)
    result = dict(version='mechanism-fresh-test-v1', protocol_sha256=digest(manifest),
                  matched=matched['test'], mate2=rows, generator_hashes=[matched['sha256'], extended['sha256']],
                  forbidden_family_overlap=0)
    result['sha256'] = digest(result)
    write_json(output, result)
    print(result['sha256'], len(result['matched']), len(result['mate2']), flush=True)
    return result


def random_matched(pairs):
    rows = [dict(family=r['family'], correct_mass=.5*sum(map(len, r['answers']))/len(r['legal_moves']),
                 adaptation=0., cross_target_mass=.5*sum(map(len, r['answers']))/len(r['legal_moves'])) for r in pairs]
    return dict(mean_correct_mass=float(np.mean([r['correct_mass'] for r in rows])), mean_adaptation=0., rows=rows)


def test_seed(corpus, manifest, test, seed, output):
    torch.set_num_threads(1)
    if test['protocol_sha256'] != digest(manifest) or seed not in manifest['training_seeds']:
        raise ValueError('Wrong frozen evaluation protocol')
    root = ROOT/'artifacts/mechanism'; parent = root/f'replication-{seed}/checkpoint_0056_0.pt'
    paths = dict(parent=parent, untrained=ROOT/f'artifacts/learning/spatial-{seed}/initial_0.pt')
    for condition, method, update in [('loo16','loo',16),('loo32','loo',32),('loo48','loo',48),
                                     ('ema48','ema',48),('restricted48','stage',48),('entropy48','zero',48)]:
        run = root/f'{method}-{seed}'
        config = json.loads((run/'history.json').read_text())['config']
        expected_method = 'loo' if method == 'stage' else method
        if (config['method'] != expected_method or config['corpus_sha256'] != corpus['sha256']
                or config['games'] != manifest['games_per_update'] or config['matched_updates'] != 32
                or config['easy_updates'] != 8 or config['lr'] != .003 or config['entropy'] != .0005):
            raise ValueError('Checkpoint run does not match the frozen design')
        paths[condition] = run/f'checkpoint_{update:04d}_0.pt'
    paths['lesioned'] = paths['loo48']; paths['random_legal'] = paths['untrained']
    opponent_path = ROOT/f'artifacts/learning/spatial-{seed}/initial_1.pt'
    opponent = load_policy(opponent_path); opponent_id = policy_identity('frozen_opponent', opponent, opponent_path)
    reports = {}
    for condition, path in paths.items():
        model = load_policy(path); lesion = condition == 'lesioned'; random = condition == 'random_legal'
        who = policy_identity(condition, model, path, lesion)
        matched = random_matched(test['matched']) if random else evaluate_pairs(model, test['matched'], lesion=lesion)
        mate2 = tactical(model, test['mate2'], lesion=lesion, random=random)
        game_report, sample = evaluate(model, opponent, test['mate2'], repeats=manifest['test_game_repeats'],
                                       cap=4, random=random, lesion=lesion)
        game_report['identity'] = who; game_report['opponent'] = opponent_id
        game_report['auxiliary_used_in_training'] = False
        reports[condition] = dict(identity=who, matched=matched, mate2=mate2, games=game_report)
        if not random:
            reports[condition]['paired_swap'] = swapped_report(matched)
        from flies.chess.selfplay import save_game
        sample['players'] = {sample['candidate']: who,
                             ('black' if sample['candidate']=='white' else 'white'):opponent_id}
        save_game(sample, Path(output).parent/condition, 0)
        write_json(output, dict(protocol_sha256=digest(manifest), test_sha256=test['sha256'], seed=seed, reports=reports))
        print(seed, condition, matched['mean_correct_mass'], matched['mean_adaptation'],
              'mate2-wins', game_report['wins'], flush=True)
    return reports


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('mode', choices=('validation','generate','test'))
    parser.add_argument('--corpus', default=str(ROOT/'artifacts/mechanism/curriculum.json'))
    parser.add_argument('--manifest', default=str(ROOT/'mechanism_manifest.json'))
    parser.add_argument('--test', default=str(ROOT/'artifacts/mechanism/fresh_test.json'))
    parser.add_argument('--seed', type=int, default=11); parser.add_argument('--output', required=True)
    args = parser.parse_args(); read = lambda p: json.loads(Path(p).read_text())
    if args.mode == 'validation':validation(read(args.corpus), args.output)
    elif args.mode == 'generate':generate_fresh(read(args.corpus), read(args.manifest), args.output)
    else:test_seed(read(args.corpus), read(args.manifest), read(args.test), args.seed, args.output)
