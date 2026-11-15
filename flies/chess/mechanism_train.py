"""Outcome-only grouped self-play: prior EMA versus independent-rollout LOO.

No proof labels, heuristic scores or strategic move generator are used here.
"""
import argparse
import json
from pathlib import Path
import time

import chess
import numpy as np
import torch
from flies.chess.batched import play_batch
from flies.chess.chess_env import SPATIAL_ENCODING, STAGE_ONLY_ENCODING
from flies.chess.learning import outcome_reward
from flies.chess.research_train import clone_source, own_turn_rewards
from flies.chess.selfplay import save_game
from flies.chess.storage import ROOT, FIXED, atomic_write, brain_payload, file_hash, validate_and_load, write_json, prepare_run
from flies.chess.train_selfplay import cpu_snapshot, policy_update, rng_state, restore_rng


def leave_one_out(scores):
    scores = np.asarray(scores, dtype=float)
    if scores.ndim != 1 or len(scores) < 2 or not np.isfinite(scores).all():
        raise ValueError('LOO requires at least two independent finite rollout scores')
    # Explicitly sum the other rows. This avoids cancellation by the row's own
    # value and makes the no-own-reward invariant exact even for extreme values.
    return np.array([np.delete(scores, i).mean() for i in range(len(scores))])


def grouped_specs(corpus, rng, update, games=128, group_size=8, matched_updates=32, easy_updates=8):
    if group_size < 2 or games % (4*group_size) or games < 8*group_size:
        raise ValueError('Budget must be >=8 groups and divisible by four groups')
    mixed = update >= matched_updates
    groups = games // group_size
    contexts = []
    # First phase uses paired boards; mixed phase reserves half the groups for
    # pairs, one quarter for suffix starts and one quarter for mate-two roots.
    n_pairs = groups // (4 if mixed else 2)
    rows = corpus['easy_train'] if update < easy_updates else corpus['matched_train']
    for _ in range(n_pairs):
        pair = rows[int(rng.integers(len(rows)))]
        contexts.extend((fen, 'matched') for fen in pair['positions'])
    if mixed:
        for tier, key in (('suffix', 'suffix_train'), ('mate2', 'mate2_train')):
            for _ in range(groups//4):
                row = corpus[key][int(rng.integers(len(corpus[key])))]
                contexts.append((row['fen'], tier))
    specs = []; tiers = []; group_ids = []
    for group, (fen, tier) in enumerate(contexts):
        board = chess.Board(fen)
        # Pair members get the same color. Groups otherwise balance attacker
        # identity and actual color over updates; both players always live.
        if (group//2 + update) % 2:
            board = board.mirror()
        attacker = (group//2 + update//2) % 2
        players = (attacker, 1-attacker) if board.turn else (1-attacker, attacker)
        seeds = rng.choice(2**31, group_size, replace=False)
        for seed in seeds:
            specs.append(dict(fen=board.fen(), seed=int(seed), players=players))
            tiers.append(tier); group_ids.append(group)
    assert len(specs) == games
    return specs, tiers, group_ids


def sample_baselines(records, specs, tiers, groups, learner, method, prior):
    if method not in ('ema', 'loo'):
        raise ValueError('Unknown baseline')
    bases = {}; score_groups = {}; contexts = {}; seeds = {}
    for i, (game, spec, tier, group) in enumerate(zip(records, specs, tiers, groups)):
        context = (spec['fen'], tuple(spec['players']), tier)
        if group in contexts and contexts[group] != context:
            raise ValueError('LOO groups must share exactly one start/color/player context')
        contexts[group] = context
        if spec['seed'] in seeds.setdefault(group, set()):
            raise ValueError('LOO groups require independent sampling seeds')
        seeds[group].add(spec['seed'])
        side = 'white' if spec['players'][0] == learner else 'black'
        role = 'attack' if side == game['candidate'] else 'defend'
        key = tier + '/' + role
        score_groups.setdefault(group, []).append((i, key, outcome_reward(game, side)))
    for members in score_groups.values():
        scores = [x[2] for x in members]
        values = leave_one_out(scores) if method == 'loo' else [prior.get(x[1], 0.) for x in members]
        for (index, _, _), value in zip(members, values):
            bases[index] = float(value)
    # Update EMA only after all current advantages have been assigned. LOO
    # never mixes contexts, counterfactual pair members, colors or roles.
    next_prior = dict(prior)
    for key in {x[1] for members in score_groups.values() for x in members}:
        scores = [x[2] for members in score_groups.values() for x in members if x[1] == key]
        next_prior[key] = .9*prior.get(key, 0.) + .1*float(np.mean(scores))
    informative = sum(len({x[2] for x in members}) > 1 for members in score_groups.values())
    return [bases[i] for i in range(len(records))], next_prior, informative


def train(corpus, sources, output, seed=11, updates=48, method='loo', encoding=SPATIAL_ENCODING,
          games=128, group_size=8, matched_updates=32, easy_updates=8, lr=.003, entropy=.0005,
          resume=None, size='tiny'):
    if method not in ('loo', 'ema', 'zero') or not 0 <= updates <= matched_updates+16:
        raise ValueError('Unknown method or invalid update budget')
    if not 0 <= easy_updates <= matched_updates or lr <= 0 or entropy < 0:
        raise ValueError('Invalid training configuration')
    torch.set_num_threads(1); torch.manual_seed(seed); rng = np.random.default_rng(seed)
    models = [clone_source(p, encoding, size=size) for p in sources]
    initial_fixed = [{key: getattr(m, key).clone() for key in FIXED} for m in models]   # wiring buffers (not all persisted)
    opts = [torch.optim.Adam(m.parameters(), lr=lr) for m in models]
    config = dict(algorithm='outcome-grouped-v1', reward_arithmetic='integer-v2', method=method,
                  corpus_sha256=corpus['sha256'], source_sha256=[file_hash(p) for p in sources],
                  seed=seed, encoding=encoding, games=games, group_size=group_size,
                  matched_updates=matched_updates, easy_updates=easy_updates, lr=lr, entropy=entropy,
                  gamma=1., chunk=64, size=size)
    output = prepare_run(output,resume); history = []; priors = [{}, {}]; start = 0
    if resume:
        ck = torch.load(resume, map_location='cpu', weights_only=False)
        if ck['config'] != config:
            raise ValueError('Mechanism resume configuration mismatch')
        for i in range(2):
            validate_and_load(models[i], ck['brains'][i]); opts[i].load_state_dict(ck['optimizers'][i])
        history, priors, start = ck['history'], ck['priors'], ck['update']
        restore_rng(ck['rng'], rng)
    else:
        for i, model in enumerate(models):
            atomic_write(output/f'checkpoint_0000_{i}.pt', brain_payload(model), binary=True)
        atomic_write(output/'resume.pt',dict(config=config,update=0,brains=[cpu_snapshot(m) for m in models],
                     optimizers=[o.state_dict() for o in opts],priors=priors,history=history,rng=rng_state(rng)),binary=True)
    if start>updates:raise ValueError('Requested updates precede resume checkpoint')
    for i, model in enumerate(models):
        if not (output/f'checkpoint_{start:04d}_{i}.pt').exists():
            atomic_write(output/f'checkpoint_{start:04d}_{i}.pt',brain_payload(model),binary=True)
    write_json(output/'history.json',dict(config=config,history=history,fixed_structure_unchanged=True))
    t0 = time.monotonic()
    for update in range(start, updates):
        specs, tiers, groups = grouped_specs(corpus, rng, update, games, group_size, matched_updates, easy_updates)
        records = [None]*games; traces = [None]*games
        for tier, cap in (('matched', 2), ('suffix', 2), ('mate2', 4)):
            indices = [i for i, t in enumerate(tiers) if t == tier]
            if not indices:
                continue
            batch, trace = play_batch(models, [specs[i] for i in indices], max_plies=cap)
            for i, game, moves in zip(indices, batch, trace):
                records[i], traces[i] = game, moves
        # Freeze every baseline before either player's parameters change.
        baseline_sets = [sample_baselines(records, specs, tiers, groups, i, 'ema' if method == 'zero' else method, priors[i]) for i in range(2)]
        metrics = []
        for learner in range(2):
            bases, priors[learner], informative = baseline_sets[learner]
            if method == 'zero':
                bases = [0.]*len(records)
            samples = []
            for game, trace, spec in zip(records, traces, specs):
                side = 'white' if spec['players'][0] == learner else 'black'
                # Terminal outcomes only; labels/suffix provenance are absent.
                rewards = own_turn_rewards(game, side, False, 'integer-v2')
                samples.append((trace[side], [0.]*len(rewards) if method == 'zero' else rewards))
            metric, _ = policy_update(models[learner], opts[learner], samples, 0., gamma=1.,
                                      entropy=entropy, chunk=64, sample_baselines=bases)
            metric['informative_groups'] = informative
            metrics.append(metric)
        outcomes = {}
        for tier in set(tiers):
            subset = [g for g, t in zip(records, tiers) if t == tier]
            outcomes[tier] = dict(games=len(subset), wins=sum(g['outcome']['winner'] == g['candidate'] for g in subset),
                                  losses=sum(g['outcome']['winner'] not in (None, g['candidate']) for g in subset),
                                  draws=sum(g['outcome']['result'] == '1/2-1/2' for g in subset),
                                  truncated=sum(g['outcome']['truncated'] for g in subset),
                                  plies=sum(g['plies'] for g in subset), illegal_attempts=sum(g['illegal_attempts'] for g in subset))
        history.append(dict(update=update+1, outcomes=outcomes, metrics=metrics, elapsed=time.monotonic()-t0))
        for model, fixed in zip(models, initial_fixed):
            assert all(torch.equal(getattr(model, key), value) for key, value in fixed.items())
        atomic_write(output/'resume.pt', dict(config=config, update=update+1, brains=[cpu_snapshot(m) for m in models],
                     optimizers=[o.state_dict() for o in opts], priors=priors, history=history, rng=rng_state(rng)), binary=True)
        if update+1 in (8, 16, 32, 40, 48) or update+1 == updates:
            for i, model in enumerate(models):
                atomic_write(output/f'checkpoint_{update+1:04d}_{i}.pt', brain_payload(model), binary=True)
        write_json(output/'history.json', dict(config=config, history=history, fixed_structure_unchanged=True))
        save_game(records[0], output/'samples', update)
        print(update+1, method, outcomes, 'gradient', [round(m['grad_norm'], 4) for m in metrics], flush=True)
    return models, history


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--corpus', default=str(ROOT/'artifacts/mechanism/curriculum.json'))
    parser.add_argument('--sources', nargs=2, required=True); parser.add_argument('--output', required=True)
    parser.add_argument('--seed', type=int, default=11); parser.add_argument('--updates', type=int, default=48)
    parser.add_argument('--method', choices=('ema', 'loo', 'zero'), default='loo'); parser.add_argument('--stage-only', action='store_true')
    parser.add_argument('--matched-updates', type=int, default=32); parser.add_argument('--easy-updates', type=int, default=8)
    parser.add_argument('--resume'); args = parser.parse_args()
    train(json.loads(Path(args.corpus).read_text()), args.sources, args.output, args.seed, args.updates, args.method,
          STAGE_ONLY_ENCODING if args.stage_only else SPATIAL_ENCODING, matched_updates=args.matched_updates,
          easy_updates=args.easy_updates, resume=args.resume)
