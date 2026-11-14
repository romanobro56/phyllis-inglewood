"""Bounded-memory REINFORCE on real FlyBrain coordinate decisions.

Only the existing FlyBrain parameters are optimized. No value/output head,
strategic teacher, engine, or random-move opponent is used here.
"""
import argparse
from pathlib import Path
import random
import time

import chess
import numpy as np
import torch
from core.brain import pick_device
from flies.chess.evaluate import compare, summarize
from flies.chess.selfplay import play_game, save_game, assert_independent
from flies.chess.storage import (ROOT, atomic_write, brain_payload, identity, load_policy,
                                 validate_and_load, write_json, prepare_run)

CURRICULUM = {
    "mate": ("7k/8/5KQ1/8/8/8/8/8 w - - 0 1", "8/8/8/8/8/5kq1/8/7K b - - 0 1"),
    "mate_rook": ("7k/R7/5KQ1/8/8/8/8/8 w - - 0 1", "8/8/8/8/8/5kq1/r7/7K b - - 0 1"),
    "endgame": ("7k/8/8/8/8/8/4PK2/8 w - - 0 1", "8/4pk2/8/8/8/8/8/7K b - - 0 1"),
    "movement": (chess.STARTING_FEN,),
    "start": (chess.STARTING_FEN,),
}


def cpu_snapshot(brain):
    ck = brain_payload(brain)
    ck["state_dict"] = {k: v.detach().cpu().clone() for k, v in ck["state_dict"].items()}
    return ck


def discounted_returns(reward, n_moves, gamma):
    """Scalar means terminal reward; a vector means per-own-turn rewards."""
    r=np.asarray(reward,dtype=float)
    if r.ndim==0:
        r=np.zeros(n_moves,dtype=float)
        if n_moves:r[-1]=float(reward)
    if r.shape!=(n_moves,) or not np.isfinite(r).all():
        raise ValueError('Reward sequence must contain one finite reward per own move')
    returns=np.zeros(n_moves);g=0.
    for i in reversed(range(n_moves)):
        g=float(r[i])+gamma*g;returns[i]=g
    return returns,float(r.sum())


def policy_update(brain, optimizer, samples, baseline, gamma=.98, entropy=.01, chunk=64, sample_baselines=None):
    """Sum stage log probabilities per own move and discount own-turn rewards.

    samples: [(own_moves, terminal_scalar_or_per_move_vector)]. A truncation
    has zero terminal outcome reward; it is never re-labeled as an official
    draw. Explicit auxiliary rewards may still be nonzero. Baseline uses prior data.
    """
    device = next(brain.parameters()).device
    optimizer.zero_grad(set_to_none=True)
    if sample_baselines is None:
        sample_baselines = [baseline] * len(samples)
    if len(sample_baselines) != len(samples):
        raise ValueError('One prior baseline is required for each episode')
    nonempty = [(sample, base) for sample, base in zip(samples, sample_baselines) if sample[0]]
    samples = [sample for sample, _ in nonempty]
    flat = []
    returns = []
    n_moves = sum(len(moves) for moves, _ in samples)
    if not n_moves:
        return dict(loss=0.0, grad_norm=0.0, moves=0, reward=0.0), baseline
    episode_totals=[]
    for (moves, reward), prior_base in nonempty:
        discounted,total=discounted_returns(reward,len(moves),gamma)
        episode_totals.append(total)
        for i, stages in enumerate(moves):
            ret = discounted[i]
            returns.append(ret)
            flat.extend((s, ret - prior_base) for s in stages)
    # Normalize by episodes, avoiding batch-length-dependent step sizes.
    # Stage log-probabilities still sum over each move; forced stages equal 0.
    denominator = len(samples)
    old = {name: p.detach().clone() for name, p in brain.named_parameters()}
    total_loss = 0.0
    for start in range(0, len(flat), chunk):
        batch = flat[start:start + chunk]
        obs = torch.as_tensor(np.stack([s["obs"] for s, _ in batch]), device=device)
        masks = torch.as_tensor(np.stack([s["mask"] for s, _ in batch]), device=device)
        actions = torch.as_tensor([s["action"] for s, _ in batch], device=device)
        advantage = torch.as_tensor([a for _, a in batch], dtype=torch.float32, device=device)
        logits, _ = brain(obs, masks)
        logits = logits / getattr(brain, "policy_temperature", 1.0)
        dist = torch.distributions.Categorical(logits=logits)
        loss = -(dist.log_prob(actions) * advantage + entropy * dist.entropy()).sum() / denominator
        if not torch.isfinite(loss):
            raise ValueError("Nonfinite policy loss; checkpoint left untouched")
        loss.backward()
        total_loss += float(loss.detach())
    grad_norm = float(torch.nn.utils.clip_grad_norm_(brain.parameters(), 1.0, error_if_nonfinite=True))
    grad_by_parameter = {name: float(p.grad.norm()) if p.grad is not None else 0.0
                         for name, p in brain.named_parameters()}
    optimizer.step()
    delta = {name: float((p.detach()-old[name]).norm()) for name, p in brain.named_parameters()}
    mean_return = float(np.mean(returns))
    return dict(loss=total_loss, grad_norm=grad_norm, moves=n_moves,
                reward=float(np.mean(episode_totals)), mean_return=mean_return,
                episodes=len(samples), nonzero_reward_episodes=sum(abs(r)>1e-12 for r in episode_totals),
                gradient_norms=grad_by_parameter, parameter_deltas=delta,
                forced_stages=sum(not s['mask'].sum() > 1 for s, _ in flat), stages=len(flat)), .9 * baseline + .1 * mean_return


def rng_state(rng):
    return dict(numpy=rng.bit_generator.state, torch=torch.get_rng_state(), python=random.getstate(),
                cuda=torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None)


def restore_rng(state, rng):
    rng.bit_generator.state = state["numpy"]
    torch.set_rng_state(state["torch"].cpu())
    random.setstate(state["python"])
    if state["cuda"] is not None and torch.cuda.is_available():
        torch.cuda.set_rng_state_all([s.cpu() for s in state["cuda"]])


def train(output, updates=2, games_per_update=2, size="tiny", device="cpu", seed=42,
          max_plies=12, curriculum="mate", resume=None, lr=1e-3, gamma=.98,
          entropy=.01, chunk=64, pool_size=3, snapshot_every=2, eval_games=2):
    if min(updates, games_per_update, max_plies, chunk, pool_size, snapshot_every, eval_games) < 1:
        raise ValueError("Training bounds must be positive")
    if not np.isfinite([gamma,lr,entropy]).all() or not 0 <= gamma <= 1 or lr <= 0 or entropy < 0:
        raise ValueError("Invalid optimization settings")
    output = prepare_run(output,resume)
    torch.manual_seed(seed)
    random.seed(seed)
    rng = np.random.default_rng(seed)
    brains = [load_policy(size=size, device=device), load_policy(size=size, device=device)]
    assert_independent(*brains)
    optimizers = [torch.optim.Adam(b.parameters(), lr=lr) for b in brains]
    config = dict(games_per_update=games_per_update, size=size, device=str(device), seed=seed,
                  algorithm="episode-reinforce-v2",
                  max_plies=max_plies, curriculum=curriculum, lr=lr, gamma=gamma, entropy=entropy,
                  chunk=chunk, pool_size=pool_size, snapshot_every=snapshot_every, eval_games=eval_games)
    pool = [cpu_snapshot(brains[0])]
    baselines, history, start = [0.0, 0.0], [], 0
    if resume:
        ck = torch.load(resume, map_location="cpu", weights_only=False)
        if ck["config"] != config:
            raise ValueError("Resume configuration or learning algorithm differs; use original code/settings (only --updates may increase)")
        for i in range(2):
            validate_and_load(brains[i], ck["brains"][i])
            optimizers[i].load_state_dict(ck["optimizers"][i])
        pool, baselines, history, start = ck["pool"], ck["baselines"], ck["history"], ck["update"]
        restore_rng(ck["rng"], rng)
    # This object is independent of both learners and never has an optimizer.
    frozen = load_policy(size=size, device=device)
    for p in frozen.parameters():
        p.requires_grad_(False)
    # Constructor consumes torch RNG; ensure resume follows the saved continuation exactly.
    if resume:
        restore_rng(ck["rng"], rng)
    t0 = time.monotonic()
    def publish_exports():
        for i, brain in enumerate(brains):
            atomic_write(output / f"brain_{i}.pt", brain_payload(brain), binary=True)
        write_json(output / 'training.json',dict(config=config,history=history,
                   identities=[identity(b,output/f'brain_{i}.pt') for i,b in enumerate(brains)],
                   elapsed_seconds=time.monotonic()-t0))
    if not resume:
        # Even an interrupt during the first rollout has a recoverable boundary.
        atomic_write(output/'resume.pt',dict(update=0,config=config,brains=[cpu_snapshot(b) for b in brains],
                     optimizers=[o.state_dict() for o in optimizers],rng=rng_state(rng),pool=pool,
                     baselines=baselines,history=history),binary=True)
    if start > updates:
        raise ValueError('Requested updates precede the committed resume checkpoint')
    publish_exports()  # Repair missing/stale exports even if no new update is requested.
    try:
        for update in range(start, updates):
            samples = [[], []]
            records, sides = [], []
            for game_index in range(games_per_update):
                swap = (update + game_index) % 2
                order = [swap, 1 - swap]
                players = [brains[i] for i in order]
                learn = [True, True]
                # Alternate live-vs-live with historical opponent games, rotating learner.
                if game_index % 2:
                    validate_and_load(frozen, pool[int(rng.integers(len(pool)))])
                    opponent_side = (update // 2) % 2
                    players[opponent_side] = frozen
                    learn[opponent_side] = False
                fen = CURRICULUM[curriculum][int(rng.integers(len(CURRICULUM[curriculum]))) ]
                game_seed = int(rng.integers(2 ** 31))
                meta = {side: identity(b) for side, b in zip(("white", "black"), players)}
                for side, b in zip(("white", "black"), players):
                    meta[side]["label"] = "FROZEN-TRAINING-SNAPSHOT" if b is frozen else "TRAINING"
                game, trace = play_game(*players, seed=game_seed, max_plies=max_plies, fen=fen, metadata=meta)
                save_game(game, output / "games", update * games_per_update + game_index)
                for side_idx, side in enumerate(("white", "black")):
                    if learn[side_idx]:
                        winner = game["outcome"]["winner"]
                        reward = 0.0 if winner is None else 1.0 if winner == side else -1.0
                        samples[order[side_idx]].append((trace[side], reward))
                records.append(game)
                sides.append("white")
            metrics = []
            for i in range(2):
                metric, baselines[i] = policy_update(brains[i], optimizers[i], samples[i], baselines[i],
                                                     gamma, entropy, chunk)
                metrics.append(metric)
            entry = dict(update=update + 1, players=metrics, games=summarize(records, sides))
            if (update + 1) % snapshot_every == 0:
                validate_and_load(frozen, pool[0])
                # Held-out positions and seeds; inference only, no updates or RNG consumption by learners.
                entry["held_out_vs_frozen"] = compare(brains[0], frozen, eval_games,
                    seed=100000 + seed, max_plies=max_plies)
                pool.append(cpu_snapshot(brains[(update + 1) % 2]))
                pool = pool[-pool_size:]
            history.append(entry)
            # Complete resume record is the commit point. Individual exports are derived artifacts.
            checkpoint = dict(update=update + 1, config=config,
                brains=[cpu_snapshot(b) for b in brains], optimizers=[o.state_dict() for o in optimizers],
                rng=rng_state(rng), pool=pool, baselines=baselines, history=history)
            atomic_write(output / "resume.pt", checkpoint, binary=True)
            publish_exports()
            print(f"update {update + 1}: {metrics}", flush=True)
    except KeyboardInterrupt:
        print("Interrupted: last fully committed resume.pt remains valid.", flush=True)
        raise
    return brains, history


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--output", default=str(ROOT / "artifacts/training"))
    ap.add_argument("--updates", type=int, default=2, help="total updates including resumed updates")
    ap.add_argument("--games-per-update", type=int, default=2)
    ap.add_argument("--size", default="tiny")
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--max-plies", type=int, default=12)
    ap.add_argument("--curriculum", choices=CURRICULUM, default="mate")
    ap.add_argument("--resume")
    ap.add_argument("--lr", type=float, default=.001)
    ap.add_argument("--gamma", type=float, default=.98)
    ap.add_argument("--entropy", type=float, default=.01)
    ap.add_argument("--chunk", type=int, default=64)
    ap.add_argument("--pool-size", type=int, default=3)
    ap.add_argument("--snapshot-every", type=int, default=2)
    ap.add_argument("--eval-games", type=int, default=2)
    args = vars(ap.parse_args())
    torch.set_num_threads(1)
    args["device"] = pick_device(args["device"])
    train(**args)


if __name__ == "__main__":
    main()
