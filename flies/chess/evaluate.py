"""Held-out comparisons; random legal moves are explicitly a baseline only."""
import argparse
from collections import defaultdict
from pathlib import Path

import chess
import torch
from core.brain import pick_device
from flies.chess.selfplay import play_game, save_game
from flies.chess.storage import ROOT, identity, load_policy, write_json, prepare_run

# Distinct from the training curriculum. Selection never uses a chess heuristic.
HELD_OUT = (chess.STARTING_FEN, "k7/8/1QK5/8/8/8/8/8 w - - 0 1")


def summarize(games, candidate_sides):
    if len(games) != len(candidate_sides):
        raise ValueError("One candidate side is required for each game")
    counts = dict(wins=0, losses=0, draws=0, truncated=0)
    stage = defaultdict(lambda: dict(decisions=0, legal=0, probability_sum=0.0))
    for game, side in zip(games, candidate_sides):
        outcome = game["outcome"]
        if outcome["truncated"]:
            counts["truncated"] += 1
        elif outcome["winner"] is None:
            counts["draws"] += 1
        else:
            counts["wins" if outcome["winner"] == side else "losses"] += 1
        for move in game["moves"]:
            if move["side"] != side:
                continue
            for s in move["stages"]:
                m = stage[s["stage"]]
                m["decisions"] += 1
                m["legal"] += int(s["mask"][s["action"]])
                m["probability_sum"] += s["selected_probability"]
    for m in stage.values():
        m["masked_legality_accuracy"] = m.pop("legal") / m["decisions"]
        m["mean_selected_probability"] = m.pop("probability_sum") / m["decisions"]
    return dict(**counts, games=len(games), average_plies=sum(g["plies"] for g in games) / max(1, len(games)),
                illegal_attempts=sum(g["illegal_attempts"] for g in games), per_stage=dict(stage),
                elapsed_seconds=sum(g["elapsed_seconds"] for g in games))


def compare(candidate, opponent, games=2, seed=100042, max_plies=80, positions=HELD_OUT,
            candidate_lesion=False, random_opponent=False, output=None, metadata=None):
    if games < 1 or not positions:
        raise ValueError("Evaluation needs positive games and held-out positions")
    results, sides = [], []
    # Each seed/position is played with both colors before advancing to the next.
    for i in range(games):
        side = "white" if i % 2 == 0 else "black"
        fen = positions[(i // 2) % len(positions)]
        white, black = (candidate, opponent) if side == "white" else (opponent, candidate)
        lesion = (candidate_lesion, False) if side == "white" else (False, candidate_lesion)
        players = None
        if metadata:
            players = dict(zip((side, "black" if side == "white" else "white"), metadata))
        game, _ = play_game(white, black, seed + i // 2, max_plies, fen, lesions=lesion,
                            random_side=("black" if side == "white" else "white") if random_opponent else None,
                            metadata=players)
        if output:
            save_game(game, output, i)
        results.append(game)
        sides.append(side)
    return summarize(results, sides)


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--checkpoint", required=True)
    ap.add_argument("--opponent-checkpoint")
    ap.add_argument("--size", default="tiny")
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--games", type=int, default=4)
    ap.add_argument("--seed", type=int, default=100042)
    ap.add_argument("--max-plies", type=int, default=80)
    ap.add_argument("--output", default=str(ROOT / "artifacts/evaluation"))
    a = ap.parse_args()
    a.output=prepare_run(a.output)
    torch.set_num_threads(1)
    torch.manual_seed(a.seed)
    dev = pick_device(a.device)
    candidate = load_policy(a.checkpoint, a.size, dev)
    untrained = load_policy(size=a.size, device=dev,encoding=candidate.encoding)
    untrained.policy_temperature=candidate.policy_temperature
    frozen = load_policy(a.opponent_checkpoint or a.checkpoint, a.size, dev)
    for p in frozen.parameters():
        p.requires_grad_(False)
    ident = identity(candidate, a.checkpoint)
    configs = {
        "trained_vs_random_legal": (candidate, frozen, False, True),
        "untrained_vs_random_legal": (untrained, frozen, False, True),
        "lesioned_vs_random_legal": (candidate, frozen, True, True),
        "trained_vs_untrained": (candidate, untrained, False, False),
        "trained_vs_frozen_checkpoint": (candidate, frozen, False, False),
    }
    reports = {}
    for label, (brain, opponent, lesion, random_opponent) in configs.items():
        meta = [identity(brain, None if brain is untrained else a.checkpoint, lesion),
                dict(label="RANDOM-LEGAL") if random_opponent else identity(opponent,
                    a.opponent_checkpoint or a.checkpoint if opponent is frozen else None)]
        reports[label] = compare(brain, opponent, a.games, a.seed, a.max_plies,
                                candidate_lesion=lesion, random_opponent=random_opponent,
                                output=Path(a.output) / label, metadata=meta)
        print(label, reports[label])
    write_json(Path(a.output) / "evaluation.json", dict(identity=ident, seed=a.seed,
               positions=HELD_OUT, reports=reports,
               metric_note="Masked legality accuracy is a rules check, not strategic move accuracy. Truncations are excluded from draws."))


if __name__ == "__main__":
    main()
