# Chess Fly: Implementation Specification for `romanobro56/phyllis-inglewood`

**Objective:** Build a legally correct, connectome-grounded chess-playing fruit fly and have two copies of the same fly-brain architecture play against each other. The implementation must be an additive contribution to this existing repository, not a replacement engine or unrelated neural network.

**Starting repository:** https://github.com/romanobro56/phyllis-inglewood

**Helpful reference (do not replace the target repository):** https://github.com/cesp99/fly-chess

**Repository contracts inspected:** `core/task.py`, `core/brain.py`, `core/train.py`, `core/check.py`, `core/registry.py`, `server/serve.py`, `flies/_template/task.py`, `flies/_template/station.js`.

## Constraints from the actual repository

1. `FlyTask` defines `name`, `actions` (2–8 fixed names), `obs_dim`, `mode`, `max_steps`, plus `reset`, `step`, optional `mask`, optional `teacher`, `caption`.
2. `FlyBrain(preset, obs_dim, n_actions, fly_name)` accepts `[batch_size, obs_dim]` observations and an optional mask; returns action logits and neural-activity statistics.
3. The model uses FlyWire-derived connectivity, fixed edge existence and signs, trainable connection strengths, leak/bias/input mapping, and a readout of fixed groups of descending neurons. It is a **rate-based recurrent approximation**, not a validated biologically exact spiking brain.
4. `core.train` trains one `FlyTask` environment and one policy at a time. It does **not** implement native two-brain chess self-play.
5. `server.serve` drives each fly's `FlyTask` independently; it does **not** have a two-agent chess session or a chess-board state event.
6. Contributor rule: implement the feature only in `flies/chess/`. Do not edit `core/`, `server/`, `web/`, `scripts/`, shared `requirements.txt`, or other fly folders without approval. Place isolated scripts and dependencies within `flies/chess/`.
7. `core.check` expects valid `float32` observations, finite rewards, a legal nonempty action mask, and a `caption` string. All valid actions must flow through `FlyBrain` output groups, not a chess engine's strategic choice.
8. `python-chess` will be needed for move legality, game outcomes, PGNs, repetition, promotion, castling and en passant. Install as an isolated feature dependency: `pip install -r flies/chess/requirements.txt`.

## Feature tree

```text
flies/chess/
  task.py                # Exactly one FlyTask subclass: ChessFly
  chess_env.py           # Chess board, canonical feature encoder, hierarchical action mask / state machine
  selfplay.py            # Two independent FlyBrain instances, alternate moves and record trajectories
  train_selfplay.py      # Policy-gradient updates and opponent snapshot pool
  evaluate.py            # Random, untrained, trained, lesioned and head-to-head baselines
  station.js             # 3D board station, or an honest visualization of the demo task
  requirements.txt       # python-chess, no implicit root dependency changes
  README.md              # Commands, methodology, attribution, known scientific limitations
  tests/
    test_env.py
    test_encoding.py
    test_selfplay.py
    test_evaluation.py
```

The repo's registry finds `flies/chess/task.py` automatically; `ChessFly.name` must equal `"chess"`. Avoid defining other `FlyTask` subclasses in that file.

## Key architectural decision: fixed eight-action policy with hierarchical legal moves

**Do not create 4,000+ actions or a learned dense output head** in the original core. The original `FlyBrain` only supports 2–8 action slots and explicitly maps them to existing descending neurons.

Use `actions = ["slot_0", ..., "slot_7"]`. The task presents one of the following decision stages, each using the same eight descending-neuron output groups:

1. Source file (a through h): mask files that have at least one legal move from a square in the file.
2. Source rank (1 through 8): mask ranks that contain a legal source square in the chosen file.
3. Destination file (a through h): mask files accessible by at least one legal move from that selected source.
4. Destination rank (1 through 8): mask ranks accessible from the selected source and destination file.
5. Promotion piece, **only when necessary**: slot 0=queen, 1=rook, 2=bishop, 3=knight. Mask promotions that are not legal. Without promotion the destination selection finalizes the move.

In each stage, `mask()` must correspond exactly to the legal moves remaining after prior partial selections; it must never be all-False unless the episode is already terminal, when no call to the brain is permitted. A full move is applied only after all selections; its legality must be verified again using `python-chess` before `board.push(move)`. If a move is rejected, fail loudly in tests; never silently choose a substitute. **The chess library may enforce legality; it must never choose strategic moves.**

Encode the board into numeric channels with canonical current-player perspective, with a defined transformation for black that is invertible for all move stages. Include 12 piece planes x 64 squares, active stage one-hot, current partial choices, castling rights, en-passant state, check flag, and normalized halfmove clock. All observations must have deterministic shape `obs_dim`, `float32`, finite values, and fixed, documented feature order. Do not encode an engine evaluation or pre-ranked legal-move list as inputs. Test that black-to-move board and square choices are transformed consistently; test inverse transformation.

**Tradeoff:** selecting a move in 4–5 neural decisions allows every legal chess move but is an engineered input/output interface. Eight real descending-neuron groups are used for each coordinate stage; this does not mean a real fly naturally understands chessboard squares.

## Phase 1: repo-compatible task / smoke test

- Use `scripts/new_fly.py chess` only if the `flies/chess/` folder does not yet exist. Fill the required `ChessFly` contract, with `mode="reward"` for the native training mode.
- The demo task can pit one fly against a uniformly random *legal* opponent. Clearly label this as **not** two-fly self-play. When a full move is finalized, apply it, optionally apply one random opponent reply as an environment step, then generate next observation for the fly. Random opponent choice is allowed *only for the phase-1 baseline/demo*, never misrepresent it as a fly-brain move.
- Give simple, bounded rewards (e.g. per legal move 0; terminal +1/-1/0; optionally material change during initial curriculum, explicitly disclose as engineered shaping). Do not reward the four intermediate coordinate selections as separate chess moves.
- `max_steps` counts brain decisions, not chess plies; enforce a bounded game horizon. A timeout is adjudicated as an evaluation truncation, **not** automatically a chess draw unless the chosen experimental rule explicitly declares it so. Distinguish official chess result and truncated outcome in logs.
- `caption()` should report complete UCI/SAN move and/or board FEN when available; use station.js for visualization. Existing server sends only action and caption, not board snapshots, so any rendering must use information explicitly passed via caption or await an approved server integration. Do not fake displayed board moves.
- Keep step fast, deterministic under injected numpy RNG, no network calls, and no model calls from within `step`.

## Phase 2: genuine two-fly evaluation / self-play

`selfplay.py` must create two separate `FlyBrain` instances with identical initial graph, `fly_name="chess"`, observation and eight-action scheme. Optional initial weights copied from one checkpoint, but **not shared Parameter objects**. Maintain independent neural computation and optimizer state. On each chess turn, pass the canonical board observation to the current player's brain for each required stage; mask legal choices; sample or take argmax from its logits. Alternate White and Black until official terminal result or an explicit, reported move/compute cap.

A full match consists of one `python-chess` board controlled by these two brains. The match runner must log each individual neural stage choice, resulting full move, position, selection probabilities, side to move, exact checkpoint hash, source graph hash, RNG seeds, game outcome, elapsed time, and optional neuronal activity. Support `--white-checkpoint`, `--black-checkpoint`, `--games`, `--seed`, `--size`, `--device`, `--max-plies`, `--output`.

If no trained checkpoints are supplied, label the runs UNTRAINED and do not claim chess competency. A lesioned control may have uniform masked choices; report the true lesion baseline. Do not use Stockfish, a material heuristic, or a random fallback to pick the fly's legal move in a reported two-fly match. Legal action masking itself is permitted.

**Both players must call `FlyBrain.forward` to pick their own chess actions.** A test should instrument calls and assert that White and Black each invoked their own separate model instances. Test checkpoint cloning creates distinct parameter storage and that updating one fly does not update the other.

## Phase 3: train rather than only evaluate

- Start with a low-cost `tiny` or `small` graph and bounded curriculum (mate-in-one recognition, simple endgames, piece movement); full 134k-neuron experiments require appropriate GPU resources.
- For native reward-mode learning, use only the repository's documented `FlyTask` interface. For actual alternating two-policy training, implement a **separate self-play trainer** inside `flies/chess/` rather than pretending `core.train` supports it.
- Use REINFORCE or another explicit, testable policy-gradient scheme on the selected hierarchical action logits; accumulate log probabilities across the 4–5 decisions per completed move. Attribute outcome rewards to each brain's own selected moves. Apply discounting, an advantage baseline, entropy regularization, gradient clipping, and stable checkpointing. Keep a frozen opponent pool to avoid co-adaptation and allow comparative evaluation.
- Separate evaluation from training; use held-out seeds and positions, alternating colors. Track wins/losses/draws, truncated games, illegal move attempts (must be zero), average game length, per-stage masked accuracy, training reward and performance against frozen versions of the policy.
- If using a rule-based or Stockfish teacher for curriculum/imitation, label such training explicitly as externally supervised. Do not interpret the resulting chess rating as a direct measurement of biological fruit-fly cognition.
- `train_selfplay.py` should be resumable, store optimizer and RNG states, protect graph and action-dimension compatibility, and allow deterministic small-scale smoke runs.

## Phase 4: integrate the live visualization honestly

`station.js` should draw a recognizable 8x8 board with correct square colors and piece layout; animate only moves actually chosen by the model. Prefer transmitting complete FEN or UCI/SAN + FEN as an explicit structured event if shared server changes are approved. With no core/server modification, make the visualization only as capable as the existing caption protocol allows and explicitly describe the limitation. For live *two-brain* matches in the existing multi-fly world, first request the `server/` owner's approval and define a minimal match-event contract, without changing the core fly-decision semantics. Show current turn, source and destination, move legality, outcome, brains/checkpoints and activity.

## Acceptance checks

1. Exactly one `FlyTask` subclass in `flies/chess/task.py`; `name="chess"`, 8 actions, valid `obs_dim`, valid observation from `reset`, full `step`, `mask`, `caption` contract. `python -m core.check chess` must PASS on the real checkout.
2. Legal move selection works for start position and tests for: check, checkmate, stalemate, castling both colors, en passant, promotion to each of Q/R/B/N, pinned pieces, only one legal move, threefold repetition claim policies, 50-move claims, fivefold repetition, and 75-move endings. No game result may be fabricated from a timeout.
3. When the stage is not terminal, exactly 8 mask elements with at least one True. After each full move, recorded UCI and reconstructed FEN agree with `python-chess`.
4. Fixed seed replay yields the same move sequences with identical checkpoints, device and sampling setup (within applicable numerical determinism constraints).
5. Two-player runner uses **two separate live brain models**. No move is picked by a strategic external engine; `python-chess` performs only rules/masking/outcomes.
6. Tiny-graph smoke test; gradient/backward pass; checkpoints round-trip with graph-hash validation; resume test; zero gradients not inadvertently copied across players.
7. Evaluation includes random legal-move, untrained, trained and lesioned baselines. Report synthetic/tiny/small/full identity conspicuously; never call the synthetic graph a biologically real fly.
8. Test long games and deadlock avoidance; bounded runtime and memory; crashes/interrupts do not corrupt final checkpoints or results.
9. Tests do not require downloading FlyWire or large models unless marked as integration tests. Optional GPU full-graph validation separately.
10. `station.js` exports the exact `build` and async `play` interface; visualization matches actual chess move stream, no invented moves.

## Contributor workflow

```bash
git clone https://github.com/romanobro56/phyllis-inglewood.git
cd phyllis-inglewood
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
python scripts/new_fly.py chess        # skip if it already exists
# add files and tests under flies/chess/
pip install -r flies/chess/requirements.txt
python -m core.graph --size synthetic # optional purely synthetic first smoke (not scientific result)
# Build actual graph/data for non-synthetic tests using the README instructions.
python -m core.check chess
python -m core.train chess --size tiny --minutes 3
python -m flies.chess.selfplay --size tiny --games 2 --seed 42  # target CLI; implement
```

Use `python -m core.check chess` before every push and follow the repo's review rules. If adding files beyond `flies/chess/` becomes necessary, open a focused upstream request first.

## Required end-of-work report

Provide: files created/changed; explicit architecture map; exact commands tested with results; features implemented vs pending; bugs and unresolved risks; evidence that both real `FlyBrain` instances controlled actual legal moves; biological fidelity limitations; performance benchmarks; and reproduction commands. **Do not claim integration tests pass unless they were actually executed in a real checkout.**

## Codex directive

Implement the above in the given existing checkout in phases, beginning with inspecting current files for changes since this specification. Fix any incompatible assumptions before changing code. Produce runnable code and tests rather than only a plan. Preserve the original shared modules unless owner-approved. Fail transparently if full self-play cannot be integrated within contributor constraints; never fake it with a random or external engine opponent. Prioritize an honest working baseline over an unverified complicated trainer.
