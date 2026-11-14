# Chess Fly

A FlyWire-derived rate-network chooses every file, rank, and promotion in a legal chess move. Two independent copies can play and train against one another. No chess engine, learned output head, ranked move list, or random-move fallback controls a fly.

**Status:** local implementation verified, including actual tiny FlyWire integration, completed two-brain games, native Safari replay/activity inspection, and full 134,181-neuron **CPU compatibility** testing. General chess competence and full-graph learning are not established. See [COMPLETION.md](COMPLETION.md) for the final audit and [AUDIT.md](AUDIT.md) for historical evidence.

The cinematic addition passes **67 Python tests, core.check, both JavaScript suites and 46 real Safari/WebGL checks**, including an official 15-ply live repetition draw with every move animation acknowledged. [Verified evidence](cinematic_verification.json) records the exact test results, model identities, completed recording and unchanged shared-source hashes.

The latest [mechanism study](MECHANISM_RESULTS.md) supports modest matched mate-one board dependence (+2.06pp, nominal98.75% interval +0.72 to +3.62), also present in the frozen parent. The new training gain and LOO advantage remain inconclusive, and mate-two/game improvement is unsupported. Integer-v2 rewards were validated; three full replications produced identical weights/Adam states to float-v1. Full-brain GPU scaling is not justified. Historical [research results](RESEARCH_RESULTS.md) remain available; their different test distributions must not be pooled.

## Start the live cinematic 3D arena

```bash
PYTHONDONTWRITEBYTECODE=1 .venv/bin/python -m flies.chess.cinematic --port 8770
```

Open **http://127.0.0.1:8770/** and click **New live match**. Two original repository flies physically carry each real FlyBrain move. Only after placement, release, return and a verified animation acknowledgment does the other brain start inference. Click either fly for its genuine simulated neuron rates in the repository's existing Matrix viewer. Trained checkpoints are selectable in Match settings; omit them for clearly labeled untrained brains. Ordinary live games never update weights.

See [CINEMATIC.md](CINEMATIC.md) for trained launch commands, cameras, pause/recovery, replay, recording, quality settings and browser verification; [CINEMATIC_REUSE.md](CINEMATIC_REUSE.md) identifies the unchanged shared components. This feature-local host is the full two-brain 3D experience. The shared native multi-station world still has the separately labeled single-fly demo contract.

## Legacy two-dimensional recording viewer

The older recording-following viewer below remains available. It does not use the new animation-to-inference handshake; use the cinematic arena above for synchronized live 3D play.

From the repository root, using existing trained local checkpoints:

```bash
PYTHONDONTWRITEBYTECODE=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 .venv/bin/python -m flies.chess.demo \
  --white-checkpoint flies/chess/artifacts/mechanism/loo-11/checkpoint_0048_0.pt \
  --black-checkpoint flies/chess/artifacts/mechanism/loo-11/checkpoint_0048_1.pt \
  --games 1 --max-plies 512 --output flies/chess/artifacts/my-demo --port 8766
```

Open the printed localhost URL. The viewer follows actual full moves, shows all eight descending-group rates, records provenance and outcomes, and offers step/autoplay/move-list replay. Paste `artifacts/my-demo/game_0000.json` into **Recording URL** and click **Load recording URL** to replay from the initial board and download its PGN. Local game JSON files also work through **Choose File**. The launcher serves only this feature folder on127.0.0.1; it keeps serving after the match. Ctrl-C stops the listener and requests a full-move-boundary stop for any ongoing match. A stopped game is unfinished, never a draw. Use a fresh output directory for a new match; existing recordings/checkpoints are protected against accidental overwrite.

Omit checkpoint arguments for clearly labeled untrained brains. If the research artifacts are absent in a clean checkout, use that untrained mode or reproduce training below. No downloaded model or setup is needed in the already configured workspace. Complete verified example games remain in `artifacts/final-demo/` (17/19 plies, both official repetition draws); these demonstrate functioning full games, not strong play.

## Architecture

```
python-chess board → canonical packed observation → existing FlyBrain sensory input
   ↑                                                   ↓
verified move ← coordinate state machine ← 8 fixed descending-neuron groups
```

`chess_env.py` supplies rules and observations only. `selfplay.py` owns two distinct `core.brain.FlyBrain` objects and calls the current side's `forward` once per coordinate stage. Models use `fly_name="chess"`, the same source graph and fixed output groups, and separate parameter storage. There are no persistent neural states shared between sides. Parameters, fixed connectivity, neuron signs, sensory indices, and descending groups remain the existing core architecture. The optimizer acts on the existing core trainable parameters: edge strengths, neuron bias/leak and sensory projection. No connections or output parameters are added.

Eight fixed actions (`slot_0`…`slot_7`) are reused for:

1. Source file, a…h.
2. Source rank, 1…8 in current-player coordinates.
3. Destination file.
4. Destination rank.
5. Promotion, only when required: queen, rook, bishop, knight in slots 0…3.

Every mask is derived from the remaining legal moves after the current partial choices. Completion rechecks legality before pushing. An invalid choice raises; it never substitutes another move. Terminal positions have an all-false mask and never call the brain.

## Canonical observation and policy conflict

AGENTS.md is authoritative and limits observations to **64 floats**; the specification's literal 768 piece-plane input cannot meet that limit. The implementation uses a lossless packed board representation and provides `decode_planes(obs)` to recover the requested 12×64 planes exactly. This is an explicit specification adaptation, not an expanded core input interface. Packed categories are harder for the sensory projection to learn than one-hot planes.

All observations have shape `(64,)`, dtype `float32`, and values in `[0,1]`. The table describes native packed-v1; spatial-v2 additionally uses the reserved geometry fields described below. Square order is a1,b1,…,h8 after canonicalization. For Black, flip **ranks only** (`square ^ 56`); files stay a…h. This is its own inverse and applies to board pieces, en passant and every coordinate stage. Piece channels always mean own P/N/B/R/Q/K, then opponent P/N/B/R/Q/K.

| Indices | Contents |
|---|---|
| 0…31 | Two square codes per float: `(code[2i] + 13*code[2i+1])/168`; empty=0, own piece types=1…6, opponent=7…12. Round `float*168` to decode exactly. |
| 32…36 | Stage one-hot, in the order above. |
| 37…40 | Partial coordinate selections `(slot+1)/8`; 0 means not selected. |
| 41…44 | Own kingside/queenside castling, opponent kingside/queenside. |
| 45 | `(canonical_ep_square+1)/64`; 0 means none. |
| 46 | Current player is in check. |
| 47 | Halfmove clock clipped to 150, divided by 150. |
| 48 | Absolute White-to-move flag. |
| 49 | Fraction of ply budget remaining. |
| 50…51 | Current position has occurred at least twice / three times. |
| 52 | Automatic claim-draw experiment policy enabled. |
| 53…63 | Zero in native packed-v1; redundant geometry in research spatial-v2. |

The board's complete history remains in python-chess for repetition adjudication. The observation exposes history flags, not the full repetition stack; the policy is therefore not fully Markov with respect to all possible future repetition claims. No engine evaluation or tactical target is input.

## Rules, rewards and caps

Standard chess, including both colors' castling, en passant, and all four promotions. By default the experiment automatically takes available threefold and 50-move claims (including claims based on a legal next move, following python-chess). This is a fixed adjudication policy, not an extra strategic fly decision. `selfplay --no-claim-draw` disables optional claims; fivefold/75-move endings, checkmate, stalemate, and insufficient material still terminate automatically. Checkmate takes precedence as specified by the library.

A cap is **truncated, result `*`, termination `null`**, never a manufactured draw. A move consists of at most five neural decisions. Self-play stops at a full-move boundary after `--max-plies` chess plies. The native demo additionally caps at 80 brain decisions; it may truncate a partial selection. No inference follows termination.

Self-play REINFORCE uses terminal rewards +1/-1/0, discounted across each player's own completed moves. All coordinate log probabilities for a move are summed. Truncated games have zero outcome reward, but remain separately reported. An EMA baseline from earlier data, entropy regularization, gradient clipping, and Adam are used. Forward/backward is recomputed in bounded chunks; full-game autograd graphs are never retained. Sparse rewards and easy draws are significant limitations. There is no teacher or externally supervised policy.

The **native world/demo task is different**: White is the fly, Black chooses uniformly from legal moves using the injected NumPy RNG. It is labeled `fly-vs-random-demo` everywhere. It supplies disclosed material-change shaping (`P=1,N=B=3,R=5,Q=9`, material delta/100 clipped to ±0.1) on completed moves and terminal ±1/0. Intermediate selections get zero reward. This shaping is never used to choose or rank a move and is not used by the self-play trainer.

## Commands

Run from the repository root with the existing virtual environment. No setup, graph generation or downloads are needed if the environment is already configured. The feature dependency is isolated in `flies/chess/requirements.txt`:

```bash
PYTHONDONTWRITEBYTECODE=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 .venv/bin/python -m core.check chess
PYTHONDONTWRITEBYTECODE=1 .venv/bin/python -m unittest discover -s flies/chess/tests -v
PYTHONDONTWRITEBYTECODE=1 CHESS_REAL_GRAPH_TESTS=1 .venv/bin/python -m unittest discover -s flies/chess/tests -v
```

Default unit tests use a tiny explicitly **synthetic test fixture** with the real `FlyBrain` implementation, not downloaded data. The opt-in integration test uses existing `data/tiny.npz` and requires it to be non-synthetic. The core check uses the real checkout and never writes a run checkpoint.

Untrained two-fly matches, with complete JSON and PGN recordings:

```bash
PYTHONDONTWRITEBYTECODE=1 .venv/bin/python -m flies.chess.selfplay \
  --size tiny --games 2 --seed 42 --max-plies 24 --activity --live \
  --output flies/chess/artifacts/matches
```

Actual two-policy local training (not `core.train` pretending to be self-play):

```bash
PYTHONDONTWRITEBYTECODE=1 .venv/bin/python -m flies.chess.train_selfplay \
  --size tiny --updates 2 --games-per-update 4 --max-plies 12 \
  --curriculum mate --snapshot-every 1 --output flies/chess/artifacts/tiny-training
# Resume with identical settings, increasing the total number of updates:
PYTHONDONTWRITEBYTECODE=1 .venv/bin/python -m flies.chess.train_selfplay \
  --size tiny --updates 3 --games-per-update 4 --max-plies 12 \
  --curriculum mate --snapshot-every 1 --output flies/chess/artifacts/tiny-training \
  --resume flies/chess/artifacts/tiny-training/resume.pt
```

Curricula: `mate` (KQK), `mate_rook` (KQRK), `endgame` (pawn endings), `movement` and `start` (standard initial board). All positions are fixed starting states; no oracle chooses actions. The curriculum is a configurable experiment, not a guaranteed progression or measured chess rating. More outcome-rich smoke run:

```bash
PYTHONDONTWRITEBYTECODE=1 .venv/bin/python -m flies.chess.train_selfplay \
  --size tiny --updates 2 --games-per-update 16 --max-plies 8 --seed 9 \
  --curriculum mate_rook --snapshot-every 1 --output flies/chess/artifacts/tiny-rook-smoke
```

Live-vs-live games alternate colors. Every other training game uses a frozen historical opponent; opponent colors and learners rotate across updates. The bounded CPU snapshot pool has no optimizer. Each learner has its own optimizer. Held-out inference against a frozen snapshot runs at snapshot intervals using disjoint seeds and positions. Optimizer/RNG state, both brains, baselines, pool, settings and update count are in `resume.pt`. Settings must match on resume, except total `--updates` may increase. Device determinism is only promised for identical software/device/thread settings; tests verify CPU continuation exactly.

`brain_0.pt` and `brain_1.pt` are individually compatible with the core brain checkpoint format. The self-play loader also checks encoding, graph hash, dimensions, fixed buffers and finite weights. Use **trusted local checkpoints only**; optimizer and RNG records use PyTorch pickle loading. Writes use fsync plus atomic replace. `resume.pt` is the authoritative training commit point, including update0 before rollouts. Exported brains/results are derived artifacts. The native trainer repairs missing/stale exports even when resuming at the same total update count; no extra rollout is needed. Fresh launches refuse existing run directories; specify `--resume` or a new output path.

```bash
PYTHONDONTWRITEBYTECODE=1 .venv/bin/python -m flies.chess.evaluate \
  --checkpoint flies/chess/artifacts/tiny-training/brain_0.pt \
  --opponent-checkpoint flies/chess/artifacts/tiny-training/brain_1.pt \
  --size tiny --games 4 --max-plies 24 --output flies/chess/artifacts/evaluation
PYTHONDONTWRITEBYTECODE=1 .venv/bin/python -m flies.chess.selfplay \
  --white-checkpoint flies/chess/artifacts/tiny-training/brain_0.pt \
  --black-checkpoint flies/chess/artifacts/tiny-training/brain_1.pt \
  --size tiny --games 2 --seed 200042 --max-plies 80 \
  --output flies/chess/artifacts/trained-matches
```

Evaluation uses paired seeds/positions with alternating candidate colors, and compares checkpoint vs uniform legal moves, untrained vs uniform legal moves, actual lesioned checkpoint vs uniform legal moves, checkpoint vs untrained, and checkpoint vs frozen checkpoint. Uniform *legal moves* differ from uniform hierarchical stage choices. Lesions call the real `FlyBrain.forward(..., lesion=True)`; trained neuron biases can leave nonuniform but input-independent preferences. Never report that as a uniformly random chess player.

Metrics include W/L/D, truncations, illegal attempts, game length, per-stage selected probability and **masked legality accuracy**. The last is a rules correctness metric, not accuracy against expert chess strategy. All games carry selected probabilities, masks, UCI/SAN, before/after FEN, seeds, outcome, graph/state/checkpoint hashes, size/synthetic identity, and optional group/mean activity. Sampled model initialization seed and device are also recorded by the CLI. JSON files are authoritative; PGNs replay-verify each move before publication.

## Visualization

`station.js` implements the shared `build` / async `play` interface. It draws 64 correctly colored squares, distinguishable labeled pieces, and animates each completed model move and explicitly random reply from real FEN caption events. FEN snapshots handle castling, en passant, promotions, and episode resets. All eight action names have explicit branches. GPU job submission uses the shared native trainer and therefore would train this **random-opponent demo**, not the separate self-play script.

```bash
PYTHONDONTWRITEBYTECODE=1 OMP_NUM_THREADS=1 .venv/bin/python -m server.serve --only chess --host 127.0.0.1
# World demo: http://127.0.0.1:8000/web/
# Two-brain recording/live viewer: http://127.0.0.1:8000/flies/chess/viewer.html
```

The viewer can load any game JSON, step/play through actual positions and stage probabilities, or follow `artifacts/matches/live.json`. Run self-play with `--live --delay 1 --activity` in another terminal for paced live two-brain play. A `?source=artifacts/trained-matches/live.json` query selects another live output. The viewer invents no moves and runs no chess engine. Alternatively serve **only this feature folder** with a standard local static HTTP server and open `/viewer.html`.

The existing server forwards captions verbatim, so world labels contain the serialized FEN event payload as well as short text. Without shared UI changes this is functional but verbose. The station sign shows concise status. This protocol does not supply opponent brain inspection in the world because Black there is random. The standalone two-brain viewer shows both model identities and optional activity.

A future owner-approved shared-world integration would use a `chess_position` event with `{match_id, ply, fen, move:{uci,san,side}, stages, players, outcome}` after a completed move, and a `chess_stage` event for each brain choice. The match runner would own the two policies; the existing server must not drive both as independent single-agent tasks. **No shared server changes have been made or approval assumed.**

Optional protocol and JavaScript tests:

```bash
# With the existing chess server running:
PYTHONDONTWRITEBYTECODE=1 .venv/bin/python -m flies.chess.tests.verify_live --seconds 60
# After creating artifacts/matches/game_0000.json, on a machine with Node:
node --experimental-vm-modules flies/chess/tests/visual_smoke.mjs
```

The JavaScript test uses lightweight DOM/Three stubs to execute the real module, all eight branches and actual recorded moves. It is not proof of browser/WebGL rendering. Manual one-minute world/console QA remains separate.

Runtime outputs in these commands stay in `flies/chess/artifacts/` and are ignored. Frozen datasets/evidence and verified reports are deliberately retained beside the source for review. No push, merge, commit, shared GPU job, remote management command, graph generation, or infrastructure modification is part of these commands.

## Scientific limits and attribution

FlyWire v783 supplies the graph; reduced `tiny`/`small` subgraphs are not the full connectome. `synthetic=True` is always identified as **not a biological fly**. The core is a rate-based recurrent approximation, not a validated spiking or biological cognition model. Chess encoding, repeated coordinate decisions, masking, reward design and optimization are engineered interfaces. Legal play alone does not establish chess skill or a biological fruit fly's ability to play chess. No Elo or improved performance is claimed by these smoke runs.

Connectome: Dorkenwald et al., *Nature* (2024); annotations: Schlegel et al., *Nature* (2024); packaged connectivity: Shiu et al., *Nature* (2024). This feature extends this repository's existing `FlyBrain`; the specification also credits the fixed-connectivity training approach in [cesp99/fly-chess](https://github.com/cesp99/fly-chess). No external engine code was substituted for the repository brain.

## Historical first learning investigation

The original two-update commands above are smoke tests. For measured learning, use [EXPERIMENTS.md](EXPERIMENTS.md) and the learning follow-up in [AUDIT.md](AUDIT.md). The unchanged real tiny FlyBrain now reaches a pooled **18.0% first-move mate rate** on held-out KQK/KQRK tactical starts, against 3.5% matching initialization and 5.7% random legal moves. This is a narrow curriculum result, not evidence of general chess skill. The corrected shuffled-board control reaches 16.2%; the intact-minus-shuffled simultaneous interval includes zero, so board-state dependence is not established.

The study uses 256 training, 64 validation and 128 test position families, excluding color/dihedral duplicates across splits; three training seeds; 60 updates ×128 genuine two-brain games/seed; fixed checkpoint measurements; and 512 color-balanced test games per learner/condition. Intervals cluster by position and training seed rather than treating repeated games as independent. Test comparisons use the exact pretraining initialization and a fixed initial opponent. Random and actual-lesion controls are explicitly labeled. Both learners retain independent optimizer/parameter state. The inference batching implementation reproduces serial neural choices in regression tests.

```bash
PYTHONDONTWRITEBYTECODE=1 .venv/bin/python -m flies.chess.curriculum
# Repeat train/evaluate/curve commands with seeds 11, 23 and 37:
PYTHONDONTWRITEBYTECODE=1 .venv/bin/python -m flies.chess.learning train \
  --seed 11 --updates 60 --games 128 --output flies/chess/artifacts/learning/spatial-11
PYTHONDONTWRITEBYTECODE=1 .venv/bin/python -m flies.chess.learning evaluate \
  --run flies/chess/artifacts/learning/spatial-11 --updates 60 --repeats 4 \
  --output flies/chess/artifacts/learning/test-spatial-11
PYTHONDONTWRITEBYTECODE=1 .venv/bin/python -m flies.chess.measure_curve \
  --corpus flies/chess/artifacts/learning/corpus.json --run flies/chess/artifacts/learning/spatial-11
PYTHONDONTWRITEBYTECODE=1 .venv/bin/python -m flies.chess.study_report \
  --output flies/chess/artifacts/learning/pooled.json \
  flies/chess/artifacts/learning/test-spatial-11/evaluation.json \
  flies/chess/artifacts/learning/test-spatial-23/evaluation.json \
  flies/chess/artifacts/learning/test-spatial-37/evaluation.json
```

The study's `--resume .../resume.pt` preserves learners, optimizer moments, baselines, snapshots, settings and RNGs. Only increasing total `--updates` is permitted; other settings must match. Game count must be divisible by eight to balance roles, colors, and live/historical opponents. The original trainer remains available, with improved episode normalization and diagnostic metrics.

The default study encoding `chess-packed-v2-spatial` retains the complete original 64-float observation and uses reserved indices 53…63: own king file/rank (53…54), opposing king file/rank (55…56), first own queen file/rank (57…58), first own rook file/rank (59…60), selected source piece type/6 (61), and source-to-enemy-king signed file/rank displacement scaled from [-7,7] to [0,1] (62…63). Coordinates use `(coordinate+1)/8`, with zero for absence; queen/rook are the first in canonical square order. There is no engine score, mate label or strategic move ranking in the inputs. Every observation still enters through the original sensory projection.

A matched seed-11 `--encoding packed` ablation retains the original inputs; `--encoding stage-only-control` zeros everything except stage flags and is an explicitly exploratory diagnostic. These preserve the architecture and eight action groups. **Study checkpoints record their encoding and must use this feature's `load_policy`, self-play runner or study evaluator.** The unchanged shared server does not interpret encoding metadata, so copying a spatial/control checkpoint into the native world would supply mismatched observations. The normal `ChessFly` task retains its original packed input and previous checkpoint compatibility.

Rules-based mate labels curate the corpus and compute reporting/chance statistics; training never reads them. A regression test corrupts those labels and verifies identical learned weights. Rewards come only from actual sampled game outcomes. Optimization now reports gradient/update norms for every existing parameter family, separates empty trajectories, and measures only the candidate's action statistics. Unfinished games remain separately logged truncations.

Legacy self-play optimizer resumes now carry `algorithm="episode-reinforce-v2"`. Older optimizer/RNG bundles without that tag are rejected instead of silently continuing under the changed loss normalization. Their original-format brain weights still load with the original packed encoder. Resume exactness is tested within the same algorithm/settings.
