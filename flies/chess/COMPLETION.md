# Chess Fly completion and review record

Local implementation is complete and verified within contributor scope. The feature is ready for code review and an honest demonstration of connectome-controlled legal chess, training, activity and replay. It is **not** a strong chess player; full-graph GPU learning and shared-world two-brain ownership remain unapproved/unexecuted. All source changes are inside `flies/chess/`; branch `feat/chess-fly` is unchanged. No setup/download, commit, push, merge, remote command or GPU job was performed.

## Architecture and specification audit

`python-chess rules/state → canonical64-float sensory observation → unchanged core FlyBrain → eight fixed descending groups →4/5 masked coordinate choices → legality recheck → board.push`.

Each strategic player has a distinct model and optimizer. Fixed edges, signs, sensory indices and output groups are preserved. Existing core trainables (edge strength, bias, leak, sensory projection) remain the only trainables. There is no learned readout, engine-selected move or random fallback. Random legal players are explicitly labeled evaluation controls or the separate native single-fly demo.

| Specification / final directive requirement | Executed evidence | Status |
|---|---|---|
| One registered ChessFly, eight actions, valid task contract | `core.check chess` actual tiny graph PASS | Complete |
| All legal moves, checks, pins, castling, EP, four promotions, draw policies and automatic endings | Rules/encoding tests, exact prefix masks, PGN/FEN replay verification | Complete |
| Original12×64 information and current-player transformation | Lossless packed decoder, Black inverse/mirror tests; fixed obs_dim64 | Adapted: AGENTS64-float rule overrides literal768-float spec |
| Genuine independent FlyBrain decisions | Instrumented tiny/full forward calls, separate storage/optimizers, recorded neural stages | Complete |
| Complete two-brain games | Actual trained tiny models from opening:17/19 plies, official repetition draws, zero illegal attempts | Complete functionality; weak repetitive play |
| Self-play RL and stable checkpoints | Native trainer, frozen pool, study/experimental trainers; separate optimizers, clipping, reward/gradient tests | Complete |
| Reliable stop/resume | Initial update0 commits; atomic resume; exact continuation; no-op export repair; match SIGINT/full-move stop | Complete |
| No accidental run loss | Fresh match/training/evaluation launches reject existing run outputs; prior artifacts retained | Complete |
| Honest baselines and evaluation | Same encoding/temperature for basic untrained control; matched family tests,3 seeds, trained restriction, lesion/random/parent/entropy controls | Complete |
| Outcome/reward/statistics accuracy | Exact-fraction integer-v2 audit; no caps-as-draws; PGN rejects inconsistent final FEN/outcome; uncertainty clustered by family/seed | Complete |
| Recordings and neural activity | JSON/PGN, masks/probabilities, checkpoint/graph hashes, per-stage real mean/group rates | Complete; dense whole-brain rate histories intentionally not recorded by default |
| Functional local visualization | Feature-local HTTP launcher; native Safari board, recording URL, next-move replay and group-rate inspection; DOM autoplay/move list/error regressions | Complete |
| Shared 3D station contract | All8 branches,64 squares,24 actual moves, FEN/piece counts tested in Node DOM/Three harness | Logic complete; shared-world WebGL/owner integration remains external |
| Full graph architecture/entrypoints | Two134,181-neuron CPU brains, sparse backward/Adam isolation, checkpoints, activity and spatial forward | Complete compatibility probe; not full learning |
| Scientific learning improvement | Narrow historical mate-one gain; fresh modest board adaptation supported; added-training/LOO/mate-two/general-game gains unsupported | Honest negative/inconclusive findings retained |
| Contributor policy | No changes to core/server/web/scripts/other flies; no GPU/remote/push/merge | Complete within authorization |

AGENTS hackathon checklist items requiring GPU-trained-above-random full results and merge are **not claimed complete**: the user prohibited those actions, and scientific scaling gates also fail. This is the exact remaining external boundary, not hidden implementation work.

## Recovery and repairs

The interrupted learning study had already completed training/evaluation. Its models/datasets were preserved. `mechanism_gradient.py` now finishes validated terminal-only diagnostics, counts unique pair families correctly, uses float64 cosine reductions and distinguishes zero signals. Frozen-data reanalysis reproduces every aggregate exactly. The numerical reward fix changes no weights/Adam state in three complete paired repetitions.

Additional repairs found during project-wide verification:

- Initial checkpoints are committed before the first rollout in all trainers. Native resumes repair exports without requiring another update. Backward update requests and accidental fresh-run overwrites are rejected.
- Match CLI supports a graceful SIGINT stop at a full-move boundary; interrupted matches remain unfinished. Initial/live FENs are canonicalized. Random baseline initial events are correctly labeled.
- PGN export now verifies the declared final FEN and library outcome as well as every move; fabricated wins or mismatched endings are rejected.
- Basic evaluator's fresh untrained brain uses the candidate's encoding and sampling temperature, rather than confounding learned weights with a different observation interface.
- Control metadata distinguishes untrained/frozen/restricted/entropy baselines. Full CPU engineering checkpoints are labeled `ENGINEERING-PROBE`, not chess-trained agents.
- Viewer validates record structure, continuity and neural probabilities before replacing a valid replay. It supports recording URLs, PGN links, move lists, autoplay and all8 group activities. Browser inspection exposed unequal grid ranks; explicit equal row sizing fixes the layout.

## Final tests and exercised workflows

Final full regression after the last code edit:

```bash
PYTHONDONTWRITEBYTECODE=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 CHESS_REAL_GRAPH_TESTS=1 \
  .venv/bin/python -m unittest discover -s flies/chess/tests -v
PYTHONDONTWRITEBYTECODE=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 \
  .venv/bin/python -m core.check chess
```

**58 Python tests passed, zero skipped**, including actual FlyWire tiny integration. Core check **PASS**. Exact final timing/output are retained in `artifacts/completion-tests.log`, `artifacts/completion-core.log` and `completion_verification.json`. Node visualization regression **PASS**:64 squares,8 branches,24 actual moves, FEN/piece counts, replay/autoplay/move list and invalid-event rejection. Native Safari rendering/replay/activity was observed. Automated browser-console/shared-world WebGL QA is not claimed.

Actual CLI integration runs (all exit0 unless described as a running viewer):

```bash
# Actual native RL smoke; all moves from two brains or frozen FlyBrain snapshots.
.venv/bin/python -m flies.chess.train_selfplay --size tiny --updates 2 \
  --games-per-update 4 --max-plies 8 --curriculum mate_rook --seed 271 --snapshot-every 1 \
  --output flies/chess/artifacts/final-training
.venv/bin/python -m flies.chess.train_selfplay --size tiny --updates 3 \
  --games-per-update 4 --max-plies 8 --curriculum mate_rook --seed 271 --snapshot-every 1 \
  --output flies/chess/artifacts/final-training --resume flies/chess/artifacts/final-training/resume.pt

# All five basic evaluator conditions,8 games each.
.venv/bin/python -m flies.chess.evaluate \
  --checkpoint flies/chess/artifacts/mechanism/loo-11/checkpoint_0048_0.pt \
  --opponent-checkpoint flies/chess/artifacts/mechanism/loo-11/checkpoint_0048_1.pt \
  --size tiny --games 8 --max-plies 24 --output flies/chess/artifacts/final-cli-evaluation

# Real completed games, live data, replay and activity; localhost listener tested.
.venv/bin/python -m flies.chess.demo \
  --white-checkpoint flies/chess/artifacts/mechanism/loo-11/checkpoint_0048_0.pt \
  --black-checkpoint flies/chess/artifacts/mechanism/loo-11/checkpoint_0048_1.pt \
  --games 2 --max-plies 512 --delay .01 --output flies/chess/artifacts/final-demo --port 8766

# Full CPU graph compatibility, including additional spatial-encoding forward.
.venv/bin/python -m flies.chess.full_smoke --output flies/chess/artifacts/full-cpu-final
```

These runs used `PYTHONDONTWRITEBYTECODE=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1`. Reusing their completed output directories now deliberately fails for a fresh run; choose new paths for new work. The native smoke produced no wins and is not performance evidence. The40-game basic evaluator produced no wins in any condition; draws/truncations are separate and all illegal counts are0. The full CPU probe used entropy-only updates because the two played opening plies had zero outcome reward. Its10.873s total/1,071,874,048-byte peak RSS are process measurements, not GPU estimates.

## Research outcome

See [MECHANISM_RESULTS.md](MECHANISM_RESULTS.md) for the protocol, raw evidence, intervals and exact reproduction. Integer-v2 correctly removes tiny false rewards, but three complete repetitions have identical model/optimizer states to legacy arithmetic. The controlled main study adds73,728 games, plus9,216 numerical replications and6,144 pilot games. Ten-condition fresh evaluation adds11,520 sampled games alongside exact probability integration on128 matched pairs. Replay audits are duplicates, not independent samples.

Final matched adaptation is **+2.063pp [0.716,3.622]**, nominal98.75%. The parent already has **+1.347pp [0.446,2.337]**. New gain **+0.716pp [−0.240,1.761]** is inconclusive; LOO-minus-EMA **−0.094pp [−0.448,0.226]** is inconclusive. Mate-two final wins4/1,152 versus parent10/1,152; paired gain **−0.521pp [−1.823,0.434]**. The unvisited subset is also inconclusive. Historical warm-start exposure is incompletely logged. There is no full-game strength or Elo claim.

This is sufficient evidence of a real limitation at the tested methods/budgets. Further tuning against the opened tests would invalidate generalization claims. Local implementation work is finished; a scientifically new learning intervention needs training/validation development and another locked test, not an unapproved larger graph run.

## Files and external boundaries

The complete feature inventory and source/data hashes are in [completion_verification.json](completion_verification.json). Core implementation: `task.py`, `chess_env.py`, `selfplay.py`, `train_selfplay.py`, `evaluate.py`, `storage.py`, `station.js`, `viewer.html`, `demo.py`. Learning/research: `batched.py`, `curriculum.py`, `learning.py`, `measure_curve.py`, `study_report.py`, `research_positions.py`, `extended_curriculum.py`, `research_train.py`, `matched_eval.py`, `generalization.py`, `gradient_diagnostics.py`, `research_report.py`, `benchmark.py`, `reward_validation.py`, `mechanism_{data,train,eval,audit,gradient,report}.py`, `full_smoke.py`. Tests are under `tests/`; frozen datasets/evidence/results and protocol/audit documents are feature-local. Large research/checkpoint artifacts stay under ignored `artifacts/` and remain available in this workspace.

Remaining limits are explicit: weak task-specific policies; no supported mate-two/endgame/general-game gain; only3 warm-start seeds and incomplete historical exposure logs; core rate-network approximation/engineered sensory/action interface; no full-graph learning or GPU measurements; shared server cannot own two brains or honor research encodings without an approved owner change. The feature-local viewer solves usable two-brain visualization without those shared edits. [SCALING.md](SCALING.md) preserves the unexecuted owner-approved queue proposal; do not submit it without permission.

## Subsequent cinematic integration

The later [cinematic arena](CINEMATIC.md) replaces the legacy 2D viewer as the primary user experience. It reuses `web/lib/fly.js` and the original `web/matrix.js` unchanged, adds feature-local animation-gated live inference and preserves all training/research artifacts. See [CINEMATIC_REUSE.md](CINEMATIC_REUSE.md) and the cinematic section of [AUDIT.md](AUDIT.md) for the later test counts, real WebGL evidence, completed-game recording and explicit integration limitations. The earlier research conclusions and full-graph CPU evidence above remain historical and unchanged.
