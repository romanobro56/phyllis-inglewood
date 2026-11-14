# Chess Fly implementation audit

**Final local completion:** see [COMPLETION.md](COMPLETION.md) for the requirement-by-requirement audit, tested commands, repairs and external limits. Final regression:58 Python tests passed with actual tiny integration, core check PASS, Node viewer/station logic PASS, native Safari replay/activity verified. Two trained tiny brains completed17/19-ply repetition draws. Two actual full134,181-neuron brains passed CPU initialization/forward/backward/checkpoint/activity compatibility; no full learning or GPU run is claimed. The fresh mechanism study supports modest matched mate-one adaptation (+2.06pp, nominal98.75% interval +0.72,+3.62), also present in the parent; added-learning/mate-two/general-game improvements remain unsupported. [MECHANISM_RESULTS.md](MECHANISM_RESULTS.md) and [mechanism_results.json](mechanism_results.json) contain the verified latest evidence. Historical sections below retain their original results and limitations.

## Scope and architecture

Implemented in the existing `feat/chess-fly` checkout. AGENTS.md, CLAUDE.md and the complete chess specification were read before implementation. The scaffold was reused; no setup, graph generation, push, merge, commit or GPU/remote command was performed. Only source/artifact files inside `flies/chess/` were created or edited. The feature dependency was installed in the existing virtual environment with approval. Existing untracked root AGENTS.md and CLAUDE.md were left untouched.

Architecture: python-chess rules/state → 64-float canonical sensory observation → unchanged `core.brain.FlyBrain` → existing eight descending groups → four/five masked coordinate selections → legality recheck → board.push. No learned output head, connection changes, strategic engine, ranked legal-move features, or random-move fallback. Black uses the self-inverse rank flip. Self-play instantiates distinct models and optimizers, with a bounded frozen opponent pool.

The native FlyTask is separately labeled **FlyBrain White vs random legal Black**. The genuine match runner/trainer uses two FlyBrains; evaluation's optional random legal player is explicitly labeled as a baseline. The real lesion control calls `FlyBrain.forward(lesion=True)` rather than replacing inference with randomness.

## Requirement-by-requirement acceptance

| Specification acceptance check | Implementation and executed evidence | Status |
|---|---|---|
| 1. Exactly one registered ChessFly, eight actions, full task contract; core check | `task.py`; actual registry/interface and real 5,000-neuron FlyBrain check, backward and lesion test | PASS |
| 2. Legal start/check/checkmate/stalemate/castling/EP/promotions/pins/one-move/repetition/50/75-move rules | `test_env.py`: all initial moves, both colors and both castles, both EP colors, all eight colored promotions, pin rejection, only-one-escape position, optional claim settings and automatic endings | PASS |
| 3. Exact legal continuation masks and replay agreement | Every stage uses filtered legal moves; tests compare masks exactly, reject masked/out-of-range choices, compare UCI/FEN and replay PGNs | PASS |
| 4. Deterministic fixed-seed replay | Identical models and seed reproduce move sequence; CPU uninterrupted vs resumed training produces bit-identical complete brain states | PASS on tested CPU/software; other devices unverified |
| 5. Two separate live brains with no strategic substitute | Instrumented 6-ply game: White forward called 12 times, Black's separate forward called 12 times; real tiny integration also instruments both. Independent storage and update isolation checked | PASS |
| 6. Tiny backward, checkpoints, resume, graph protection, gradient isolation | Actual tiny forward/backward; distinct storage; nonzero edge gradients; unchanged opponent; strict graph/dimension/encoding/fixed-buffer validation; checkpoint round-trip; interrupted-write test; exact resume | PASS |
| 7. Random/untrained/trained/lesioned/head-to-head controls; honest graph labels | Five comparisons executed, paired seeds/colors and held-out positions; JSON includes graph/state/checkpoint hashes and explicit `synthetic`/preset/label | PASS for tiny; no full-graph or competence claim |
| 8. Bounded long games, memory and atomic artifacts | 160-ply-cap neural match test, 300-ply-cap stage walk, bounded 4–5 choices/move, chunked backward, bounded snapshot pool, atomic checkpoint interruption test | PASS for tested bounds; full-scale GPU memory unmeasured |
| 9. No unit-test downloads | Default fixture patches graph loading only, executes real core FlyBrain on an explicitly synthetic 176-neuron fixture. Optional integration uses existing actual data | PASS |
| 10. Station interface and actual move visualization | Real station module executed with DOM/Three stubs: 64 squares, 32 starting pieces, all eight actions, 24 recorded moves, exact FEN/piece counts, invalid-event rejection; viewer replay tested | Logic PASS; browser/WebGL and one-minute console QA NOT VERIFIED |

Additional implementation requirements:

- Reinforcement learning: separate two-policy REINFORCE trainer, summed stage log probabilities per move, own-move terminal returns, discounting, prior-data EMA baseline, entropy, clipping, Adam, CPU historical snapshots. No externally supervised teacher.
- Curriculum: KQK, KQRK mate positions, pawn endings and standard movement/start positions. Positions are configured environment starts, never a move oracle.
- Checkpoints: both learners, both optimizers, Python/NumPy/Torch/CUDA RNG states, baseline, opponent pool, configuration and update count. `resume.pt` is the atomic commit point; per-brain exports use the existing core checkpoint layout plus encoding metadata.
- Recordings: individual stage masks/probabilities/actions/side/FEN, full UCI/SAN and before/after FEN, hashes, seed, outcome, elapsed time and optional activity. PGN replay checks run before publishing JSON/PGN artifacts.
- Evaluation: W/L/D, unfinished truncations, illegal attempts, mean plies, per-stage masked legality accuracy/selected probability, training reward and held-out frozen-opponent comparisons. Masked legality accuracy is not expert strategic accuracy.
- Visualization: caption-based 3D random-opponent world demo; standalone JSON playback and polling of atomic `live.json` for genuine two-brain matches. Shared-world two-brain integration is pending owner approval; README defines a minimal future event contract. No shared infrastructure was changed.
- Observation adaptation: AGENTS.md's 64-float limit takes precedence over the spec's literal 768-float planes. Lossless pair-of-square packing encodes all pieces; `decode_planes` reconstructs the 12×64 planes. Stage, partial choice, castling, EP, check, clock and history flags have documented fixed positions. This is an explicit deviation from the literal input representation, preserving the authoritative contributor policy.

## Exact executed checks and results

Environment: macOS, Python 3.9.6, PyTorch 2.8.0, NumPy 2.0.2, chess 1.11.2 / python-chess 1.999. CPU graph: **actual FlyWire tiny**, 5,000 neurons, `synthetic=False`, graph hash `cc2e5ce35f`. CLI runners set PyTorch threads to 1.

1. Dependency install: `.venv/bin/python -m pip install -r flies/chess/requirements.txt` — SUCCESS. Initial sandbox network denial was retried with approval. Both final pinned versions are installed.
2. `PYTHONDONTWRITEBYTECODE=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 .venv/bin/python -m core.check chess` — **PASS**, exit 0. Final output:

   ```text
   interface ok (240 random steps, 295 us/step)
   random baseline: avg return 0.010
   brain ok: 5,000 neurons, mean firing rate 0.170
   lesion test ok: intact spread 0.0141, lesioned spread 0.0e+00
   quick train ok (2 updates)
   PASS
   chess:PASS
   ```

3. `PYTHONDONTWRITEBYTECODE=1 .venv/bin/python -m unittest discover -s flies/chess/tests -v` — **21 discovered, 20 passed, 1 optional integration skipped**, exit 0, 1.125 s.
4. `PYTHONDONTWRITEBYTECODE=1 CHESS_REAL_GRAPH_TESTS=1 .venv/bin/python -m unittest discover -s flies/chess/tests -v` — **21 passed, zero skipped**, exit 0, 1.417 s. This is an actually executed FlyWire tiny integration, not a mocked claim.
5. `PYTHONDONTWRITEBYTECODE=1 .venv/bin/python -m flies.chess.selfplay --size tiny --games 2 --seed 42 --max-plies 24 --activity --live --output flies/chess/artifacts/matches` — exit 0; UNTRAINED actual tiny vs UNTRAINED actual tiny; both matches 24 plies, unfinished `*`, zero illegal attempts; 0.870570917 s and 0.842697166 s (about 28 plies/s, roughly 112 neural stages/s). Live JSON, PGN and full stage logs produced.
6. `PYTHONDONTWRITEBYTECODE=1 .venv/bin/python -m flies.chess.train_selfplay --size tiny --updates 2 --games-per-update 4 --max-plies 12 --curriculum mate --snapshot-every 1 --output flies/chess/artifacts/tiny-training` — exit 0, two updates and frozen-opponent evaluation, valid checkpoints. These sparse-reward games had no decisive rewards; only entropy gradients provided updates.
7. Same training command with `--updates 3 --resume flies/chess/artifacts/tiny-training/resume.pt` — exit 0; resumed update 3 completed. Unit tests additionally verify *exact* model-state equivalence across a split run.
8. `PYTHONDONTWRITEBYTECODE=1 .venv/bin/python -m flies.chess.train_selfplay --size tiny --updates 2 --games-per-update 16 --max-plies 8 --curriculum mate --seed 9 --snapshot-every 1 --output flies/chess/artifacts/tiny-reward-smoke` — exit 0; valid pipeline but no decisive rewards. This motivated a separate KQRK curriculum smoke run, not a claim of performance.
9. `PYTHONDONTWRITEBYTECODE=1 .venv/bin/python -m flies.chess.train_selfplay --size tiny --updates 2 --games-per-update 16 --max-plies 8 --curriculum mate_rook --seed 9 --snapshot-every 1 --output flies/chess/artifacts/tiny-rook-smoke` — exit 0, **32 games, two decisive outcomes, 27 actual draws, 3 truncations, zero illegal attempts**. Training plus snapshot evaluation/serialization took 3.943344709 s. Player 0 gradients: 0.30389598, 0.87609118; player 1: 0.00454915, 0.00223212. Player 0 mean sampled terminal rewards: 0.0625 and 0.125. Confirms genuine outcome-driven policy updates, not improved chess strength.
10. `PYTHONDONTWRITEBYTECODE=1 .venv/bin/python -m flies.chess.evaluate --checkpoint flies/chess/artifacts/tiny-training/brain_0.pt --opponent-checkpoint flies/chess/artifacts/tiny-training/brain_1.pt --size tiny --games 4 --max-plies 24 --output flies/chess/artifacts/evaluation` — exit 0; all five baselines complete.
11. Same evaluator with checkpoints under `tiny-rook-smoke/` and `--output flies/chess/artifacts/evaluation-rook` — exit 0; all five comparisons, results below. Evaluation uses default held-out seed 100042 and alternating colors.
12. `PYTHONDONTWRITEBYTECODE=1 .venv/bin/python -m flies.chess.selfplay --white-checkpoint flies/chess/artifacts/tiny-rook-smoke/brain_0.pt --black-checkpoint flies/chess/artifacts/tiny-rook-smoke/brain_1.pt --size tiny --games 2 --seed 200042 --max-plies 40 --output flies/chess/artifacts/trained-matches` — exit 0; two TRAINED actual tiny models; 40 plies each, zero illegal attempts, both unfinished `*`; 1.404974542 s and 1.400698083 s.
13. Existing server started with `PYTHONDONTWRITEBYTECODE=1 OMP_NUM_THREADS=1 .venv/bin/python -m server.serve --only chess --host 127.0.0.1` (local networking approved). Port 8000 was already occupied; the shared static-server thread reported that collision. Our independent websocket on 8765 worked. No pre-existing process was stopped. `PYTHONDONTWRITEBYTECODE=1 .venv/bin/python -m flies.chess.tests.verify_live --seconds 60` — exit 0: **60.002759791 s, 938 decisions, 468 legal plies, 11 resets**, exact FEN continuity. The test websocket server was subsequently stopped using Ctrl-C. This validates the real protocol, not rendering.
14. JavaScript module/logic test executed through the available Node REPL: `await import('file:///Users/arabeea/Documents/Projects/phyllis-inglewood/flies/chess/tests/visual_smoke.mjs?final=1')` — **PASS**. 64 station squares, all eight slots, 24 recorded moves, exact FEN/piece counts, viewer start/final replay, rejected invalid events. Equivalent reproduction on a machine with Node: `node --experimental-vm-modules flies/chess/tests/visual_smoke.mjs`. Standalone `node` was not installed on the shell PATH here; the Node REPL's VM runtime executed the test.

An initial unit fixture incorrectly claimed one escape where the position had two; corrected the fixture to guard the rook and leave only Kg1. Final tests above all pass. An initial patch application was rejected before modifying files; corrected the edit operation. Neither issue required a shared-code change.

## Held-out evaluation results

KQRK-trained checkpoint, four games per comparison, max 24 plies, seed 100042. These sample sizes and horizons are deliberately small smoke tests.

| Candidate vs opponent | Wins | Losses | Draws | Truncated | Average plies | Illegal attempts |
|---|---:|---:|---:|---:|---:|---:|
| Trained vs random legal | 0 | 0 | 1 | 3 | 18.25 | 0 |
| Untrained vs random legal | 0 | 0 | 1 | 3 | 18.25 | 0 |
| Lesioned trained vs random legal | 0 | 0 | 1 | 3 | 18.25 | 0 |
| Trained vs untrained | 0 | 0 | 2 | 2 | 12.5 | 0 |
| Trained vs frozen other checkpoint | 0 | 0 | 2 | 2 | 12.5 | 0 |

Every observed stage's masked legality accuracy was 1.0. **No improvement over random is established**, and no Elo estimate is justified. Draws were library-adjudicated; none of the caps was counted as a draw.

KQRK checkpoint SHA-256s:

- `brain_0.pt`: `07b4e1ad6e7661f54b54e8b74c963c4d33d035287688ae89e5e66418d22aed4a`
- `brain_1.pt`: `65c91312c953bd22c1f0618d08f9ea05fb3c335118274a5c039327d260bda52b`

## Implemented files

- `task.py` — native FlyTask contract and honest random-opponent demo; disclosed material shaping.
- `chess_env.py` — canonical encoder/decoder, exact hierarchy/masks, legality/outcomes/caps.
- `selfplay.py` — separate brains, inference, live stream, deterministic sampling, detailed records, replay-verified PGNs and CLI.
- `train_selfplay.py` — two independent optimizers, REINFORCE, curricula, snapshot pool, held-out evaluation, atomic save/resume.
- `evaluate.py` — five labeled baseline/head-to-head comparisons and metrics.
- `storage.py` — atomic artifacts, hashes, strict checkpoint compatibility and local output restriction.
- `station.js` — actual-FEN-driven 3D demo animation for every action.
- `viewer.html` — game playback and live two-brain recording display with provenance/activity.
- `requirements.txt` — feature-only pinned chess dependencies.
- `tests/test_env.py`, `test_encoding.py`, `test_selfplay.py`, `test_evaluation.py`, `support.py` — unit and optional actual-graph integration tests.
- `tests/visual_smoke.mjs`, `tests/verify_live.py` — JS behavior and actual-server protocol integration.
- `.gitignore` — feature-generated artifacts and bytecode.
- `README.md`, `AUDIT.md` — architecture, commands, methodology, attribution, audit and limitations.

`chess_fly_spec.md` is unchanged. All generated checkpoints, PGNs, JSONs and live streams are retained locally under ignored `flies/chess/artifacts/`. No shared runs directory was written.

## Remaining limitations and risks

1. No full/small graph or GPU experiment was run. Full-graph memory, speed and comparative scores remain unverified, and GPU submission needs the user's approval. No shared remote command was used.
2. Tiny-graph smoke training demonstrates legal decisions and real gradient updates, not useful playing strength or trained-above-random acceptance. Sparse terminal rewards, engineered coordinate factorization and packed inputs make optimization difficult. A longer pre-registered experiment is needed.
3. Browser automation did not navigate reliably to the local world and the browser access host was unavailable. DOM/Three behavior and the one-minute *real server protocol* pass; actual WebGL render correctness and one-minute browser-console QA remain unverified. Static port 8000 was already in use during the server test.
4. Live two-brain play is available in the feature-local viewer. Shared multi-fly world two-agent ownership/events and opponent brain inspection require server-owner approval and integration; no approval is implied. Native captions include verbose JSON because shared label rendering is unchanged.
5. Each checkpoint/result file is atomic. `resume.pt` is the authoritative training transaction; a crash between derived exports can leave different export versions. Resume and complete the next update to regenerate them. Disk use grows with explicitly requested recordings/updates; game trajectory memory, gradient chunks and snapshot pool are bounded.
6. Repetition adjudication keeps full board history, while the neural input contains repetition flags rather than the whole history. Board packing is lossless but does not give the sensory projection one-hot planes. These are documented engineered compromises.
7. The model is the repository's rate-based FlyWire approximation, not a validated biological/spiking chess brain. Reduced graphs, learned bias/leak/sensory projections, action grouping, legality masking, reward shaping in the native demo, and the optimizer limit biological interpretation. Synthetic unit fixtures are never presented as real flies.

Demo pitch: **Two fruit-fly connectomes argue over a chessboard, one descending-neuron coordinate at a time.**

## Learning investigation follow-up

This follow-up investigates actual learning, rather than replacing the original smoke-test history. All source and artifact changes remain inside `flies/chess/`. No GPU job, remote command, commit or push was performed. Existing real `core.brain.FlyBrain` is unchanged: 5,000-neuron FlyWire tiny graph, `synthetic=False`, graph hash `cc2e5ce35f`. Both strategic players call independent FlyBrain models, with independent parameter and optimizer storage. Source connectivity, signs, sensory indices and fixed descending groups match initialization exactly in every trained model inspected, including the control run.

### Diagnosis

1. The original color-paired curriculum supplied only **one canonical position per curriculum**. KQK has 1 mate and 7 stalemates among 23 legal first moves; KQRK has 5 mates and **24 stalemates among 37 moves**. Both learners therefore often received no useful outcome signal. A two-update run with zero decisive rewards mainly optimized entropy. Pinning/masking and reward signs were correct; no reversed winner/loser credit or detached policy gradient was found.
2. Four evaluation games, an often-unfinished opening position, and one easy endgame cannot resolve performance changes. Many truncations obscured the question; counting only completed games would further bias the result. Masked legality accuracy is not chess-playing accuracy.
3. Uniform hierarchical slot sampling is **not** uniform legal-move sampling. The new test corpus has analytical first-move mate probabilities of 3.9821% under uniform coordinate branches and 5.0122% under uniform legal moves. These rules-only chance calculations are not policy inputs. The sampled random baseline was 5.6641%.
4. Packing two categorical squares into one scalar preserves information but does not make that information easy for a sensory projection to decode. Reusing eight groups for all stages also creates opportunities to learn stage/action priors. Both hypotheses were measured, not assumed resolved.
5. Fixed several implementation/measurement issues, each covered by regression tests: empty episodes contaminated reward averages; empty updates retained stale gradients; total-move normalization caused batch-length-dependent update scaling; candidate stage metrics included the opponent; a changing historical validation opponent confounded checkpoint W/L/D curves; and a row-shift shuffle mostly preserved identical boards when adjacent rows repeated the same position. New batched games also reject same-model or invalid-index pairings before they can mislabel self-play or deadlock.

### Implemented changes

- `curriculum.py`: reproducible rules-curated KQK/KQRK positions with a legal mate in one and at most 15% immediate stalemate moves. Full color/dihedral symmetry families are split before training. Train: 256 positions (69 KQK / 187 KQRK); validation: 64 (20/44); test: 128 (40/88). Corpus SHA-256: `79f2b27708f8f96ed5c2c1629aa3f95e47baea2ed95f8ad7ce79c5a4f0e93965`. Independently regenerating the corpus reproduced it exactly.
- `batched.py`: throughput batching of independent legal games using the same actual FlyBrain.forward, exact masks and per-game RNG streams. No engine or random substitute in training. Regression tests compare serial and batched move streams and instrument both models.
- `learning.py`: outcome-based, two-learner REINFORCE study with balanced roles/colors, live and frozen opponents, prior-update attacker/defender EMA baselines, terminal own-side rewards, fixed four-ply cap, independent Adam optimizers and resumable checkpoints. Historical snapshots are independent models; no learned value/output head was introduced. Training never reads corpus move labels, verified by corrupting labels and obtaining identical trained states.
- `train_selfplay.py`: episode normalization, empty-gradient cleanup, correct nonempty reward averages and per-parameter gradient/update diagnostics; the original trainer remains supported. Full moves use the sum of their hierarchical log probabilities. Forced stages contribute exactly zero. A positive-reward regression verifies that the selected probability increases; actual FlyBrain-sampled mate outcomes verify winner/loser attribution.
- `chess_env.py`, `storage.py`, `selfplay.py`: optional spatial encoding in 11 formerly reserved input floats, retaining the lossless packed board and 64-float limit; explicit encoding/temperature metadata and validation; matching inference/recomputed training probabilities. The normal native task remains on its original compatible input. No connectivity or output groups changed.
- `measure_curve.py`: remeasure every saved checkpoint against fixed untouched initial opponents. The initial study's changing pool opponent affected secondary W/L/D interpretation; first-move mate probability was unchanged. Fixed-opponent curves were rerun for all four planned runs, and future study runs use the fixed opponents directly.
- `study_report.py`: paired position-cluster and two-way training-seed/position bootstrap summaries, with nominal 98.75% intervals for four planned contrasts (Bonferroni familywise 5%). Duplicate report/state inputs are rejected. Three seeds limit confidence about initialization variation.
- `tests/test_learning.py`: regression coverage for all listed issues, geometry and Black transforms, encoder compatibility, stage-only control isolation, real forward calls, gradient flow, independent optimizer behavior, exact resume, split integrity, statistics and absence of teacher-label dependence.
- `EXPERIMENTS.md`, `learning_results.json`, README and this audit: frozen protocol, machine-readable verified results, commands and limitations.

### Protocol and executed experiments

The protocol in `EXPERIMENTS.md` was fixed after a two-update **validation-only** pilot and before the main 60-update runs/test evaluations. Primary endpoint: **first-move library-verified mate** on the held-out tactical distribution. It does not measure full chess competence.

Main seeds 11, 23, 37 each trained two independent brains for 60 updates ×128 genuine two-brain games, gamma=1, Adam lr=.003, entropy=.003, cap=4 plies, with balanced actual colors. That is **23,040 main training games**, plus 7,680 games for a matched seed-11 original-input ablation. A later exploratory stage-only run adds another 7,680 training games. No teacher loss, heuristic move choice, dense head, random training opponent, engine, or synthetic graph was used. Rewards are terminal +1/-1/0; capped games receive zero curriculum outcome reward and remain official result `*`, not chess draws.

Validation measured checkpoints 0/10/20/30/40/50/60; checkpoint 60 was fixed in advance and was **not** selected as the best checkpoint. Final testing used all 128 test families ×4 color-balanced sampled games per learner, **512 games per learner/condition**. Main five-control testing totals **15,360 games**, with another 5,120 for the input ablation. Initial comparisons use that learner's exact saved pretraining state; opponents are separate fixed initial states. Random legal candidate, actual lesioned trained candidate, and shuffled-board candidate are explicit controls.

An initial shuffle used adjacent row shifts, which preserved most repeated boards. Its reports are archived as `evaluation.invalid-shuffle-control.json` and excluded from conclusions. A tested board-identity derangement corrected the null; all fixed models were reevaluated, without retraining, changing hyperparameters or selecting checkpoints. Every corrected test run has **zero same-board donors across its 2,048 first-move neural stages per learner**. Stage and actual partial choices are preserved while board features are shuffled. Consequently this null probes the incremental board features, not all information supplied by legal masks and prior coordinate choices.

### Verified held-out results

Rates pool six learners across three training seeds, with **3,072 games per condition**. Repeated games are not treated as 3,072 independent positions: uncertainty resamples 128 position families and three training seeds, averaging learners within a seed. Common evaluation seeds deliberately reuse the random first-move draws across learners/runs; that random baseline has 512 unique sampled first-move draws, not 3,072 independent draws.

| Condition | First-move mates / games | Rate | Nominal 95% two-way bootstrap interval |
|---|---:|---:|---:|
| Trained | 553 / 3,072 | **18.00%** | 13.87–22.43% |
| Exact untouched initialization | 108 / 3,072 | 3.52% | 2.02–5.21% |
| Random legal moves | 174 / 3,072 | 5.66% | 3.91–7.62% |
| Actual lesioned trained brain | 194 / 3,072 | 6.32% | 4.20–8.63% |
| Trained, shuffled board features | 499 / 3,072 | 16.24% | 12.47–20.15% |

| Trained minus control | Difference | Nominal simultaneous 98.75% interval |
|---|---:|---:|
| Initialization | +14.49 percentage points | +9.20 to +20.41 |
| Random legal | **+12.34 points** | **+6.71 to +18.55** |
| Lesioned | +11.69 points | +7.01 to +16.86 |
| Shuffled board | +1.76 points | **−0.29 to +4.69** |

Training-seed means (averaging that seed's two learners): 17.68%, 19.34%, 16.99%; respective untouched initial means: 3.91%, 3.42%, 3.22%. The trained policy improves this **specific tactical success metric** over initialization and random legal play in the tested experiment. It also exceeds the actual lesion control. **Dependence on the board-feature input is not established**, because the intact-minus-shuffled interval includes zero. No biological chess intelligence, general chess competence or Elo improvement is claimed.

Secondary complete/capped outcomes, from the candidate's side:

| Condition | Wins within cap | Losses | Actual draws | Unfinished truncations | Illegal attempts |
|---|---:|---:|---:|---:|---:|
| Trained | 765 | 0 | 128 | 2,179 | 0 |
| Initialization | 154 | 0 | 209 | 2,709 | 0 |
| Random legal | 265 | 0 | 236 | 2,571 | 0 |
| Lesioned | 294 | 0 | 141 | 2,637 | 0 |
| Shuffled board | 699 | 0 | 131 | 2,242 | 0 |

These are prewinning endgames with a four-ply compute cap. Do not reinterpret completed-game fractions as a normal chess win rate. Most games remain unfinished; no timeout was manufactured into a draw. Descriptive first-move rates by piece set: KQK trained 14.90% vs random 3.13%; KQRK trained 19.41% vs random 6.82%. These subgroup figures are exploratory, not separately corrected confirmatory tests.

### Learning across checkpoints and encoding

Mean validation first-move mate rate over the six main learners, 64 families ×2 games/learner, against fixed initial opponents:

| Update | 0 | 10 | 20 | 30 | 40 | 50 | 60 |
|---|---:|---:|---:|---:|---:|---:|---:|
| Mate rate | 2.99% | 4.95% | 9.11% | 11.85% | 13.41% | 13.67% | 14.19% |

Individual learners are not monotonic; for example seed 11's second learner ends below its update-50 validation score. Reporting the fixed final checkpoint avoids selecting a favorable peak.

The original packed input **can learn** under this expanded curriculum/protocol: seed-11 final held-out mean is 15.43%, compared with 17.68% for the spatial input from the same initialization/data schedule. Their paired position-bootstrap difference is +2.25 points (95% interval +0.59 to +4.10). This is a **one-seed exploratory encoding ablation**, not reliable evidence of a generally superior encoder or an isolated explanation for all improvement. The coordinate factorization is workable for this restricted task; its masks and partial choices carry substantial task structure.

After seeing the first test controls, an explicitly exploratory stage-only seed-11 run retained only stage flags and zeroed board, metadata and partial choices. The actual legal masks remained state dependent. It reached **12.01%** (learner rates 13.48%, 10.55%), versus 17.68% for the corresponding full-input seed. The paired descriptive difference is +5.66 points (95% interval +2.93 to +8.59). This follow-up is outside the original four planned contrasts. Together with the 16.24% shuffled-board result, it shows that legal masking, stage-conditioned preferences and partial coordinates can explain substantial success without demonstrating board understanding. It is not a replacement policy for the normal chess task.

### Gradients, optimizer behavior and architectural integrity

Both learners update their own Adam states; frozen opponents never have optimizers. Empty trajectories produce no update and leave no stale gradients. Selected action probabilities increase under positive reward in a direct directional regression. Winner reward is +1, the other side is −1, and draws/caps are zero; own earlier completed moves receive the outcome through their own joint stage likelihoods. Forced single-action selections have zero log-probability gradient. No reversed credit or detached backward path was found.

Every main run has 7,680 training games and zero illegal attempts. Official mates observed during training: seeds 11/23/37 = 1,265 / 1,477 / 1,312. Mean clipped gradient norms by parameter family:

| Seed | Edge strengths | Sensory projection | Neuron bias | Leak |
|---|---:|---:|---:|---:|
| 11 | 0.01321 | 0.06663 | 0.13208 | 0.00344 |
| 23 | 0.01326 | 0.06697 | 0.12822 | 0.00387 |
| 37 | 0.01031 | 0.05299 | 0.10512 | 0.00290 |

Existing core trainability (edge strengths, bias, leak, sensory projection) was retained; no trainable readout was added. Parameter update norms are recorded per update. Exact initial/final comparisons of `edge_index`, transpose buffers, `sign`, `input_idx` and `groups` all passed for every main, encoding-ablation and stage-only learner. Checkpoint/resume regression reproduces final weights bit-for-bit on the tested CPU. Full-graph/GPU memory, speed and learning remain untested.

### Exact commands and verification

Executed from the real checkout, using existing Python 3.9.6 / PyTorch 2.8.0 / NumPy 2.0.2 / chess 1.11.2 environment:

```bash
PYTHONDONTWRITEBYTECODE=1 .venv/bin/python -m flies.chess.curriculum --output flies/chess/artifacts/learning/reproduced-corpus.json
PYTHONDONTWRITEBYTECODE=1 .venv/bin/python -m flies.chess.learning train --updates 60 --games 128 --seed 11 --output flies/chess/artifacts/learning/spatial-11
PYTHONDONTWRITEBYTECODE=1 .venv/bin/python -m flies.chess.learning train --updates 60 --games 128 --seed 23 --output flies/chess/artifacts/learning/spatial-23
PYTHONDONTWRITEBYTECODE=1 .venv/bin/python -m flies.chess.learning train --updates 60 --games 128 --seed 37 --output flies/chess/artifacts/learning/spatial-37
PYTHONDONTWRITEBYTECODE=1 .venv/bin/python -m flies.chess.learning train --updates 60 --games 128 --seed 11 --encoding packed --output flies/chess/artifacts/learning/packed-11
PYTHONDONTWRITEBYTECODE=1 .venv/bin/python -m flies.chess.learning train --updates 60 --games 128 --seed 11 --encoding stage-only-control --output flies/chess/artifacts/learning/stage-only-11
```

All exited 0. The original corpus was generated by the identical `make_corpus()` function; CLI regeneration compared equal, including its full SHA. Main/packed observed wall times were 236.37 / 233.47 / 237.97 / 242.11 seconds; stage-only 573.51 seconds. These include process scheduling, serialization and console I/O during concurrent local work, not a controlled hardware speed benchmark.

For each run name `spatial-11`, `spatial-23`, `spatial-37`, `packed-11`, and `stage-only-11`, executed:

```bash
PYTHONDONTWRITEBYTECODE=1 .venv/bin/python -m flies.chess.learning evaluate --run flies/chess/artifacts/learning/<run> --updates 60 --repeats 4 --output flies/chess/artifacts/learning/test-<run>
```

All exited 0. Planned four runs have five conditions (20,480 corrected evaluation games total); stage-only has four conditions because shuffling an absent board is meaningless (4,096 more exploratory games). Invalid-shuffle earlier evaluations are excluded. Main/packed checkpoint curves were independently remeasured using:

```bash
PYTHONDONTWRITEBYTECODE=1 .venv/bin/python -m flies.chess.measure_curve --corpus flies/chess/artifacts/learning/corpus.json --run flies/chess/artifacts/learning/<run>
PYTHONDONTWRITEBYTECODE=1 .venv/bin/python -m flies.chess.study_report --output flies/chess/artifacts/learning/pooled.json flies/chess/artifacts/learning/test-spatial-11/evaluation.json flies/chess/artifacts/learning/test-spatial-23/evaluation.json flies/chess/artifacts/learning/test-spatial-37/evaluation.json
```

All exited 0. Full test command after implementation:

```bash
PYTHONDONTWRITEBYTECODE=1 CHESS_REAL_GRAPH_TESTS=1 .venv/bin/python -m unittest discover -s flies/chess/tests -v
```

**34 passed, no skips, exit 0, 9.926 seconds**, including actual FlyWire tiny forward/backward integration, all chess rules tests and 13 learning regressions. Unit fixtures remain explicitly synthetic; the optional real-graph integration and every reported learning experiment used actual tiny data.

```bash
PYTHONDONTWRITEBYTECODE=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 .venv/bin/python -m core.check chess
```

**PASS**, exit 0: 240 interface steps, 1,006 us/step; random return 0.010; 5,000 neurons; mean firing rate 0.176; intact spread 0.0104, lesioned spread 0.0; two quick backward updates passed. Earlier concurrent checks produced a timing warning under heavier CPU contention; no I/O was added to task.step. Timing/firing statistics vary between runs.

```bash
PYTHONDONTWRITEBYTECODE=1 .venv/bin/python -m flies.chess.selfplay --white-checkpoint flies/chess/artifacts/learning/spatial-11/checkpoint_0060_0.pt --black-checkpoint flies/chess/artifacts/learning/spatial-11/checkpoint_0060_1.pt --size tiny --games 2 --max-plies 8 --output flies/chess/artifacts/learning/spatial-cli-smoke
```

**PASS**, exit 0: two actual trained spatial-encoder brains played 8 plies each from the normal starting board; zero illegal attempts; both unfinished `*`, honestly capped. This verifies feature-loader/serial-runner compatibility, not opening competence. JavaScript logic smoke was also rerun via the available Node REPL importing `tests/visual_smoke.mjs?learning_followup=1`: **PASS**, 64 squares, eight action branches, 24 actual recorded moves, FEN/piece counts and viewer replay. No new browser/WebGL verification is claimed.

### Remaining limitations and next scientific question

The new evidence supports improved tactical success **under this engineered interface and curated distribution**. It does not establish board-state-dependent chess reasoning: shuffled inputs retain most of the gain, and stage-only inputs also learn useful priors. Legal masks and partial coordinate selections supply substantial structure. This is now an explicit negative result, rather than confusing improved scores with chess understanding.

The experiment uses only three training seeds, 128 test families, related endgame configurations, four-ply horizons and a fixed frozen initial opponent. Longer games, adversarial/defensive positions, position-template-held-out transfer and genuinely board-dependent counterfactual tasks remain untested. The bootstrap intervals are nominal approximations, particularly with only three initialization seeds. The encoding and stage-only comparisons have one seed; stage-only was post hoc and is exploratory. No Elo, generic-game improvement, biological cognition or full-connectome result is supported.

Normal task checkpoints remain compatible. Spatial/control checkpoints must use the feature loader, which honors their encoding; the unchanged shared server cannot do that automatically. No shared infrastructure was modified to conceal that limitation. All saved checkpoints, aggregate training diagnostics, test-game binary outcomes, sample recordings, fixed-opponent curves and invalid-control archives remain in ignored `artifacts/learning/`; `learning_results.json` preserves the compact verified results alongside the source.

A final compatibility regression tags the original self-play trainer's optimizer checkpoints with `episode-reinforce-v2` and rejects pre-change optimizer bundles. This prevents silently resuming an old loss-normalization experiment under different update semantics; original packed brain-weight loading remains compatible. Validation regression also deliberately corrupts the historical pool and verifies the *actual frozen model state* used for evaluation still equals the untouched initial opponent, rather than merely checking a reported filename.

## Board-dependence and progression follow-up (2026-11-14)

The four requested phases were implemented and locally exercised. None of the experiments establishes general chess competence or a reason to scale to the shared GPU yet. The new matched corpus asks a different question from the earlier mate-one study; its rates must not be pooled with the older corpus.

| Requirement | Implemented / observed evidence | Conclusion |
|---|---|---|
| Preserve biological architecture and genuine two-brain play | Actual tiny FlyWire, two independent learners/optimizers plus frozen historical opponents; all twelve learners' fixed graph/sign/input/group buffers unchanged; no external playing engine | PASS |
| Isolate board information from masks, priors and coordinates | Same own K/Q/R coordinates, exact entire legal UCI set and every hierarchy-prefix mask, disjoint correct targets; exact neural probability integration, both colors | Implemented; stricter than legal-count matching |
| Correctly paired controls across seeds | Seeds 11/23/37; fixed learner 0; intact/full-input swap/stage+partial/own-coordinate/packed/coordinate-only/actual lesion; independently stage-trained models and uniform-legal baselines | PASS; useful adaptation remains inconclusive |
| Prevent answer/family/input leakage | Own-piece symmetry/color families disjoint across splits, corpora and known pretraining layouts; zero measured overlaps; label-corruption tests leave both-phase training weights identical; proof labels never enter policy | PASS on known corpus; curated distribution remains narrow |
| Improve curriculum and preserve self-play | 40 matched updates, 16 mixed updates per run; genuine sampled defender replies; proof-verified mate-two, uncertified KPK/KRK endings, larger reachable legal positions and rehearsal | Implemented and run: 18,432 new training games |
| Disclose rewards and verify credit/gradients/optimizer | Outcome rewards plus explicitly engineered material/pawn-progress deltas on two tiers; own-turn credit includes opponent reply; exact resume, parameter-family gradients, independent frozen state | PASS correctness; gradient existence does not imply stronger play |
| Generalization and checkpoint comparisons | Frozen updates 0/40/56; original untrained, pretrained parent, random, lesion, independent restriction; 48 matched test pairs, 24 test families per harder tier | Measured, with seed/family uncertainty and raw evidence preserved |
| Meaningful game results separate from tactics | 6,576 sampled games, zero illegal attempts; final mate-two 14W/100D/174 unfinished of 288; endgame 0W/262D/26 unfinished; larger positions 0W/1D/287 unfinished | No supported harder-task or general-game gain |
| Full games / strength | Long endings: final 0W/72D; standard opening: 0W/3L/3D/18 unfinished of 24 | Insufficient for strength estimation; no Elo |
| Prepare full-graph scaling | Actual tiny CPU timings/RSS, metadata-only full storage lower bounds, conservative pilot command and gates in SCALING.md | Prepared; no full model/GPU/remote action executed |
| Tests and scope | 46/46 Python tests, core check PASS, JS logic PASS; all edits within flies/chess | PASS; browser/WebGL/live-minute QA not run |

Primary result: final matched adaptation **+0.881 pp [−0.581, +2.565]**. Change from pretrained parent **+0.873 pp [−0.341, +2.112]**. Correct mate-one mass rose from 8.91% to 11.83%, while swapped-board mass is 10.95% and independently stage-trained mass 7.41%. This supports a restricted probability-mass gain, but does not establish choosing the appropriate different move when only the enemy king changes. Post-training zero-input knockouts can be out of distribution. Total variation/reactivity alone is not correct adaptation. Three seeds and this limited family distribution constrain inference.

Mate-two root-proof mass after extended training changes by **+0.960 pp [−3.485, +6.573]**; no supported gain. With entropy disabled, 24 validation mate-two games produced no reward and exactly zero reward gradients, while endgame/larger-position rewards propagated through edges, bias, leak and sensory projection. The bottleneck includes sparse multi-ply rewards, not a generally disconnected backward path. Dense auxiliary gradients did not translate into demonstrated endgame conversion or wins.

A floating-order issue in the new auxiliary reward was discovered after the six trials. The default now accumulates integer material/progress units; reporting ignores magnitudes below 1e−12. Reported trials remain explicitly labeled float-v1; the new arithmetic was verified by regression tests and a frozen reward-gradient diagnostic, not a fresh full learning trial. Historical reproduction uses `--reward-version float-v1`; reward versions cannot silently change on resume. Other new regressions cover exact paired masks/nulls, swap-derived MAP statistics, proof-search unknowns, all defender replies, stale probability caches, opponent-reply credit, mixed-tier budget coverage, answer leakage, exact resume and honest boundary/single-family uncertainty. Existing tests remain intact.

Exact final commands/results:

```text
PYTHONDONTWRITEBYTECODE=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 CHESS_REAL_GRAPH_TESTS=1 .venv/bin/python -m unittest discover -s flies/chess/tests -v
Ran 46 tests in 15.458s — OK, zero skips, exit 0.

PYTHONDONTWRITEBYTECODE=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 .venv/bin/python -m core.check chess
interface ok (240 random steps, 927 us/step)
random baseline: avg return 0.010
brain ok: 5,000 neurons, mean firing rate 0.179
lesion test ok: intact brain reacts to input (spread 0.0143), lesioned brain doesn't (0.0e+00)
quick train ok (2 updates)
PASS
chess:PASS — exit 0.

Node runtime import of flies/chess/tests/visual_smoke.mjs
PASS: 64 station squares, 8 slot branches, 24 actual recorded moves,
FEN/piece counts, viewer replay, invalid-event rejection. No WebGL render claimed.
```

Added implementation: `research_positions.py`, `matched_eval.py`, `extended_curriculum.py`, `research_train.py`, `generalization.py`, `gradient_diagnostics.py`, `research_report.py`, `benchmark.py`, `tests/test_research.py`. Updated `train_selfplay.py` for per-own-move discounted rewards and diagnostic threshold. Added `RESEARCH_PROTOCOL.md`, `RESEARCH_RESULTS.md`, `SCALING.md`, frozen `research_data/{matched,extended}.json`, `research_results.json`, `research_evidence.json`; linked README/EXPERIMENTS/earlier results. Complete exact reproduction commands and limitations are in RESEARCH_RESULTS.md. Archive-only reanalysis was executed and reproduced every aggregate exactly; both frozen dataset hashes passed independent recomputation. Large checkpoint/PGN/stage artifacts remain locally under ignored `artifacts/research/`. No push, merge, commit, remote command, shared infrastructure edit, or GPU job occurred.

## Cinematic live 3D integration — verified November 14, 2026

The primary interface is now the feature-local [Connectome Table](CINEMATIC.md), launched with `PYTHONDONTWRITEBYTECODE=1 .venv/bin/python -m flies.chess.cinematic --port 8770` at `http://127.0.0.1:8770/`. The older 2D recording viewer and native caption station remain available. This section supersedes their earlier UI limitations for the independent feature-local arena, without claiming a new shared-world backend API.

### Requirement audit

| Requirement | Implemented evidence / practical boundary |
|---|---|
| Contributor scope and biology | Only `flies/chess/` writes. Before/after hashes of 21 shared/root source files are identical; tracked Git diff is empty. `AGENTS.md`/`CLAUDE.md` were already untracked and were not edited. No GPU jobs, remote commands, commits, push or merge. |
| Actual independent FlyBrains | `live_match.py` calls the existing `selfplay.choose_stage` for the current side. Separate models/parameter storage, CPU evaluation mode, no-grad, no optimizer. Synthetic fixture and actual tiny FlyWire tests prove both call inference, weights remain fixed, and Black's call count stays zero until White's ACK. Existing core lesion and connectivity tests pass. |
| Eight biological action groups and legal chess | Existing coordinate hierarchy, sensory encoding, masks and `ChessEnv` reused. No output head, random-legal fallback, strategic chess engine, engine score or answer label introduced. Python-chess supplies rules/outcomes and rules-only visual metadata. |
| Actual fly assets | Directly instantiate the original `Fly` from `web/lib/fly.js` twice, with its original anatomy, eyes, wings and leg nodes. No imported model loader, rig or animation clips exist in this repository. Piece attachment/leg grip is a feature-local extension, not a replacement fly model. |
| Existing brain visualization | Import original `web/matrix.js`; runtime-import original Matrix host DOM/styles from `web/index.html`; reuse original points/meta files. Serialize source changes and measured rate playback; dispose old point resources. No copied neural renderer. |
| Cinematic board and scene | 3D board/plinth, engraved coordinates, cached sculpted pawn/rook/knight/bishop/queen/king geometry, ivory/obsidian materials, warm/rim/environment lighting, soft shadows, optional SSAO, tone mapping, atmospheric fog. Same Three.js r170 and OrbitControls stack. |
| Complete physical moves | Director explicitly visits approach, pickup, carry, place, release, return and completion. Uses actual fly root attachment; own-color source asserted. Curved, eased, pausable flight and asynchronous rendering. Capture removal, both colors' castling, en passant, promotion/underpromotion, check, checkmate and stalemate rendering verified against FEN. |
| Strict inference/animation handshake | One serial neural worker; it stops after each selected full move. UUID, capability token, connection epoch, monotonic event/move IDs, explicit lifecycle events, before/after FEN. ACK checks sequence, full FEN and rendered placement. Duplicate/out-of-order/stale ACKs fail. Next inference has no path around a valid ACK. Per-move inference and animation-ACK timestamps persist. |
| Pause, restart, disconnect, timeout | Stage/animation pause; restart cancels old callbacks and preserves previous records. Heartbeat lease and ACK timeout pause/error without ACK. Recovery reconstructs a pending animation from before FEN and rotates epoch. Superseded tabs wait for deliberate Take control, rather than fighting over epochs. WebGL restoration follows recovery. Live process/RNG state is not persisted across server-process restart: use saved Replay or start a fresh match. |
| Actual neural provenance | Full actual 8-iteration rates collected with `FlyBrain(record=True)`, shared uint8 quantizer reproduced exactly in a regression. Side/match/decision/stage/observation/mask/probabilities/group means/timestamp and checkpoint identity supplied. Latest packet per side only. Shared decorative base colors/rain clearly distinguished from measured simulated rates. Live updates optional while the director continues off-screen. |
| Interface and replay | New Live, pause/resume, restart, preset/checkpoint selection, pace, five camera modes, SAN history, official result versus cap, JSON/PGN, replay file/current recording, playback and seek, ray-picked fly inspection and White/Black switching. Replay performs no inference and is explicitly labeled. Live inference never silently trains. |
| Performance and reliability | Shared piece geometries/materials, bounded telemetry/event history, cached unchanged packets, disposal on brain switching, adaptive quality/DPR limits, resize handling, reduced motion, hidden-tab pause, WebGL context loss/recovery. Repeated board resets and brain switches have bounded GPU geometry counts. Desktop Safari observed about 60 FPS at balanced quality; this is not a cross-device guarantee. |
| Existing training/evaluation/scaling | All existing tests remain intact and pass; no research dataset/checkpoint overwritten. Existing independent training/optimizer/resume and evaluation workflows remain separate. Full graph loader/CPU compatibility evidence remains valid; full-graph cinematic throughput and new GPU learning were not executed. Scientific scaling gates remain unmet. |

### Final executed checks

After the last production code change:

- `PYTHONDONTWRITEBYTECODE=1 CHESS_REAL_GRAPH_TESTS=1 .venv/bin/python -m unittest discover -s flies/chess/tests -v`: **67 tests PASS in 7.423 s**, including actual FlyWire tiny live gating, actual backward integration, independent parameters/optimizers, checkpoint/resume, rules, reward, evaluation and all prior research regressions.
- `PYTHONDONTWRITEBYTECODE=1 .venv/bin/python -m core.check chess`: **chess:PASS**. Interface, actual 5,000-neuron forward, input dependence versus disconnected lesion, and two training updates pass.
- `tests/cinematic_smoke.mjs`: **PASS** for exact FEN reconstruction, special-move plans, duplicate/order/stale-event protection, real Director overlap rejection, complete animation lifecycle, pause and cancellation.
- Existing `tests/visual_smoke.mjs`: **PASS** for original station and 2D viewer. Both suites executed in the available built-in Node runtime using `vm.SourceTextModule`; the shell has no `node` executable. The documented `node --experimental-vm-modules ...` commands are equivalent entrypoints when Node is installed, not falsely reported as successful shell executions here. All five arena JS modules parsed as well.
- **Actual Safari 27.0.1/WebGL browser pass: 46 checks, zero failures, zero unexpected console errors.** The deliberate context-loss test generates exactly its expected pause notification. The page and original Matrix were visually inspected through native Safari; this is not just a mocked scene. `artifacts/cinematic-browser.json` records user agent, lifecycle states, observed 60 FPS, 43 arena geometries and detailed results. Other browser engines/mobile hardware were not executed.
- A standard-initial-position **trained-versus-trained live game finished in 15 plies, official THREEFOLD_REPETITION, result 1/2-1/2, not truncated**. All 15 physical moves were acknowledged. Both actual 5,000-neuron models use graph `cc2e5ce35f`; distinct trained checkpoint SHA256 values are recorded. The final example is `artifacts/cinematic/0415781dfce6470497c23c55f3a96e97/game_0000.json` and its PGN. Independent replay verifies every legal move, final FEN and outcome; each prior ACK timestamp precedes the next inference timestamp. This demonstrates engineering correctness, not chess strength.

Machine-readable evidence and implementation hashes are in [cinematic_verification.json](cinematic_verification.json). Full logs are `artifacts/cinematic-python-final.log`, `artifacts/cinematic-core-final.log`, `artifacts/cinematic-js-final.json`. Earlier passing and failing browser reports were retained with timestamps; final failures were fixed, not removed from the tests.

### Discovered defects and repairs

Real browser testing exposed late HTTP responses crossing recovery/restart boundaries. Generation checks, monotonic-state rejection, and serialized new/recovery operations now prevent old tokens/epochs from rolling back the current state or triggering spurious reconnects. The browser regression exercises both WebGL recovery and deliberate owner takeover. Restart during inference is also tested against stale move publication. Inspection serializes the original Matrix's asynchronous 8-step rate playback before switching brains. Piece normals preserve the original smooth lathe/sphere normals rather than recomputing flat normals after geometry merging. A final audit found that cancelling a finished capped game could relabel its cap as user stop; finished records are now preserved and a regression checks that reason.

### Files in this addition

New runtime files: `cinematic.py`, `live_match.py`, `arena.html`, `arena.css`, `arena.js`, `arena_scene.js`, `arena_state.js`, `director.js`, `inspector.js`. `selfplay.py` adds optional true rate recording, preserving its default behavior.

New tests: `tests/test_cinematic.py`, `tests/cinematic_smoke.mjs`, `tests/cinematic_browser.html`, `tests/cinematic_fixtures.json`. New documentation/evidence: `CINEMATIC.md`, `CINEMATIC_REUSE.md`, `cinematic_verification.json`. Updated existing documentation: `README.md`, `AUDIT.md`, `SCALING.md`, `COMPLETION.md`. New generated records/logs remain under `artifacts/`; previous research artifacts remain intact.

### Remaining integration and scientific limits

The independently functioning arena is ready for review and a local public demonstration. Shared `/web/` multi-station embedding would require the core owner's supported world/session integration; the existing host is a non-exported application and its server owns separate one-policy tasks, not a two-agent board. No protected implementation was copied or changed to bypass that boundary. The original low-poly fly has no skeletal gripping rig; attachment to its existing nodes is the supported alternative. The arena requires the same external Three.js CDN and browser WebGL2/import-map support as its host. It has one controlling local connection and is not an internet multi-user service. Dense activity is latest-measurement inspection, not an indefinitely recorded rate movie. Local server-process resume is explicitly limited as above.

No new training experiment, rating or performance improvement is claimed from this UI work. The existing mechanism study's limited task-specific findings and inconclusive incremental training/mate-two/game gains remain the scientific result. Full-brain training is still not justified and was not submitted.
