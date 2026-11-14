# Board dependence and harder-task study — observed results

**Historical study.** The subsequent [mechanism investigation](MECHANISM_RESULTS.md) completed integer-v2 replications and a larger fresh evaluation. It supports modest mate-one board dependence, including in the frozen parent, while added-learning/mate-two gains remain unsupported. Results below retain their original datasets and conclusions; do not pool the two studies.

The actual 5,000-neuron FlyWire policy reacts to board changes, but **useful board-state adaptation remains unestablished**. New training increased probability mass on mate-one answers on the matched test distribution; the paired correct-versus-swapped advantage remains small and uncertain. There is no supported improvement in mate-two reasoning, endgame conversion, or general chess play. No playing-strength estimate or shared-GPU run is warranted.

This is an exploratory follow-up to [EXPERIMENTS.md](EXPERIMENTS.md), using the protocol in [RESEARCH_PROTOCOL.md](RESEARCH_PROTOCOL.md). It does not invalidate the earlier distribution-specific mate-one improvement. The new corpus and endpoint ask a stricter question. Results below are measured CPU experiments, not projected training performance.

## What was implemented and run

All decisions in training come from two independent instances of the unchanged `core.brain.FlyBrain`, or a separate frozen historical FlyBrain. No move engine, answer label, ranked candidate list, output head, new connection, or random training fallback was introduced. Each sampled chess move still comprises four/five choices from the existing eight descending-neuron groups. The rule-based proof procedure described below runs only during dataset curation and evaluation.

Three seeds, 11/23/37, warm-start both learners from their existing 60-update spatial-encoding checkpoints. Focal learner **0** was fixed before the new evaluations. An independently trained stage-only restriction starts from identical source weights for each seed. Both conditions run 40 updates ×64 matched games, then 16 updates ×32 mixed games: **3,072 games per run, six runs, 18,432 new training games**. Roles, actual colors, live/historical opponents and tiers are balanced. Frozen checkpoints 0/20/40/48/56, optimizers, baselines, pool and RNG are retained; test comparisons use fixed 0/40/56, never the best checkpoint. Parent and truly untrained initialization are different controls.

The reported learning trials used reward arithmetic `float-v1`, Adam lr=.003, entropy=.003, gamma=.98 and backward chunk 64. The current default is the numerically corrected `integer-v2`, explained below. No full 56-update learning trial of that correction has been run; do not attribute the reported performance to it.

## Exact legal-tree matching and leakage prevention

The mate-one corpus contains **96 training / 24 validation / 48 test pairs**. Each pair has the same own K/Q/R coordinates, the exact same complete legal UCI move set, and disjoint mate-one answer sets. Only the enemy king changes. Thus every hierarchical prefix has exactly the same legal mask across the pair. A board-blind stage/coordinate preference cannot use masking to distinguish the targets. This is stronger than matching the number of legal moves.

All enemy placements sharing an own-piece arrangement, its dihedral transforms and color mirror form one family. Families are disjoint across splits, matched/extended corpora and all known earlier pretraining corpus layouts. Verified overlap counts are zero. Both actual colors are evaluated. Only FEN-derived observations and legal masks reach FlyBrain; tests replace answer labels with nonsense and verify identical training weights in both phases. No policy cache contains labels or model probabilities. Labels do determine the curated task distribution, which is explicitly restricted to these piece sets and horizons.

The extended corpus contains **144 training / 36 validation / 72 test positions**, equally divided among: KQK forced mate-two positions with no immediate mate; KPK/KRK basic endgames; larger legal positions reached by 12–28 construction plies. The construction sampler is not either self-play policy. A bounded Boolean AND/OR proof through all defender replies produces mate-two labels for curation/scoring. Search exhaustion raises unknown, never a negative answer; the observed generator needed 4,373 candidates and had zero unknowns. Endgame theoretical WDL was **not** certified. Frozen dataset JSON and content hashes are preserved in `research_data/`; policy code never reads answer fields.

## Phase 1: what the policy depends on

For pair A/B with disjoint correct sets M_A/M_B, the primary endpoint is

`adaptation = [P_A(M_A) + P_B(M_B) - P_A(M_B) - P_B(M_A)] / 2`.

Probabilities integrate the actual FlyBrain's conditional outputs at every legal hierarchy prefix. There is no sampling noise from action draws in this endpoint. An input swap exchanges A/B under their identical legal tree; it is evaluated against the original targets. Stage-only, stage+partial-only, own-coordinate-only, random-legal and actual-lesion policies have zero adaptation algebraically. Zero here is a structural identity, not a statistical estimate from no events. Uniform legal moves are a separate baseline: uniform coordinate-stage choices do not generally induce a uniform legal-move distribution.

Uncertainty uses 5,000 two-way bootstrap draws over **three training seeds and whole held-out families**, averaging actual colors within family. Intervals are nominal 98.75%; four main intervention contrasts were planned, while later decompositions and training-phase contrasts are descriptive. Three seeds and 48 pairs leave substantial uncertainty; these are not population-wide guarantees.

| Frozen policy | Exact mate-one probability mass | Adaptation, percentage points (98.75% interval) |
|---|---:|---:|
| Original untrained initialization | 2.69% | −0.013 [−0.061, +0.033] |
| Parent pretrained policy, new update 0 | 8.91% | +0.009 [−1.118, +1.007] |
| After 40 matched updates | 11.13% | +0.808 [−0.643, +2.901] |
| After 56 total updates | 11.83% | **+0.881 [−0.581, +2.565]** |
| Independently stage-trained, update 56 | 7.41% | 0, by construction |
| Uniform legal baseline | 3.58% | 0, by construction |

Final per-seed adaptation is +1.060 / +0.072 / +1.512 percentage points. The all-training adaptation change from the parent is +0.873 pp [−0.341, +2.112]; the extended phase contributes +0.073 pp [−1.059, +1.040]. Neither establishes learning of the required board distinction. The upper bounds still allow useful positive dependence; this is not proof that no dependence exists.

The final intact correct mass exceeds the parent by +2.925 pp [+0.893, +5.040] and the independently stage-trained model by +4.427 pp [+1.133, +7.604] in descriptive comparisons. These are restricted-distribution probability gains, **not evidence that the agent selects the appropriate different move across the matched boards**.

| Final policy intervention | Correct mass | Adaptation (pp) |
|---|---:|---:|
| Intact | 11.83% | +0.881 |
| Paired full-input swap | 10.95% | −0.881 |
| Stage + partial choice only | 7.33% | 0 |
| Own coordinates + stage/partial only | 8.95% | 0 |
| Coordinate features, packed board removed | 10.99% | +0.482 |
| Packed board, redundant geometry removed | 9.31% | +0.318 |
| Stage flags only | 6.36% | 0 |
| Actual connectome lesion | 4.80% | 0 |

Intact-vs-own-coordinate, intact-vs-stage and intact-vs-lesion contrasts are positive, but intact-vs-paired-swap is inconclusive. Zeroing inputs shifts the policy away from its training distribution; the independently stage-trained control helps address this, but does not make knockouts an additive causal decomposition. The result is consistent with a large contribution from stage priors, own coordinates and legal masks, plus board-driven reactivity that is not reliably aligned with the changed targets. Intact pair-distribution total variation is 8.45%; reactivity alone is not tactical adaptation. Both paired joint-MAP moves are correct only 0.69% of family/color/seed comparisons; joint MAP is a diagnostic, not the sampled game policy.

## Phases 2–3: harder tasks, gradients and actual outcomes

Matched-phase games allow a real second-brain reply but cap at two plies; only immediate mate yields a positive outcome. Mixed-phase caps are mate-one 8, mate-two 12, endgame 24, larger positions 16. Outcome rewards are actual +1/−1/0. Endgame/larger-position tiers additionally receive `.25 * clip(Phi_next - Phi_before, -1, 1)` per own turn, including the opponent reply. `Phi = (material in pawn units + .05 * signed pawn advancement) / 10`. This engineered horizon objective can reward material/progress without a chess win and is **not policy-invariant**. Mate-two receives no proof or distance reward. Caps remain unfinished `*`, never fabricated draws.

All parameter groups had nonzero gradients during training; all twelve learners retained identical fixed wiring/signs/sensory indices/descending groups before/after training. Exact split-run resume and isolation tests pass. In a frozen seed-11 **validation-only** reward diagnostic, entropy is disabled and SGD lr=0 prevents weight changes:

| Tier, 24 actual games each | Nonzero move rewards | Gradient norms: edges / bias / leak / sensory projection |
|---|---:|---|
| Mate-two | 0 | 0 / 0 / 0 / 0 |
| Endgame | 23 | .01267 / .10282 / .00387 / .05181 |
| Larger positions | 125 | .00768 / .04386 / .00198 / .03625 |

This diagnostic distinguishes insufficient reward opportunities from a broken backward path. It does not prove that nonzero gradients improve chess. Mate-two success can be lost at any of several neural choices, making terminal learning much sparser than mate-one. An EMA baseline/entropy can move parameters when current rewards are zero; that must not be reported as new tactical evidence.

Each generalization tier uses 24 held-out families ×4 color-balanced sampled games ×3 seeds = **288 games per condition**. The opponent is the corresponding separate frozen untrained learner 1, with identity hashes retained. Seven conditions, three tiers yield 6,048 games; long endgames add 504; opening-start matches add 24: **6,576 sampled evaluation games total**, zero illegal attempts. More games on the same few families/seeds do not create new independent evidence.

Mate-two exact first-move mass on proof-winning roots: final 8.74% [2.73, 17.68], matched40 7.78%, parent 8.53%, untrained 10.71%, independent restriction 8.02%, lesion 9.33%, uniform legal 6.09%. The extended-phase change is **+0.96 pp [−3.48, +6.57]**. Root accuracy does not certify continuation, and the current data do not establish improvement.

Actual W/L/D/unfinished counts:

| Candidate | Mate-two, cap 12 | Endgame, cap 48 | Larger positions, cap 24 |
|---|---|---|---|
| Final trained56 | 14 / 0 / 100 / 174 | 0 / 0 / 262 / 26 | 0 / 0 / 1 / 287 |
| Matched40 | 18 / 0 / 105 / 165 | 2 / 0 / 262 / 24 | 0 / 0 / 1 / 287 |
| Parent pretrained | 16 / 0 / 108 / 164 | 2 / 0 / 264 / 22 | 0 / 0 / 2 / 286 |
| Original untrained | 4 / 0 / 47 / 237 | 0 / 0 / 176 / 112 | 0 / 0 / 0 / 288 |
| Uniform legal | 0 / 0 / 51 / 237 | 0 / 0 / 150 / 138 | 6 / 1 / 0 / 281 |
| Actual lesion | 8 / 0 / 68 / 212 | 2 / 0 / 179 / 107 | 0 / 0 / 0 / 288 |
| Independent stage restriction | 10 / 0 / 77 / 201 | 0 / 0 / 234 / 54 | 0 / 0 / 2 / 286 |

Long endgames (12 families ×2 games ×3 seeds, cap 160) finish: final trained **0W/0L/72D**, comprising 54 repetition and 18 insufficient-material draws. Matched40 wins once; every other condition has zero wins. These starts include uncertified KPK/KRK positions; this is not a theoretical endgame-conversion benchmark. Opening-start trained games finish **0W/3L/3D/18 unfinished** (24 total, cap 160). One opening family cannot support a position-general uncertainty interval or an Elo estimate. Repetitions and material loss explain much of the endgame completion rate; completion alone does not indicate competence.

Per-family outcomes, checkpoint identities, terminations, auxiliary scores and intervals are in `research_evidence.json` / `research_results.json`. Ordinary zero-event bootstrap intervals were suppressed or replaced by conservative per-run position-cluster boundary bounds; a single opening gets no interval. Auxiliary return is separate from W/L/D and is never used as a playing-strength claim.

## Discovered issues and regression coverage

- Ordinary unmatched legal masks and board shuffles did not isolate strategic use of board information. New exact-full-tree counterfactuals make blind policies' null adaptation provably zero; both colors and every prefix are checked.
- Paired members must not use each other's contemporaneous outcome as a baseline. New training uses independent action RNG and prior-update, tier/role-specific EMA baselines only.
- Dense rewards must include material lost during an opponent reply. New own-turn reward vectors and discounted-return tests check a real queen-capture fixture, signs, terminal credit and cap handling.
- Floating summation order gave tiny spurious material changes and misleading nonzero-reward counts. Integer material/progress units now yield exact zero for unchanged totals; diagnostics use a 1e−12 reporting threshold. Legacy `--reward-version float-v1` remains for reproduction and incompatible resumes are rejected. Full learning effects of the correction remain unmeasured.
- Derived swap reports initially retained the intact joint-MAP statistic. It now exchanges correct/cross statistics; a real explicit-swap regression checks equivalence. Probability endpoints and trained weights were unaffected.
- Cached decision trees must not cache a stale policy. A parameter-mutation test proves the next integration changes probabilities.
- A proof search cap must not become a false target; an exhausted search raises unknown. A mate-two fixture verifies every defender reply.
- Zero-event bootstraps and one-position opening samples must not imply certainty; tests cover conservative boundary bounds and missing single-family intervals.
- A reduced training budget could skip mixed tiers. Extended training now requires at least 32 nominal games; tests exercise all four tiers and prove both phases ignore label changes. Resume, fixed-buffer and observation-restriction compatibility are checked.

## Verified checks and source changes

Environment: Python 3.9.6, torch 2.8.0, numpy 2.0.2, python-chess 1.999/chess 1.11.2; actual FlyWire tiny `cc2e5ce35f`, `synthetic=False`, CPU. Final Python run: **46 tests, all passed, zero skipped, 15.458 seconds**. `core.check chess`: PASS; intact input spread .0143, lesion spread 0, two quick training updates passed. JavaScript logic smoke: PASS, 64 squares, all eight slots, 24 recorded moves, FEN/piece counts, replay and invalid-event rejection. Browser/WebGL rendering and live one-minute world QA were not performed.

Added source: `research_positions.py`, `matched_eval.py`, `extended_curriculum.py`, `research_train.py`, `generalization.py`, `gradient_diagnostics.py`, `research_report.py`, `benchmark.py`, and `tests/test_research.py` (12 new regression tests). Updated `train_selfplay.py` to support validated per-own-move discounted rewards and robust diagnostic counts. Added protocol/results/scaling documents, frozen datasets and machine-readable evidence; linked the previous audit/results. Shared infrastructure and native task/observation defaults remain untouched.

## Reproduction

Use the already configured virtual environment/data; no setup or GPU command is required. Export `PYTHONDONTWRITEBYTECODE=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1`. All commands below run locally. Existing artifact paths contain the completed experiment; choose fresh output directories when repeating a trial.

```bash
# Tests (the optional real-data integration is enabled).
CHESS_REAL_GRAPH_TESTS=1 .venv/bin/python -m unittest discover -s flies/chess/tests -v
.venv/bin/python -m core.check chess

# Recompute every aggregate from archived family-level evidence; no checkpoints needed.
.venv/bin/python -m flies.chess.research_report \
  --evidence flies/chess/research_evidence.json \
  --root flies/chess/artifacts/research/reanalysis

# The frozen datasets avoid dependence on an absent historical exclusion corpus.
mkdir -p flies/chess/artifacts/research
cp flies/chess/research_data/matched.json flies/chess/artifacts/research/matched.json
cp flies/chess/research_data/extended.json flies/chess/artifacts/research/extended.json
# Generator reproduction, if the original pretraining corpus is present:
.venv/bin/python -m flies.chess.research_positions --output flies/chess/artifacts/research/matched-regenerated.json
.venv/bin/python -m flies.chess.extended_curriculum --output flies/chess/artifacts/research/extended-regenerated.json
```

Known dataset hashes: matched `497987cf7619b1e17806666df18d06eb706a1600e870b84428a6c8e1e6be59c6`; extended `6694f1b96feff6f603699c709738507f1adb3ae3436f5cb604a11d2d001b2d2f`. Generation changes if historical exclusion data is absent; the frozen JSON is authoritative for this study. Archived reanalysis was actually executed and reproduced **every aggregate exactly**, with both frozen dataset hashes verified. If parent checkpoints are missing, reproduce the earlier 60-update study first; research training deliberately depends on those parent weights:

```bash
# Only if the existing parent study artifacts are unavailable; this is not setup.
.venv/bin/python -m flies.chess.curriculum
for seed in 11 23 37; do
  .venv/bin/python -m flies.chess.learning train --seed "$seed" \
    --updates 60 --games 128 --encoding spatial \
    --output "flies/chess/artifacts/learning/spatial-$seed"
done
```

```bash
for seed in 11 23 37; do
  .venv/bin/python -m flies.chess.research_train --seed "$seed" --updates 56 \
    --reward-version float-v1 \
    --sources "flies/chess/artifacts/learning/spatial-$seed/checkpoint_0060_0.pt" "flies/chess/artifacts/learning/spatial-$seed/checkpoint_0060_1.pt" \
    --output "flies/chess/artifacts/research/intact-$seed"
  .venv/bin/python -m flies.chess.research_train --seed "$seed" --updates 40 --stage-only \
    --reward-version float-v1 \
    --sources "flies/chess/artifacts/learning/spatial-$seed/checkpoint_0060_0.pt" "flies/chess/artifacts/learning/spatial-$seed/checkpoint_0060_1.pt" \
    --output "flies/chess/artifacts/research/stage-$seed"
  .venv/bin/python -m flies.chess.research_train --seed "$seed" --updates 56 --stage-only \
    --reward-version float-v1 --resume "flies/chess/artifacts/research/stage-$seed/resume.pt" \
    --sources "flies/chess/artifacts/learning/spatial-$seed/checkpoint_0060_0.pt" "flies/chess/artifacts/learning/spatial-$seed/checkpoint_0060_1.pt" \
    --output "flies/chess/artifacts/research/stage-$seed"
done

# These six exact probes select learner 0 from every seed (never the best).
for label in untrained pretrained matched40 matched56 restricted40 restricted56; do
  case "$label" in
    untrained) pattern='learning/spatial-SEED/initial_0.pt' ;;
    pretrained) pattern='research/intact-SEED/checkpoint_0000_0.pt' ;;
    matched40) pattern='research/intact-SEED/checkpoint_0040_0.pt' ;;
    matched56) pattern='research/intact-SEED/checkpoint_0056_0.pt' ;;
    restricted40) pattern='research/stage-SEED/checkpoint_0040_0.pt' ;;
    restricted56) pattern='research/stage-SEED/checkpoint_0056_0.pt' ;;
  esac
  .venv/bin/python -m flies.chess.matched_eval --label "$label" \
    --checkpoints "flies/chess/artifacts/${pattern/SEED/11}" "flies/chess/artifacts/${pattern/SEED/23}" "flies/chess/artifacts/${pattern/SEED/37}" \
    --output "flies/chess/artifacts/research/${label}_probe.json"
done
for seed in 11 23 37; do
  .venv/bin/python -m flies.chess.generalization \
    --run "flies/chess/artifacts/research/intact-$seed" \
    --parent "flies/chess/artifacts/learning/spatial-$seed" \
    --stage-run "flies/chess/artifacts/research/stage-$seed" \
    --output "flies/chess/artifacts/research/generalization-$seed"
  .venv/bin/python -m flies.chess.generalization --tactical-only \
    --run "flies/chess/artifacts/research/intact-$seed" \
    --parent "flies/chess/artifacts/learning/spatial-$seed" \
    --stage-run "flies/chess/artifacts/research/stage-$seed" \
    --output "flies/chess/artifacts/research/tactics-$seed.json"
done
.venv/bin/python -m flies.chess.research_report
.venv/bin/python -m flies.chess.gradient_diagnostics \
  --checkpoint flies/chess/artifacts/research/intact-11/checkpoint_0056_0.pt \
  --opponent flies/chess/artifacts/learning/spatial-11/initial_1.pt \
  --output flies/chess/artifacts/research/gradient_diagnostics-integer.json
```

Recorded trial configs/checkpoint hashes, final per-family evidence and source hashes are in `research_results.json` and `research_evidence.json`; large checkpoints, PGNs and full stage recordings remain in ignored local `artifacts/research/`. Exact model-state resume is verified on the tested CPU stack; checkpoint container bytes and other hardware need not reproduce bit-for-bit. Current default `integer-v2` is suitable for future validation experiments, but use explicit `float-v1` to reproduce the historical trials above. Next-step gates, measured memory/latency and the unexecuted full-graph proposal are in [SCALING.md](SCALING.md).
