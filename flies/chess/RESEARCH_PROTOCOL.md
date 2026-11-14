# Matched-board dependence, progression and scaling protocol

This completed historical protocol is followed by [MECHANISM_PROTOCOL.md](MECHANISM_PROTOCOL.md) and its [verified results](MECHANISM_RESULTS.md). Full-graph CPU compatibility has since been verified; no GPU learning experiment has been executed.

This is an exploratory local study, not an external preregistration. It follows the prior fixed mate-in-one study. A two-family validation timing pilot was used to debug exact probability integration. Previously frozen learner 0 from each of seeds 11/23/37 is the focal policy; no best learner/checkpoint selection is allowed.

## Phase 1

Generate 96 training, 24 validation and 48 test **pairs**. Members share the complete legal UCI move set and identical own-piece coordinates, but have disjoint mate-in-one answer sets. Therefore every coordinate-stage mask is identical for every shared prefix. Vary the enemy king, not the labels or policy outputs. Both actual colors are evaluated. Split by the own-piece arrangement's color/dihedral family; all enemy placements for that arrangement stay together. Exclude all known earlier corpus families, including earlier validation/test layouts, from this new corpus.

Primary endpoint: exact adaptation `(P_A(M_A)+P_B(M_B)-P_A(M_B)-P_B(M_A))/2`, averaging colors within pair families and then seeds. This is zero for any board-blind policy on the matched legal tree, even with unequal answer-set sizes. Positive total variation alone is not successful adaptation. Integrate actual FlyBrain conditional probabilities across all legal prefixes; this supplies no oracle actions. Report intact, paired full-input swap, stage+partial-only, own-coordinate-only, stage-only, packed-only, coordinate-only and actual-lesion interventions. The swap's correct mass equals the intact cross-target mass algebraically; explicitly test that equality.

Use the three frozen pretraining seeds, then 40 updates ×64 matched self-play games/seed, capped at two plies so only an immediate mate supplies a positive outcome reward. Failed attempts still allow a real second-brain reply and remain unfinished rather than drawn. Train two separate FlyBrains; warm-start both from their existing seed-matched 60-update weights. Train paired members with independent sampling seeds; never use the other member's current outcome as an advantage baseline. Three separately trained stage-only restrictions start from identical source parameters and use the same matched protocol. Labels are for curation/scoring only; all training actions and terminal outcomes come from sampled FlyBrain self-play. Checkpoints 0/20/40 are retained.

## Phase 2 and 3

Continue both intact and separately trained restricted models for 16 updates ×32 games using a disclosed mixture of matched mate-one, proof-verified mate-two, pawn/rook endgames and reachable larger legal positions. Caps are 8/12/24/16 plies respectively. Keep previous-task rehearsal. A bounded AND/OR rules verifier supplies mate-two labels for dataset/evaluation only, never actions or policy inputs. Search-limit cases are unknown and excluded, never labeled as failures. Candidate first-move proof accuracy and actual game wins are different metrics.

Rewards: actual terminal +1/-1/0 plus explicitly engineered per-own-turn material/pawn-progress changes on the endgame/larger-position tiers. Clip each auxiliary increment to ±0.25; include changes during the opponent reply. This auxiliary objective can reward horizon material/progress and is not claimed to preserve optimal full-chess policies. Caps retain official result `*`. Report auxiliary return separately from official outcome. All samples remain on-policy; frozen opponents have separate state, no optimizer. Keep checkpoint 56 fixed; no test-based checkpoint selection.

Generalization comparisons use held-out families, fixed untouched initial opponents, actual intact/lesioned/random policies, original untrained initial states and independently trained restricted observations. Report exact tactical mass on mate-two proofs, sampled W/L/D/truncation for each tier, longer bounded endgame games and opening-start matches. Never convert caps to draws or estimate Elo from these small/unfinished samples.

Uncertainty clusters by pair/position family and training seed, not by repeated games or legal prefixes. Use nominal 98.75% two-way intervals for the four principal control contrasts; other decompositions are descriptive. Three seeds and a restricted piece distribution limit population inference. The matched test set may be used to diagnose frozen and final fixed checkpoints, not tune hyperparameters.

## Phase 4

Measure actual tiny CPU forward/backward wall times and process memory in an isolated benchmark. Read existing full-graph metadata only to project storage/activation requirements. Distinguish measured values from lower-bound/rough projections. No full model, remote command or GPU job is needed to prepare the proposal.

Minimum proposed GPU gates: (1) matched correct-minus-swapped adaptation interval above zero with point gain at least 2 percentage points across all three seeds; (2) improvement beyond matched pretraining and restricted controls, not just reaction/legality; (3) demonstrable held-out mate-two or endgame outcome gains with frozen opponents; (4) stable gradients/resume and measured memory headroom; (5) substantial meaningful game completion before any playing-strength estimate. If these fail, recommend continued tiny research rather than GPU scaling. Full-graph commands are proposals only and require user/core-owner approval before execution.

## Execution notes and corrections

Observed results are in [RESEARCH_RESULTS.md](RESEARCH_RESULTS.md). All six completed learning trials used the original floating arithmetic (`float-v1`); a subsequently found summation-order issue was corrected with integer material/progress units (`integer-v2`, now the default). Full learning effects of that correction remain unmeasured. Exact historical reproduction must explicitly request float-v1. Frozen diagnostic trajectories were rerun with integer-v2, entropy disabled and lr=0; no policy update was applied. Single-opening and zero-event uncertainty reporting was corrected without changing trajectories/counts. Later descriptive probability-mass changes and independent-restriction contrasts were added to the report after the primary evaluations; they are not additional preregistered confirmatory endpoints. The original four-control comparison and its inconclusive paired-swap result remain visible.
