# Corrected rewards and learning mechanism: verified results

The fresh test supports **modest board-dependent mate-one behavior on the curated matched distribution**. It does not establish a benefit from the new learning mechanism: the frozen parent also shows dependence, the training gain over that parent is inconclusive, and leave-one-out (LOO) does not outperform EMA. Mate-two continuation and general-game improvement remain unsupported. Full-brain GPU training is not scientifically justified by these results.

This completes the interrupted study in [MECHANISM_PROTOCOL.md](MECHANISM_PROTOCOL.md). The original [frozen manifest](mechanism_manifest.json), datasets, model/optimizer checkpoints and raw results are retained. No model was selected by the fresh test, and that test is now retired for development. Later operational repairs do not retroactively change the experimental algorithm or results.

## Numerical reward correction

An independent exact-fraction oracle audited **72 actual FlyBrain validation games, 543 own-turn rewards**. Among 399 exactly zero reference rewards, integer-v2 produced **zero false nonzeros**; float-v1 produced eight residuals. Maximum absolute errors were 4.16e−17 and 2.82e−16 respectively. Tests additionally cover quiet moves, opponent captures, both colors' promotions, en passant, castling and terminal credit. Own-turn deltas include the opponent's reply.

Three complete repetitions of the earlier intact 56-update protocol used integer-v2: seeds11/23/37, **9,216 games**. Both learners' tensors at updates40 and56, final optimizer states and NumPy RNG states matched the corresponding float-v1 trials **exactly**. Reward-only gradients on the audit trajectories also matched at float32 precision. This validates the numerical correction but gives no evidence that it improves learning. It must not be presented as a performance intervention.

## Controlled mechanism and curriculum experiments

The training/validation-only pilot used seed101 with seed-11 warm-start weights, comparing EMA, LOO and entropy-only for 16×128 games each. Final validation adaptation was 1.94% LOO versus1.89% EMA; the descriptive one-seed difference was +0.053 percentage points (95% family interval −0.180,+0.277). Earlier checkpoints fluctuated, including the entropy-only control. No best checkpoint or hyperparameter was selected.

At **2026-11-14T06:30:48.445944+00:00**, the protocol was frozen before fresh test generation/main training. Main experiments use three warm-start seeds11/23/37 and four budget-matched conditions: full-input LOO, full-input EMA, independently trained stage-only LOO, and full-input entropy-only. Every condition has two separate live FlyBrains, parameters and Adam states; no strategic training opponent is random. All fixed connectivity/sign/sensory/group buffers remain unchanged. The existing eight descending groups and 64-float spatial observation are preserved; there is no new critic or output head.

Each run has **48×128=6,144 games**, making **73,728 main training games**. First32 updates rehearse exact-legal-tree mate-one pairs; last16 use 64 matched /32 suffix /32 mate-two games per update, capped at2/2/4 plies. The first eight updates use the training quartile with highest minimum uniform-legal mate opportunity across pair members. This is disclosed oracle curation of starting states, not a policy feature. The suffix bridge retains38 training own-piece families after exclusions and only **one** validation suffix family; no suffix-generalization claim is justified. The bridge is common to all conditions, so its causal effect is not isolated.

Eight independent action-RNG streams share each exact FEN/color/player context. A LOO baseline uses the other seven outcomes, never its own outcome or the other counterfactual board. Regression tests verify context/seed isolation, no own-reward dependence and the unbiased score-function gradient on an exhaustively enumerated Bernoulli task. All baselines are frozen before either learner updates. No group standard-deviation normalization, label loss, proof-distance reward or engine score is used. Rewards are actual terminal ±1/0, gamma1, lr.003, entropy.0005, chunk64. The entropy control zeros rewards and baselines while still recording actual outcomes and making every move through FlyBrain.

Across the three seeds:

| Condition | Matched wins /15,360 training games | Suffix wins /1,536 | Mate-two wins /1,536 | Actual neural stages |
|---|---:|---:|---:|---:|
| LOO | 2,348 | 131 | 45 | 141,012 |
| EMA | 2,363 | 136 | 53 | 140,872 |
| Stage-only LOO | 1,341 | 96 | 37 | 144,856 |
| Entropy-only | 1,220 | 69 | 13 | 144,772 |

Every condition has18,432 games and zero illegal attempts. Different terminations yield different realized compute despite equal games/caps. Only647/2,304 LOO rollout groups had differing actual outcomes; sparse information remains a limitation. Training wins and stage counts are descriptive, not held-out evidence. Including the pilot and integer replication, this phase executed **89,088 distinct-budget training games**. Another18,432 deterministic replay games audit logging and are **duplicate verification**, not independent learning evidence.

## Frozen fresh evaluation

The fresh corpus contains **128 matched mate-one pairs and48 mate-two roots**, excluding all known curated-start/suffix families. Matched members share own K/Q/R coordinates, the complete legal UCI set and every hierarchical prefix mask, with disjoint answers and different enemy king locations. Both actual colors are integrated. Correct mass and adaptation are exact sums of real FlyBrain conditional probabilities over the legal tree, not sampled action accuracy or engine-selected moves.

`adaptation = [P_A(M_A)+P_B(M_B)-P_A(M_B)-P_B(M_A)]/2`.

LOO learner0 at checkpoints0/16/32/48 is fixed; no best learner/checkpoint selection. Compare final EMA, independently stage-trained, entropy-only, untouched initial, actual lesion and uniformly random legal controls. The paired full-input swap has the same legal tree. Blinded controls' zero adaptation is an algebraic null, not a no-event uncertainty estimate. Random legal-move probabilities differ from uniform hierarchical actions.

Intervals below are **nominal98.75% two-way bootstraps over three seeds and whole families**, 5,000 draws. Pair members and actual colors are averaged within families. Repeated games/prefixes are not independent samples. Four primary contrasts were frozen; checkpoint/control decompositions are descriptive. Three seeds limit population inference. The analytic random mate-one policy has no training-seed variation; its uncertainty comes from families.

| Policy | Mate-one correct mass | Adaptation pp [98.75% interval] |
|---|---:|---:|
| Corrected frozen parent | 8.46% | +1.347 [+0.446,+2.337] |
| Untouched untrained initialization | 2.50% | −0.044 [−0.098,+0.011] |
| LOO update16 | 8.66% | +1.507 [+0.501,+3.035] |
| LOO update32 | 9.67% | +2.021 [+0.804,+3.416] |
| LOO update48 | **9.99%** | **+2.063 [+0.716,+3.622]** |
| EMA update48 | 10.15% | +2.158 [+0.803,+3.765] |
| Independently stage-trained | 6.11% | 0, structural null |
| Entropy-only update48 | 2.53% | +0.122 [+0.065,+0.192] |
| Actual lesioned LOO | 4.09% | 0, structural null |
| Uniform legal baseline | 3.52% | 0, structural null |

| Frozen primary contrast | Estimate pp [98.75% interval] |
|---|---:|
| Final LOO correct-versus-swapped adaptation | **+2.063 [+0.716,+3.622]** |
| Adaptation gain over frozen parent | +0.716 [−0.240,+1.761] |
| Mate-two win gain over frozen parent | −0.521 [−1.823,+0.434] |
| LOO adaptation minus EMA | −0.094 [−0.448,+0.226] |

Final per-seed adaptation is1.967/1.885/2.337pp. These results establish a small appropriate response to enemy-king changes on this distribution, while excluding a purely board-blind preference as the entire explanation. They do **not** establish general chess reasoning, nor a reliable new-training benefit. The parent is also positive on this larger fresh sample. Its result is not numerically comparable to the earlier48-pair corpus as though only the model changed.

LOO exceeds the entropy-only control by1.942pp [+0.623,+3.459], descriptively: outcome rewards help retain/use task information relative to entropy-only updates. This does not substitute for improvement over the frozen parent. The mixed phase changes matched adaptation by only+0.042pp [−0.527,+0.612]; no progression gain is established.

## Mate-two outcomes and credit assignment

Each condition plays48 roots ×8 color-balanced games ×3 seeds = **1,152 games**, cap4, against separate untouched initial learner1 opponents. Ten conditions produce **11,520 sampled evaluation games**, all legal. A cap stays unfinished `*`. Winning against the fixed weak opponent and placing probability mass on a forced-mate root are separate metrics; neither alone proves an adversarially sound continuation.

| Candidate | Wins | Losses | Actual draws | Unfinished |
|---|---:|---:|---:|---:|
| Frozen parent | 10 | 0 | 93 | 1,049 |
| Untrained | 3 | 0 | 68 | 1,081 |
| LOO16 | 8 | 0 | 100 | 1,044 |
| LOO32 | 8 | 0 | 103 | 1,041 |
| LOO48 | **4** | 0 | 95 | 1,053 |
| EMA48 | 9 | 0 | 99 | 1,044 |
| Independently stage-trained | 9 | 0 | 87 | 1,056 |
| Entropy-only | 5 | 0 | 69 | 1,078 |
| Actual lesion | 5 | 0 | 93 | 1,054 |
| Uniform legal | 6 | 0 | 94 | 1,052 |

Parent win probability is0.87% [0,2.08]; final LOO0.35% [0,1.13]. The primary paired gain spans zero and has a negative point estimate. Forced-mate root mass is parent5.77%, untrained11.65%, LOO32 4.23%, LOO48 4.19%, EMA4.11%, restriction6.11%, lesion7.40%, random5.52%. Mixed-phase root-mass change is−0.037pp [−1.121,+1.755]. No mate-two improvement is supported; narrow mate-one specialization is a plausible concern, not a proved causal explanation.

Validation-only diagnostics, with **entropy0, baseline0 and SGD lr0**, preserve both models exactly. Final seed11 has13/96 matched wins (24 pair families),5/8 suffix wins (one family), and2/96 mate-two wins (12 families); the parent has11/96,3/8,1/96. Final mate-two reward gradient norms are edges.00374 /bias.02513 /leak.00086 /sensory projection.02503. Nonzero gradients verify the backward path, not useful learning. Root rewards remain rare and are dwarfed by mate-one signals. Single-seed gradient cosines vary by checkpoint; they do not establish a general gradient-interference explanation.

## Exposure audit, corrections and limitations

Passive replay of all three primary runs recorded179/189/187 learner0 own-piece decision families and reproduced checkpoints16/32/48 and both optimizers exactly. None of the128 fresh matched families was visited during new training. Mate-two overlaps are7/9/8 families by seed,12 in their union. The descriptive36-family common-unvisited subset has win gain−0.463pp [−2.055,+0.463], still inconclusive. The full48-root primary endpoint remains unchanged. Historical warm-start runs did not retain every visited position, so strict never-seen exposure over their entire history cannot be certified.

The investigation repaired control identity labels: the generic checkpoint constructor called initial checkpoints `TRAINED`; the experiment now explicitly labels untrained, parent, restricted, entropy-only and frozen opponents. Only metadata/PGN headers changed; probabilities, moves, counts, checkpoint hashes and primary protocol did not. [mechanism_corrections.json](mechanism_corrections.json) retains the frozen/current evaluator hashes and reason. Gradient cosines now accumulate in float64 after a float32 self-cosine exceeded1; regression tests also distinguish zero signal from a numerical cosine.

No external engine selects a move. Rule proofs/answers curate starts and score outcomes only. This is still an engineered task with supplied legal masks, short horizons, a restricted piece distribution and weak fixed opponents. One validation suffix is inadequate for strong curriculum selection. The old endgame study also found no conversion gain (final0W/72D in long endings). No Elo, biological chess intelligence or general-game improvement is supported.

## Reproduce and inspect

Frozen inputs are [mechanism_curriculum.json](research_data/mechanism_curriculum.json) and [mechanism_test.json](research_data/mechanism_test.json). Their hashes are `03f91a6096957bc75ea34fec8d25d08fdc9ac6a1602fe32f520a31f423c2a785` and `86efc312cc7f47733082535627f976f2e9f08df996136197fb7cd79df9086979`. [mechanism_evidence.json](mechanism_evidence.json) contains per-family evidence; [mechanism_results.json](mechanism_results.json) contains verified aggregates/diagnostics. Large model, optimizer and PGN artifacts remain under ignored `artifacts/mechanism/`. They were preserved, not replaced by new trials.

```bash
export PYTHONDONTWRITEBYTECODE=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1
# Reanalyze without training, checkpoints, downloads or a GPU:
.venv/bin/python -m flies.chess.mechanism_report \
  --evidence flies/chess/mechanism_evidence.json \
  --output flies/chess/artifacts/mechanism-reproduced-summary.json

# Repeat a training condition into a NEW directory; frozen parent sources already exist.
.venv/bin/python -m flies.chess.mechanism_train --seed 11 --updates 48 --method loo \
  --corpus flies/chess/research_data/mechanism_curriculum.json \
  --sources flies/chess/artifacts/mechanism/replication-11/checkpoint_0056_0.pt flies/chess/artifacts/mechanism/replication-11/checkpoint_0056_1.pt \
  --output flies/chess/artifacts/mechanism-repeat-loo-11
# Other matched conditions: --method ema, --method zero, or --method loo --stage-only.
# Repeat at seeds23/37; resume with identical settings and --resume <run>/resume.pt.
```

Exact original CLI arguments/settings are in each archived config and MECHANISM_PROTOCOL.md. For the standard frozen test runner, completed runs live at `artifacts/mechanism/{loo,ema,stage,zero}-{seed}`. Fresh run directories are now protected against accidental overwrite; use existing evidence for reanalysis rather than rerunning completed training. Dataset generation refuses to replace a locked test.

**Scaling decision:** the matched-adaptation gate passes on the fresh curated distribution. New adaptation learning, LOO superiority and mate-two progression gates do not pass; meaningful general full-game gains are absent. Continue only carefully controlled local work on training/validation data if pursuing research further. The local software is usable and full-graph CPU compatibility has been verified, but a shared-GPU learning experiment remains unapproved and scientifically premature. See [SCALING.md](SCALING.md) and the final specification audit in [AUDIT.md](AUDIT.md).
