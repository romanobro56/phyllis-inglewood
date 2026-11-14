# Learning audit

The three historical completions are **practice records**, not evidence of
learning or employer submissions. SQLite contains three completed legacy rows
and one pending row under `practice`, with no action events. Their saved form
schema exposes semantic keys (`name`, `email`, etc.) and prebuilt candidate
feature vectors. Without their original action traces, it is impossible to
attribute those completions to a neural checkpoint. They are preserved and
excluded from the new queue namespace and training metrics.

At inspection time, `task.py` already used a private validator, randomized forms,
eight neural action slots and reward learning. There was no active automatic
answer fallback in that implementation. The unused older `harness.py` retains
semantic-key validation; the current policy and browser runner do not import it.
It would be inaccurate to claim the historical source of the three successes
has been conclusively reconstructed.

## Corrections

- Resume slots previously had a fixed order and their literal values did not
  reach the neural sensors. Slots are now shuffled and both labels and values
  are independently encoded from public text. No field-to-answer similarity,
  matching rule or correctness flag is supplied.
- Shuffled slots persist with pending form state. Changing the private answer
  oracle does not change sensors, candidates, masks or navigation (tested).
- Browser mode previously executed actions in HTML but sourced neural inputs
  from simulator state. It now reads labels, control types, required flags,
  options and entered values from DOM controls, using the same public encoder.
  Grid/card layouts are rendered, rather than merely named in metadata.
- The station still expected old action names and colored most entries green.
  It now animates all eight current actions, displays actual outcome captions
  and labels the activity as simulation.
- Old checkpoints with the same observation dimension could be accepted by the
  sequential runner despite different sensor semantics. Environment manifests
  are now checked. Simulation writes to live history are explicitly rejected.
- Experiment v5 showed reward improving through cheap navigation and repeated
  entries, without field accuracy improving. Version 6 penalizes repeated wrong
  entries more than first errors and adds escalating, bounded penalties for
  repeating the same operation and resulting public state. These counters are
  reward accounting only, never action selectors or sensor features.
- The README contained obsolete action, observation, reward and workflow
  descriptions; it now documents the actual experiment and its limitations.

## Evidence and limits

`artifacts/audit_v5/` preserves the first experiment and its avoidance behavior.
`artifacts/audit_v6/REPORT.md` and `progress.json` contain fresh baseline,
interval evaluation, final paired controls and confidence intervals for the
corrected rewards. Action traces include observations, chosen actions, entered
values, outcomes, rewards, episode totals, checkpoint references and parameter
hashes. Training saves weights separately from the application ledger, along
with optimizer and random-generator state; v5 training was resumed from its
saved state rather than reinitialized.

The local brain is a real 5,000-neuron FlyWire subgraph with the existing fixed
descending-neuron readout. It is not the full 134,181-neuron connectome. All
experiments are reward-only; no demonstrations or teacher actions are used.
Comparisons must improve both completion and field accuracy over random,
untrained and lesioned controls. Better reward alone is not acceptance.

Browser results cover unseen **local** fixtures with external requests blocked
and native submission prevented. DOM values are checked after every operation.
They do not establish transfer to employer websites. No real application was
submitted during this audit. Full-brain GPU validation and a minute of the 3D
world demonstration remain outside the evidence from these local runs.

## Final measured outcome

Version 6 completed 784 training episodes (49 updates), including a verified
resume from update 48. On the same 64 final held-out forms/seeds:

| Policy | Mean reward | Completion | Field accuracy | Actions | Error rate |
|---|---:|---:|---:|---:|---:|
| Random | -35.09 | 1.56% | 12.11% | 39.86 | 77.65% |
| Untrained FlyBrain | -34.50 | 0% | 14.04% | 40.00 | 75.90% |
| Trained FlyBrain | -25.28 | 1.56% | 12.99% | 39.94 | 43.88% |
| Lesioned trained FlyBrain | -34.12 | 0% | 18.05% | 40.00 | 75.23% |

**The objective is not achieved.** Completion and accuracy improvements are not
statistically established. Higher return mainly reflects learning to avoid
costly mistakes, not learning to fill applications. The stricter repetition
penalties reduce a cheap-repeat incentive but do not solve this optimization
problem. `acceptance_passed` is false; these weights must not be presented as a
competent applicant. Further reward/credit-assignment and sensory representation
experiments are needed, followed by full-brain GPU training and fresh evaluation.
No change to shared core files was made in this audit, as required by AGENTS.md.

Validation: 14 task tests and the core interface/training/lesion check pass. A
separate real Chromium integration test passes: changing DOM label text changes
the observation, changing the answer oracle does not, and an incorrect neural
slot selection remains incorrect in the DOM. Station actions were updated but
Node is unavailable and a minute-long 3D-world run has not been verified.

The final browser comparison used eight paired unseen local fixtures per policy.
All four policies completed 0/8. Field accuracy: random 20.83%, untrained 10.83%,
trained 17.08%, lesioned 20.83%. See `artifacts/audit_v6/browser_report.json` and
its action log. `parameter_changes.json` confirms weights changed while wiring,
signs, sensory-neuron indices and descending-neuron action groups stayed fixed.
