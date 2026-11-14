# Local experiments (excluded from source handoff)

Each directory is an independent experiment/evaluation. Do not mix its checkpoints,
optimizer state, private vault, generator version or results with another run.

- `pilot`, `reward_v4`, `audit_v5`, `audit_v6`: historical five-field experiments.
  Their scripts remain under `legacy_*`; their limitations are recorded in `AUDIT.md`.
- `workday_phase1_smoke`, `workday_phase1_v1`: exploratory Workday pilots with
  earlier reward/action-budget settings. Retained as historical evidence, not
  current proficiency claims or recommended starting checkpoints.
- `workday_phase1_v2`: local tiny-brain pipeline pilot; not a competent full-workflow agent.
- `workday_phase1_smoke_evaluation`: small held-out plumbing evaluation, not
  statistical evidence of learning.
- `readiness_smoke_*`: abbreviated initialization/resume checks for deployment.
- `elton_workday_gpu_source.zip`: source-only handoff, rebuilt with
  `python -m flies.jobs.package`; excludes all experiment state and credentials.

A fresh GPU experiment belongs in its own `workday_full/` directory. Historical
outputs are retained because deleting them would destroy reproducibility evidence.
No generated static dataset is required: the generators use the verified profile,
explicit split namespaces, RNG seeds and checkpointed RNG state.
