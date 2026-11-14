# Elton learns job applications

Elton uses the existing `core.brain.FlyBrain`: sensory inputs enter the FlyWire
connectome and eight fixed descending-neuron groups select all actions. Training
uses reward-only REINFORCE and the shared rollout/gradient infrastructure. There
is no teacher, output classifier, field-to-answer policy, or fallback fill routine.
The validator alone maps randomized field IDs to expected resume facts.

This is a **local simulation and local browser fixture**, not an employer-site
integration. The supplied `elton_fly_job_links.txt` remains the ordered job queue.
No simulated completion represents a real submitted application. Original
`practice` ledger records are preserved for inspection, not learning evidence.

## Environment

Each form samples three to five fields, shuffled labels/order and dropdowns,
text/textarea/select controls, required flags, one to three pages, and stack/grid/
card layouts. Training starts with simpler forms. Held-out forms always use full
complexity, disjoint seeds, and different label prefixes. Error wording varies.
The neural sensor encodes literal field labels, shuffled resume snippet labels
and values, dropdown options, and public control state into 64 floats. It does
not compute field/candidate similarities or receive oracle correctness features.
The finite hashed lexical sensor and five-fact resume limit language generality.

Actions 0–4 enter a selected public candidate; 5 moves focus; 6 continues/submits;
7 switches typing/selecting mode. Navigation and submission are brain decisions.
Masks reflect only structural availability. Correctness never masks choices.
First correct entries earn +1, first correction +2, incorrect entries -1,
independently validated completion +15 (plus up to +5 known skill relevance).
Repeated wrong entries cost at least -1.15; repeated operations in the same state add escalating penalties capped at -1.5; per-field reward payments cannot recur.
URL-only jobs do not invent relevance or requirements.

## Reproduce training and evidence

Use a Python environment with the repository dependencies and real `data/tiny.npz`.
In this workspace `/tmp/jobs-check-venv/bin/python` supplies those dependencies.

```bash
python3 -m unittest flies.jobs.test_task
python3 -m core.check jobs
python3 -m flies.jobs.learning --out flies/jobs/artifacts/new_run --updates 180 --minutes 15
# Continue from weights, optimizer, episode/update counters and RNG states:
python3 -m flies.jobs.learning --out flies/jobs/artifacts/new_run --resume --updates 180 --minutes 15
```

Existing checkpoints cannot be silently overwritten. The runner validates the
observation schema version. Previous schemas require new training. The training
runner rejects a synthetic graph. Local experiments use the 5,000-neuron real
FlyWire subgraph, not the full connectome; full GPU validation is still required
by root AGENTS.md before a finished demo can be claimed.

Each experiment saves `untrained.pt`, `latest.pt`, interval checkpoints,
`optimizer.pt`, `actions.jsonl.gz`, `progress.json` and `REPORT.md`.
Every action log includes the numeric observation, action, entered value, result,
reward, cumulative reward, policy and checkpoint/revision hash. Newly started
processes also include the literal public observation. Evaluation uses paired
forms and action seeds for random, fresh untrained, trained and lesioned policies.
The final test partition differs from interval evaluation. Paired bootstrap
intervals must show improvements in completion AND field accuracy against all
three controls before `acceptance_passed` is true. Failed results remain visible.
A passing interface/lesion check alone is not evidence of learned competence.

## Browser and queue

```bash
python3 -m flies.jobs.browser_evaluate --out flies/jobs/artifacts/new_run --episodes 16
python3 -m flies.jobs.run --policy brain --checkpoint flies/jobs/artifacts/new_run/latest.pt --browser --episodes 5 --verbose
python3 -m flies.jobs.run --status
python3 -m flies.jobs.run --details
# Inspect original records, which lack action provenance:
python3 -m flies.jobs.run --legacy-history --details
```

Install Playwright/Chromium for browser commands. Browser evaluation uses fresh
seeded local DOM fixtures, reads labels/options/values from the DOM, executes only
selected neural actions and verifies DOM contents after every action. External
HTTP(S) requests are blocked and form submission is prevented. It writes paired
`browser_report.json` and `browser_actions.jsonl.gz`. This does not demonstrate
transfer to arbitrary employer websites.

The sequential runner persists history separately in SQLite under `practice_v6`.
A 40-action chunk pauses unfinished work; it resumes the SAME form, values,
shuffled resume slots and reward accounting next time. Completion or an explicit
blocking adapter error advances the queue. Training never updates that ledger.
Live-mode history is separate. Retryable errors preserve pending work; unexpected
browser errors save state and stop. Run only one queue worker per history file.

The latest audit is documented in `AUDIT.md`; reproducible measured results are
in `artifacts/audit_v6/REPORT.md`. The learning acceptance test currently fails.
