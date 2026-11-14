# Elton: Phase 1 Workday-inspired reinforcement-learning simulation

This is a local simulator and training harness. It never opens employer websites
or submits real applications. Phase 2 merging and Phase 3 live Workday integration
are not performed here. All implementation changes are inside `flies/jobs/`.

The implementation is configured for the existing full FlyWire `FlyBrain` on a
CUDA GPU. Local validation used the real 5,000-neuron `tiny` graph; **full-size GPU
runtime/memory must be checked on the 4090**. Local proficiency is not a release
requirement for this handoff. The local pilot is not a competent applicant.

## On the RTX 4090 machine

Run from the repository root, with the same shared core APIs. These commands do
not merge branches or contact employer websites:

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt -r flies/jobs/requirements.txt
# PyTorch must be a CUDA-enabled installation suitable for that machine.
python -c "import torch; print(torch.__version__, torch.cuda.is_available()); print(torch.cuda.get_device_name(0))"

python scripts/download_data.py
python -m core.graph --size tiny
python -m core.graph --size full
python -m unittest discover -s flies/jobs
python -m core.check jobs
python -m flies.jobs.preflight --size full --device cuda --batch-size 32
```

Preflight constructs the requested graph and tests neural forward/backward and an
optimizer update at the requested batch size, printing neuron count, graph hash
and peak CUDA allocation. It saves no trained policy and is not learning evidence.
If it reports missing data, CUDA or API incompatibility, resolve that before
training. Graph construction and GPU preflight are intentionally not simulated.

Choose a persistent **private Linux filesystem** for simulated credential keys
before the first run. Do not use a Windows-mounted directory that ignores chmod:

```bash
export ELTON_SECRET_DIR="$HOME/.local/share/elton-simulation-secrets"

# Fresh full connectome: start with a short GPU run.
python -m flies.jobs.learning \
  --size full --device cuda \
  --out flies/jobs/artifacts/workday_full \
  --episodes 16 --chunk 32 --checkpoint-every 10 --updates 100 \
  --eval-every 10 --eval-episodes 64 --minutes 15

# Continue the SAME experiment: weights, optimizer, RNG, accounts and curriculum.
python -m flies.jobs.learning \
  --resume --device cuda \
  --out flies/jobs/artifacts/workday_full \
  --updates 2000 --eval-every 25 --eval-episodes 64 --minutes 120
```

The full graph must start a **new** experiment. A tiny-brain checkpoint cannot be
expanded into a larger graph. The graph size/hash and observation/action schema
are checked at load; incompatible policies fail instead of falling back to a
random brain. `--resume` honors saved settings; explicit `--episodes`, `--chunk`
and `--lr`, `--gamma` and `--entropy` override their saved values. `--checkpoint-every` defaults to 10 updates; first, final and validation updates also checkpoint. Ctrl+C/SIGTERM finish the current update and save; a hard kill can lose updates since the last checkpoint. Lower `--chunk` (e.g. 16) and `--episodes`
if memory is limited. No specific throughput or full-brain VRAM requirement has
been measured on this CPU-only development machine.

The shared `python -m core.train jobs` command still works for a fixed Level 9
task, but **does not provide this fly's curriculum or resumable optimizer/state**.
Use `flies.jobs.learning` for the intended training run. The shared remote submit
helper invokes the generic trainer; do not assume it invokes this custom loop.
Use only remote commands authorized by root AGENTS.md; no GPU job was submitted
by this implementation.

## Held-out evaluation

Training uses only `train` and `validation` generators. Explicit test evaluation
never changes training weights or curriculum. Use a new output directory when
retaining multiple test reports:

```bash
python -m flies.jobs.evaluate \
  --checkpoint flies/jobs/artifacts/workday_full --device cuda \
  --out flies/jobs/artifacts/workday_full_test \
  --episodes 64 --levels 1 4 5 6 9 10 --split test
```

Random, saved untrained, trained and lesioned policies receive equivalent forms,
initial account states and action RNG seeds. Reports include paired bootstrap
intervals. `full_workflow_learning_passed` requires improvements in completion
AND field accuracy against all three controls at Levels 4, 5, 6, 9 and 10. A manual
review, skipped verification workflow or seeded prior submission is never a
completed application. Small local smoke samples are explicitly not evidence of
statistical significance.

Outputs: `progress.json`, `progress.csv`, `REPORT.md`, `learning_curve.svg`,
compressed action logs, `untrained.pt`, immutable checkpoint generations, and an
atomic `latest.json` pointer. Generations contain separate model, optimizer/RNG/
curriculum state and simulation-account/application state. An interrupted partial
generation cannot replace the last complete pointer. Credentials are **not** in
model files or action logs. Preserve the same private vault when moving/resuming
an experiment with simulated accounts; missing or mismatched keys fail clearly.
The default key directory is private OS temporary storage; set `ELTON_SECRET_DIR`
as above for durable storage across reboots.

## Watch the actual neural policy in a local browser and 3D station

```bash
python -m playwright install chromium
python -m flies.jobs.setup_assets

python -m flies.jobs.run \
  --policy trained --checkpoint flies/jobs/artifacts/workday_full --device cuda \
  --browser --headed --station --level 9 --episodes 1 \
  --state-dir flies/jobs/local_state/workday --delay 0.3
```

The headed browser remains open after execution. Ctrl+C exits it. The policy will
perform poorly until sufficiently trained. To watch simpler tasks use `--level 1`
and a fresh `--state-dir`; an existing pending form resumes its original level.
There is no automatic correct-answer filling when the policy struggles.

The embedded station imports `station.js` and the existing shared Fly renderer.
It consumes the **same** action/reward/checkpoint events as the application
browser and acknowledges each animation. It does not select another action or
run a separate simulator. All assets are loopback/local; the pinned Three.js
module is downloaded only by the explicit setup command, with checksum validation.
The asset server cannot serve credentials, checkpoints, or arbitrary repository
files. Employer/external browser requests are blocked.

A local browser-based paired evaluation can also be run:

```bash
python -m flies.jobs.evaluate \
  --checkpoint flies/jobs/artifacts/workday_full --device cuda \
  --out flies/jobs/artifacts/workday_browser_test \
  --episodes 16 --levels 9 10 --browser --headed

python -m flies.jobs.run --status --state-dir flies/jobs/local_state/workday
```

For an explicit fresh baseline: `python -m flies.jobs.run --policy untrained
--browser --headed --level 1 --state-dir flies/jobs/local_state/untrained`.
Uniform random inference is available with `--policy random`. Neither mode is
presented as trained. Training and browser inference are separate processes;
restart the browser runner to load a later checkpoint generation.

**Do not use `server.serve` for this integrated execution view.** It loads only
`runs/jobs/brain.pt`, can fall back to an untrained graph, creates its own simulation,
and does not share the persistent queue. Its old 52-input jobs checkpoint is
incompatible. Changing shared server/web behavior is prohibited by AGENTS.md;
the fly-local runner supplies correct provenance and shared events instead.

## What the environment implements

- Tenant-isolated account registration, confirmation-password constraints, existing
  accounts, login, incorrect credentials, email-first flows and auth at different
  workflow positions. Account authentication does not transfer to another tenant.
- **No inbox, received-email, email code or verification actions.** Workflows that
  require verification are skipped, with zero completion reward.
- Actual local PDF selection/upload, correct/wrong files, and none/partial/accurate/
  corrupt parsing. Parsing happens only after the policy selects upload. The policy
  must inspect/correct fields and pass independent final validation.
- Personal/contact information, repeatable experience and education sections,
  dates, status, adding/removing entries, searchable option menus, yes/no radio
  controls, dropdowns, free text and screening questions. Empty extra entries
  must be removed. Only resume-backed records can pass validation.
- Multi-page navigation, back/edit links, review, required/optional constraints,
  invalid-email/credential/form feedback, duplicate detection and simulated receipts.
- Random field order, labels, options, layouts, field counts, page counts, auth
  order/states, parsing outcomes, questions, prior submissions and error wording.
  Train/validation/test label namespaces and generator seed ranges are disjoint.

The supplied URL text file remains the sequential practice queue. URLs identify
local simulations; their content is not fetched. Pending forms, accounts, receipts,
reward accounting and RNG state persist separately from weights. A timeout pauses
the current queue item rather than skipping it. Run one queue worker per state
folder. Training uses separate simulation state and never updates the browser
queue's ledger.

## Neural interface and reward integrity

`contract.py` defines `ApplicationEnvironment`, public `Action` records and a
64-float sensor. `SimulatedWorkdayEnvironment` and `SimulatedBrowserEnvironment`
share this contract. Eight fixed descending-neuron groups select dynamic slots
containing concrete public actions: focus, enter a profile-backed literal, clear,
choose, click, upload, search, page choices, select a profile section, and request
review. More than eight choices use an explicit paginated palette. The palette
and profile picker are tools, not hidden Workday fields; a future browser adapter
must provide the same tool interface.

Candidates are enumerated from public controls/options and the complete profile
catalogue, never ranked or filtered by the answer oracle. Entering a field does
not select the right profile section or value. The policy selects every operation.
Strings are independently compressed into lexical sensors; no field/candidate
similarity, expected answer, semantic key, correctness flag or computed accuracy
enters the encoder. These finite sensors are a significant language/transfer
limitation; more neurons alone do not guarantee language understanding.

FlyBrain's wiring, signs, sensory paths and fixed descending-neuron readout are
unchanged. Training uses shared `core.train.rollout` and gradient batches with
reward-only REINFORCE. There are no demonstrations, teachers, learned output
heads or action-repair fallbacks.

Rewards are approximately: first correct field +1; first correction +1 extra;
registration +5; login +3; first correct upload +3; validated page +1; independently
validated and reviewed submission +20. Each action costs .02; wrong actions -1,
failed submit -10, unnecessary review -1 and timeout -5. Repeating the same action
in the same public state adds a bounded escalating penalty. Intermediate rewards
are paid once per field/stage/event; correction cycles cannot repeatedly farm them.
Safe manual review is a separate low-reward outcome. There is currently no
relevance bonus, because the URL-only input supplies no verified job requirements.

Curriculum levels: 1 simple personal; 2 reordered multiple fields; 3 options and
questions; 4 auth; 5 PDF/parsing; 6 experience/education; 7 more pages; 8 recovery;
9 full workflows; 10 unfamiliar combinations. Promotion requires two successive
validation evaluations of at least 32 forms, >=75% completion and >=85% field
accuracy. Thirty percent of training samples rehearse earlier levels. Time and
update count do not promote the fly.

## Profile provenance

`verified_profile.json` contains resume facts and the user-confirmed email/phone.
Employment dates, degree, institution, eligibility and availability remain unknown.
Missing required facts require explicit manual review and never earn completion.
The user-requested address is a **simulation-only public-building fixture**, not
verified residence: Boston Public Library, 700 Boylston Street, Boston MA 02116,
[official address](https://www.bpl.org/locations/central/). Public observations carry
its provenance. A future live adapter must exclude simulation-only values and
request a verified personal address before use.

## Tests and legacy behavior

```bash
python -m unittest discover -s flies/jobs
python -m core.check jobs
# Opt-in real Chromium mechanics/DOM/station tests (use the installed browser path):
PLAYWRIGHT_BROWSERS_PATH=/path/to/playwright/browsers \
  python -m unittest flies.jobs.test_workday_browser flies.jobs.test_browser
```

The fixture driver in test files uses the oracle solely to exercise mechanics. It
is never imported by production policy/training code, and its successes are not
reported as neural learning. Earlier five-field code remains under `legacy_*`:
`python -m flies.jobs.legacy_run --help`, `legacy_learning`, and
`legacy_browser_evaluate`. Earlier history and checkpoints are retained; no
schema migration silently converts old completions into new submissions.

See [GPU_TRAINING_READINESS.md](../../GPU_TRAINING_READINESS.md) for the audit, tests, measured local limitations and handoff
status. Source changes are on `job-apps`; no merge or live integration is performed.
