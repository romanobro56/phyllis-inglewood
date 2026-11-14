# GPU Training Readiness

**Status: ready to transfer for a hardware-gated full-connectome training run.**
The simulator and training/inference software passed the local correctness checks
below. The RTX 4090 must pass the supplied full-size CUDA preflight before a long
run. **Full-size CUDA execution, VRAM use and throughput are not verified on this
CPU-only host.** Learned proficiency is not a readiness requirement or claim.

Prepared on 2026-11-14, branch `job-apps`. No merge, push, live integration or
full-scale local training was performed. The implementation and report are intended to travel with the local `job-apps`
commit. A source-only handoff archive is also provided for deployment without Git. Apply it to a compatible Phyllis Inglewood checkout.

## 1. Implementation summary

`flies/jobs/task.py` exposes `SimulatedWorkdayEnvironment` to the existing core
registry. The Phyllis Inglewood `FlyBrain` is unchanged: sensory-neuron input,
fixed descending-neuron action groups, fixed wiring and excitatory/inhibitory
signs, and trainable neural parameters. No learned output head, teacher, field
matching policy or correct-answer fallback was introduced.

The simulation includes:

- Tenant-specific registration, password confirmation/constraints, existing
  accounts, login, bad credentials, email-first auth, and different auth positions.
- Persistent simulated accounts, drafts and application receipts, separate from
  model/optimizer files. Seeded prior submissions have an existing tenant account.
- Resume PDF selection/upload and simulated none/partial/accurate/corrupt parsing.
- Personal/contact fields; repeatable experience and education entries; adding,
  removing, clearing and correcting entries; dropdown/combobox/radio/text/date
  controls; screening and eligibility questions; back/edit navigation and review.
- Independent required-field, format, record-count, upload, authentication and
  final submission validation. Failed validation produces public feedback and
  allows recovery. Duplicates cannot earn another completion reward.
- Explicit review for missing facts. Unknown education credentials, dates,
  eligibility and availability are not fabricated. Email/inbox/verification work
  is excluded: verification-required workflows are skipped without completion.
- Varied labels, field/option order, stages, layouts, account states, questions,
  parsing results and errors. Train/validation/test use disjoint label namespaces
  and seed ranges. Generator and action RNG states are saved.

The user-confirmed email is `eltoninglewood@gmail.com` and phone is `5082089276`.
The user-requested address fixture is [Boston Public Library's](https://www.bpl.org/locations/central/) publicly listed
address, **not verified residence**; `verified_profile.json` and public observations
mark it simulation-only. A future live adapter must exclude it until verified.

The browser-neutral `contract.py` provides public views, concrete actions and a
64-float sensor with eight dynamic action slots. Values come from explicit public
profile choices or visible options. The policy selects the element, profile
section/value, operation and navigation. The validator's expected-answer mapping
is not included in observations, masks or action selection.

## 2. Test results and evidence

Machine-readable evidence is in [`flies/jobs/readiness/`](flies/jobs/readiness/).
Raw command/test logs are retained locally alongside the JSON summaries.

| Check | Result | Scope / explanation |
|---|---|---|
| Unit suite | **35 passed; 7 skipped; 0 failures/errors** | 42 discovered tests. Six browser tests are opt-in; one CUDA test requires a GPU. |
| Real Chromium suite | **6 passed; 0 failures/errors** | The six opt-in tests ran separately with local Chromium outside the process sandbox. |
| Core integration check | **PASS** | Interface, 5,000-neuron real FlyWire brain, short RL update and lesion response. |
| CPU hardware/API preflight | **PASS** | Real tiny graph, forward/backward, optimizer update, batch size 4. |
| Training initialization | **PASS** | Two tiny-brain updates, two episodes per update; checkpoint/report/log creation. |
| Separate-process resume | **PASS** | Update 2 → 3; episodes 4 → 6; prior model hash loaded; optimizer state retained; baseline file unchanged. |
| Paired evaluation plumbing | **PASS** | Random/untrained/trained/lesioned controls produced identical held-out form hashes. Two forms only; not proficiency evidence. |
| Full-size CUDA preflight | **NOT EXECUTED** | No CUDA GPU or CUDA-enabled PyTorch here. CLI correctly exited with a clear diagnostic (code 2). This expected guard result is not a successful CUDA test. |
| Full-size prolonged training / learned multi-stage proficiency | **NOT REQUIRED / NOT RUN** | Deferred to the GPU computer; no fabricated success criterion. |

Key evidence:

- [`unit_tests.json`](flies/jobs/readiness/unit_tests.json)
- [`browser_tests.json`](flies/jobs/readiness/browser_tests.json)
- [`smoke_tests.json`](flies/jobs/readiness/smoke_tests.json)
- [`local_environment.json`](flies/jobs/readiness/local_environment.json)
- [`core_compatibility.json`](flies/jobs/readiness/core_compatibility.json)
- [`syntax.json`](flies/jobs/readiness/syntax.json)
- [`gpu_full_config.json`](flies/jobs/readiness/gpu_full_config.json)
- [`cleanup.json`](flies/jobs/readiness/cleanup.json)

The unit tests verify that reward optimization changes neural parameters while
wiring/signs/readout buffers remain unchanged. Checkpoint tests use **nonempty Adam
momentum state** and verify that the next identical optimizer step after loading
matches the uninterrupted result. Missing simulation state, mismatched vaults and
old observation/action schemas fail instead of silently resetting or substituting
an untrained policy.

Browser tests verify literal selected values (including incorrect values), DOM
sensors, auth and actual local file inputs, radio controls, credential redaction
even when a secret is selected for the wrong field, and station acknowledgment
of the same execution event. Test-only oracle-assisted fixture drivers exercise
mechanics; their successes are never reported as learned behavior.

The local environment is Python 3.12.3, Linux/WSL2 ARM64, PyTorch 2.15.0+cpu,
NumPy 2.5.4, Playwright 1.65.2 and Matplotlib 3.11.2. Only `data/tiny.npz` is present.
These are recorded observations, not GPU wheel/version requirements.

## 3. Simulation validation and repository cleanup

Observations, masks and candidates depend on public controls/profile data, not
oracle correctness. Correctness affects rewards and post-action validation
feedback. Invalid actions and unsuccessful submissions are penalized. Terminal
applications reject further steps. Time-limited browser episodes preserve the
pending application rather than advancing the queue.

Intermediate rewards have one-time payment accounting. Correction cycles cannot
repeatedly farm positive rewards; removed records lose their values but retain
reward-payment history. Registration, login, upload, stage progress and final
submission have separate reward events. Full completion requires independent
validation plus review. Seeded prior applications, safe manual reviews and skipped
verification are tracked separately from completed applications.

Cleanup removed rebuildable Python caches and a stale missing-learning-curve
warning where a valid SVG already existed. No useful datasets, weights, application
history or failed-experiment evidence were deleted. `artifacts/README.md` identifies
historical pilots and the current smoke experiment. Legacy modules and their
dependencies remain because tests and compatibility entry points still use them.
The original resume, URL file, historical audits and ledger are preserved.

Generated weights, experiments, private state, caches, editor swaps and logs are
ignored under `flies/jobs/.gitignore`. A source-only archive excludes them. Shared
core/server/web files and unrelated existing repository changes were not edited.
The root report is the one explicitly requested exception to the fly-directory
scope. The working tree is **not globally clean**: unrelated pre-existing changes
remain. Use `git log` for current local commit status; this report does not imply
a remote push.

## 4. Training pipeline status

Use `python -m flies.jobs.learning`, not the generic core trainer, for this
curriculum/resume pipeline. It reuses `core.train.rollout` and gradient batches
for reward-only REINFORCE. Network output selects every action.

Configurable flags include graph size/device, episodes per update, gradient chunk,
learning rate, discount, entropy, seed, update/time limits, validation interval/sample
count and checkpoint interval. Validation-gated curriculum requires two consecutive
checks of at least 32 forms with >=75% completion and >=85% field accuracy; 30%
rehearsal retains previous skills. Elapsed time does not advance levels.

Checkpoints are immutable generations selected by an atomic `latest.json` pointer.
The default interval is ten updates; first, final and validation updates also save.
Ctrl+C/SIGTERM requests completion of the current update followed by a save. A hard
kill can lose work after the latest completed checkpoint. Partial/orphan generations
cannot overwrite the latest complete pointer. Checkpoints are retained; budget disk
space and archive older generations deliberately. Time limits are checked between
updates; baseline, validation and checkpoint work can extend total wall time.

No known local implementation test is failing. Numerical execution and persistence
are verified; full-size GPU fit and learning effectiveness remain target-machine
measurements, not promises. Sparse-connectome training is not distributed/DDP.

## 5. GPU requirements and setup

Target: NVIDIA RTX 4090 (24 GB) with a compatible driver and CUDA-enabled PyTorch.
Planning recommendation: Linux, Python 3.12, 32–64 GB host RAM and ample SSD space
(start with 50–100 GB free). These are planning recommendations, **not measured
minimums**. Full-graph memory/throughput have not been benchmarked locally.

Install a CUDA PyTorch build matching the target machine using the
[official PyTorch installer](https://pytorch.org/get-started/locally/), then verify
CUDA before starting. Do not install this development machine's CPU-only wheel on
the GPU host. Root `requirements.txt` supplies shared dependencies;
`flies/jobs/requirements.txt` adds plotting and optional browser inspection.
Training does not require Chromium, a display, real email, an LLM API or Workday
credentials. Headed visual inspection requires a working desktop/display.

From a compatible repository checkout with the new `flies/jobs` source applied:

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt -r flies/jobs/requirements.txt
# If needed, install the correct CUDA-enabled torch build using the official selector.
nvidia-smi
python -c "import torch; assert torch.cuda.is_available(), 'CUDA unavailable'; print(torch.__version__, torch.cuda.get_device_name(0))"

python scripts/download_data.py
python -m core.graph --size tiny
python -m core.graph --size full
python -m unittest discover -s flies/jobs
python -m core.check jobs
python -m flies.jobs.preflight --size full --device cuda --batch-size 32
```

The preflight constructs the **actual full graph**, runs neural forward/backward
and one optimizer update, and reports neuron count, graph hash and peak CUDA memory.
It is a hardware/API check, not a proficiency run. If it fails, resolve that error
before long training. No shared-core changes are required by this implementation.

## 6. Exact training, resume, evaluation and visual commands

Use a private persistent Linux directory for synthetic credential masters. Do not
use a Windows-mounted filesystem that ignores mode 0600. Keep this directory when
resuming account state; it is intentionally excluded from the transfer archive.

```bash
export ELTON_SECRET_DIR="$HOME/.local/share/elton-simulation-secrets"
mkdir -p "$ELTON_SECRET_DIR"
chmod 700 "$ELTON_SECRET_DIR"

# Fresh, conservative full-size GPU run (a NEW experiment).
python -m flies.jobs.learning \
  --size full --device cuda --seed 2026 \
  --out flies/jobs/artifacts/workday_full \
  --episodes 16 --chunk 32 --lr 0.0005 --gamma 0.997 --entropy 0.04 \
  --checkpoint-every 10 --eval-every 10 --eval-episodes 32 \
  --updates 100 --minutes 15

# Longer continuation, preserving model, Adam state, RNG, accounts and curriculum.
python -m flies.jobs.learning \
  --resume --device cuda --out flies/jobs/artifacts/workday_full \
  --checkpoint-every 10 --eval-every 25 --eval-episodes 64 \
  --updates 2000 --minutes 120

# Explicit paired held-out evaluation; does not alter training/curriculum.
python -m flies.jobs.evaluate \
  --checkpoint flies/jobs/artifacts/workday_full --device cuda \
  --out flies/jobs/artifacts/workday_full_test \
  --episodes 64 --levels 1 4 5 6 9 10 --split test
```

If memory is limited, lower `--chunk` and/or `--episodes`; explicit overrides are
honored on resume. If no completed generation exists after an early failed run,
choose a fresh output directory. **A tiny checkpoint cannot be expanded into the
full connectome**; start the full graph with untrained parameters. If moving an
already-running full experiment, separately retain its vault and use `--vault-file`
to identify that same key on the new machine. Do not copy private keys into Git.

For visible local inference and the shared station:

```bash
python -m playwright install chromium
python -m flies.jobs.setup_assets
python -m flies.jobs.run \
  --policy trained --checkpoint flies/jobs/artifacts/workday_full --device cuda \
  --browser --headed --station --level 9 --episodes 1 \
  --state-dir flies/jobs/local_state/workday --delay 0.3

python -m flies.jobs.run --status --state-dir flies/jobs/local_state/workday
```

This runs inference, not online training. The browser and station consume the same
selected-action/outcome/reward stream and identify the actual checkpoint. They
block employer requests and show a simulation watermark. No random fallback is
presented as trained. Use a fresh state directory to change the initial level;
pending forms otherwise resume their saved level and values.

The generic `server.serve` uses its own simulation and `runs/jobs/brain.pt`; its old
jobs checkpoint is incompatible. Use the fly-local runner for integrated viewing.
The shared remote submit helper invokes the generic trainer, not this custom
curriculum loop; do not assume it supplies these resume/curriculum features.

## 7. Expected outputs and transfer

A training experiment contains:

```text
manifest.json                  initial seed/settings/graph identity
untrained.pt                   untouched baseline model
latest.json                    atomic pointer to last complete generation
checkpoints/update_XXXXXX/
  brain.pt                     schema-tagged FlyBrain weights
  training.pt                  Adam, RNG, counters, curriculum, settings
  simulation_state.json        simulated accounts/applications, no raw passwords
progress.json / progress.csv   validation and training metrics
REPORT.md / learning_curve.svg
actions.jsonl.gz              observations, actions, rewards, hashes and provenance
```

Evaluation directories contain `evaluation.json`, paired confidence intervals,
reports, metrics and action traces. Reports track reward, field accuracy, account
creation/login/upload opportunities and success rates, application completion,
actions, validation errors and failed/skipped/manual-review outcomes. Learned
proficiency acceptance is separate from software readiness.

Create the source-only handoff with:

```bash
python -m flies.jobs.package
# On the GPU computer, from the existing repository root:
python -m zipfile -e /path/to/elton_workday_gpu_source.zip .
```

The archive includes the new fly source, tests, profile/PDF, URLs, local station
assets, this report and JSON readiness evidence, plus a SHA-256 file manifest. It
contains **no** model weights, experiments, accounts, ledger, vault, cache or virtual
environment. Shared core files, graph datasets and dependencies come from the
existing repository/setup commands. Core compatibility hashes are included.

## 8. Known limitations

- This is Workday-inspired, not a verified replica of each employer's platform.
  No live Workday navigation, submission, SSO/MFA, CAPTCHA handling, inbox or email
  verification is implemented. Verification-required simulations are skipped.
- Parsing is a simulated response based on the supplied profile; it is not a
  production PDF parser. Network latency, upload transport, asynchronous remote
  autocomplete, dynamic employer-specific rules and every ATS widget are not modeled.
- The paginated action/profile palette is an adapter tool, not literal Workday UI.
  A future adapter must reproduce this same contract. Finite 64-float lexical
  compression and a fixed resume can limit transfer/language understanding;
  a larger connectome alone does not establish generalization.
- Extra empty entries are tracked and must be removed; there is not an unlimited
  invented work/education catalogue. Only supplied verified records can validate.
- JSON account persistence and the sequential browser queue are single-worker.
  Long runs retain history and checkpoint generations; monitor host RAM, disk,
  compression and data-generation throughput. No distributed training is implemented.
- Synthetic credentials/hash mechanics are for local simulations, not a production
  credential store. Back up vaults separately. Bitwise CPU/GPU equivalence and
  cross-version CUDA determinism are not promised by RNG-state restoration.
- Local pilots are not competent applicants. Some failed learning criteria remain
  in historical reports deliberately; these do not indicate an implementation
  test failure or block this training-environment handoff.
- CUDA kernels, full-graph VRAM fit, prolonged GPU stability, graceful shutdown
  under every failure mode and eventual policy convergence still need target-host
  validation. The interval/safe-pointer design does not protect against every
  power-loss/filesystem failure or arbitrary deletion of checkpoint files.

## 9. Next steps

1. Transfer the source package or make the new local `job-apps` commit available
   to the GPU computer. The earlier baseline commit `a8fa705` lacks this new work.
2. Install target dependencies/graphs and pass the full CUDA preflight.
3. Run the short GPU experiment, verify GPU memory, outputs and resume, then extend
   training. Tune measured learning bottlenecks without inserting scripted answers.
4. Evaluate paired held-out distributions and lesioned controls at increasing
   curriculum levels. Use separate test outputs and retain failed results.
5. Only after simulation learning is established, perform the planned Phase 2 merge.
6. On a later separate branch, implement a real `WorkdayBrowserEnvironment` against
   this contract. Start with non-submitting evaluation, verified profile data and
   explicit approval before any real submission. Do not bypass site restrictions.

**Deliverable: validated simulation/training software and a hardware-checkable GPU
handoff, not a fully trained model and not evidence of real applications.**
