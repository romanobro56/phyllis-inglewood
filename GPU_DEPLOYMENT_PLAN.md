# Deploying custom-module queue support to the shared Runpod GPU

**Status: DEPLOYED 2026-11-14 ~21:20 UTC (steps 1–7).** Backup: `/workspace/flyhack-backups/20261114T211855Z`.
New worker pid 2565 (`worker.caps` verified). Full CUDA preflight job `212026-jobs-8b57`: PASS, 134,181 neurons,
graph `85c53631ff`, batch 32, peak 1,010,720,256 B. During step 3, the upload's `tar` failed to chown files on
MooseFS (exit 2, contents correct); fixed in `3fb8a07` (`--no-same-owner`). Laptops need that commit to `submit`.

Code: commit `ff36020` on `job-apps` (not merged to `main`).
Pod: `103.196.86.68:36812`, `/workspace/flyhack`. Every step below is run from the laptop repo root.
Steps 1–5 are owner actions (Roman authorized them); steps 6 onward are normal `submit`/`pull` use.

## What changes on the pod

| | File | Expected sha256 |
|---|---|---|
| **Upload (only this)** | `core/queue.py` | before `893d06b753583898fc708395458af7766cd54c517ce3c8f17fbe73fa45f3870a` → after `80887d985309fb0a1280135037209ff8c5a580651973e3aefc292dcc45dceb92` |
| Uploaded later by normal `submit jobs` | `flies/jobs/**` (67 files, no `.pt`, vault or sim state) | — |
| Created | `jobs/worker.caps` (by the new worker), `/root/.flyhack-secrets/` (vault dir) | — |

**Must stay untouched**: `core/brain.py` (`7a38bc6b0d25ad7b…`) and `core/graph.py` (`f423a66ce08b48ab…`),
which are *newer on the pod than in git*; every other `core/`, `server/`, `web/`, `scripts/`, `docs/` file;
`data/`; all of `runs/` (including `runs/jobs/brain.pt` and `result.json` from the earlier generic run);
other flies; queue records in `jobs/{pending,running,done,failed,cancelled,logs}`.

**Do not run `sync-core` or `bootstrap`**: both upload the repo's older `brain.py`/`graph.py` over the pod's.

The local `scripts/remote.py` change (upload fix, `--module` pass-through) needs no deployment: it runs on the
laptop. Teammates who only use generic `submit` are unaffected whether or not they pull it.

## 1. Announce and confirm the queue is idle

Post a short maintenance note in the team chat. Then:

```bash
python scripts/remote.py status        # require: no RUNNING line (PENDING jobs are fine; they're kept)
python scripts/remote.py shell <<'EOF'
cd /workspace/flyhack; ls jobs/running; pgrep -af "core.train|core.train_all|flies\." || echo "no trainer processes"
EOF
```

Proceed only if `jobs/running` is empty and no trainer process exists. Reason: a starting worker moves
anything in `running/` back to `pending/` and re-runs it, which would duplicate a live job.

## 2. Back up the current state on the pod

```bash
python scripts/remote.py shell <<'EOF'
cd /workspace/flyhack
B=/workspace/flyhack-backups/$(date -u +%Y%m%dT%H%M%SZ); mkdir -p "$B"
cp -p core/queue.py jobs_worker.log "$B/"
tar czf "$B/flies_jobs.tgz" flies/jobs
tar czf "$B/queue_records.tgz" jobs
sha256sum core/*.py > "$B/core.sha256"
sha256sum "$B/queue.py"; echo "BACKUP=$B"
EOF
```

Check that the backed-up `queue.py` hash is `893d06b7…`, and record the `BACKUP=` path for rollback.

## 3. Upload only `core/queue.py` and verify

```bash
git rev-parse --short HEAD                     # ff36020 (or a later job-apps commit containing it)
python -c "import sys; sys.path.insert(0, 'scripts'); import remote; remote.upload(['core/queue.py'])"
python scripts/remote.py shell <<'EOF'
cd /workspace/flyhack; sha256sum core/queue.py core/brain.py core/graph.py
EOF
```

Expect `80887d98…` for `queue.py`, and `brain.py`/`graph.py` still `7a38bc6b…`/`f423a66c…`.
Between this step and step 4, the old worker keeps running its in-memory code: generic submits work
exactly as before, and custom submits are refused because the old worker writes no `worker.caps`.

## 4. Restart the worker

Repeat the step 1 idle check, then:

```bash
python scripts/remote.py worker-stop
python scripts/remote.py worker-start          # note: this overwrites jobs_worker.log (backed up in step 2)
```

## 5. Verify the new worker supports custom modules

```bash
python scripts/remote.py status                # "worker: running (pid N)", history unchanged
python scripts/remote.py shell <<'EOF'
cd /workspace/flyhack; cat jobs/worker.pid; echo; cat jobs/worker.caps; echo; ps -o pid,args -p $(cat jobs/worker.pid); head -1 jobs_worker.log
EOF
```

Expect `worker.caps` = `{"pid": N, "custom_module": true}` with N equal to `worker.pid`.
Do **not** smoke-test the generic path with a real fly on the pod: `core.train` overwrites `runs/<fly>/`, so
a short test would replace that fly's trained brain. The generic path is verified locally (identical command
and job record), and the first line of the next teammate's job log (`$ ... -m core.train <fly> ...`) confirms it.

## 6. Prepare the private simulation vault directory

`/workspace` is a MooseFS FUSE mount that ignores `chmod` (a probe created mode 0600 and read back 0666; dirs
0777), so the vault's 0600 check correctly refuses it. The container disk enforces modes (probe: file 0600,
dir 0700, both checks pass). Loop-mounting an ext4/encrypted image on `/workspace` is impossible here (no
`/dev/loop*`, no `CAP_SYS_ADMIN`).

```bash
python scripts/remote.py shell <<'EOF'
mkdir -p -m 700 /root/.flyhack-secrets && chmod 700 /root/.flyhack-secrets && stat -c '%a %n' /root/.flyhack-secrets
EOF
```

Always pass `--vault-file /root/.flyhack-secrets/<experiment>.json` explicitly: the worker's environment has no
`ELTON_SECRET_DIR`, and the default would be `/tmp`.

## 7. Full-size CUDA preflight through the queue (hardware/API check, not training)

```bash
python -m core.check jobs                      # must print PASS
python scripts/remote.py submit jobs --minutes 5 --module flies.jobs.preflight \
    --args="--size full --device cuda --batch-size 32"
python scripts/remote.py logs jobs -f
```

Expect `"status": "PASS"`, `"preset": "full"`, `"synthetic": false` and a `peak_cuda_memory_bytes` value.
On out-of-memory errors, resubmit with `--batch-size 16` (then 8). `submit` uploads `flies/jobs`, replacing the
pod's older copy (backed up in step 2). Old files that no longer exist locally stay on the pod, which is harmless.

## 8. Short validation run (only after the preflight passes and is reviewed)

```bash
python scripts/remote.py submit jobs --minutes 5 --module flies.jobs.learning --args="--size full --device cuda \
    --seed 2026 --out runs/jobs/workday_full --vault-file /root/.flyhack-secrets/workday_full.json \
    --episodes 16 --chunk 32 --lr 0.0005 --gamma 0.997 --entropy 0.04 \
    --checkpoint-every 10 --eval-every 10 --eval-episodes 32 --updates 20 --minutes 5"
```

`--minutes 5` sets both the worker's hard stop (2 × 5 + 5 = 15 min, SIGTERM then SIGKILL 120 s later) and the
trainer's own deadline. Baseline and validation can run past the trainer deadline, but stay inside the hard stop.
`cancel <job id>` sends SIGTERM: the trainer finishes the current update and saves (verified locally).
`learning.py` refuses an existing `--out` without `--resume`, so earlier results can't be overwritten.

**Back up the vault right after the first run creates it.** The key is written once and never changes, so
one copy is enough. Without it, `--resume` is impossible after the container disk is reset by a pod restart.

```bash
mkdir -p -m 700 ~/.local/share/flyhack-secrets
(umask 077; python -c "import sys; sys.path.insert(0, 'scripts'); import remote; \
  sys.stdout.buffer.write(remote.run('cat /root/.flyhack-secrets/workday_full.json', capture=True).stdout)" \
  > ~/.local/share/flyhack-secrets/workday_full.json)
```

Restore after a pod restart, before any `--resume`:

```bash
python -c "import sys; sys.path.insert(0, 'scripts'); import remote, pathlib; \
  remote.run('mkdir -p -m 700 /root/.flyhack-secrets && umask 077 && cat > /root/.flyhack-secrets/workday_full.json \
  && chmod 600 /root/.flyhack-secrets/workday_full.json', \
  stdin=pathlib.Path('~/.local/share/flyhack-secrets/workday_full.json').expanduser().read_bytes())"
```

The vault is never committed, never uploaded (`simulation_vault.json` is in `SKIP`; this file is outside the
repo), and never pulled (`pull` only fetches `runs/<fly>/`). Note: every teammate SSHes in as `root`, so 0600
protects the vault from non-root processes, not from teammates. It only keys synthetic simulation passwords.

## 9. Retrieve results and test resume

```bash
python scripts/remote.py pull jobs             # -> runs/jobs/workday_full/{latest.json,checkpoints/,progress.*,REPORT.md,...}
python scripts/remote.py submit jobs --minutes 5 --module flies.jobs.learning --args="--resume --device cuda \
    --out runs/jobs/workday_full --vault-file /root/.flyhack-secrets/workday_full.json --updates 5 --minutes 3"
```

The resume log must start with `RESUMED update=N ... hash=<model_hash from latest.json>`.
Size: each checkpoint folder is about 37 MB (3.1M parameters → about 12 MB `brain.pt` plus about 25 MB of Adam state in `training.pt`).
`pull jobs` downloads all of `runs/jobs/`, including every checkpoint and the old generic `brain.pt`. The volume
uses 410 MB today. For long runs, raise `--checkpoint-every` or archive old checkpoint folders on purpose.

## Rollback

1. `python scripts/remote.py status`, then `cancel` every **pending custom** job (`--module` jobs). An old worker
   would run them as `core.train jobs` and overwrite `runs/jobs/brain.pt` and `result.json`.
2. Wait until nothing is running (step 1 check), then `python scripts/remote.py worker-stop`.
3. `python scripts/remote.py shell`, then `cp -p <BACKUP>/queue.py /workspace/flyhack/core/queue.py`. Check that
   `sha256sum` gives `893d06b7…`, then `rm -f jobs/worker.caps`.
4. `python scripts/remote.py worker-start`, then `status`.
5. Optional: restore the old fly code with `tar xzf <BACKUP>/flies_jobs.tgz` in `/workspace/flyhack`.
   Laptop side: `git revert ff36020` (this only affects `remote.py`/docs behavior locally).

Rollback doesn't touch `runs/`, `data/`, queue history or `/root/.flyhack-secrets`.

## Verification evidence (local; the pod steps above confirm them on hardware)

| # | Claim | Evidence |
|---|---|---|
| 1 | Generic path unchanged | Jobs without `--module` produce the same job JSON keys and the same `core.train` command (sandbox logs for `gym`). `status` output for old jobs is unchanged. |
| 2 | Custom submission works | The exact `remote_py` strings, run through `sh`, queued and ran `flies.jobs.preflight` (PASS) and `flies.jobs.learning`. Guards refuse an old worker, a module outside the fly, and `--args` without `--module`. |
| 3 | jobs fly uploads | `tar_bytes(['flies/jobs'])` extracted into an empty folder matches the working tree exactly (67 files, 0 tracked files missing, 0 `.pt`/vault/state files). The bootstrap upload still excludes the top-level `runs/` and `jobs/` and `.git`. |
| 4 | Checkpoints and metrics retrievable | `pull`'s own extraction code on a real `runs/jobs/<exp>` gives `latest.json` → the newest checkpoint, with matching model hash, `progress.csv`, `simulation_state.json` and `REPORT.md`. |
| 5 | No overwrite of jobs or results | The worker uses the same queue folders and format. Output goes to a new `runs/jobs/<exp>/`, and `learning.py` refuses an existing `--out` without `--resume`. The idle check prevents re-running a live job. |
| 6 | Existing GPU jobs compatible | The pod's `queue.py` equals the old git version (`893d06b7…`). The new worker reads old job records unchanged, and `queue.py` imports nothing from `brain`/`graph`. Jobs unit tests (36: 35 OK, 1 CUDA skip), `core.check jobs` and queued learning plus resume all pass against the pod's `brain.py`/`graph.py`. |
| 7 | CUDA preflight submittable | Step 7 was exercised through the queue with `--size tiny --device cpu --batch-size 32`. The pod has torch `2.8.0+cu128` with CUDA available, `data/full.npz`, Python 3.12.3 and an idle RTX 4090. Training needs only numpy and torch. |
