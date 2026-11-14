"""A tiny file-based job queue that runs on the shared GPU machine.

One worker trains one fly at a time (one GPU, no out-of-memory fights).
Teammates normally use scripts/remote.py from their laptop, which calls these:

    python -m core.queue submit gym --size full --minutes 15 --by alice
    python -m core.queue status
    python -m core.queue logs gym            # newest job for that fly (or a job id)
    python -m core.queue cancel <job id>
    python -m core.queue worker              # the loop that runs jobs (started once)
"""
import argparse
import json
import os
import signal
import subprocess
import sys
import time
import uuid
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
JOBS = ROOT / "jobs"
STATES = ["pending", "running", "done", "failed", "cancelled"]


def _dirs():
    for s in STATES + ["logs"]:
        (JOBS / s).mkdir(parents=True, exist_ok=True)


def _jobs(state):
    return sorted((json.loads(p.read_text()) for p in (JOBS / state).glob("*.json")),
                  key=lambda j: j["submitted"])


def _move(job, src, dst):
    (JOBS / dst / f"{job['id']}.json").write_text(json.dumps(job, indent=2))
    (JOBS / src / f"{job['id']}.json").unlink(missing_ok=True)


def submit(fly, size, minutes, by):
    _dirs()
    if not (ROOT / "flies" / fly / "task.py").exists():
        raise SystemExit(f"flies/{fly}/task.py not found on the GPU machine (upload it with remote.py submit)")
    for j in _jobs("pending"):
        if j["fly"] == fly:
            _move(j, "pending", "cancelled")
            print(f"replaced your earlier pending job {j['id']}")
    job = dict(id=time.strftime("%H%M%S") + "-" + fly + "-" + uuid.uuid4().hex[:4], fly=fly, size=size,
               minutes=minutes, by=by, submitted=time.time())
    (JOBS / "pending" / f"{job['id']}.json").write_text(json.dumps(job, indent=2))
    ahead = len(_jobs("pending")) - 1 + len(_jobs("running"))
    print(f"queued {job['id']} ({ahead} job(s) ahead of it)")


def status():
    _dirs()
    now = time.time()
    worker = _worker_pid()
    print(f"worker: {'running (pid %d)' % worker if worker else 'NOT RUNNING - start it: remote.py worker-start'}")
    for j in _jobs("running"):
        print(f"RUNNING  {j['id']:28s} by {j['by']:10s} {(now - j['started']) / 60:5.1f} min of {j['minutes']}")
    for i, j in enumerate(_jobs("pending")):
        print(f"PENDING  {j['id']:28s} by {j['by']:10s} #{i + 1} in line")
    finished = sorted(_jobs("done") + _jobs("failed"), key=lambda j: j.get("ended", 0))[-10:]
    for j in reversed(finished):
        state = "DONE   " if j.get("returncode") == 0 else "FAILED "
        res = ROOT / "runs" / j["fly"] / "result.json"
        extra = ""
        if state.startswith("DONE") and res.exists():
            r = json.loads(res.read_text())
            extra = f"trained {r['score_trained']:.2f} / random {r['score_random']:.2f} / lesioned {r['score_lesioned']:.2f}"
        print(f"{state}  {j['id']:28s} by {j['by']:10s} {extra}")


def logs(key, follow):
    _dirs()
    files = sorted((JOBS / "logs").glob(f"*{key}*.log"), key=lambda p: p.stat().st_mtime)
    if not files:
        raise SystemExit(f"no logs matching '{key}'")
    path = files[-1]
    print(f"--- {path.name} ---")
    with open(path) as f:
        sys.stdout.write(f.read())
        while follow:
            line = f.readline()
            if line:
                sys.stdout.write(line); sys.stdout.flush()
            else:
                time.sleep(1)


def cancel(job_id):
    _dirs()
    for j in _jobs("pending"):
        if j["id"] == job_id:
            _move(j, "pending", "cancelled"); print("cancelled"); return
    for j in _jobs("running"):
        if j["id"] == job_id and j.get("pid"):
            os.kill(j["pid"], signal.SIGTERM); print("stopping running job"); return
    raise SystemExit("no such pending or running job")


def _worker_pid():
    p = JOBS / "worker.pid"
    if not p.exists():
        return None
    pid = int(p.read_text())
    try:
        os.kill(pid, 0)
        return pid
    except OSError:
        return None


def worker():
    _dirs()
    if _worker_pid():
        raise SystemExit("a worker is already running")
    (JOBS / "worker.pid").write_text(str(os.getpid()))
    for j in _jobs("running"):          # left over from a crash/restart: retry them
        _move(j, "running", "pending")
    print("worker started; waiting for jobs", flush=True)
    while True:
        pending = _jobs("pending")
        if not pending:
            time.sleep(3)
            continue
        job = pending[0]
        log_path = JOBS / "logs" / f"{job['id']}.log"
        cmd = [sys.executable, "-u", "-m", "core.train", job["fly"], "--size", job["size"],
               "--minutes", str(job["minutes"])]
        with open(log_path, "w") as log:
            log.write(f"$ {' '.join(cmd)}\n"); log.flush()
            proc = subprocess.Popen(cmd, cwd=ROOT, stdout=log, stderr=subprocess.STDOUT)
            job.update(started=time.time(), pid=proc.pid)
            _move(job, "pending", "running")
            print(f"started {job['id']}", flush=True)
            # hard stop at 2x the requested budget (the trainer stops itself at 1x)
            try:
                rc = proc.wait(timeout=job["minutes"] * 120 + 300)
            except subprocess.TimeoutExpired:
                proc.kill(); rc = -9
                log.write("\nKILLED: ran far past its time budget\n")
        job.update(ended=time.time(), returncode=rc)
        _move(job, "running", "done" if rc == 0 else "failed")
        print(f"finished {job['id']} rc={rc}", flush=True)


def main():
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    s = sub.add_parser("submit"); s.add_argument("fly"); s.add_argument("--size", default="full")
    s.add_argument("--minutes", type=float, default=15); s.add_argument("--by", default="someone")
    sub.add_parser("status")
    l = sub.add_parser("logs"); l.add_argument("key"); l.add_argument("-f", "--follow", action="store_true")
    c = sub.add_parser("cancel"); c.add_argument("job_id")
    sub.add_parser("worker")
    a = ap.parse_args()
    if a.cmd == "submit":
        submit(a.fly, a.size, a.minutes, a.by)
    elif a.cmd == "status":
        status()
    elif a.cmd == "logs":
        logs(a.key, a.follow)
    elif a.cmd == "cancel":
        cancel(a.job_id)
    else:
        worker()


if __name__ == "__main__":
    main()
