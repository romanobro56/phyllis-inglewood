"""Use the shared GPU machine from your own laptop (Mac, Linux or Windows; needs `ssh`).

One-time, everyone:
    python scripts/remote.py setup <host> <port> [--key ~/.ssh/id_ed25519]

Everyone, all night:
    python scripts/remote.py submit gym [--size full] [--minutes 15]   # uploads flies/gym, queues training
    python scripts/remote.py submit jobs --minutes 5 --module flies.jobs.learning --args="--size full ..."
                                                     # a fly's own training loop instead of core.train
    python scripts/remote.py status                                      # queue + recent results
    python scripts/remote.py logs gym [-f]                               # training output
    python scripts/remote.py pull [gym inbox ...]                        # download runs/<fly>/ (all if none)
    python scripts/remote.py shell                                       # interactive ssh

GPU owner only:
    python scripts/remote.py deploy            # push the committed code (core, server, web, every fly) and
                                               # restart the 3D world on the GPU; prints its public URL
    python scripts/remote.py world-status | world-logs [-f] | world-stop
    python scripts/remote.py bootstrap         # first time: upload repo, install, build data, start worker
    python scripts/remote.py add-key "ssh-ed25519 AAAA... alice@laptop"   # give a teammate access
    python scripts/remote.py sync-core         # push your latest core/, server/, scripts/, web/
    python scripts/remote.py worker-start | worker-stop

Settings live in .remote.json (not committed - it's per person).
"""
import argparse
import getpass
import io
import json
import shlex
import subprocess
import sys
import tarfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
CONF = ROOT / ".remote.json"
REMOTE_DIR = "/workspace/flyhack"     # /workspace survives pod restarts on RunPod
CORE_PARTS = ["core", "server", "scripts", "web", "docs", "requirements.txt", "README.md"]
SKIP = {".git", "__pycache__", ".remote.json", "raw", "simulation_vault.json"}   # skipped at any depth
SKIP_TOP = {"runs", "jobs"}   # the pod's own queue/results dirs; only at the top so flies/jobs still uploads


def conf():
    if not CONF.exists():
        raise SystemExit("Not set up. Run: python scripts/remote.py setup <host> <port>  (ask the GPU owner)")
    return json.loads(CONF.read_text())


def ssh_cmd(c, tty=False):
    cmd = ["ssh", "-p", str(c["port"]), "-o", "StrictHostKeyChecking=accept-new",
           "-o", "ServerAliveInterval=30"]
    if c.get("key"):
        cmd += ["-i", str(Path(c["key"]).expanduser())]
    if tty:
        cmd.append("-t")
    return cmd + [f"{c.get('user', 'root')}@{c['host']}"]


def run(remote, stdin=None, capture=False, tty=False, check=True):
    c = conf()
    full = f"cd {REMOTE_DIR} 2>/dev/null || mkdir -p {REMOTE_DIR} && cd {REMOTE_DIR}; {remote}"
    p = subprocess.run(ssh_cmd(c, tty) + [full], input=stdin, capture_output=capture)
    if check and p.returncode != 0:
        if capture and p.stderr:
            sys.stderr.write(p.stderr.decode(errors="replace"))
        raise SystemExit(f"remote command failed ({p.returncode}): {remote}")
    return p


def _keep(ti):
    parts = Path(ti.name).parts
    if Path(ti.name).name in SKIP or (len(parts) == 1 and parts[0] in SKIP_TOP):
        return None
    return ti


def tar_bytes(paths):
    buf = io.BytesIO()
    with tarfile.open(fileobj=buf, mode="w:gz") as t:
        for p in paths:
            src = ROOT / p
            if src.exists():
                t.add(src, arcname=p, filter=_keep)
    return buf.getvalue()


def upload(paths):
    # --no-same-owner: as root, tar would chown files to the laptop's uid, which /workspace (MooseFS) refuses
    run("tar xzf - --no-same-owner", stdin=tar_bytes(paths))


def upload_tracked(paths):
    """Upload only the files git tracks under `paths` (never local state, secrets, raw media or caches)."""
    files = subprocess.run(["git", "ls-files", "-z", "--", *paths], cwd=ROOT, capture_output=True,
                           check=True).stdout.decode().split("\0")
    buf = io.BytesIO()
    with tarfile.open(fileobj=buf, mode="w:gz") as t:
        for f in filter(None, files):
            if (ROOT / f).is_file():
                t.add(ROOT / f, arcname=f)
    run("tar xzf - --no-same-owner", stdin=buf.getvalue())
    return sum(1 for f in files if f)


WORLD_PORT = 8000     # exposed by the pod as https://<pod id>-8000.proxy.runpod.net
# anchored at the start of the command line, so it never matches the ssh shell that runs these commands
WORLD_PIDS = f"$(pgrep -f '^python[0-9.]* -m server[.]serve .*--port {WORLD_PORT}' | tr '\\n' ' ')"
WORLD_URL = ("echo https://$(tr '\\0' '\\n' < /proc/1/environ | sed -n 's/^RUNPOD_POD_ID=//p')"
             f"-{WORLD_PORT}.proxy.runpod.net/web/")


def world_start(device):
    run(f"kill {WORLD_PIDS} 2>/dev/null; sleep 1; "
        f"setsid nohup python -m server.serve --host 0.0.0.0 --port {WORLD_PORT} --device {device} "
        f"> world.log 2>&1 < /dev/null & "
        f"for i in $(seq 90); do grep -q 'Open http' world.log && break; sleep 2; done; cat world.log; {WORLD_URL}")


def remote_py(args):
    return "python -m core.queue " + " ".join(shlex.quote(a) for a in args)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    s = sub.add_parser("setup"); s.add_argument("host"); s.add_argument("port", type=int)
    s.add_argument("--key", help="private key file, e.g. ~/.ssh/id_ed25519"); s.add_argument("--user", default="root")
    s.add_argument("--name", default=getpass.getuser(), help="your name in the queue")
    sb = sub.add_parser("submit"); sb.add_argument("fly"); sb.add_argument("--size", default="full")
    sb.add_argument("--minutes", type=float, default=15)
    sb.add_argument("--module", help="fly-local training module to run instead of core.train, e.g. flies.jobs.learning")
    sb.add_argument("--args", default="", help='arguments for --module as one string, e.g. --args="--size full"')
    sub.add_parser("status")
    lg = sub.add_parser("logs"); lg.add_argument("key"); lg.add_argument("-f", "--follow", action="store_true")
    cn = sub.add_parser("cancel"); cn.add_argument("job_id")
    pl = sub.add_parser("pull"); pl.add_argument("flies", nargs="*")
    sub.add_parser("shell")
    sub.add_parser("bootstrap")
    sub.add_parser("sync-core")
    sub.add_parser("worker-start"); sub.add_parser("worker-stop")
    dp = sub.add_parser("deploy"); dp.add_argument("--device", default="cuda")
    ws = sub.add_parser("world-start"); ws.add_argument("--device", default="cuda")
    sub.add_parser("world-stop"); sub.add_parser("world-status")
    wl = sub.add_parser("world-logs"); wl.add_argument("-f", "--follow", action="store_true")
    ak = sub.add_parser("add-key"); ak.add_argument("pubkey")
    a = ap.parse_args()

    if a.cmd == "setup":
        CONF.write_text(json.dumps(dict(host=a.host, port=a.port, key=a.key, user=a.user, name=a.name), indent=2))
        p = run("echo connected to $(hostname); nvidia-smi --query-gpu=name,memory.total --format=csv,noheader || true",
                capture=True, check=False)
        print(p.stdout.decode(errors="replace") or p.stderr.decode(errors="replace"))
        print("saved .remote.json" if p.returncode == 0 else
              "could not connect - is your public key added? (ask the GPU owner to run add-key)")
    elif a.cmd == "submit":
        if not (ROOT / "flies" / a.fly / "task.py").exists():
            raise SystemExit(f"flies/{a.fly}/task.py not found")
        upload([f"flies/{a.fly}"])
        extra = ([f"--module={a.module}", f"--args={a.args}"] if a.module else [])
        run(remote_py(["submit", a.fly, "--size", a.size, "--minutes", str(a.minutes),
                       "--by", conf().get("name", "someone")] + extra))
    elif a.cmd == "status":
        run(remote_py(["status"]))
    elif a.cmd == "logs":
        run(remote_py(["logs", a.key] + (["-f"] if a.follow else [])), tty=a.follow)
    elif a.cmd == "cancel":
        run(remote_py(["cancel", a.job_id]))
    elif a.cmd == "pull":
        targets = " ".join(shlex.quote(f"runs/{f}") for f in a.flies) or "runs"
        p = run(f"tar czf - {targets} 2>/dev/null", capture=True, check=False)
        if not p.stdout:
            raise SystemExit("nothing to pull yet")
        with tarfile.open(fileobj=io.BytesIO(p.stdout), mode="r:gz") as t:
            names = sorted({Path(m.name).parts[1] for m in t.getmembers() if len(Path(m.name).parts) > 1})
            t.extractall(ROOT, filter="data")
        print(f"pulled runs/ for: {', '.join(names)}")
    elif a.cmd == "shell":
        subprocess.run(ssh_cmd(conf(), tty=True))
    elif a.cmd == "bootstrap":
        print("uploading repo ...")
        upload(CORE_PARTS + ["flies"])
        print("installing packages, downloading the connectome and building brains (a few minutes) ...")
        run("pip install -q -r requirements.txt && python scripts/download_data.py && "
            "python -m core.graph --size all")
        run("nohup python -m core.queue worker > jobs_worker.log 2>&1 & sleep 2; python -m core.queue status")
    elif a.cmd == "sync-core":
        upload(CORE_PARTS)
        print("core synced. Restart the worker if you changed core/: remote.py worker-stop && remote.py worker-start")
    elif a.cmd == "worker-start":
        run("nohup python -m core.queue worker > jobs_worker.log 2>&1 & sleep 2; python -m core.queue status")
    elif a.cmd == "worker-stop":
        run("kill $(cat jobs/worker.pid) 2>/dev/null; rm -f jobs/worker.pid; echo stopped")
    elif a.cmd == "deploy":
        dirty = subprocess.run(["git", "status", "--porcelain", "--untracked-files=no"], cwd=ROOT,
                               capture_output=True, text=True).stdout.strip()
        if dirty:
            print("note: uncommitted changes are NOT deployed (only committed files go up):\n" + dirty)
        n = upload_tracked(CORE_PARTS + ["flies"])
        print(f"uploaded {n} files; installing requirements ...")
        run("python -m pip install -q --break-system-packages -r requirements.txt 2>&1 | grep -v 'as the .root' || true")
        world_start(a.device)
    elif a.cmd == "world-start":
        world_start(a.device)
    elif a.cmd == "world-stop":
        run(f"kill {WORLD_PIDS} 2>/dev/null; echo stopped")
    elif a.cmd == "world-status":
        run("pgrep -af '^python[0-9.]* -m server[.]serve' || echo 'world not running'; "
            "nvidia-smi --query-gpu=name,memory.used,utilization.gpu --format=csv,noheader; " + WORLD_URL)
    elif a.cmd == "world-logs":
        run("tail -n 40 " + ("-f " if a.follow else "") + "world.log", tty=a.follow)
    elif a.cmd == "add-key":
        key = a.pubkey.strip()
        if not key.startswith(("ssh-ed25519 ", "ssh-rsa ", "ecdsa-")) or "PRIVATE" in key:
            raise SystemExit("that doesn't look like a PUBLIC key (it should start with ssh-ed25519 or ssh-rsa)")
        run("mkdir -p ~/.ssh && chmod 700 ~/.ssh && "
            f"grep -qxF {shlex.quote(key)} ~/.ssh/authorized_keys 2>/dev/null || "
            f"echo {shlex.quote(key)} >> ~/.ssh/authorized_keys; chmod 600 ~/.ssh/authorized_keys; "
            "echo added. Keys with access: $(wc -l < ~/.ssh/authorized_keys)")


if __name__ == "__main__":
    main()
