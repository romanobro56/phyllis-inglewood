"""Use the shared GPU machine from your own laptop (Mac, Linux or Windows; needs `ssh`).

One-time, everyone:
    python scripts/remote.py setup <host> <port> [--key ~/.ssh/id_ed25519]

Everyone, all night:
    python scripts/remote.py submit gym [--size full] [--minutes 15]   # uploads flies/gym, queues training
    python scripts/remote.py status                                      # queue + recent results
    python scripts/remote.py logs gym [-f]                               # training output
    python scripts/remote.py pull [gym inbox ...]                        # download runs/<fly>/ (all if none)
    python scripts/remote.py shell                                       # interactive ssh

GPU owner only:
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
SKIP = {".git", "runs", "jobs", "__pycache__", ".remote.json", "raw"}


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


def tar_bytes(paths):
    buf = io.BytesIO()
    with tarfile.open(fileobj=buf, mode="w:gz") as t:
        for p in paths:
            src = ROOT / p
            if src.exists():
                t.add(src, arcname=p, filter=lambda ti: None if Path(ti.name).name in SKIP else ti)
    return buf.getvalue()


def upload(paths):
    run("tar xzf - --no-same-owner", stdin=tar_bytes(paths))   # volume refuses chown to the laptop uid


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
    sub.add_parser("status")
    lg = sub.add_parser("logs"); lg.add_argument("key"); lg.add_argument("-f", "--follow", action="store_true")
    cn = sub.add_parser("cancel"); cn.add_argument("job_id")
    pl = sub.add_parser("pull"); pl.add_argument("flies", nargs="*")
    sub.add_parser("shell")
    sub.add_parser("bootstrap")
    sub.add_parser("sync-core")
    sub.add_parser("worker-start"); sub.add_parser("worker-stop")
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
        run(remote_py(["submit", a.fly, "--size", a.size, "--minutes", str(a.minutes),
                       "--by", conf().get("name", "someone")]))
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
