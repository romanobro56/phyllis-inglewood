# Shared GPU: set it up once, everyone trains from their own laptop

One person (the **GPU owner**) rents a single cloud GPU. Everyone else gets their own SSH access to it and
submits training jobs with `scripts/remote.py`. A queue on the GPU trains one fly at a time, so jobs never
fight over memory, and anyone can pull the trained brains back.

**Nobody shares a password, an account login, an API key or a private key.** Each person has their own SSH
key pair and only sends the *public* half (the `.pub` file) to the owner. Access is per person and can be
revoked by deleting one line.

```
 alice's laptop ─┐                      ┌──────── GPU pod (RunPod) ────────┐
 bob's laptop   ─┼── ssh (own key) ───► │ /workspace/flyhack               │
 cara's laptop  ─┤   remote.py submit   │   jobs/ queue  ─► worker ─► GPU  │
 you (owner)    ─┘   remote.py pull  ◄──│   runs/<fly>/brain.pt            │
                                        └──────────────────────────────────┘
```

---

## 1. Everyone: make an SSH key (2 minutes)

Mac, Linux, or Windows PowerShell (OpenSSH ships with Windows 10 and 11):

```bash
ssh-keygen -t ed25519 -C "your-name-flyhack"      # press Enter for the defaults
cat ~/.ssh/id_ed25519.pub                          # Windows: type $env:USERPROFILE\.ssh\id_ed25519.pub
```

Send the one line that starts with `ssh-ed25519` to the GPU owner in the team chat.
Never send `id_ed25519` itself (the file without `.pub`). That's your private key.

## 2. GPU owner: rent the GPU (10 minutes)

1. Create a RunPod account and add credit. A $100 budget is plenty: a 24 GB card costs well under $1/hour
   (check the current price on the deploy page), and billing is per second.
2. **Add everyone's keys before you deploy.** Go to Settings, then SSH Public Keys, and paste all four
   `ssh-ed25519 ...` lines, **one per line**. RunPod copies them into every pod you start, so the
   whole team gets access at once.
3. Deploy a pod:
   - **GPU:** RTX 4090 or any 24 GB card. An A100 or H100 is only worth it if training is slow.
   - **Template:** *Runpod PyTorch*. It already exposes TCP port 22 and runs an SSH server, which `scp`-style transfers need.
   - **Volume:** 20 GB mounted at `/workspace`. Everything we keep lives there and survives restarts.
   - Pick a pod with a **public IP**. Full SSH (needed here) requires one.
4. When it's running, open the pod's **Connect** tab and copy the **SSH over exposed TCP** command. It looks like
   `ssh root@203.0.113.7 -p 12345 -i ~/.ssh/id_ed25519`. Don't use the `ssh.runpod.io` command: that proxy
   can't transfer files.

## 3. GPU owner: install everything on the pod (5 minutes)

From the repo on your laptop:

```bash
python scripts/remote.py setup 203.0.113.7 12345 --key ~/.ssh/id_ed25519 --name roman
python scripts/remote.py bootstrap
```

`bootstrap` uploads the repo, installs the packages, downloads the connectome, builds every brain size and
starts the queue worker. It finishes by printing `worker: running`.

Then post the **IP and port** (nothing else) in the team chat:
`GPU is up: python scripts/remote.py setup 203.0.113.7 12345 --name <you>`

### Adding someone later

If someone's key wasn't in RunPod settings when the pod started, add it to the running pod:

```bash
python scripts/remote.py add-key "ssh-ed25519 AAAAC3Nz... alice-flyhack"
```

To remove someone, run `python scripts/remote.py shell`, then edit `~/.ssh/authorized_keys` and delete their line.

## 4. Everyone: connect and train (1 minute)

```bash
python scripts/remote.py setup 203.0.113.7 12345 --name alice      # add --key if yours isn't the default
```

It should print `connected to ...` and the GPU name. A password prompt means your key isn't on the pod yet,
so ping the owner.

Then, all night:

```bash
python scripts/remote.py submit gym                 # uploads flies/gym/ and queues training (full brain, 15 min)
python scripts/remote.py submit gym --minutes 5     # shorter run
python scripts/remote.py status                     # what's running, what's queued, recent scores
python scripts/remote.py logs gym -f                # watch training live (Ctrl+C to stop watching)
python scripts/remote.py pull gym                   # download runs/gym/ to your laptop
python scripts/remote.py cancel <job id>            # take a pending job out of the queue
```

A fly with its own training loop (resume, curriculum, checkpoints) can run that instead of `core.train`.
The module must live in the fly's folder, and the worker still hard-stops it at 2x `--minutes`
(SIGTERM first, so it can save). Write outputs under `runs/<fly>/<name>/` so `pull` brings them back:

```bash
python scripts/remote.py submit jobs --minutes 5 --module flies.jobs.learning \
    --args="--size full --device cuda --out runs/jobs/workday_full --updates 20 --minutes 5"
```

How it works:
- `submit` uploads only **your fly's folder**. Your fly trains with your latest local edits, even before they're merged.
- If you submit the same fly again while it's still waiting, the new job replaces the old one. No double-booking.
- Jobs run one at a time in submit order. Each stops itself at its time budget (`--minutes`, default 15).
- A typical fly takes 5–15 minutes on the full brain. With 13 flies the queue can back up, so test locally on
  `--size tiny` first and submit only flies that pass `core.check`.

## 5. The final run (GPU owner)

```bash
python scripts/remote.py sync-core          # make sure the pod has the latest core/ and web/
python scripts/remote.py shell
cd /workspace/flyhack
python -m core.queue status                 # wait until nothing is running
python -m core.train_all --size full --minutes 15 --skip-done
exit
python scripts/remote.py pull               # everyone's brains -> runs/ on the demo laptop
```

`--skip-done` skips flies that already have a full-size result from the queue, so this only fills the gaps.
`train_all` keeps going past a broken fly and prints a summary table at the end.

## 6. Keeping the cost down

- **Stop the pod** when nobody needs it for a while. Billing is per second. The `/workspace` volume keeps
  the repo, data and trained brains, and stopped storage costs only cents.
- **Terminate the pod** after the hackathon, once you've pulled `runs/`. A stopped pod's volume keeps billing until it's deleted.
- `remote.py status` shows whether anything is running before you stop it.

## Troubleshooting

| Problem | Fix |
| --- | --- |
| Asked for a password | Your public key isn't on the pod. The owner runs `remote.py add-key "<your .pub line>"`. |
| `Connection refused` or timeout | The pod restarted or was stopped. Its IP and port may have changed: get the new ones from the Connect tab, and everyone re-runs `remote.py setup`. |
| `worker: NOT RUNNING` | `python scripts/remote.py worker-start`. This happens after a pod restart. |
| `ModuleNotFoundError` after a restart | The container disk was reset (only `/workspace` survives). The owner re-runs `remote.py bootstrap`: data already in `/workspace` isn't downloaded again. |
| Job FAILED | `remote.py logs <fly>` shows the error. Usually it's the task code, so reproduce with `python -m core.check <fly>` locally. |
| Out of GPU memory | Lower the batch with `train = {"chunk": 128}` in your task class, or submit with `--size central`. |
| `brain.pt ... different data/<preset>.npz` on the demo laptop | Run `python -m core.graph --size all` on that laptop (same data gives the same graph). |
| Windows: `ssh` not found | Settings, Apps, Optional features, add **OpenSSH Client**. |

## Not using RunPod?

Any Linux machine with an NVIDIA GPU and SSH works the same way: Lambda, Vast.ai, a university box, or a
teammate's desktop. Put everyone's public keys in `~/.ssh/authorized_keys`, then run `remote.py setup <host> <port>`
and `bootstrap`. The repo lives in `/workspace/flyhack` unless you edit `REMOTE_DIR` in `scripts/remote.py`.
