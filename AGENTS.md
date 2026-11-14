# Agent brief: build and train a fly brain for the flyhack project

You are helping one teammate build **one or more "flies"** for a hackathon. Read this whole file before acting.

## What the project is

Each fly is a copy of the real fruit-fly connectome (FlyWire v783: 134,181 neurons, 2.7M connections),
trained to make decisions in a small simulated task (gym workouts, triaging email, ...) and shown live in a
three.js world. The fun claim we must keep true: **the fly's real neurons make every decision you see.**
The shared code in `core/` guarantees that if you follow the task template:
- the observation enters only through real sensory neurons;
- each action is read from a fixed group of real descending neurons (no learned output layer);
- training changes connection strengths only; the wiring and the excitatory/inhibitory signs stay the fly's;
- the lesion test (`core.check`) proves a disconnected brain stops reacting.

## Hard rules

1. **Only create or edit files inside `flies/<your_fly>/`** (`task.py`, `station.js`, optional `demos.npz`).
   Never edit `core/`, `server/`, `web/`, `scripts/`, `docs/` or another person's fly. If the core seems
   wrong, stop and tell your human; Roman (the core owner) fixes it.
2. **Never bypass the brain.** Don't add a learned output layer, don't read actions from anywhere but
   `FlyBrain`, and don't make `step()` choose actions itself.
3. **Shared GPU etiquette.** Use only these remote commands: `setup`, `submit`, `status`, `logs`, `pull`,
   `cancel` (your own jobs only), `shell` (read-only looking around). Never run `bootstrap`, `sync-core`,
   `worker-start`, `worker-stop` or `add-key`, and never stop, restart or delete the pod. Those belong to Roman.
4. **Secrets.** Never print, copy or commit a private SSH key. Only the `.pub` line is ever shared.
5. Run `python -m core.check <fly>` before every push and every GPU submit. It must print PASS.

## 1. One-time setup (about 10 minutes)

```bash
git clone <repo URL from Roman> flyhack && cd flyhack     # skip if already cloned
python -m venv .venv && source .venv/bin/activate          # Windows: .venv\Scripts\activate
pip install -r requirements.txt
python scripts/download_data.py                            # ~110 MB of FlyWire tables
python -m core.graph --size all                            # builds the brains, ~10 s
python -m core.check --all                                 # every fly should PASS
```

## 2. Get access to the shared GPU

The GPU is a Runpod RTX 4090 that Roman runs. Access is per person, by SSH key.

1. Make a key if your human doesn't have one (`ls ~/.ssh/*.pub` to check):
   `ssh-keygen -t ed25519 -C "<name>-flyhack"` (press Enter at every prompt).
2. Have your human send Roman the **one line** from `~/.ssh/id_ed25519.pub` (it starts with `ssh-ed25519`).
   Never the file without `.pub`.
3. Once Roman confirms the key is added:
   ```bash
   python scripts/remote.py setup 103.196.86.68 28818 --name <your-name>
   ```
   Add `--key <path>` if the key isn't `~/.ssh/id_ed25519`. Success prints `connected to ...` and the GPU name.
   A password prompt means the key isn't on the pod yet. `Connection refused` usually means the pod was
   restarted and its address changed: ask Roman for the new IP and port and run `setup` again.

Address as of Nov 14, 2026, 02:20 UTC: **IP 103.196.86.68, port 28818**.

## 3. Build a fly

```bash
python scripts/new_fly.py <name>       # lowercase, e.g. karaoke; copies flies/_template
```

Then edit `flies/<name>/task.py` (one `FlyTask` subclass) and `flies/<name>/station.js` (its look in the
3D world). Study the two worked examples first: `flies/gym` (reward mode) and `flies/inbox` (teacher mode).

**Design rules**
- **Actions: 2 to 8**, each small and always sensible. Use `mask()` to hide actions that are invalid right
  now (an untrained fly then still never does anything impossible). Never raw mouse or keyboard control.
- **Observation: 4 to 64 floats, roughly 0 to 1.** Only what the decision needs: counters, levels, flags.
  `obs_dim` must equal its length.
- **Mode**
  - `"reward"`: you can simulate the task and score each step. This is the default. Give small rewards
    every step, not only at the end.
  - `"teacher"`: it's easier to say what the right move is than to score it. Implement `teacher()` (a rule;
    an LLM call is allowed but only runs while collecting demos). Demos are cached in
    `flies/<name>/demos.npz`; commit that file so nobody pays to regenerate it.
- **`step()` must be fast**: pure Python/numpy, no network or file I/O. It runs tens of thousands of times.
- `caption(action)` is called right after `step()`; return one short, funny line for the world
  ("rep #7 on the bench", "fails the rep!").
- Tasks with changing options (e.g. job-application forms): use fixed slots `slot_0..slot_7`, put each
  candidate's features into the observation, and mask the empty slots.

**Station (`station.js`)**: `build({THREE, group, fly})` once; `async play(action, caption, {fly})` after
every decision, resolving when the animation ends (aim for 1 to 2 s; the brain waits for it). Helpers come
from `/web/lib/fly.js`: `tween`, `wait`, `box`, `cylinder`, and `fly.walkTo`, `fly.hop`, `fly.buzz`, `fly.face`.
There must be a branch for every action name in `task.py`.

## 4. Test locally, then train on the GPU

```bash
python -m core.check <name>                                  # must PASS (about 1 minute)
python -m core.train <name> --size tiny --minutes 3           # optional quick local training on CPU
python scripts/remote.py submit <name>                        # full brain on the GPU, default 15 min
python scripts/remote.py status                               # queue + recent scores
python scripts/remote.py logs <name> -f                       # watch it (Ctrl+C stops watching only)
python scripts/remote.py pull <name>                          # results -> runs/<name>/
```

- `submit` uploads only your fly folder, so you can train before merging. Resubmitting a fly that is still
  waiting replaces the earlier job.
- The queue runs **one job at a time for the whole team**. Submit only flies that pass `core.check`. Use
  `--minutes 5` for experiments, and don't submit the same fly over and over.
- Read `runs/<name>/result.json`: `score_trained` should beat `score_random`, and `score_lesioned` should
  fall back to around random. Teacher mode also reports `teacher_agreement` against
  `teacher_agreement_chance`. How good the score is doesn't matter much; it just has to be trained.
- Out of GPU memory: add `train = {"chunk": 128}` to your task class and resubmit.

## 5. See it in the world

```bash
python -m server.serve --only <name>     # then open http://localhost:8000/web/
```

Flies without a pulled `runs/<name>/brain.pt` run an untrained brain, labelled "untrained". Click the
fly to enter its brain (the matrix view) and check that the lesion toggle makes its choices go flat.

## 6. Done means

- [ ] `python -m core.check <name>` prints PASS
- [ ] Trained on the GPU (`--size full`), and `runs/<name>/result.json` shows trained above random, lesioned around random
- [ ] `station.js` animates every action and the fly ran in the world for 1 minute with no console errors
- [ ] Committed only `flies/<name>/` on your own branch, merged to `main` (CI runs the check again)
- [ ] Told your human the fly's one-line pitch for the demo

Full background: `README.md`, `docs/COMPUTE.md`, and the team doc "Fly Brain Hackathon: Universal Task Training Guide".
