# flyhack: fruit-fly brains doing silly jobs

Each fly is a copy of the real *Drosophila* connectome (FlyWire v783: 134,181 neurons,
2.7M connections), trained on a GPU for one task and shown live in a three.js world.

**The world (live, on the GPU):** https://ape9a1r3z9a8ef-8000.proxy.runpod.net/web/ while the pod runs
(`python scripts/remote.py world-status` prints the current address).

| Fly | Folder | Job | Brain in the world |
| --- | --- | --- | --- |
| Bard | `flies/bard` | types Shakespeare one word at a time from a tray of 8 | full, trained (teacher) |
| Chess | `flies/chess` | plays White against a random Black, 4-5 neuron decisions per move | full, trained |
| Gordo (critic) | `flies/critic` | tastes the dish you feed it and scores it | full, trained (teacher) |
| Marlo (gym) | `flies/gym` | reps versus fatigue | full, trained |
| Inbox | `flies/inbox` | archive, reply, flag or unsubscribe | full, trained (teacher) |
| Elton (jobs) | `flies/jobs` | fills in simulated Workday applications | full, best held-out checkpoint |
| Mona Lisa | `flies/painting` | paints a 64-cell Mona Lisa from 8 pigments | full, trained |
| Scrolly (reels) | `flies/reels` | likes, comments or scrolls reels | full, trained |
| The Fly Band (5 flies) | `flies/band` | four players each hear one part of a song, a fifth conducts | spiking 8,161-neuron brains, in your browser |
Use the < > buttons (or click a fly) to visit each fly and watch its live brain at the top; ^ shows the pods.

**Why "the fly decided" is true here**
- Input enters only through real sensory neurons.
- Each action is read from a fixed group of 16 real descending neurons. There is no learned output layer.
- Training changes connection *strengths* only. The wiring and the excitatory/inhibitory signs are the fly's.
- Lesion test: zero the connectome and the fly stops reacting to its world (`core.check` proves it, and the brain panel has a toggle).

## Rules for working together

1. **Your fly lives in `flies/<your_fly>/` and you only commit inside it.** That's `task.py`, `station.js` and optionally `demos.npz`.
2. `core/`, `server/`, `web/` and `scripts/` belong to their owners (see the team split). Need a change? Ask them.
3. Run `python -m core.check <fly>` before every push. CI runs it on every push too.
4. Merge to `main` as soon as your fly passes. Don't wait for the end.

Because nobody touches anyone else's folder, combining the work is just merging branches.

## Setup (everyone, ~5 minutes)

```bash
git clone <this repo> && cd flyhack
python -m venv .venv && source .venv/bin/activate      # Windows: .venv\Scripts\activate
pip install -r requirements.txt
python scripts/download_data.py                        # ~110 MB of FlyWire tables
python -m core.graph --size all                        # builds the brains, ~10 s
python -m core.check --all                             # should print PASS for every fly
```

## Make a fly

```bash
python scripts/new_fly.py karaoke                 # copies flies/_template -> flies/karaoke
# edit flies/karaoke/task.py   (actions, observation, reward or teacher)
# edit flies/karaoke/station.js (what it looks like in the world)
python -m core.check karaoke                      # 1-minute smoke test on a tiny brain
python -m core.train karaoke --size tiny --minutes 3   # optional quick local training
python scripts/remote.py submit karaoke           # real training on the shared GPU (docs/COMPUTE.md)
```

Two worked examples:
- `flies/gym`: **reward mode.** The fly chooses between doing a rep and switching between bench and squats, and is scored on "gains" against fatigue.
- `flies/inbox`: **teacher mode.** The fly triages email by copying a rule-based expert. Its demos are cached in `demos.npz`.

The full guide (designing actions, choosing a mode, the done checklist) is the team doc *Fly Brain Hackathon: Universal Task Training Guide*.

## Train

| Where | Command | Brain |
| --- | --- | --- |
| Your laptop (CPU) | `python -m core.train <fly> --size tiny --minutes 3` | 5k neurons |
| Your laptop (GPU, 6 GB+) | `python -m core.train <fly> --size small` | 20k neurons |
| Shared GPU | `python scripts/remote.py submit <fly>` | full, 134k neurons |
| Final run, GPU owner | `python -m core.train_all --size full --minutes 15` | full |

Outputs go to `runs/<fly>/brain.pt` and `runs/<fly>/result.json`. The JSON holds the trained, random and lesioned scores, plus which FlyWire neurons drive each action.

## Run the world

```bash
python scripts/remote.py pull        # get everyone's trained brains from the GPU
python -m server.serve               # open http://localhost:8000/web/
```

Flies without a trained brain still appear, running an **untrained** brain and labelled as such. A fly with no `station.js` gets a placeholder pad. Locally inference runs on CPU (`--device cuda` if you have one). It needs every preset that was used for training (`python -m core.graph --size all`).

### On the GPU pod (the production world)

```bash
python scripts/remote.py deploy        # GPU owner: push the committed code, restart the world on CUDA, print the URL
python scripts/remote.py world-status  # is it up, GPU memory, the public URL
python scripts/remote.py world-logs -f
```

`deploy` uploads only git-tracked files (never local state, secrets or raw media) and runs
`server.serve --device cuda` on port 8000, which Runpod exposes at `https://<pod id>-8000.proxy.runpod.net/web/`.
The brains it runs are the pod's own `runs/<fly>/` (trained there), so nothing has to be copied back first.

### Hooks a fly can use in the world (all optional)

- `world_checkpoint = "<path in runs/<fly>/>"`: run this checkpoint instead of `brain.pt` (the jobs fly runs its best one).
- `world_args = {...}`: constructor arguments for the world's copy of the task (the bard writes freely, jobs uses level 1).
- `view()`: a small dict of what the fly is choosing between, sent to `station.play` as `ctx.view` (the bard's word tray).
- `step()`'s info dict reaches `station.play` as `ctx.info` when it is JSON-able and under 50 kB (the chess board).
- A folder with `station.js` and **no** `task.py` is a browser exhibit: shown in the world, never trained or run by the
  server. Its station may call `setCaption(text)` from `build()` and gets `focus(true/false)` when it is visited (the band).

## Repo map

```
core/graph.py      FlyWire tables -> data/<preset>.npz (+ browser files)
core/brain.py      FlyBrain: connectome as a trainable recurrent network
core/task.py       FlyTask base class (the interface every fly implements)
core/train.py      reward (REINFORCE) and teacher (imitation) training
core/check.py      smoke test + lesion test
core/train_all.py  trains every fly, keeps going past failures
core/queue.py      job queue on the shared GPU
scripts/remote.py  submit / status / logs / pull from your laptop
server/serve.py    runs all brains live + serves the web world
web/               three.js world + views/nav (main.js), live brain point clouds (brain.js), the pod tower (pods.js), fly model (lib/fly.js)
flies/<name>/      ONE FOLDER PER FLY: task.py, station.js, demos.npz (flies/band: a browser exhibit, see above)
docs/COMPUTE.md    setting up and sharing the GPU
```

## Data and credit

Connectome: FlyWire v783 (Dorkenwald et al., *Nature* 2024). Annotations: Schlegel et al., *Nature* 2024.
Packaged connectivity: Shiu et al., *Nature* 2024. The approach of keeping the wiring and signs fixed and learning
strengths follows github.com/cesp99/fly-chess. Cite the papers if you present this.
