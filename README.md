# flyhack: 13 fruit-fly brains doing silly jobs

Each fly is a copy of the real *Drosophila* connectome (FlyWire v783: 134,181 neurons,
2.7M connections), trained overnight for one task and shown live in a three.js world.
Click a fly to enter its brain and watch the decision form.

**Why "the fly decided" is true here**
- Input enters only through real sensory neurons.
- Each action is read from a fixed group of 16 real descending neurons. There is no learned output layer.
- Training changes connection *strengths* only. The wiring and the excitatory/inhibitory signs are the fly's.
- Lesion test: zero the connectome and the fly stops reacting to its world (`core.check` proves it, and the matrix view has a toggle).

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

Flies without a trained brain still appear, running an **untrained** brain and labelled as such. A fly with no `station.js` gets a placeholder pad. Inference runs on CPU, so the demo laptop doesn't need a GPU. It does need every preset that was used for training (`python -m core.graph --size all`).

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
web/               three.js world (main.js), matrix view (matrix.js), fly model (lib/fly.js)
flies/<name>/      ONE FOLDER PER FLY: task.py, station.js, demos.npz
docs/COMPUTE.md    setting up and sharing the GPU
```

## Data and credit

Connectome: FlyWire v783 (Dorkenwald et al., *Nature* 2024). Annotations: Schlegel et al., *Nature* 2024.
Packaged connectivity: Shiu et al., *Nature* 2024. The approach of keeping the wiring and signs fixed and learning
strengths follows github.com/cesp99/fly-chess. Cite the papers if you present this.
