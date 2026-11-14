# Job Application fly: hackathon demo

**One-line pitch:** the real 134,181-neuron FlyWire connectome, with only its connection strengths trained by
reinforcement learning, fills in and submits simulated Workday-style applications about **twice as often as a
random or lesioned brain**: 14.8% against 6–7%, on 256 never-seen forms.

Everything is a **local simulation**. No employer site is contacted and nothing is really submitted.

## Best checkpoint

`runs/jobs/workday_v3_p2/checkpoints/update_000020/brain.pt`
(run 2, update 20, reward v3; graph `85c53631ff`; on the GPU pod at `/workspace/flyhack/runs/jobs/...`).
It was chosen by **held-out completion**, not by training reward. The checkpoint with the highest reward
(run 2, update 50, reward −22.6) completes only 7.8%.

### Held-out evidence (level 1, test split, 256 identical forms per policy; `runs/jobs/eval_final/comparison.json`)

| Policy | Completed | Completion | Field accuracy | Failed actions / form | Reward (v3) |
|---|---:|---:|---:|---:|---:|
| Random | 16 | 6.2% | 23.4% | 35.1 | −31.1 |
| Untrained full brain | 9 | 3.5% | 11.7% | 20.9 | −44.7 |
| Original 20-update run (`workday_full`) | 24 | 9.4% | 23.4% | 20.2 | −29.0 |
| Run 1, update 20 (`workday_v2_p1`) | 35 | 13.7% | 34.0% | 26.6 | −23.9 |
| Run 1, final (update 50) | 11 | 4.3% | 22.3% | 28.1 | −23.1 |
| **Run 2, update 20 (best)** | **38** | **14.8%** | **34.0%** | 27.1 | −23.7 |
| Run 2, final (update 50) | 20 | 7.8% | 24.2% | 26.8 | −22.6 |
| Best brain, lesioned (sensory input cut) | 19 | 7.4% | 23.8% | 34.9 | −30.5 |

Paired 95% intervals for the best brain: completion +8.6 points against random [+3.5, +13.7], +7.4 against its own
lesioned brain [+2.3, +12.9] and +11.3 against untrained [+6.6, +16.0]; field accuracy +10.5 against random
[+3.5, +17.6]. It is **not** significantly better than run 1 at update 20 (+1.2 [−1.6, +3.9]).

## Run the demo

One-time setup on the demo laptop (the repository's `core/` must include commit `c39c450`, so it can load
brains trained on the pod):

```bash
source .venv/bin/activate
pip install -r requirements.txt -r flies/jobs/requirements.txt
python -m playwright install chromium
python -m flies.jobs.setup_assets
python scripts/remote.py pull jobs          # fetches runs/jobs/ (checkpoints and evaluations)
```

**Live visual demo** (browser form, the 8 neural action choices, and the 3D fly station animating every decision):

```bash
B=runs/jobs/workday_v3_p2/checkpoints/update_000020/brain.pt
python -m flies.jobs.run --policy trained --checkpoint $B --device cpu --level 1 \
    --browser --headed --station --fresh-forms --seed 7 --episodes 20 \
    --state-dir /tmp/fly-demo-trained --delay 0.2
```

**Side-by-side comparison on the same forms** (same `--seed`, each run in a fresh `--state-dir`):

```bash
python -m flies.jobs.run --policy lesioned --checkpoint $B --device cpu --level 1 --fresh-forms --seed 7 \
    --episodes 20 --delay 0 --state-dir /tmp/fly-demo-lesioned
python -m flies.jobs.run --policy random --device cpu --level 1 --fresh-forms --seed 7 \
    --episodes 20 --delay 0 --state-dir /tmp/fly-demo-random
# untrained FULL brain (not --policy untrained, which builds a tiny 5,000-neuron brain):
python -m flies.jobs.run --policy brain --checkpoint runs/jobs/workday_v3_p2/untrained.pt --device cpu --level 1 \
    --fresh-forms --seed 7 --episodes 20 --delay 0 --state-dir /tmp/fly-demo-untrained
```

Each run ends with a `SCOREBOARD` line. With seed 7, the 20 forms gave trained 2/20, random 1/20 and lesioned 0/20.
Twenty forms is noisy, so present the 256-form table above as the evidence and the live run as an illustration.
Do not hunt for a lucky seed.

Notes:
- `--fresh-forms` makes every round a new application, and a timeout ends that job. The forms and the action
  random draws come from `--seed` and the round number, so every policy sees the same forms.
- Without it, the runner keeps working on a timed-out form in the next round (its original persistent mode).
- CPU inference is about 0.15 s per decision. With `--station`, each decision waits for its animation
  (about 0.8 s), so 20 forms take about 15 minutes.
- Use a new `--state-dir` for every demo run: simulated accounts and the job queue persist inside it.

## What to point at
- **The action panel:** the brain chooses one of these 8 slots at every step. There is no output layer and no
  scripted fallback: the choice is read from fixed groups of real descending neurons.
- **The feedback is real simulation feedback:** "Invalid email format", "Please check…", and after a passed review
  "Review complete: ready to submit." A completed form ends with a simulated receipt.
- **Lesion contrast:** cut the sensory input and completion drops to the random level.

## How it was trained (about 20 minutes on the shared RTX 4090)

| Run | Reward | Updates / episodes | GPU time | Notes |
|---|---|---|---|---|
| `workday_full` (earlier) | v1 | 20 / 320 | 87 s | baseline pilot |
| `workday_v2_p1` | v2 | 50 / 3,200 | 579 s | 64 episodes per update, chunk 128, lr 5e-4, entropy .04 |
| `workday_v3_p2` | v3 | 50 / 3,200 | 587 s | same hyperparameters, fresh start |

Reward changes, each backed by trace evidence (see the `git log` for `flies/jobs/workday.py`):
1. **v2:** a wrong field value costs −0.25 instead of −1, and the correction bonus is +0.2 instead of +1. At −1,
   trying a value was worth less than idling in the editor, so the fly stopped entering values.
2. **Observation:** a successful review is now visible ("Review complete: ready to submit."). Before, the page
   looked identical, so Submit (+20 or −10) was a blind gamble for a memoryless policy.
3. **v3:** field reward = the change in the number of correct fields (potential-based). Run 1 kept overwriting
   its correct value; now overwriting, clearing or removing a correct value costs what it earned.

## Known limitations
- **Level 1 only:** one personal field, then review, then submit. The curriculum (it needs ≥75% completion)
  never advanced, and harder levels (auth, uploads, experience) are untested for this policy.
- **Both 10-minute runs peaked around update 20 and then declined,** while reward kept rising. Reward v3 slowed
  the decline but did not remove it. Checkpoints are therefore chosen on held-out completion.
- **Largest remaining bottleneck: value selection.** There are about 20 wrong entries per form, and field
  accuracy is 34%. A second one: in browse mode the observation does not include the field's current value, so
  Next is a partly blind choice. Fixing that would change the browser contract (`browser.py` matches actions by
  equality), so it was left alone.
- The simulation is Workday-*inspired*, not a replica, and there is no live submission path.
