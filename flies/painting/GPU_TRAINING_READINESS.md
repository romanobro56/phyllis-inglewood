# Mona Lisa painting fly

Implemented for the existing generic reward trainer. **Not trained.** No remote
commands or GPU jobs were executed. All painting decisions in the demo come from
the standard `FlyBrain` sensory input and fixed descending-neuron action groups.
No trainer, learned output head, teacher, optimizer, or painting policy was added.

## Simulation and decisions

An episode is 64 decisions on an 8×8 RGB canvas. The neutral gray blank canvas
is cleared at reset. The brush visits each cell exactly once in row-major order.
The fly chooses one of eight fixed pigments (`paint_0` through `paint_7`) for that
cell. Every color is legal; `mask()` returns eight true values. Applying paint,
advancing the cursor, ending at cell 64, and clearing on reset are deterministic
environment mechanics. They never select a color. Invalid actions and steps
after termination raise errors without changing the painting.

Eight direct pigment choices avoid navigation and color-cycling credit assignment.
The 8×8 data grid is stretched into portrait proportions on the easel. This is
fixed-artwork color matching, not evidence of general artistic intelligence.

## Observations

Twelve finite float32 values in [0,1], in this order:

| Indices | Meaning |
| --- | --- |
| 0–2 | Current cell's reference RGB |
| 3–5 | Current cell's canvas RGB |
| 6–7 | x/7, y/7 |
| 8 | Decisions completed / 64 |
| 9–11 | Painted left neighbor RGB; blank at the left edge |

Reference RGB is ordinary observable task data, not a quantized action index.
No nearest-pigment answer, reward, similarity, or hidden evaluation state enters
the policy. Current cell RGB is blank under the one-pass rules; it is included
to describe the visible painting state. The terminal observation uses the final
cell with progress 1 and is discarded by normal reset handling.

## Reward and scoring

`error = mean((canvas - reference)**2)` across all normalized RGB channels and
cells. `similarity = 1 - error` (0–1). For the chosen cell:

```text
reward = mean((old_RGB - target_RGB)**2)
       - mean((chosen_pigment_RGB - target_RGB)**2)
```

Thus reward is **64 times** the whole-image similarity improvement. Worse paint
gets negative reward; better paint gets positive reward. There is no completion
bonus. Each cell is visited only once, so repeated repaint cycles cannot farm
rewards. The sum of episode rewards is exactly
`64 * (final_similarity - blank_similarity)`, up to float rounding.

The task sets `gamma=0`: future cell rewards are unrelated to the present pigment
choice, so immediate reward avoids unnecessary return-to-go noise. There is no
long-horizon navigation decision. Generic REINFORCE remains responsible for all
weight updates. Image score updates use the local error delta; there is no image
processing, disk access, network access, animation, or waiting inside `step()`.

## Reference provenance and reproducibility

Leonardo da Vinci, *Mona Lisa*, Musée du Louvre. Wikimedia Commons describes this
faithful reproduction as public domain:
[source and public-domain notice](https://commons.wikimedia.org/wiki/File:Mona_Lisa.jpg).
The downloaded original URL and SHA-256 are stored in `reference.json`.

Asset preparation used Pillow: convert the full original to RGB, resize its whole
extent to 8×8 with BOX averaging, derive eight colors using MEDIANCUT on that
small image, then sort the colors by mean RGB. `reference.json` stores raw target
RGB values and palette values, not a target action map. `source.jpg` is a small
160×242 preview of the same original. Training reads JSON once at module import;
there is no runtime Pillow dependency or download. Committed data is authoritative
and avoids dependence on future image-tool versions. Reset is intentionally
identical for every seed; all variation comes from the selected actions.

## Validation

Local CPU validation on 2026-11-14:

- Registry discovers exactly one `FlyTask` subclass with reward mode, 12 inputs,
  eight unique actions and 64-step episodes.
- Four environment tests passed (including all 512 cell/pigment combinations):
  exact single-cell mutation, correct RGB, both reward signs, full-image reward
  equivalence, finite bounded observations, masks, terminal/reset behavior,
  invalid-action rejection, controlled determinism and telescoping return.
- Reset, observations, captions and episode steps work with file opening and
  socket creation disabled after import.
- Standard generic rollout produced 2,048 transitions across 32 random episodes.
- `python -m core.check painting --device cpu`: **PASS**. Real tiny graph,
  5,000 neurons; intact probability spread 0.0115, lesion spread 0.0; random
  mean return 4.075. This required check performs **two tiny CPU smoke updates**
  in memory; it saves no checkpoint. No proficiency training was run.
- 64,000 environment steps with resets: 1.585 seconds, approximately 40,373
  steps/second (24.8 microseconds/step). This excludes brain inference. NumPy
  allocations and Python calls dominate; no obvious environment bottleneck.
- Chromium ran the unchanged world on CPU inference for approximately two minutes:
  65 neural decisions, one completed 64-cell attempt and the next episode's first
  decision, with no page/console errors. Existing local services occupied default
  ports, so the test used HTTP 8137 / WebSocket 8876 and a browser-only socket URL
  override; no shared file was modified. The existing project Python environment
  at `/home/mitch/venvs/phyllis-inglewood` supplied the server's dependencies.
- A separate browser fixture checked every pigment's displayed RGB and that
  unselected cells remain unchanged, using the final station implementation.
  Ordinary animations measured 1,591–1,651 ms. Final paint was observed before
  clearing; admiration/retreat finished in 3,472.5 ms, below the 4,000 ms watchdog;
  the next first stroke painted correctly after reset.
- `preview.png` is a close-up from that eight-action browser fixture. Its fixed
  test captions and colors illustrate the station, **not learned performance**.

Reproduce from the repository root using a Python environment with the project's
existing requirements installed:

```bash
PYTHONDONTWRITEBYTECODE=1 python -m unittest discover -s flies/painting -p 'test_*.py' -v
PYTHONDONTWRITEBYTECODE=1 python -m core.check painting --device cpu
```

The validation machine's `python` command was unavailable; tests used
`/tmp/jobs-check-venv/bin/python` with existing torch/numpy, and two CPU threads
for the core check. These machine-specific paths are not deployment requirements.

## Generic GPU compatibility and eventual commands

The actual `core.train` implementation was inspected: registry loading, batched
rollouts, masks, reward learning, task `train` settings, evaluation, and checkpoint
saving all accept this task unchanged. `core.brain`, the template, gym, inbox,
server/viewer protocol, `scripts/remote.py`, README and COMPUTE guide were also
inspected. Full-GPU execution itself has not been tested at this stage.

Conservative initial settings in the task: 16 episodes/update, gradient chunk
128, gamma 0, with generic defaults lr 0.001, entropy 0.01, 300 update ceiling,
32 evaluation episodes. Chunk 128 bounds gradient batch memory, not all rollout
memory. No new package or CUDA operation is required by the task.

**Later only, after authorization to train and with existing remote setup:**

```bash
python -m core.check painting
python scripts/remote.py submit painting --size full --minutes 5
python scripts/remote.py status
python scripts/remote.py logs painting -f
python scripts/remote.py pull painting
```

Start with one five-minute full-brain run. Review it before another submission.
For a later 15-minute run, repeat the check and submit with `--minutes 15`.
The equivalent supported direct command on the prepared GPU host is:

```bash
python -m core.train painting --size full --minutes 15
```

These are verified CLI interfaces, **not claims that a job has been executed**.
No setup, owner command, pod change, or custom queue/training workflow is needed.
The generic time budget is checked between updates; an update and final evaluation
can extend execution beyond the nominal minutes.

## Outputs and later evaluation

The unchanged pipeline writes `runs/painting/brain.pt` and
`runs/painting/result.json`. Expected JSON fields include `preset`, `synthetic`,
`mode`, `actions`, `graph_hash`, `action_groups`, `settings`, `train_seconds`,
`score_trained`, `score_random`, and `score_lesioned`. Reward mode has no
teacher-agreement metric. Logs report average episode return every five updates.

Require `preset=full` and `synthetic=false`. Trained mean return should exceed
random; lesioned should fall back near random. Because reset is fixed and episodes
always complete, convert each mean return to mean final image similarity using
`blank_similarity + score/64`. Obtain blank similarity as `1-env.error` just after
reset. Completion is 100% by construction, not evidence of skill. The generic
trainer discards `info`, so image similarity/completion are not separate JSON
fields; this conversion requires no trainer changes. Evaluate over more seeds
if the default 32-episode estimate is inconclusive. Do not call the fly trained
or successful until results establish an advantage.

## Visualization

```bash
python -m server.serve --only painting
# Open http://localhost:8000/web/
```

No checkpoint means the standard viewer labels the fly untrained. The station
has a wooden easel, enlarged individual paint cells, reference portrait, palette,
brush and score panel. The eight action names map directly to eight pigment
indices. Captions transport authoritative cell numbers and similarity because
the unchanged server discards step `info`; JS never chooses a pigment or improves
the canvas. It only replays that selected action. No JS learning loop exists.

Ordinary brush actions take about 1.5 seconds. On cell 64 the stroke is compressed
to 260 ms, followed by a 180 ms retreat and a 3,000 ms admiration pause, then the
canvas clears. This 3,440 ms sequence fits the existing viewer's 4,000 ms watchdog.
The server resets its environment immediately on terminal; the viewer waits for
the animation before requesting the next decision. A full attempt takes about
98 seconds. Repetition is inference, not ongoing training. A very slow browser
can still hit the existing watchdog; shared viewer code was left untouched.

## Known limits

RGB MSE similarity can look numerically high even for gray or poor paintings;
it is not perceptual accuracy or a percentage of artistic mastery. Report the
random baseline and inspect canvases, not an isolated percentage. The eight-color
palette imposes an approximation ceiling. This asset has blank similarity
0.881235, analytically expected uniform-random similarity 0.947296 (episode
return 4.227905), and best possible palette approximation similarity 0.998921.
The ceiling is an offline metric, never an action selector. One pass prevents corrections, brush
shape learning, and free composition. RGB reference observations make this a
small color-matching problem; learning success is still unproven. A fixed target
and deterministic order permit position memorization. Full-connectome GPU speed
and proficiency require later measurement. The existing world loads three.js
from its existing CDN; simulation/training assets are entirely local.
