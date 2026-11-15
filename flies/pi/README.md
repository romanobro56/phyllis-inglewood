# Pi Fly

A native reward-mode FlyTask and 3D station. The real shared fruit fly carries a
neural-selected digit from a ten-tile deck to the sequence strip. Errors return
that same physical object; accepted tiles remain in the strip and replenish the
deck once. The reference begins `3.14159265358979323846264338327950288419716939937510`.

## Run

From the repository root, using the existing virtual environment and data:

```sh
PYTHONDONTWRITEBYTECODE=1 .venv/bin/python -m server.serve --only pi
# Open http://localhost:8000/web/ (uses shared runs/pi/brain.pt if present).

# Use the checkpoint produced by the feature-local experiment:
PYTHONDONTWRITEBYTECODE=1 .venv/bin/python -m flies.pi.serve --only pi
# If HTTP port 8000 is occupied, add --port 8001 and open /web/ there.
```

The host labels an absent checkpoint **untrained**. `flies.pi.serve` changes only
the host's in-memory checkpoint directory for Pi; it reuses the unchanged server
and writes no shared files. The frontend fixes WebSocket port **8765**; retain
that port even when changing the HTTP port. Only one brain server can use it.
Click the actual fly to use the existing neural inspector and lesion toggle;
drag and scroll use the host's orbit and zoom controls. The inspector receives
recorded FlyBrain rates, including for untrained brains. Inference is under
`torch.no_grad()` and never trains weights. The shared fly is the repository's
stylized 3D fruit-fly model, without a substitute model.

## Neural decisions and state

`PiFly` is discovered by `core.registry`, with exactly eight fixed actions
`choose_0` through `choose_7`, 40 sensory floats, reward mode, and sampled choices.
Stage 0 permits outputs 0 and 1 (digit groups 0–4 / 5–9). Stage 1 permits outputs
0–4 (offset). The completed digit is `group * 5 + offset`. There is a fresh real
FlyBrain inference for **each stage**, using the same fixed descending neuron
groups. Outputs 5–7 are always masked; they are declared for the supported eight
output contract. There is no policy in the station, teacher, output head,
strategic override, or change to connectivity.

The environment owns the full accepted prefix, current index/stage/group/digit,
completed/correct/incorrect guesses, per-position attempts, streak/best streak,
reward, prior guess/feedback, terminal/truncated flags, and first-attempt counts.
Every successful guess advances exactly one position; errors retain the position.
`step()` rejects masked actions and stepping a finished episode. Calling
`caption()` repeatedly cannot apply another transition or reward.

Observation schema (float32, finite, all in [-1,1]):

| Indices | Meaning |
| --- | --- |
| 0 | Constant 1 sensory bias |
| 1–3 | Public index /1024, (index mod 32)/31, floor(index/32)/32 |
| 4–5 | Stage 0/1; last selected group: 0 absent, .5 low, 1 high |
| 6–9 | Position attempts /20, guesses /64, streak/best clipped at 10 /10 |
| 10–11 | Previous guess `(d+1)/10` (0 absent), previous feedback -1/0/+1 |
| 12–15 | Correct /64, incorrect /64, reward /(3*64), decisions /128 |
| 16–31 | Last 16 accepted historical digits, `(d+1)/10`, left padded with 0 |
| 32–39 | Last four (guess, feedback) pairs, same encoding, left padded with 0 |

Normalizers for configurable limits use the corresponding environment settings.
The index is available history, not an answer lookup. The observation and mask
never read the current or future reference digits. The current target is read
only after a completed guess to grade it. Held-out evaluation explicitly supplies
a known historical prefix before the test interval; those supplied digits are
excluded from its verified progress metric.

Reward settings are feature-local class attributes: correct
`+1 + .1 * min(previous_streak, 10)`; incorrect `-.2`; intermediate group `0`.
A wrong guess resets the streak. There are no other bonuses or duplicate payouts.
Rewards are bounded [-.2,2] with the defaults; configuration validation keeps
rewards bounded [-1,3]. Defaults stop at 32 accepted digits, 64 guesses, 128
neural decisions, or 20 failed attempts at one position. Limits truncate;
only completing the finite prefix terminates successfully. All decision budgets
are even, so a limit never cuts a hierarchical choice in half. The shared
trainer's return-to-go propagates the final reward to both neural decisions;
`gamma=.995` per decision discounts approximately .990 per complete digit.
No unlimited episode reward farming is possible.

## Reference accuracy

`pi_digits.txt` is a versioned 1,024-digit string including the integer digit 3,
with no separator. SHA-256 (including final newline):
`cc362dbd5fd9161a2ed268b6f2f8a86cb07c4fc2a68a7cc65bf442e12b8a6a78`.
`reference.machin_digits()` uses integer arctangent series and guard digits;
tests independently verify all digits with 1,060-digit Decimal Chudnovsky
arithmetic and check the known first 100 digits. Ordinary `math.pi` is not used.
The decimal dot is attached visually after the first accepted digit; it is
never an action and scrolls away with that accepted slot.

## Training and evaluation

Use the feature-local runner so **all outputs remain under `flies/pi/`**:

```sh
PYTHONDONTWRITEBYTECODE=1 OMP_NUM_THREADS=1 VECLIB_MAXIMUM_THREADS=1 \
  .venv/bin/python -m flies.pi.experiment \
  --updates 3 --episodes 4 --eval-episodes 16 --prefix 16 --minutes 2

# Evaluate the saved checkpoint with a fresh reproducible sampling seed:
PYTHONDONTWRITEBYTECODE=1 OMP_NUM_THREADS=1 VECLIB_MAXIMUM_THREADS=1 \
  .venv/bin/python -m flies.pi.experiment --evaluate-only --eval-episodes 32 \
  --prefix 16 --eval-seed 456
```

This invokes **shared `core.train.train_reward`**, unchanged REINFORCE, and
shared `save_brain`/`load_brain`. It uses actual non-synthetic tiny FlyWire data on
CPU only. Extend local budgets deliberately for learning experiments. No shared
GPU, remote jobs, or network training are invoked. `core.train.rollout`, settings,
backpropagation and native checkpoint compatibility are tested directly.
The shared `python -m core.train pi ...` interface is compatible but writes to
`runs/pi/`; it was not used for this feature's saved outputs.

The four controls are matched-initialization untrained, trained, uniformly random
legal hierarchical actions, and trained with zeroed connectome. All use identical
masks, budgets, episode count and evaluation seed. `experiment.json` records each
episode, pooled **first-attempt accuracy**, incorrect guesses, attempts/accepted
digit (null if none), longest correct streak, mean verified prefix length,
cumulative reward, graph identity, checkpoint hash and a training progress curve.
Evaluation interval 0–15 is the training prefix; 16–31 is held out, initialized
with known context, rather than with hidden target hints. Reported trials are
visited positions, including an unsuccessfully attempted final position. A model
that stalls sees fewer positions; inspect prefix length and failures alongside
accuracy. Random first-attempt chance is 10%; eventual acceptance after retries
is **not** evidence of prediction.

Actual smoke evidence (seed 7 training / 123 evaluation, three updates, 16 episodes
per control/interval, 17.2 s CPU including evaluations):

| Training-prefix control | First-attempt accuracy | Mean verified digits | Misses (all episodes) | Attempts / accepted digit | Best streak | Mean reward |
| --- | --- | --- | --- | --- | --- | --- |
| Untrained | 12.75% (13/102) | 5.44 | 778 | 9.94 | 2 | -4.21 |
| Trained | 12.75% (13/102) | 5.44 | 778 | 9.94 | 2 | -4.21 |
| Random | 13.13% (13/99) | 5.25 | 750 | 9.93 | 3 | -4.06 |
| Lesioned | 14.02% (15/107) | 5.75 | 740 | 9.04 | 3 | -3.40 |

Held-out first-attempt accuracy was 8.62% untrained, 10.94% trained, 7.81% random,
and 4.48% lesioned. These are small, correlated adaptive episodes, not evidence
of statistically reliable improvement or unseen-digit prediction. Evaluations
after updates 0 and 2 each accepted 42 digits across eight sampled episodes,
with 8% first-attempt accuracy. The purpose of this short run is compatibility.

Model provenance: `tiny`, graph `cc2e5ce35f`, 5,000 neurons, 147,905 connections;
FlyWire v783 data from the repository's Shiu/Schlegel sources. Rebuilding the graph
in memory from the existing raw tables matched every root ID, edge, sign,
synapse count, sensory index and descending index. Backward tests verify fixed
buffers/groups/signs and finite nonzero gradients on actual synaptic weights.
Training changed only the supported `log_w`, bias, leak, and sensory projection.
Smoke checkpoint SHA-256:
`4ba959119212afaff47416f1ffe55959addb5834e8ea45a0cfa4e07fc588f6ca`.
The local checkpoint is ignored by git; the measurements and training log are
kept in this folder. Retraining replaces it and records a new identity.

## Animation and integration

The native contract remains `build({THREE, group, fly})` and
`play(action, caption, {fly})`. The compact readable `π1` caption protocol carries
an episode/transition ID, group or completed digit and authoritative ✓/✗,
accepted index/suffix, streak, errors, reward and completion status. The server
discards step-info and resets finished episodes before transmitting a saved
caption; this protocol works with that exact behavior. There is **no second pi
reference in JavaScript** and no renderer-side correctness prediction.

There are seven physical slots (six recent accepted digits plus one active slot).
On acceptance they tween exactly one pitch left, dispose the offscreen slot,
add one empty destination and replenish one consumed tile. On rejection the
same object is retrieved and released at its original home; no strip movement
or tile creation occurs. Sprites keep numbers facing the camera. The tile is
attached to the actual fly body while carried. All movement, lowering, feedback,
return and replenishment complete before the returned promise resolves. A
feature-local queue ignores duplicates/stale IDs and serializes overlap.

The host has a fixed **4,000 ms** fallback timeout. Normal complete paths are
2,060 ms correct / 2,690 ms incorrect (plus frame scheduling), group thinking
180 ms. In a background tab, severe frame stall or disconnected browser, the
unchanged host can request another neural choice before animation completion;
local serialization prevents overlapping animations but cannot guarantee host
lockstep under that condition. Transition gaps fail closed and require refresh.
This shared-host constraint cannot be removed within the authorized folder.

## Validation

```sh
PYTHONDONTWRITEBYTECODE=1 .venv/bin/python -m core.check pi
PYTHONDONTWRITEBYTECODE=1 OMP_NUM_THREADS=1 \
  .venv/bin/python -m unittest discover -s flies/pi/tests -v

# JS tests with Node 18+ (native Node is optional for running the world):
# Download the exact module URL already used by web/index.html to /tmp.
curl -fsSL https://cdn.jsdelivr.net/npm/three@0.170.0/build/three.module.js \
  -o /tmp/pi-three.module.mjs
node --experimental-vm-modules flies/pi/tests/run_js.mjs /tmp/pi-three.module.mjs
```

`core.check pi` passed on actual tiny data, including two training updates and
the lesion test. All 14 Python tests pass and cover task/state/masks/reference/leakage, limits,
rewards, real gradients, inference weight immutability, native server activity,
terminal-caption transport and repository scope. The JS suite passed 436
assertions with real Three.js 0.170.0 objects and the shared Fly/model/helpers,
using a virtual animation clock and stubbed canvas drawing. It ran through the
available Node REPL because this shell does not have Node installed. This is
scene/logic testing, **not** a WebGL renderer test.

Separately, the actual world was opened in Safari at
`http://127.0.0.1:8001/web/`, connected to actual untrained tiny inference.
The shared fly, ten-number deck, moving carried tile, strip and stats rendered
in WebGL. Live authoritative prefixes progressed through `3`, `314`, `3141`
and `314159` with interspersed wrong guesses. Clicking the shared fly opened the existing matrix inspector; genuine activity
points, fixed action groups and current action probabilities rendered in WebGL.
Native-server integration tests also verify the actual recorded activity payload. No automated
browser console audit, mobile device or cross-browser matrix was run.

See `AUDIT.md` for the reuse inventory. No tracked source outside `flies/pi/`
was edited. No commit, merge, push or shared GPU job was performed.
