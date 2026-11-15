# Checkout audit and reuse inventory

Audited before implementation on branch `feat/pi-fly`, base `83c87c3` (the
checkout has one commit). Root `AGENTS.md` and `CLAUDE.md` do not exist. The
initial working tree contained only untracked `flies/pi/task.py` and
`flies/pi/station.js`, both unchanged copies of the starter template.

Read the README, task/brain/train/check/registry/graph modules, server, templates,
gym and inbox examples, fly model, world, matrix, HTML, data setup, CI and compute
workflow. No newer shared implementation exists in this checkout.

| Existing component | Reuse |
| --- | --- |
| `core.task.FlyTask` + `core.registry` | Native reward task, automatic discovery, eight fixed outputs |
| `core.brain.FlyBrain` | Actual FlyWire tiny graph, sensory projection, fixed descending groups, sparse backward, checkpoint format |
| `core.train` | Existing masked rollout and REINFORCE optimizer; feature-local runner redirects artifacts into this folder |
| `server.serve` | Genuine CPU inference, caption transport, independent browser session, ready handshake |
| `web/lib/fly.js` | Actual shared Fly, `tween(ms, fn)`, `box`, `cylinder`, `walkTo({x,z}, ms)`, `face`, `hop`, `buzz` |
| `web/main.js` | Scene, lighting, shadows, orbit/zoom, fly click inspection, trained/untrained labels |
| `web/matrix.js` | Actual recorded neuron rates and lesion toggle; no duplicate inspector |

Constraints discovered: observations are sensory inputs; no learned action head.
The server discards step info but transmits `caption()` after stepping, so a
compact versioned caption carries completed transition feedback and recent
accepted history. It resets finished episodes before sending that saved caption.
The host races play against a **4000 ms** timeout. Full animations must stay
comfortably below this; a station queue also prevents overlap if a slow frame or
background tab triggers that host timeout. An unchanged host cannot guarantee
brain/animation lockstep in a background or stalled tab; document that limit.

The supplied local tiny graph is non-synthetic: 5,000 neurons, 147,905 edges,
graph hash `cc2e5ce35f`. Raw FlyWire tables are already present. No remote jobs,
GPU work, shared source edits, commit, merge or push are needed.
