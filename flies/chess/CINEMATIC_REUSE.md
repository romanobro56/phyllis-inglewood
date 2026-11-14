# Cinematic reuse inventory

Inspected before implementation. Shared sources remain read-only.

| Source | Supported API reused | Feature-local usage |
|---|---|---|
| `web/lib/fly.js` | `new Fly(color)`, `.root`, `.body`, `.wings`, `.face(target)` | Two original brown-bodied, red-eyed fruit flies. No external model files, skeletal rigs, loaders or AnimationMixer clips exist. Pausable flight and attachment extend these public scene nodes. |
| `web/matrix.js` | `Matrix({onClose,onLesion})`, `open(info,title)`, `show(message)`, `close()` | Single existing neural renderer, serialized source switching; actual rate packets from the selected brain. No copied neural viewer. |
| `web/index.html` | Existing `#matrix` DOM and inline shared styles | Fetched and imported at runtime so the original viewer's host contract and styling are reused. |
| `web/main.js` | Three.js r170 import map, renderer/OrbitControls conventions | Shared top-level world is not exported; chess creates its scene using the same library and controls. |
| `core/brain.py` | `FlyBrain.forward(...,record=True)` and fixed groups | Both independent live models, with unmodified biological buffers. |
| `core/graph.py`, `data/*.points.bin`, `data/*.meta.json` | Normalized measured neuron coordinates and metadata | Original Matrix geometry, FlyWire IDs and sensory/action indices. |
| `server/serve.py` | Documented uint8 sqrt(rate/2) telemetry format | Compatible encoding only; no shared backend changes. Native ready protocol cannot own a two-brain chess board. |
| `flies/chess/chess_env.py` | `ChessEnv.observe/mask/select/outcome` | Authoritative rules, legal hierarchical moves and FEN. |
| `flies/chess/selfplay.py` | `choose_stage`, `assert_independent`, `save_game` | Actual independent inference, validated JSON and PGN. |
| `flies/chess/storage.py` | `load_policy`, `identity`, atomic feature-local records | Existing trained/untrained checkpoints and immutable connectivity validation. |
| `flies/gym/station.js`, `flies/_template/station.js` | Existing fly animation patterns | Reviewed; shared horizontal wall-clock tween cannot support cancellable, paused 3D transport. |

No new rendering framework or animation dependency. Three.js r170 addons use the same version and MIT license as the existing host. Official source APIs checked: [SSAOPass](https://github.com/mrdoob/three.js/blob/r170/examples/jsm/postprocessing/SSAOPass.js), [RoomEnvironment](https://github.com/mrdoob/three.js/blob/r170/examples/jsm/environments/RoomEnvironment.js), [license](https://github.com/mrdoob/three.js/blob/r170/LICENSE). Bloom, SSAO and environment rendering are presentation effects, never measured neural activity. Matrix glyph rain is decorative; measured rate playback is explicitly labeled.

The shared world is a non-exported application and its backend has no two-agent match/animation acknowledgment contract. A local HTTP adapter serves the existing shared assets read-only and owns only chess sessions. Embedding the arena inside the shared world remains a core-owner integration, not a reason to alter protected files.
