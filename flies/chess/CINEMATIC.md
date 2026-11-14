# The Connectome Table

A feature-local live 3D chess arena. Two original repository flies carry moves selected by two independent, actual FlyWire-derived `FlyBrain` models. It uses the existing Three.js r170 stack and the existing Matrix neural viewer. Nothing in `core/`, `server/`, `web/`, `scripts/` or biological data is changed.

## Launch

From the configured repository root:

```bash
PYTHONDONTWRITEBYTECODE=1 .venv/bin/python -m flies.chess.cinematic --port 8770
```

Open **http://127.0.0.1:8770/** and press **New live match**. Defaults are two untrained 5,000-neuron FlyWire models, CPU, seed 42, 160-ply cap. No training runs inside the arena. The shared Three.js CDN must be accessible, as in the original world. No new frontend packages or installation are required. WebGL2 and browser import maps are required.

To use the existing independently trained checkpoints:

```bash
PYTHONDONTWRITEBYTECODE=1 .venv/bin/python -m flies.chess.cinematic --port 8770 \
  --white-checkpoint flies/chess/artifacts/mechanism/loo-11/checkpoint_0048_0.pt \
  --black-checkpoint flies/chess/artifacts/mechanism/loo-11/checkpoint_0048_1.pt
```

The dropdowns in **Match settings** also select trusted checkpoints under this feature's artifacts directory. Preset must match each checkpoint. Missing or incompatible checkpoints produce an explicit error; there is no substituted agent. A TRAINED label means saved trained weights, not demonstrated general chess competence. Both brains remain separately allocated even if the same checkpoint is chosen twice. Available checkpoint artifacts are local and ignored by Git; clean checkouts can use untrained mode or reproduce the existing training commands in README.

`--size tiny|small|central|full`, `--seed`, `--max-plies`, `--greedy`, and `--fen '...'` configure the match. Full graph uses **local CPU only**; loading it is deliberately explicit and may be expensive. No GPU jobs or remote commands are invoked. Ctrl-C stops the listener, cancels the session and saves its actual board. A capped or stopped game remains unfinished (`*`) unless chess rules already supply a result. Training and evaluation continue through the existing documented CLI tools, separate from the arena.

## At the table

- **New live match / Restart:** create a unique match and independently initialize/load both models. New matches preserve all earlier recordings.
- **Pause / Resume:** freeze the carry director and block new neural stages. A core forward pass already running may finish its current stage; no other player's turn can start without ACK.
- **Pace:** scales animation time only; it never changes a selected move. Reduced-motion preferences use shorter transport and no camera shake.
- **Cameras:** cinematic three-quarter, active-fly follow, elevated board, free orbit/zoom and slow showcase. Dragging switches to Free. Camera transitions retain the board as target.
- **Moves:** SAN history, official outcome versus truncation, verified game JSON and PGN links.
- **Replay:** load a recorded JSON file or replay the current saved recording, play/pause, change pace or seek by completed ply. The banner says RECORDED REPLAY. Replay never calls inference. Recordings with no full per-neuron rates cannot open a claimed live brain inspection; start Live to inspect actual models.
- **Click either fly**, or use its brain button, to open the repository's original Matrix viewer. White/Black switches serialize old rate playback before changing data source. Inspection initially pauses the game. **Observe live updates** continues the actual director off-screen and receives bounded latest measurements; **Pause game updates** freezes it again. Closing returns to the physical board. The rate timestamp, decision stage, selected slot, eight group means, observation encoding, graph, model preset and checkpoint identify the measurement. No inference yet means no measured activity yet.

Neural colors replay eight **simulated neuron rate** iterations, not biological spikes. They use the shared viewer's sqrt(rate/2), uint8 quantization, clipped above rate 2. Base class/input colors and the shared glyph rain are decorative and labeled as such. Real normalized FlyWire neuron coordinates and FlyWire IDs come from the original metadata. No substitute neural renderer or decorative firing trace is used.

## Authoritative pipeline and handshake

```
READY → explicit /api/next → THINKING (4–5 FlyBrain stages)
      → ANIMATING (before FEN + selected move + after FEN)
      → physical approach/pickup/carry/place/release/return
      → validate rendered placement → explicit /api/ack → READY or FINISHED
```

`live_match.py` owns one `ChessEnv`, two independent models, a NumPy sampling stream and one serial neural worker. Its worker returns after selecting one move. It cannot enqueue the next turn. Both the live controller and the ordinary self-play runner reuse `selfplay.choose_stage`. Legal masks and `python-chess` enforce rules; all strategic coordinates come from the existing eight descending-neuron groups. Inference is no-grad, evaluation mode, and has no optimizer. Fixed graph buffers and weights are immutable in live mode.

Each match has a UUID, capability token, connection epoch, monotonic event IDs and move sequences. ACK requires the current identity/epoch, exactly the pending sequence, the authoritative full FEN and a placement reconstructed from actual rendered pieces. Duplicate, stale, out-of-order and mismatched ACKs are rejected. The client independently reconstructs captures, castling, en passant and promotions from before/after positions and verifies the server's rules-only visual metadata. The director asserts source ownership, piece attachment state, exact positions, final map and fly return before acknowledging. All components are presentation/rules adapters, never move selectors.

A connection claim rotates its epoch and pauses. Reload/reconnection cancels stale visual callbacks and reconstructs any pending move from its before FEN; Resume animates it again before ACK. Lost ACK responses recover against the server's acknowledged state. Client generations ignore late responses from the old connection. A 20-second lost-heartbeat lease pauses play. A 120-second active-animation timeout reports an error with the pending move retained. Neither timeout is an ACK. Paused time is excluded. WebGL loss pauses; restoration claims a new epoch. No hanging animation silently advances a turn.

There is one active local match and one controlling browser connection. A second tab claims the session, superseding the first; the old tab stops and offers **Take control** rather than repeatedly claiming it back. This is not a multi-user public server. Match control is bound to 127.0.0.1, rejects cross-origin mutations, serves shared assets read-only and never serves `.pt` or `.npz`. Load only trusted local checkpoints (PyTorch payloads are trusted pickle files).

## Recording and memory

Unique `artifacts/cinematic/<match-id>/game_0000.json`, `.pgn` and `protocol.json` files are saved atomically at selection/ACK/cancellation. `visually_acknowledged_through` distinguishes the selected authoritative move from completed presentation. PGN export replays and verifies every recorded legal move and outcome. Recordings persist; live model/RNG/controller state is in memory. After a server-process restart, use Replay for the saved game or start a fresh live match; exact process-resume for a half-selected live move is not claimed. Existing **training** resume remains supported separately.

Only the latest full rate packet per brain is retained, not a full-game neuron-rate movie. At full size this is two 8×134,181-byte quantized arrays (plus JSON/base64 and the current forward's transient tensors). Per-stage activity summaries persist in game records; dense rates are inspection-only. Unchanged packets are cached in the client. Event history is bounded to 64 entries; complete move histories are bounded by the explicit 1–1,024-ply cap. Pieces share cached geometry/materials; captures are removed, repeated board resets reuse resources, and the Matrix adapter disposes previous point geometry/materials when switching brains.

Default DPR is capped; quality levels choose shadow resolution, render resolution and optional restrained SSAO. The environment uses same-version Three.js RoomEnvironment/PMREM, physically based stone/ivory materials, warm directional lights, soft shadows, ACES tone mapping and OutputPass. Persistent low measured frame rate selects low quality. Resize, hidden-tab pause and WebGL recovery are supported. A 60-FPS observation on one desktop is not a guarantee for other hardware or full graphs.

## Reuse and scope

See [CINEMATIC_REUSE.md](CINEMATIC_REUSE.md) for exact source paths/APIs. The original fly has no skeletal gripping rig, imported model loader or animation clips. Chess attaches the piece using `Object3D.attach` beneath the actual fly and animates its existing leg nodes. Its wing/body identity is retained. The shared horizontal wall-clock tween cannot be paused/cancelled for curved 3D carrying, so the local Director owns the pausable transport clock. The shared Matrix implementation is imported unchanged; its host DOM and styles are imported directly from `web/index.html` at runtime.

The shared world is a non-exported application and the shared backend does not support two models owning one board. This local arena is independently functional and reuses its actual shared assets; inserting the chess arena into the shared `/web/` multi-station world would require the core owner's integration approval. No protected file is edited to achieve it.

## Reproduce verification

With this server running, open **http://127.0.0.1:8770/tests/cinematic_browser.html**. This clearly labeled test harness uses actual WebGL, the production arena, actual tiny FlyWire models and existing trained checkpoints. It creates its own recorded test matches; keep just this controlling tab open. It writes its detailed observed result to `artifacts/cinematic-browser.json` and shows PASS/FAIL. Deliberate context loss emits an expected notification; unexpected console errors fail the harness.

```bash
PYTHONDONTWRITEBYTECODE=1 CHESS_REAL_GRAPH_TESTS=1 .venv/bin/python -m unittest discover -s flies/chess/tests -v
PYTHONDONTWRITEBYTECODE=1 .venv/bin/python -m core.check chess
# Optional Node >=20 (not needed to launch the arena):
node --experimental-vm-modules flies/chess/tests/cinematic_smoke.mjs
node --experimental-vm-modules flies/chess/tests/visual_smoke.mjs
```

The new JS smoke suite verifies state reconstruction, duplicate/order protection, stale callbacks, full carry lifecycle, pause and cancellation using a deterministic presentation fixture. It is separate from the real WebGL harness. Tests for special chess moves use explicitly labeled rules fixtures, not invented neural decisions. Live tests never override the actual neural move selection.

The arena is an engineering/visualization improvement. It supplies **no new evidence of stronger chess play**. Existing scientific results, negative findings and full-graph scaling gates remain in [MECHANISM_RESULTS.md](MECHANISM_RESULTS.md) and [SCALING.md](SCALING.md).
