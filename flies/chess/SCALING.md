# Scaling evidence and proposed full-graph experiment

The current experiments do **not** justify shared GPU training. The fresh matched-adaptation gate now passes, but improvement over the frozen parent, mate-two progression and meaningful full-game gains do not. See [MECHANISM_RESULTS.md](MECHANISM_RESULTS.md). Full-graph **CPU compatibility** is now tested; full-graph learning remains unexecuted. No remote command or GPU job was run.

## Full CPU compatibility actually executed

`python -m flies.chess.full_smoke --output flies/chess/artifacts/full-cpu-final` passed: two independent actual full brains (134,181 neurons, 2,700,513 edges, graph9ae1232b36), four forward calls per side in a legal two-ply game, group activity recording, separate backward/Adam updates, fixed-buffer preservation, checkpoint round-trip and an additional spatial-encoding forward. The probe took10.873224292s end-to-end with process peak1,071,874,048 bytes. This includes three resident model instances at checkpoint verification plus optimizer/backend/interpreter costs; it is not an inference latency or GPU-memory estimate. Both learners had zero outcome rewards: entropy alone exercised backward. The exported full checkpoints are **engineering probes, not chess-trained agents**.

```bash
PYTHONDONTWRITEBYTECODE=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 .venv/bin/python -m flies.chess.full_smoke \
  --output flies/chess/artifacts/full-cpu-repeat
```

Use a fresh output directory; the completed full probes are retained. This is optional local verification, not a remote job. The native self-play/training entrypoints already support `--size full --device cpu`; a future approved GPU job would use `--device cuda`. Tiny checkpoints cannot be loaded into the full graph. The CPU probe closes the initialization/forward/backward/checkpoint compatibility gap; it does not meet the scientific training gates below.

## Measured locally

Actual FlyWire tiny: 5,000 neurons, 147,905 edges, 1,024 sensory neurons, graph `cc2e5ce35f`; CPU, one PyTorch thread. Each number is a median of five measurements after warmup in a fresh process. Forward/backward includes an Adam step with lr=0 to allocate optimizer state without changing the policy. This is one coordinate decision, not one chess move.

| Batch | Forward | Forward + backward + Adam |
|---|---:|---:|
| 1 | 24.94 ms | 61.66 ms |
| 8 | 32.53 ms | 117.54 ms |
| 32 | 36.66 ms | 210.40 ms |

Process peak RSS rose from 145,096,704 to 194,363,392 bytes. This is a process high-water mark, including interpreter/libraries, not tensor storage or GPU memory. An earlier run measured 29.46/84.28 ms at batch 1; host load and allocator effects are uncontrolled. Both runs are retained. Timings should not be extrapolated into a promised GPU speedup.

One chess ply requires four or five decisions, each with the core's eight rate iterations. Batch-1 forward timing therefore suggests roughly 100–125 ms per ply under this benchmark, excluding rules, scheduling and recording. Exact integration over a legal decision tree requires many more forwards than a sampled move. Activity recording stores optional dense neuron rates; eight full-graph rate arrays alone occupy 4,293,792 bytes per decision. Prefer ordinary descending-group summaries for sustained games.

## Full-graph storage projection (metadata only)

Existing `data/full.npz`: 134,181 neurons, 2,700,513 edges, 2,048 sensory neurons, graph `9ae1232b36`. The original estimates below used metadata only; subsequent CPU model verification is described above. Parameter count is `E + 2N + 64I`; fixed buffers include sparse indices, transpose indices/permutation, signs, sensory indices and the eight fixed groups. Float32 single-model storage is approximately **131,239,768 bytes**.

| Backward batch | Two Adam learners + one frozen brain + eight saved rate states: lower bound |
|---|---:|
| 1 | 472,411,824 bytes |
| 8 | 502,468,368 bytes |
| 32 | 605,519,376 bytes |

These bounds exclude other autograd intermediates, sparse-backend scratch, gathered edge products, CUDA context, allocator overhead and CPU snapshot/checkpoint pools. They are **not** an estimate of required VRAM. Measure `torch.cuda.max_memory_allocated/reserved`, host RSS and wall time at batch 1, then chunk 8, before choosing a production batch. GPU inference and training requirements are unmeasured.

## Gates before requesting shared compute

1. On a new frozen held-out matched corpus, correct-minus-swapped adaptation has a nominal simultaneous interval above zero, a mean of at least 2 percentage points, and a positive direction in all three seeds. Improvement must also exceed the parent policy and an independently trained restricted model.
2. Held-out mate-two continuation wins or independently justified endgame conversions improve with uncertainty excluding zero. Root proof mass, auxiliary material score, and legality alone do not qualify.
3. Exact resume, fixed-buffer integrity, reward-only gradient checks and stable optimizer diagnostics pass at the intended difficulty. The three-seed integer-v2 repetition is complete and exactly matches legacy weights/Adam states; no learning benefit can be attributed to the arithmetic correction.
4. Meaningful opening-start games finish across varied starts/opponents. No Elo until a sufficiently large, controlled full-game study exists. The current 24 games from one opening are inadequate.

Next tiny work should target reward density and representation using training/validation data: tune an easier-to-harder forced-mate continuation curriculum with actual sampled neural replies; report every auxiliary reward separately; compare coordinate encodings under matched training budgets. The current post-training input knockouts are distribution shifts and cannot prove an optimal encoding. Any rules-derived progress reward must be disclosed as an engineered objective and must never select the fly's move. Test only on a fresh locked corpus after these choices are fixed.

## Proposed full-graph pilot — NOT EXECUTED

Tiny checkpoints cannot load into the full graph. After the gates and explicit approval, initialize two fresh independent full brains through `storage.load_policy(size='full', encoding=SPATIAL_ENCODING)` using different fixed seeds and save `brain_payload` exports under `flies/chess/artifacts/full-proposal/initial_{0,1}.pt`. Preserve the eight existing descending groups and all fixed buffers. The named full-proposal experiment remains unexecuted; the separate CPU engineering probe is not its training initialization.

An owner-approved job could then run this conservative five-update matched-only smoke pilot:

```bash
python -m core.check chess
python -m flies.chess.research_train \
  --size full --device cuda --seed 11 --updates 5 --games 32 --chunk 8 \
  --lr .001 --entropy .003 --reward-version integer-v2 \
  --matched flies/chess/research_data/matched.json \
  --extended flies/chess/research_data/extended.json \
  --sources flies/chess/artifacts/full-proposal/initial_0.pt flies/chess/artifacts/full-proposal/initial_1.pt \
  --output flies/chess/artifacts/full-proposal/pilot-11
```

This pilot only establishes numerical/compute feasibility; it cannot establish learning. A subsequent three-seed full study would need a fresh, fixed protocol and budget. The repository's normal `remote.py submit chess` invokes the native single-fly demo training path, **not** this research runner. The core owner would need to approve a custom entrypoint within the permitted queue workflow; no shared scripts should be changed by this fly. Do not run remote shell training or bypass the queue.

Reproduce the local measurements with:

```bash
PYTHONDONTWRITEBYTECODE=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 .venv/bin/python -m flies.chess.benchmark \
  --output flies/chess/artifacts/research/benchmark-repeat.json
```

## Cinematic live arena (local integration addition)

`python -m flies.chess.cinematic --size full --port 8770` uses the existing full-graph loader on **CPU**, with two independent models and no training/optimizer allocations. No GPU job is submitted. This entrypoint and all model compatibility checks are shared with tiny; the full-graph CPU probe documented above remains the executed architecture evidence. Full-graph cinematic browser throughput has **not** been measured.

The arena's worker records eight actual rate iterations for the current hierarchical decision. The latest quantized 8×N packet is retained per side, about 2.05 MiB raw for two full graphs, plus base64/JSON, original model tensors, transient float rates and viewer color buffers. This is not a full memory estimate. Original full-graph point metadata is reused by the shared Matrix, unchanged packets are cached, and full-game dense activity is not accumulated. Move/event summaries remain bounded by the ply cap / 64-event ring. Browser DPR, shadows and SSAO have selectable quality; use low quality for constrained clients. Board drawing stays asynchronous from CPU inference.

Use the [cinematic protocol and commands](CINEMATIC.md) for local demonstration and its explicit Live/Replay distinction. This UI does not change the scientific scaling decision: **general-game improvement and a reliable incremental learning advantage remain unestablished; further full-brain training is not justified by these engineering tests.** Shared GPU use still requires the user's authorization and the team's queue. Shared-world embedding needs the core owner's API integration; no shared server or renderer is modified.
