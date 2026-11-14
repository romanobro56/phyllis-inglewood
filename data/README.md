# data/

Nothing here is committed. Build it on each machine:

```
python scripts/download_data.py      # ~110 MB into data/raw/
python -m core.graph --size all      # ~10 s, writes data/<preset>.npz/.points.bin/.meta.json
```

The build is deterministic, so a brain trained on the GPU loads on any laptop that built the same preset.
Offline? `python -m core.graph --size synthetic` makes a random test graph (never use it for the demo).

Data: FlyWire v783 (Dorkenwald et al., Nature 2024; Schlegel et al., Nature 2024), packaged by
Shiu et al., Nature 2024. Cite these if you show or publish anything built on it.
