"""Build brain graphs from the FlyWire v783 connectome.

    python -m core.graph --size small        # one preset
    python -m core.graph --size all          # every preset
    python -m core.graph --size synthetic    # fake graph, no download needed (testing only)

Presets (all share the same rules, only the neuron count differs):
    full       every proofread neuron (~139k)                    rented GPU
    central    full minus the optic lobes (~62k)                  24 GB GPU or big laptop GPU
    small      20k most-connected neurons (+ all inputs/outputs)  laptop GPU
    tiny       5k most-connected neurons (+ all outputs)          CPU, smoke tests
    synthetic  random wiring, 3k neurons                          offline testing only, never for the demo

Writes data/<preset>.npz (used for training) and data/<preset>.points.bin +
data/<preset>.meta.json (used by the matrix view in the browser).
"""
import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
RAW = ROOT / "data" / "raw"
DATA = ROOT / "data"

MIN_SYNAPSES = 5          # FlyWire's standard threshold for a real connection
N_INPUTS = 2048           # sensory neurons that receive the task's observation
INPUT_CLASSES = ["sensory", "sensory_ascending", "ascending"]
OUTPUT_CLASS = "descending"
PRESETS = ["full", "central", "small", "tiny", "synthetic"]
KEEP = {"small": 20000, "tiny": 5000}


def load_raw():
    for f in ["Connectivity_783.parquet", "Completeness_783.csv", "neuron_annotations.tsv"]:
        if not (RAW / f).exists():
            raise SystemExit(f"Missing data/raw/{f}. Run: python scripts/download_data.py")
    con = pd.read_parquet(RAW / "Connectivity_783.parquet",
                          columns=["Presynaptic_ID", "Postsynaptic_ID", "Connectivity", "Excitatory"])
    con = con[con.Connectivity >= MIN_SYNAPSES]
    ann = pd.read_csv(RAW / "neuron_annotations.tsv", sep="\t", low_memory=False,
                      usecols=["root_id", "pos_x", "pos_y", "pos_z", "super_class", "cell_type", "top_nt"])
    ann = ann.drop_duplicates("root_id").set_index("root_id")
    return con, ann


def _top(series, k):
    """Indices of the k largest values; ties broken by neuron ID so every machine picks the same set."""
    order = np.lexsort((series.index.values, -series.values))
    return series.index.values[order[:k]]


def select_neurons(con, ann, size):
    ids = np.union1d(con.Presynaptic_ID.unique(), con.Postsynaptic_ID.unique())
    cls = np.asarray(ann.super_class.reindex(ids).fillna("unknown").tolist(), dtype=str)
    if size == "central":
        ids = ids[cls != "optic"]
    elif size in KEEP:
        deg = (con.groupby("Presynaptic_ID").Connectivity.sum()
               .add(con.groupby("Postsynaptic_ID").Connectivity.sum(), fill_value=0))
        deg = deg.reindex(ids).fillna(0)
        must_out = ids[cls == OUTPUT_CLASS]
        inputs = ids[np.isin(cls, INPUT_CLASSES)]
        # keep the most connected input neurons so the observation can actually propagate
        top_inputs = _top(deg.reindex(inputs), N_INPUTS // 2)
        forced = np.union1d(must_out, top_inputs)
        rest = _top(deg.drop(forced), max(0, KEEP[size] - len(forced)))
        ids = np.sort(np.union1d(forced, rest))
    return ids


def build(size):
    if size == "synthetic":
        return build_synthetic()
    print(f"[{size}] loading FlyWire v783 tables ...")
    con, ann = load_raw()
    ids = select_neurons(con, ann, size)
    index = pd.Series(np.arange(len(ids)), index=ids)
    con = con[con.Presynaptic_ID.isin(index.index) & con.Postsynaptic_ID.isin(index.index)]
    pre = index[con.Presynaptic_ID.values].values.astype(np.int64)
    post = index[con.Postsynaptic_ID.values].values.astype(np.int64)
    count = con.Connectivity.values.astype(np.float32)
    sign = np.where(con.Excitatory.values > 0, 1, -1).astype(np.int8)

    a = ann.reindex(ids)
    super_class = np.asarray(a.super_class.fillna("unknown").tolist(), dtype=str)
    cell_type = np.asarray(a.cell_type.fillna("").tolist(), dtype=str)
    pos = a[["pos_x", "pos_y", "pos_z"]].values.astype(np.float64)
    missing = np.isnan(pos).any(1)
    pos[missing] = np.nanmean(pos, 0)

    outdeg = np.bincount(pre, weights=count, minlength=len(ids))
    cand = np.where(np.isin(super_class, INPUT_CLASSES))[0]
    input_idx = np.sort(cand[np.lexsort((cand, -outdeg[cand]))][:N_INPUTS])
    output_idx = np.where(super_class == OUTPUT_CLASS)[0]
    return dict(root_id=ids.astype(np.int64), pre=pre, post=post, count=count, sign=sign,
                super_class=super_class, cell_type=cell_type, pos=pos,
                input_idx=input_idx, output_idx=output_idx, synthetic=False)


def build_synthetic(n=3000, n_in=300, n_out=200, avg_deg=40, seed=0):
    print("[synthetic] building a RANDOM graph (testing only, not a fly)")
    rng = np.random.default_rng(seed)
    e = n * avg_deg
    pre, post = rng.integers(0, n, e), rng.integers(0, n, e)
    keep = pre != post
    pre, post = pre[keep], post[keep]
    pairs = np.unique(np.stack([pre, post], 1), axis=0)
    pre, post = pairs[:, 0], pairs[:, 1]
    count = rng.geometric(0.15, len(pre)).astype(np.float32) + 4
    neuron_sign = np.where(rng.random(n) < 0.7, 1, -1)
    super_class = np.array(["central"] * n, dtype=object)
    super_class[:n_in] = "sensory"
    super_class[n - n_out:] = "descending"
    pos = rng.normal(size=(n, 3)) * [3e4, 1.5e4, 1e4] + [5e5, 3e5, 1.5e5]
    return dict(root_id=np.arange(n, dtype=np.int64), pre=pre, post=post, count=count,
                sign=neuron_sign[pre].astype(np.int8), super_class=super_class.astype(str),
                cell_type=np.array([""] * n), pos=pos,
                input_idx=np.arange(n_in), output_idx=np.arange(n - n_out, n), synthetic=True)


def save(size, g):
    DATA.mkdir(exist_ok=True)
    # identifies the wiring AND which neurons are inputs/outputs, so a brain never loads onto a mismatched graph
    h = hashlib.sha1(np.concatenate([g["pre"], g["post"], g["input_idx"], g["output_idx"]])
                     .astype(np.int64).tobytes()).hexdigest()[:10]
    np.savez_compressed(DATA / f"{size}.npz", n=len(g["root_id"]), graph_hash=h, **g)

    # browser files: positions centred and scaled to about [-1, 1], y up
    p = g["pos"] - g["pos"].mean(0)
    p = p / np.abs(p).max()
    xyz = np.stack([p[:, 0], -p[:, 1], -p[:, 2]], 1).astype(np.float32)
    (DATA / f"{size}.points.bin").write_bytes(xyz.tobytes())
    classes = sorted(set(g["super_class"]))
    meta = dict(preset=size, n=len(g["root_id"]), graph_hash=h, synthetic=bool(g["synthetic"]),
                classes=classes,
                class_index=[classes.index(c) for c in g["super_class"]],
                root_id=[str(x) for x in g["root_id"]],
                cell_type=list(g["cell_type"]),
                input_idx=g["input_idx"].tolist(), output_idx=g["output_idx"].tolist())
    (DATA / f"{size}.meta.json").write_text(json.dumps(meta, separators=(",", ":")))
    print(f"[{size}] {meta['n']:,} neurons, {len(g['pre']):,} connections, "
          f"{len(g['input_idx'])} input neurons, {len(g['output_idx'])} descending neurons -> data/{size}.npz")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--size", default="small", choices=PRESETS + ["all"])
    a = ap.parse_args()
    sizes = PRESETS if a.size == "all" else [a.size]
    for s in sizes:
        save(s, build(s))


if __name__ == "__main__":
    main()
