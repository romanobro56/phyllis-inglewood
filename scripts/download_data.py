"""Download the FlyWire v783 connectome tables into data/raw/ (about 110 MB).

    python scripts/download_data.py

Sources (public, no login):
  - Connectivity_783.parquet, Completeness_783.csv: github.com/philshiu/Drosophila_brain_model
    (Shiu et al., Nature 2024; built from FlyWire v783, Dorkenwald et al., Nature 2024)
  - neuron_annotations.tsv: github.com/flyconnectome/flywire_annotations
    (Schlegel et al., Nature 2024) - cell classes, neurotransmitters, positions
Cite these papers if you show or publish anything built on the data.
"""
import sys
import urllib.request
from pathlib import Path

RAW = Path(__file__).resolve().parents[1] / "data" / "raw"

FILES = {
    "Connectivity_783.parquet":
        "https://raw.githubusercontent.com/philshiu/Drosophila_brain_model/main/Connectivity_783.parquet",
    "Completeness_783.csv":
        "https://raw.githubusercontent.com/philshiu/Drosophila_brain_model/main/Completeness_783.csv",
    "neuron_annotations.tsv":
        "https://raw.githubusercontent.com/flyconnectome/flywire_annotations/main/"
        "supplemental_files/Supplemental_file1_neuron_annotations.tsv",
}


def progress(name):
    def hook(blocks, block_size, total):
        done = blocks * block_size
        if total > 0:
            pct = min(100, done * 100 // total)
            sys.stdout.write(f"\r  {name}: {pct:3d}% of {total / 1e6:.0f} MB")
            sys.stdout.flush()
    return hook


def main():
    RAW.mkdir(parents=True, exist_ok=True)
    for name, url in FILES.items():
        dest = RAW / name
        if dest.exists() and dest.stat().st_size > 1000:
            print(f"  {name}: already downloaded")
            continue
        tmp = dest.with_suffix(dest.suffix + ".part")
        urllib.request.urlretrieve(url, tmp, progress(name))
        tmp.rename(dest)
        print()
    print("Done. Next: python -m core.graph --size small   (or --size all)")


if __name__ == "__main__":
    main()
