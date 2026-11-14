"""Train every fly in flies/ (the final overnight run).

    python -m core.train_all --size full --minutes 15
    python -m core.train_all --only gym inbox
    python -m core.train_all --skip-done          # only flies without runs/<fly>/result.json

Keeps going if one fly fails and prints a summary table at the end.
"""
import argparse
import json
import traceback

from core.registry import fly_names, run_dir
from core.train import train_fly


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--size", default="full")
    ap.add_argument("--minutes", type=float, default=15)
    ap.add_argument("--device", default="auto")
    ap.add_argument("--only", nargs="*")
    ap.add_argument("--skip-done", action="store_true")
    a = ap.parse_args()
    names = a.only or fly_names()
    summary = {}
    for n in names:
        res_file = run_dir(n) / "result.json"
        if a.skip_done and res_file.exists() and json.loads(res_file.read_text()).get("preset") == a.size:
            summary[n] = "skipped (done)"
            continue
        print(f"\n===== {n} =====")
        try:
            r = train_fly(n, a.size, a.minutes, a.device, log=lambda m, n=n: print(f"[{n}] {m}"))
            summary[n] = (f"trained {r['score_trained']:.2f} | random {r['score_random']:.2f} | "
                          f"lesioned {r['score_lesioned']:.2f} | {r['train_seconds']:.0f}s")
        except Exception as e:
            traceback.print_exc()
            summary[n] = f"FAILED: {e}"
    print("\n===== summary =====")
    for n, s in summary.items():
        print(f"{n:20s} {s}")


if __name__ == "__main__":
    main()
