"""Index the real reel videos the phone plays.

    python flies/reels/media.py

Put videos (mp4, webm, mov, m4v) you have the rights to show in one folder per kind of reel:

    flies/reels/raw/cat/      flies/reels/raw/cooking/   flies/reels/raw/dance/
    flies/reels/raw/fail/     flies/reels/raw/lecture/   flies/reels/raw/ad/

then run this script. It writes flies/reels/raw/reels.json, which the station loads. Whenever the fly's
world serves, say, a 9 s cooking reel, the phone plays the cooking video closest to 9 s. Kinds with no
videos keep the drawn cartoon reels. The brain and task.py don't change, so a trained brain keeps working.

Re-running keeps anything you typed into reels.json by hand for a video:
  "user", "caption", "likes" (number), "comment_count" (number), "comments" (list of strings).

raw/ is git-ignored, and scripts/remote.py skips folders called raw, so the videos never go to the GPU.
"""
import json
import shutil
import subprocess
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
RAW = HERE / "raw"
OUT = RAW / "reels.json"
CATS = ["cat", "cooking", "dance", "fail", "lecture", "ad"]   # must match task.py
EXTS = {".mp4", ".webm", ".mov", ".m4v"}
KEEP = ("user", "caption", "likes", "comment_count", "comments")


def duration(path):
    if not shutil.which("ffprobe"):
        return None
    p = subprocess.run(["ffprobe", "-v", "error", "-show_entries", "format=duration", "-of", "csv=p=0", str(path)],
                       capture_output=True, text=True)
    try:
        return round(float(p.stdout.strip()), 2)
    except ValueError:
        return None


def main():
    for c in CATS:
        (RAW / c).mkdir(parents=True, exist_ok=True)
    data = json.loads(OUT.read_text()) if OUT.exists() else {}
    old = {r["file"]: r for r in data.get("reels", [])}
    is_video = lambda f: f.is_file() and f.suffix.lower() in EXTS
    reels = []
    for cat in CATS:
        for f in sorted(filter(is_video, (RAW / cat).iterdir())):
            rel = f.relative_to(HERE).as_posix()              # e.g. raw/cat/loaf.mp4
            entry = {"file": rel, "category": cat, "seconds": duration(f)}
            entry.update({k: v for k, v in old.get(rel, {}).items() if k in KEEP})
            reels.append(entry)
    stray = [f.relative_to(HERE).as_posix() for f in sorted(RAW.iterdir()) if is_video(f)]
    data["reels"] = reels                                     # other keys (the Instagram bridge's note) stay
    OUT.write_text(json.dumps(data, indent=2, ensure_ascii=False) + "\n")
    counts = {c: sum(r["category"] == c for r in reels) for c in CATS}
    print(f"wrote {OUT.relative_to(HERE.parents[1])}: {len(reels)} videos  " +
          "  ".join(f"{c}={n}" for c, n in counts.items()))
    if stray:
        print("skipped (put them in raw/<kind>/, kinds: " + ", ".join(CATS) + "):\n  " + "\n  ".join(stray))
    if not shutil.which("ffprobe"):
        print("note: ffprobe not found, so durations are unknown; videos are then picked at random per kind")
    return 0


if __name__ == "__main__":
    sys.exit(main())
