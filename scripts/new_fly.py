"""Create a new fly from the template.

    python scripts/new_fly.py karaoke
"""
import re
import shutil
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def main():
    if len(sys.argv) != 2 or not re.fullmatch(r"[a-z][a-z0-9_]{1,30}", sys.argv[1]):
        raise SystemExit("usage: python scripts/new_fly.py <name>   (lowercase letters, digits, _)")
    name = sys.argv[1]
    dest = ROOT / "flies" / name
    if dest.exists():
        raise SystemExit(f"flies/{name} already exists")
    shutil.copytree(ROOT / "flies" / "_template", dest)
    cls = "".join(p.capitalize() for p in name.split("_")) + "Fly"
    for f in dest.iterdir():
        if f.is_file():
            f.write_text(f.read_text().replace("TEMPLATE_NAME", name).replace("TemplateFly", cls))
    print(f"Created flies/{name}/. Next:\n"
          f"  1. edit flies/{name}/task.py and flies/{name}/station.js\n"
          f"  2. python -m core.check {name}\n"
          f"  3. python -m core.train {name} --size tiny --minutes 3   (or submit to the GPU: see docs/COMPUTE.md)")


if __name__ == "__main__":
    main()
