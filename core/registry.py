"""Finds every fly in flies/<name>/task.py. Folders starting with '_' are ignored."""
import importlib.util
import inspect
from pathlib import Path

from core.task import FlyTask

ROOT = Path(__file__).resolve().parents[1]
FLIES = ROOT / "flies"


def fly_names():
    return sorted(p.name for p in FLIES.iterdir()
                  if p.is_dir() and not p.name.startswith(("_", ".")) and (p / "task.py").exists())


def load_task(name):
    path = FLIES / name / "task.py"
    if not path.exists():
        raise SystemExit(f"No fly called '{name}'. Known: {', '.join(fly_names())}")
    spec = importlib.util.spec_from_file_location(f"flies.{name}.task", path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    classes = [c for _, c in inspect.getmembers(mod, inspect.isclass)
               if issubclass(c, FlyTask) and c is not FlyTask and c.__module__ == mod.__name__]
    if len(classes) != 1:
        raise SystemExit(f"flies/{name}/task.py must define exactly one FlyTask subclass (found {len(classes)})")
    cls = classes[0]
    if cls.name != name:
        raise SystemExit(f"flies/{name}/task.py: name = '{cls.name}' must match the folder name '{name}'")
    return cls


def run_dir(name):
    return ROOT / "runs" / name
