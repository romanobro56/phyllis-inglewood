"""Launch the unchanged host with the feature-local checkpoint, if present."""
from pathlib import Path
from server import serve as host


def main():
    shared_run_dir = host.run_dir
    local = Path(__file__).resolve().parent / 'artifacts'
    host.run_dir = lambda name: local if name == 'pi' else shared_run_dir(name)
    host.main()


if __name__ == '__main__':
    main()
