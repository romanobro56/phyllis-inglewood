"""Compatibility entry point; use --browser with flies.jobs.evaluate."""
import sys
from flies.jobs.evaluate import main

if __name__ == '__main__':
    if '--browser' not in sys.argv:
        sys.argv.append('--browser')
    main()
