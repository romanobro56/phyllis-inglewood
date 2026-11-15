"""Only test fixtures choose known digits. Production decisions always use FlyBrain."""
import json
from pathlib import Path
import numpy as np
from flies.pi.task import PiFly
from flies.pi.reference import PI_DIGITS


def make():
    env = PiFly(); rng = np.random.default_rng(3)
    env.reset(rng)
    rows = []
    # All ten digits, repeated wrong selections, long strip overflow, then reset.
    for digit in list(range(10)) + [9,9] + [int(d) for d in PI_DIGITS[1:12]]:
        for action in [digit // 5, digit % 5]:
            env.step(action, rng)
            rows.append(dict(action=env.actions[action], caption=env.caption(action)))
    env.reset(rng)
    for action in [0, 3, 0, 0]:
        env.step(action, rng)
        rows.append(dict(action=env.actions[action], caption=env.caption(action)))
    return rows


if __name__ == '__main__':
    Path(__file__).with_name('fixtures.json').write_text(json.dumps(make(), indent=2, ensure_ascii=False) + '\n')
