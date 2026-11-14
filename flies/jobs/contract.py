"""Browser-visible contract v1. No validator, policy, or label-to-answer rules.

An adapter exposes a public view and executes ONE concrete Action. Eight dynamic
slots page through those actions. Adapters may implement DOM focus/typing/clicks,
never choose which field or profile value the policy should select.
"""
from dataclasses import dataclass, asdict
import hashlib
import json
import re
from typing import Protocol
import numpy as np

SCHEMA = 'elton-workday-public-v1'
ACTIONS = [f'workday_slot_{i}' for i in range(8)]
KINDS = ('focus', 'enter', 'clear', 'choose', 'click', 'upload', 'review', 'more', 'catalog', 'cancel', 'search')


@dataclass(frozen=True)
class Action:
    kind: str
    label: str
    target: str = ''
    ref: str = ''
    value: str = ''
    secret: bool = False

    def public(self):
        d = asdict(self)
        # Passwords are opaque handles, including in sensors.
        if self.secret:
            d['value'] = '[protected]'
        return d


class ApplicationEnvironment(Protocol):
    def public_view(self) -> dict: ...
    def available_actions(self) -> list[Action]: ...
    def execute(self, action: Action) -> dict: ...
    def obs(self) -> np.ndarray: ...
    def mask(self) -> list[bool]: ...
    def step(self, action: int, rng): ...


def lexical(text, n):
    """Independent token/character sensor, never compares field and candidate.

Finite lexical compression is an explicit limitation, not semantic understanding.
Public word overlap must be learned by FlyBrain, not computed by this function.
"""
    words = re.findall(r'[a-z0-9]+', str(text).lower())
    if not words:
        return np.zeros(n, np.float32)
    vectors = [np.frombuffer(hashlib.blake2b(w.encode(), digest_size=n).digest(), np.uint8) / 255. for w in words]
    return np.mean(vectors, axis=0).astype(np.float32)


def encode(view):
    """64 floats: eight public slot descriptions x 7, eight public context values."""
    features = []
    for action in view['actions']:
        if action is None:
            features.extend([0.] * 7)
            continue
        features.append((KINDS.index(action['kind']) + 1) / len(KINDS))
        features.extend(lexical(action['label'], 3))
        features.extend(lexical(action['value'], 2))
        features.append(float(bool(action.get('secret'))))
    focus = next((e for e in view['elements'] if e['id'] == view['focus']), {})
    features.extend(lexical(focus.get('label', view['title']), 2))
    features.extend(lexical(focus.get('value', ''), 2))
    features.extend(lexical(view['stage'], 1))
    features.extend(lexical(' '.join(view['messages']), 1))
    features.append(float(view['mode'] == 'edit'))
    features.extend(lexical(focus.get('constraint', ''), 1))
    result = np.asarray(features, np.float32)
    if result.shape != (64,) or not np.isfinite(result).all():
        raise ValueError('invalid public observation')
    return result


def view_digest(view):
    return hashlib.sha256(json.dumps(view, sort_keys=True).encode()).hexdigest()
