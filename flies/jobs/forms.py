"""Public form generation and independent reward oracle.

The policy sees labels/options and resume snippets, never expected values,
semantic field keys, answer indices, or validator accuracy/progress.
"""
import hashlib
import re
import numpy as np

VERSION = 6
LABELS = {
    'name': ('Applicant name', 'Full name', 'Your name', 'Legal name'),
    'email': ('Contact email', 'Email address', 'Your email', 'Applicant email'),
    'experience': ('Work experience', 'Professional experience', 'Experience summary', 'Relevant experience'),
    'education': ('Education history', 'Education summary', 'Your education', 'Highest education'),
    'operating_environment': ('Operating environment', 'Preferred environment', 'Work environment', 'Suitable environment'),
}
DISTRACTORS = {
    'name': ('Morgan Reed', 'Alex Smith', 'Taylor Lane', 'Jordan Quinn'),
    'email': ('alex@example.invalid', 'team@example.invalid', 'morgan@example.invalid', 'jobs@example.invalid'),
    'experience': ('Software engineering', 'Retail cashier', 'Accounting', 'Customer service'),
    'education': ('High school diploma', 'Bachelor of Arts', 'Master of Science', 'Doctorate'),
    'operating_environment': ('Indoor', 'Outdoor', 'Remote only', 'Office only'),
}


def encode(text, dimensions=8):
    """Fixed, nonlearned lexical sensor. Encodes one string independently.

No expected answer, current field, lookup table, or comparison enters this
function. This is a small vocabulary sensor, not an LLM/text understanding model.
"""
    tokens = re.findall(r'[a-z0-9@.]+', text.lower().replace('_', ' ')) or ['']
    vectors = [np.frombuffer(hashlib.sha256(word.encode()).digest()[:dimensions], dtype=np.uint8)
               .astype(np.float32) / 255 for word in tokens]
    return np.mean(vectors, axis=0)


class Validator:
    """Only environment reward calculation and final validation use this object."""
    def __init__(self, expected):
        self.expected = dict(expected)

    def correct(self, field, values):
        value = self.expected.get(field['id'])
        return bool(value) and values.get(field['id']) == value

    def accepted(self, fields, values):
        return all(self.correct(f, values) for f in fields
                   if f['required'] or bool(values.get(f['id'])))

    def accuracy(self, fields, values):
        return sum(self.correct(f, values) for f in fields) / len(fields)


def generate(profile, rng, split='train', difficulty=1.0):
    count = int(rng.integers(3, 6)) if difficulty >= 1 else 1 if difficulty < .7 else int(rng.integers(2, 4))
    keys = rng.choice(list(LABELS), count, replace=False).tolist()
    fields, expected = [], {}
    pages = int(rng.integers(1, min(count, 3) + 1))
    # Different layout/label combinations at evaluation; seeds are disjoint too.
    prefix = str(rng.choice(('Please provide ', 'Applicant details: ', 'Enter '))) if split == 'test' else str(rng.choice(('', 'Application: ', 'Candidate ')))
    for i, key in enumerate(keys):
        identity = f'field_{i}'
        value = profile['facts'].get(key, '')
        expected[identity] = value
        kind = 'text' if difficulty < .3 else str(rng.choice(('text', 'textarea', 'select'), p=(.45, .20, .35)))
        choices = [value] + list(DISTRACTORS[key])
        rng.shuffle(choices)
        fields.append({'id': identity, 'label': prefix + str(rng.choice(LABELS[key])),
                       'kind': kind, 'required': bool(i == 0 or rng.random() < .8),
                       'page': min(pages - 1, i * pages // count),
                       'options': choices if kind == 'select' else [],
                       'x': float(rng.random()), 'y': float(rng.random())})
    return fields, Validator(expected), {
        'layout': str(rng.choice(('stack', 'grid', 'cards'))),
        'error_message': str(rng.choice(('Please check this page.', 'Application rejected; review entries.', 'Some entries need attention.'))),
        'split': split, 'pages': pages,
    }
