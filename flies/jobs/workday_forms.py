"""Workday-inspired generators and independent oracle, never policy code.

Train/validation/test label namespaces and seed namespaces are disjoint. The
public field dictionaries contain opaque IDs, labels and browser constraints,
not profile keys, answer indices or correctness indicators.
"""
import copy
import hashlib
import json
import re
import numpy as np

SPLITS = {'train': 100000000, 'validation': 200000000, 'test': 300000000}
PREFIX = {'train': ('Candidate: ', 'Please enter ', 'Application / '),
          'validation': ('Your details — ', 'Provide your ', 'Applicant / '),
          'test': ('For this position: ', 'Enter the following: ', 'About you / ')}
LABELS = {
    'first_name': ('First name', 'Given name'), 'last_name': ('Last name', 'Family name'),
    'email': ('Email address', 'Contact email'), 'phone': ('Telephone number', 'Phone'),
    'address': ('Street address', 'Home address'), 'location': ('City and country', 'Location'),
    'website': ('Portfolio URL', 'Personal website'),
    'title': ('Job title', 'Position title'), 'organization': ('Organization', 'Employer name'),
    'description': ('Responsibilities', 'Experience description'), 'start_date': ('Start date', 'Date started'),
    'end_date': ('End date', 'Date ended'), 'current': ('Currently employed here?', 'Current position?'),
    'program': ('Education program', 'Program name'), 'institution': ('School or institution', 'Institution name'),
    'degree': ('Degree', 'Qualification'), 'field_of_study': ('Field of study', 'Specialization'),
    'status': ('Program completion status', 'Education status'),
    'work_authorization': ('Authorized to work in this country?', 'Work authorization'),
    'sponsorship': ('Will you need sponsorship?', 'Sponsorship requirement'),
    'availability': ('When can you start?', 'Availability date'),
    'operating_environment': ('Preferred work environment', 'Operating environment'),
    'experience_summary': ('Relevant experience', 'Brief work experience summary'),
}


def label(key, rng, split):
    return str(rng.choice(PREFIX[split])) + str(rng.choice(LABELS[key]))


def generate(profile, rng, split, level):
    if split not in SPLITS or not 1 <= level <= 10:
        raise ValueError('split must be train/validation/test and level must be 1..10')
    nonce = int(rng.integers(1 << 30))
    tenant = f'sim-tenant-{split}-{int(rng.integers(6))}'
    expected, sections = {}, {}
    ordinal = 0

    def field(key, value, required=True, kind=None):
        nonlocal ordinal
        identity = 'el_' + hashlib.sha256(f'{split}:{nonce}:{ordinal}'.encode()).hexdigest()[:12]
        ordinal += 1
        if kind is None:
            kind = 'textarea' if key in ('description', 'experience_summary') else 'text'
            if key=='phone':
                kind='tel'
            elif key == 'email':
                kind = 'email'
            elif key in ('start_date', 'end_date', 'availability'):
                kind = 'date'
            elif key in ('current', 'work_authorization', 'sponsorship'):
                kind = str(rng.choice(('radio', 'select')))
            elif level >= 3 and value and key not in ('description', 'email', 'phone', 'address') and rng.random() < .3:
                kind = 'combobox'
        options = []
        if kind in ('select', 'combobox', 'radio'):
            options = ['Yes', 'No'] if key in ('current', 'work_authorization', 'sponsorship') else list(dict.fromkeys([
                v for v in (value, 'Other', 'Not applicable', 'In progress') if v]))
            rng.shuffle(options)
        expected[identity] = value
        return {'id': identity, 'label': label(key, rng, split), 'kind': kind,
                'required': bool(required), 'options': options,
                'constraint': 'YYYY-MM-DD' if kind == 'date' else 'valid email' if kind == 'email' else '',
                'value': '', 'unavailable': False}

    personal = list(profile['personal'])[:(1 if level == 1 else 3 if level <= 3 else int(rng.integers(3, 8)))]
    if level == 1:
        personal = [str(rng.choice(('first_name', 'last_name', 'email'))) ]
    rng.shuffle(personal)
    sections['personal'] = [field(k, profile['personal'][k], required=(profile['personal'][k] is not None or (level >= 8 and rng.random() < .15))) for k in personal]
    if level >= 6:
        for section in ('experience', 'education'):
            sections[section] = []
            count = int(rng.integers(1, len(profile[section]) + 1))
            for record_index,record in enumerate(profile[section][:count]):
                keys = list(record)
                rng.shuffle(keys)
                entry=[field(k, record[k], required=(record[k] is not None or (level >= 9 and rng.random() < .08))) for k in keys]
                for item in entry:
                    item['group']=f'{section.title()} entry {record_index+1}'
                    item['label']=item['group']+' / '+item['label']
                sections[section].append(entry)
    if level >= 3:
        question_keys = ['operating_environment'] if level < 7 else rng.choice(list(profile['answers']), int(rng.integers(2, 6)), replace=False).tolist()
        sections['questions'] = [field(k, profile['answers'][k], required=(profile['answers'][k] is not None or rng.random() < .25),
                                       kind=('select' if k == 'operating_environment' else None)) for k in question_keys]
    stages = []
    if level >= 5:
        stages.append('resume')
    stages.append('personal')
    if level >= 6:
        stages += ['experience', 'education']
    if 'questions' in sections:
        stages.append('questions')
    if level >= 7 and rng.random() < .5:
        # Split personal details across two real pages instead of cosmetic metadata.
        fields = sections['personal']
        cut = max(1, len(fields) // 2)
        if fields[cut:]:
            sections['contact'] = fields[cut:]
            sections['personal'] = fields[:cut]
            stages.insert(stages.index('personal') + 1, 'contact')
    stages.append('review')
    auth = level >= 4
    if auth:
        position = str(rng.choice(('start', 'after_resume', 'before_review'))) if level >= 9 else 'start'
        where = 0 if position == 'start' else stages.index('review') if position == 'before_review' else min(1, len(stages)-1)
        stages.insert(where, 'auth')
    scenario = {'split': split, 'level': level, 'nonce': nonce, 'tenant': tenant,
            'stages': stages, 'sections': sections, 'expected': expected,
            'layout': str(rng.choice(('stack', 'grid', 'cards'))),
            'auth_start': str(rng.choice(('register', 'login', 'email_first'))),
            'seed_existing': bool(rng.random() < .5), 'seed_duplicate': bool(level >= 9 and rng.random() < .15),
            'verification_required': bool(level >= 9 and rng.random() < .08),
            'parse': str(rng.choice(('none', 'partial', 'accurate', 'corrupt'))) if level >= 5 else 'none',
            'error_prefix': str(rng.choice(('Please check: ', 'Unable to continue: ', 'Action needed: '))) }
    if scenario['seed_duplicate']:
        scenario['seed_existing']=True
    return scenario


class IndependentValidator:
    def __init__(self, expected):
        self.expected = copy.deepcopy(expected)

    def correct(self, field):
        expected = self.expected.get(field['id'])
        if expected is None:
            return not field['value'] and bool(field.get('unavailable'))
        value = field['value']
        if field['kind'] == 'email' and not re.fullmatch(r'[^\s@]+@[^\s@]+\.[^\s@]+', value):
            return False
        if field['kind'] == 'date' and value and not re.fullmatch(r'\d{4}-\d{2}-\d{2}', value):
            return False
        return value == expected

    def errors(self, fields):
        messages = []
        for f in fields:
            if self.expected.get(f['id']) is None:
                if f['required']:
                    messages.append((f['id'], 'Verified information unavailable; manual review required.'))
                elif f['value']:
                    messages.append((f['id'], 'Unverified information; clear or request review.'))
            elif (f['required'] or f['value']) and not self.correct(f):
                message = 'Invalid email format.' if f['kind'] == 'email' and not re.fullmatch(r'[^\s@]+@[^\s@]+\.[^\s@]+', f['value']) else 'Check the supplied value.'
                messages.append((f['id'], message))
        return messages


def fingerprint(scenario):
    # For paired evaluation only, never part of policy observation.
    return hashlib.sha256(json.dumps(scenario, sort_keys=True).encode()).hexdigest()
