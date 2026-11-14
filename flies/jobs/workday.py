"""Phase 1 simulation. Every transition executes the selected public action.

Private scenario/oracle/state objects are never passed to the encoder or policy.
No network/filesystem I/O occurs in step(), including simulated persistence;
runners flush SimulationState outside the RL hot path.
"""
import copy
import json
from pathlib import Path
import numpy as np
from core.task import FlyTask
from flies.jobs.contract import ACTIONS, SCHEMA, Action, encode, view_digest
from flies.jobs.sim_state import SecretVault, SimulationState
from flies.jobs.workday_forms import generate, IndependentValidator, PREFIX
from flies.jobs.jobs import load_jobs, job_id

ROOT = Path(__file__).resolve().parent


class SimulatedWorkdayEnvironment(FlyTask):
    name = 'jobs'
    actions = ACTIONS
    obs_dim = 64
    mode = 'reward'
    max_steps = 768
    train = {'gamma': .997, 'entropy': .04, 'lr': .0005, 'episodes': 16, 'chunk': 64}
    REWARD_VERSION = 2          # 1: wrong entry -1, correction +1 (runs/jobs/workday_full)
    WRONG_ENTRY = -.25
    CORRECTION_BONUS = .2

    def __init__(self, profile=None, state=None, vault=None, split='train', level=9, jobs=None, trace=False):
        self.profile = copy.deepcopy(profile if profile is not None else json.loads((ROOT / 'verified_profile.json').read_text()))
        self.state = state if state is not None else SimulationState()
        self.vault = vault if vault is not None else SecretVault()
        self.state.bind_vault(self.vault)
        self.jobs = jobs if jobs is not None else load_jobs(ROOT / 'elton_fly_job_links.txt')
        self.split, self.level, self.trace_enabled = split, level, trace
        self.max_steps = 48+16*level if level<=3 else 128 if level<=5 else 768
        self.traces = []
        self.catalogs = {'personal': self.profile['personal'], 'answers': self.profile['answers']}
        for section in ('experience', 'education'):
            for i, record in enumerate(self.profile[section]):
                self.catalogs[f'{section} {i+1}'] = record
        self.catalogs['credentials'] = {f'Credential {c}': None for c in 'ABC'}
        self.documents = {'resume_pdf': {'label': 'Elton_Inglewood_Fly_Resume.pdf', 'path': str(ROOT / 'Elton_Inglewood_Fly_Resume.pdf')},
                          'other_document': {'label': 'Application notes (not a resume)', 'path': str(ROOT / 'README.md')}}

    def reset(self, rng):
        self.scenario = generate(self.profile, rng, self.split, self.level)
        self._validator = IndependentValidator(self.scenario['expected'])
        self.job = self.jobs[int(rng.integers(len(self.jobs)))]
        self.tenant, self.job_identity = self.scenario['tenant'], job_id(self.job)
        self.index, self.t, self.total_actions = 0, 0, 0
        self.done, self.completed, self.task_success = False, 0, 0
        self.status = 'in_progress'
        self.focus, self.mode, self.catalog, self.palette = '', 'browse', None, 0
        self.messages, self.paid, self.repeats, self.valuerefs = [], set(), {}, {}
        self.cumulative_reward, self.errors, self.validation_failures, self.entry_attempts = 0., 0, 0, 0
        self.account_created = self.logged_in = self.uploaded = 0
        self.added = {'experience': 0, 'education': 0}
        self.extra_entries = {'experience': 0, 'education': 0}
        self.reviewed = self.review_ok = False
        self.auth_view = self.scenario['auth_start']
        self.authenticated = False
        self.email = self.profile['personal']['email']
        self.auth_fields = [self._auth_field('auth_email', str(rng.choice(PREFIX[self.split]))+'Email address', 'email'),
                            self._auth_field('auth_password', str(rng.choice(PREFIX[self.split]))+'Password', 'password'),
                            self._auth_field('auth_confirm', str(rng.choice(PREFIX[self.split]))+'Confirm password', 'password')]
        # Each pristine seeded world can contain an existing tenant account.
        # Existing persisted tenants are never replaced by a newly drawn scenario.
        if self.scenario['seed_existing'] and not self.state.account(self.tenant, self.email):
            self.state.register(self.tenant, self.email, self.vault.resolve(self.tenant, 'password_A'))
        if self.scenario['seed_duplicate']:
            key = self.state.application_id(self.tenant, self.email, self.job_identity)
            self.state.applications.setdefault(key, {'simulated': True, 'receipt': 'seeded-prior-submission'})
        self.initial_account_exists = bool(self.state.account(self.tenant,self.email))
        self.upload_name = ''
        self.traces = []
        self.last_caption = 'Simulation: unfamiliar candidate application'
        # Verification workflows are explicitly out of scope; no inbox or token actions.
        if self.scenario['verification_required']:
            self.status = 'skipped_email_verification'
            self.messages = ['This workflow requires email verification and is skipped.']
        return self.obs()

    @staticmethod
    def _auth_field(identity, label, kind):
        return {'id': identity, 'label': label, 'kind': kind, 'required': True,
                'options': [], 'constraint': '8+ characters, upper/lowercase, digit' if kind == 'password' else 'valid email',
                'value': '', 'unavailable': False}

    @property
    def stage(self):
        return self.scenario['stages'][self.index]

    def all_fields(self):
        result = []
        for name, fields in self.scenario['sections'].items():
            if name in ('experience', 'education'):
                result.extend(f for entry in fields[:self.added[name]] for f in entry)
            else:
                result.extend(fields)
        return result

    def current_fields(self):
        if self.stage == 'auth':
            return self.auth_fields[:1] if self.auth_view == 'email_first' else self.auth_fields if self.auth_view == 'register' else self.auth_fields[:2]
        fields = self.scenario['sections'].get(self.stage, [])
        if self.stage in ('experience', 'education'):
            return [f for entry in fields[:self.added[self.stage]] for f in entry]
        return fields

    def _buttons(self):
        # These are visible controls. None is hidden based on correctness.
        if self.done:
            return []
        suffix = {'train': '', 'validation': ' / save', 'test': ' / application'}[self.split]
        buttons = []
        if self.stage == 'auth':
            buttons += [('register', 'Create account'), ('login', 'Sign in'), ('auth_email', 'Continue with email')]
        else:
            if self.stage in ('experience', 'education'):
                buttons += [('add_entry', 'Add another entry'), ('remove_entry', 'Remove last entry')]
            if self.stage == 'review':
                buttons += [('review', 'Review application'), ('submit', 'Submit application')]
                for i, name in enumerate(self.scenario['stages']):
                    if name not in ('review', 'auth'):
                        buttons.append((f'edit_{i}', f'Edit {name}'))
            else:
                buttons.append(('next', 'Save and continue'))
            if self.index > 0:
                buttons.append(('back', 'Previous page'))
        return [{'id': 'button_' + k, 'label': v + suffix, 'kind': 'button', 'value': '', 'required': False,
                 'constraint': '', 'options': []} for k, v in buttons]

    def elements(self):
        elements = copy.deepcopy(self.current_fields())
        if self.stage == 'resume':
            elements.append({'id': 'resume_upload', 'label': 'Upload resume or CV', 'kind': 'file',
                             'constraint': 'PDF resume', 'required': True, 'options': [],
                             'value': self.upload_name})
        for f in elements:
            f['protected'] = self.valuerefs.get(f['id'],'').startswith('password_')
            if (f['kind'] == 'password' or f['protected']) and f['value']:
                f['value'] = '[protected]'
        return elements + self._buttons()

    def _raw_actions(self):
        if self.done:
            return []
        if self.status == 'skipped_email_verification':
            return [Action('review', 'Skip verification-required workflow')]
        if self.mode == 'browse':
            result = []
            for e in self.elements():
                kind = 'click' if e['kind'] == 'button' else 'focus'
                result.append(Action(kind, e['label'], e['id']))
            result.append(Action('review', 'Request manual review / information unavailable'))
            return result
        focus = next((e for e in self.elements() if e['id'] == self.focus), None)
        if focus is None:
            raise ValueError('focused element is no longer visible')
        result = []
        if focus['kind'] == 'file':
            result = [Action('upload', d['label'], self.focus, ref=k) for k, d in self.documents.items()]
        elif focus['kind'] in ('select', 'combobox', 'radio'):
            search = getattr(self, 'search_text', '')
            options = [o for o in focus['options'] if search.lower() in o.lower()]
            result = [Action('choose', o, self.focus, value=o) for o in options]
            if focus['kind'] == 'combobox':
                # Search queries are arbitrary literal profile snippets, not inferred answers.
                result += [Action('search', 'Search for: ' + str(v), self.focus, value=str(v))
                           for v in self.profile['personal'].values() if v]
                result.append(Action('search', 'Clear option search', self.focus))
        elif self.level <= 3:
            result = [Action('enter', k.replace('_', ' '), self.focus, ref='personal|' + k, value=v)
                      for k, v in self.profile['personal'].items() if v is not None]
        elif self.catalog is None:
            result = [Action('catalog', name, ref=name) for name in self.catalogs]
        else:
            for key, value in self.catalogs[self.catalog].items():
                if self.catalog == 'credentials':
                    code = key[-1]
                    result.append(Action('enter', key, self.focus, ref='password_' + code, secret=True))
                elif value is not None:
                    result.append(Action('enter', key.replace('_', ' '), self.focus,
                                         ref=self.catalog + '|' + key, value=value))
            result.append(Action('catalog', 'All profile sections'))
        result += [Action('clear', 'Clear field', self.focus),
                   Action('review', 'Information unavailable / request review', self.focus),
                   Action('cancel', 'Back to application controls')]
        return result

    def available_actions(self):
        raw = self._raw_actions()
        if not raw:
            return []
        # Paginated action palette is explicit UI, shared with the DOM adapter.
        # No sorting/ranking by answer correctness or field meaning.
        start = (self.palette * 7) % (max(1, ((len(raw)+6)//7)) * 7)
        page = raw[start:start+7]
        if len(raw) > 7:
            page += [Action('more', 'More choices')]
        return page

    def public_view(self):
        actions = [a.public() for a in self.available_actions()]
        actions += [None] * (8-len(actions))
        profile = {k: {key: value for key, value in group.items()} for k, group in self.catalogs.items() if k != 'credentials'}
        return {'schema': SCHEMA, 'simulation': True, 'title': 'Elton candidate application — LOCAL SIMULATION',
                'tenant': self.tenant, 'job_url': self.job['url'], 'stage': self.stage,
                'steps': list(self.scenario['stages']), 'page': self.index,
                'elements': self.elements(), 'focus': self.focus, 'mode': self.mode,
                'palette': self.palette, 'messages': self._messages(), 'actions': actions,
                'profile': profile, 'profile_provenance': {'simulation_only_fields': self.profile.get('simulation_only_fields',[]),
                     'field_sources': self.profile.get('field_sources',{})}, 'section_entry_count': {k:self.added[k]+self.extra_entries[k] for k in self.added}, 'documents': [{'ref': k, 'label': d['label']} for k, d in self.documents.items()],
                'layout': self.scenario['layout'], 'status': self.status,
                'summary': copy.deepcopy(self.all_fields()) if self.stage == 'review' else []}

    def _messages(self):
        # A passed review is visible on the page, like a failed one. Without this the view before and
        # after a successful review was identical, so Submit (+20 reviewed / -10 unreviewed) was a blind
        # gamble for a memoryless policy. Derived from state, so any later edit removes it.
        ready = self.stage == 'review' and self.reviewed and getattr(self, 'review_ok', False)
        return list(self.messages) + (['Review complete: ready to submit.'] if ready else [])

    def obs(self):
        return encode(self.public_view())

    def mask(self):
        n = len(self.available_actions())
        # A terminal adapter can still be inspected by core.check; no decision may execute it.
        return [i < n for i in range(8)] if n else [True] + [False]*7

    def _field(self, identity):
        return next((f for f in self.current_fields() if f['id'] == identity), None)

    def _password(self, identity):
        ref = self.valuerefs.get(identity, '')
        # C is deliberately too short, B valid but different, A the seeded credential.
        if ref.startswith('password_'):
            return self.vault.resolve(self.tenant, 'password_weak' if ref == 'password_C' else ref)
        return next((f['value'] for f in self.auth_fields if f['id'] == identity), '')

    def _once(self, key, amount):
        if key in self.paid:
            return 0.
        self.paid.add(key)
        return float(amount)

    def _move(self, index):
        self.index = index
        self.mode, self.focus, self.catalog, self.palette = 'browse', '', None, 0
        self.search_text = ''
        if self.stage == 'auth' and self.authenticated:
            # No action is selected here; authenticated sessions do not re-login.
            self.index = min(len(self.scenario['stages'])-1, self.index+1)

    def _validation(self, whole=False):
        fields = self.all_fields() if whole else self.current_fields()
        errors = self._validator.errors(fields)
        for name in ('experience', 'education'):
            if name in self.scenario['sections'] and (whole or self.stage == name):
                if self.added[name] != len(self.scenario['sections'][name]) or self.extra_entries[name]:
                    errors.append((name, 'Complete the resume-backed entries; remove extra empty entries.'))
        if (whole or self.stage == 'resume') and 'resume' in self.scenario['stages'] and not self.uploaded:
            errors.append(('resume_upload', 'A PDF resume is required.'))
        if whole and 'auth' in self.scenario['stages'] and not self.authenticated:
            errors.append(('auth', 'Sign in before submitting.'))
        return errors

    def _feedback(self, errors):
        self.validation_failures += 1
        self.messages = [self.scenario['error_prefix'] + message for _, message in errors]
        return {'event': 'validation_error', 'reward': -1., 'successful': False}

    def _auth(self, button):
        if button == 'auth_email':
            if self.auth_view != 'email_first':
                self.auth_view = 'email_first'
                return {'event': 'auth_view_changed', 'reward': -.05, 'successful': True}
            email = self.auth_fields[0]['value']
            self.auth_view = 'login' if self.state.account(self.tenant, email) else 'register'
            return {'event': 'account_lookup', 'reward': -.05, 'successful': True}
        if button != self.auth_view:
            self.auth_view = button
            return {'event': 'auth_view_changed', 'reward': -.05, 'successful': True}
        email = self.auth_fields[0]['value']
        if email != self.email:
            return self._feedback([('auth_email', 'Use the verified applicant email address.')])
        if button == 'register':
            if self.state.account(self.tenant, email):
                return self._feedback([('auth_email', 'An account already exists. Sign in instead.')])
            password, confirmation = self._password('auth_password'), self._password('auth_confirm')
            if len(password) < 8 or not any(c.isupper() for c in password) or not any(c.islower() for c in password) or not any(c.isdigit() for c in password):
                return self._feedback([('auth_password', 'Password needs 8+ characters, upper/lowercase and a digit.')])
            if password != confirmation:
                return self._feedback([('auth_confirm', 'Passwords do not match.')])
            self.state.register(self.tenant, email, password)
            self.account_created = 1
            reward = self._once('register', 5.)
        else:
            if not self.state.login(self.tenant, email, self._password('auth_password')):
                return self._feedback([('auth_password', 'Incorrect credentials or account not found.')])
            self.logged_in = 1
            reward = self._once('login', 3.)
        self.authenticated = True
        self._move(min(self.index + 1, len(self.scenario['stages'])-1))
        if self.state.submitted(self.tenant, email, self.job_identity):
            self.messages = ['This application has already been submitted.']
        return {'event': 'account_created' if button == 'register' else 'login_success', 'reward': reward, 'successful': True}

    def _click(self, target):
        button = target.removeprefix('button_')
        if self.stage == 'auth':
            return self._auth(button)
        if button in ('add_entry', 'remove_entry'):
            section = self.stage
            self.reviewed=False
            if button == 'add_entry':
                if self.added[section] < len(self.scenario['sections'][section]):
                    self.added[section] += 1
                else:
                    self.extra_entries[section] += 1
            elif self.extra_entries[section]:
                self.extra_entries[section] -= 1
            elif self.added[section]:
                self.added[section] -= 1
                for field in self.scenario['sections'][section][self.added[section]]:
                    field['value']=''
                    field['unavailable']=False
                    self.valuerefs.pop(field['id'],None)
            else:
                return {'event': 'invalid_action', 'reward': -1., 'successful': False}
            return {'event': button, 'reward': -.1, 'successful': True}
        if button == 'back':
            self._move(max(0, self.index-1))
            return {'event': 'page_back', 'reward': -.1, 'successful': True}
        if button.startswith('edit_'):
            self._move(int(button[5:]))
            return {'event': 'edit_page', 'reward': -.1, 'successful': True}
        if button == 'review':
            self.reviewed = True
            errors = self._validation(whole=True)
            self.review_ok = not errors
            if errors:
                return self._feedback(errors)
            self.messages = []          # the confirmation replaces stale error banners (see public_view)
            return {'event': 'review_passed', 'reward': self._once('review', 1.), 'successful': True}
        if button == 'submit':
            if self.state.submitted(self.tenant, self.email, self.job_identity):
                self.messages = ['Duplicate submission: this job was already submitted.']
                return {'event': 'duplicate_submission', 'reward': -10., 'successful': False}
            errors = self._validation(whole=True)
            if not self.reviewed:
                errors.append(('review', 'Review the completed application before submitting.'))
            if errors:
                result = self._feedback(errors)
                result['reward'], result['event'] = -10., 'submission_rejected'
                return result
            key = self.state.application_id(self.tenant, self.email, self.job_identity)
            self.state.applications[key] = {'simulated': True, 'receipt': f'SIM-{self.scenario["nonce"]}',
                                            'fields': copy.deepcopy(self.all_fields())}
            self.completed = self.task_success = 1
            self.done, self.status = True, 'simulated_completed'
            return {'event': 'simulated_submission', 'reward': 20., 'successful': True}
        if button == 'next':
            errors = self._validation()
            if errors:
                return self._feedback(errors)
            stage = self.stage
            self._move(min(self.index+1, len(self.scenario['stages'])-1))
            return {'event': 'page_advanced', 'reward': self._once('stage:'+stage, 1.), 'successful': True}
        return {'event': 'invalid_action', 'reward': -1., 'successful': False}

    def execute(self, action):
        """Execute a concrete selected action; no policy or corrective fallback."""
        if self.status == 'skipped_email_verification':
            self.done = True
            return {'event': 'skipped_email_verification', 'reward': 0., 'successful': False}
        if action not in self.available_actions():
            return {'event': 'invalid_action', 'reward': -1., 'successful': False}
        if action.kind == 'more':
            self.palette = (self.palette + 1) % max(1, (len(self._raw_actions())+6)//7)
            return {'event': 'choices_paged', 'reward': -.1, 'successful': True}
        if action.kind == 'focus':
            self.focus, self.mode, self.palette, self.catalog = action.target, 'edit', 0, None
            self.search_text = ''
            return {'event': 'element_selected', 'reward': -.05, 'successful': True}
        if action.kind == 'cancel':
            self.focus, self.mode, self.catalog, self.palette = '', 'browse', None, 0
            return {'event': 'editor_closed', 'reward': -.05, 'successful': True}
        if action.kind == 'catalog':
            self.catalog, self.palette = action.ref or None, 0
            return {'event': 'profile_section_selected', 'reward': -.05, 'successful': True}
        if action.kind == 'click':
            return self._click(action.target)
        if action.kind == 'search':
            self.search_text, self.palette = action.value, 0
            return {'event': 'options_searched', 'reward': -.05, 'successful': True}
        if action.kind == 'review':
            field = self._field(action.target)
            unknown_field = bool(field and field['id'] in self._validator.expected and self._validator.expected[field['id']] is None)
            if unknown_field:
                field['value'], field['unavailable'] = '', True
                self.valuerefs.pop(field['id'], None)
                if not field['required']:
                    return {'event': 'optional_information_unavailable', 'reward': self._once('unknown:'+field['id'], .2), 'successful': True}
            # A manual review is a separate safe outcome, never a completed application.
            duplicate = self.state.submitted(self.tenant, self.email, self.job_identity) and self.authenticated
            unknown_required = any(f['required'] and self._validator.expected.get(f['id']) is None for f in self.all_fields())
            if duplicate or (unknown_field and field['required']) or unknown_required:
                self.done, self.status, self.task_success = True, 'manual_review_duplicate' if duplicate else 'manual_review_missing_information', 1
                return {'event': self.status, 'reward': 1., 'successful': True}
            self.messages = ['No missing required information is identified; continue or correct the application.']
            return {'event': 'manual_review_unnecessary', 'reward': -1., 'successful': False}
        if action.kind == 'upload':
            self.reviewed=False
            self.upload_name = Path(self.documents[action.ref]['path']).name
            if action.ref != 'resume_pdf':
                self.uploaded = 0
                self.messages = ['Wrong document: upload the applicant PDF resume.']
                return {'event': 'upload_rejected', 'reward': -1., 'successful': False}
            self.uploaded = 1
            if 'upload' not in self.paid:
                # Only the explicit upload can trigger parsing. Corrupt/partial output
                # is a deterministic environment response, never policy answer repair.
                for field in self.all_fields():
                    expected = self._validator.expected.get(field['id'])
                    if expected is not None and self.scenario['parse'] in ('partial', 'accurate', 'corrupt'):
                        if self.scenario['parse'] == 'accurate' or int(field['id'][-1], 16) % 2:
                            field['value'] = expected if self.scenario['parse'] != 'corrupt' else 'Unrecognized resume text'
            self.messages = ['Resume upload successful. Check the extracted information.']
            return {'event': 'resume_uploaded', 'reward': self._once('upload', 3.), 'successful': True}
        field = self._field(action.target)
        if field is None:
            return {'event': 'invalid_action', 'reward': -1., 'successful': False}
        if action.kind == 'clear':
            self.reviewed=False
            field['value'], field['unavailable'] = '', False
            self.valuerefs.pop(field['id'], None)
            return {'event': 'field_cleared', 'reward': -.1, 'successful': True}
        if action.kind in ('enter', 'choose'):
            previous = field['value']
            value = '[protected]' if action.secret else action.value
            if field['kind']=='date':
                import re
                value = value if re.fullmatch(r'\d{4}-\d{2}-\d{2}',value) else ''
            field['value'], field['unavailable'] = value, False
            if value:self.valuerefs[field['id']] = action.ref
            else:self.valuerefs.pop(field['id'],None)
            self.entry_attempts += 1
            self.reviewed = False
            # Authentication is validated at register/login, never by field-answer lookup.
            if self.stage == 'auth':
                return {'event': 'auth_value_entered', 'reward': -.05 if previous != field['value'] else -1., 'successful': True}
            correct = self._validator.correct(field)
            # A wrong entry is a cheap, recoverable attempt (-0.25), not an invalid action (-1):
            # at -1, trying a value was worth less than idling in the editor, so policies stopped
            # entering values at all (workday_full traces: correct entries fell 3.3 -> 2.5/episode).
            reward = self.WRONG_ENTRY
            if correct:
                reward = self._once('field:'+field['id'], 1.)
                if previous and previous != field['value']:
                    # Correction bonus is paid once and stays below the wrong-entry cost, so
                    # wrong-then-right (-.25 + 1 + .2 - 2 x .02) never outscores immediate correctness.
                    reward += self._once('correction:'+field['id'], self.CORRECTION_BONUS)
            return {'event': 'field_correct' if correct else 'field_incorrect', 'reward': reward, 'successful': correct}
        return {'event': 'invalid_action', 'reward': -1., 'successful': False}

    def step(self, action, rng):
        if self.done:
            raise ValueError('episode is terminal')
        public = self.public_view()
        available = self.available_actions()
        selected = available[int(action)] if isinstance(action, (int, np.integer)) and 0 <= action < len(available) else None
        result = self.execute(selected) if selected else {'event': 'invalid_action', 'reward': -1., 'successful': False}
        self.t += 1
        self.total_actions += 1
        # Repeated operations cannot farm intermediate rewards.
        self.errors += int(not result['successful'])
        signature = view_digest(public) + ':' + str(action)
        repeat = self.repeats.get(signature, 0)
        self.repeats[signature] = repeat+1
        penalty = min(2., repeat * .25)
        reward = float(result.pop('reward')) - .02 - penalty
        if self.t >= self.max_steps and not self.done:
            self.done, self.status = True, 'timed_out'
            reward -= 5.
        self.cumulative_reward += reward
        self.last_caption = 'Simulation: ' + result['event'].replace('_', ' ')
        info = {**result, 'reward': reward, 'cumulative_reward': self.cumulative_reward,
                'completed': self.completed, 'status': self.status, 'repetition_penalty': penalty}
        if self.trace_enabled:
            self.traces.append({'observation': encode(public).tolist(), 'public_observation': public,
                                'action_index': int(action), 'selected_action': selected.public() if selected else None, **info})
        return self.obs(), reward, self.done, info

    def caption(self, action):
        return self.last_caption

    def metrics(self):
        fields = self.all_fields()
        all_generated = []
        for name, group in self.scenario['sections'].items():
            all_generated.extend([f for entry in group for f in entry] if name in ('experience', 'education') else group)
        known = [f for f in all_generated if self._validator.expected.get(f['id']) is not None]
        return {'reward': self.cumulative_reward, 'completed': self.completed, 'task_success': self.task_success,
                'field_accuracy': sum(self._validator.correct(f) for f in known)/max(1, len(known)),
                'account_creation_success': self.account_created, 'login_success': self.logged_in,
                'account_creation_opportunity': int('auth' in self.scenario['stages'] and not self.initial_account_exists),
                'login_opportunity': int('auth' in self.scenario['stages'] and self.initial_account_exists),
                'resume_upload_opportunity': int('resume' in self.scenario['stages']),
                'resume_upload_success': self.uploaded, 'actions': self.total_actions,
                'error_rate': self.errors/max(1, self.total_actions),
                'validation_error_rate': self.validation_failures/max(1, self.total_actions),
                'failed_application': int(not self.completed and self.status not in ('manual_review_missing_information', 'manual_review_duplicate', 'skipped_email_verification')),
                'skipped_email_verification': int(self.status == 'skipped_email_verification')}

    def snapshot(self):
        # Only secret references/password hashes, never raw credentials.
        keys = ('max_steps', 'scenario', 'job', 'tenant', 'job_identity', 'index', 't', 'total_actions', 'done', 'completed', 'task_success',
                'status', 'focus', 'mode', 'catalog', 'palette', 'messages', 'repeats', 'valuerefs', 'cumulative_reward',
                'errors', 'validation_failures', 'entry_attempts', 'account_created', 'logged_in', 'uploaded', 'added',
                'extra_entries', 'initial_account_exists', 'upload_name', 'reviewed', 'review_ok', 'auth_view', 'authenticated', 'email', 'auth_fields', 'last_caption')
        result = {k: copy.deepcopy(getattr(self, k)) for k in keys}
        result.update(schema=SCHEMA, level=self.level, split=self.split, paid=list(self.paid), search_text=getattr(self, 'search_text', ''))
        return result

    def restore(self, snapshot, resume_chunk=False):
        if snapshot['schema'] != SCHEMA:
            raise ValueError('incompatible application snapshot')
        for key, value in snapshot.items():
            if key != 'schema':
                setattr(self, key, copy.deepcopy(value))
        self.paid = set(self.paid)
        self._validator = IndependentValidator(self.scenario['expected'])
        if resume_chunk and self.status == 'timed_out':
            self.t, self.done, self.status = 0, False, 'in_progress'
        self.traces = []
