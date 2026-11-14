"""Unscripted FlyBrain application environment, version 6. No oracle observations."""
import copy
import json
import os
from pathlib import Path
import numpy as np
from core.task import FlyTask
from flies.jobs.forms import VERSION, encode, generate, Validator
from flies.jobs.jobs import load_jobs
from flies.jobs.errors import BLOCKING, validate_error

ROOT = Path(__file__).resolve().parent


class JobsFly(FlyTask):
    name = 'jobs'
    actions = [f'slot_{i}' for i in range(5)] + ['next_field', 'continue_or_submit', 'toggle_input_mode']
    obs_dim = 64
    mode = 'reward'
    max_steps = 40
    train = {'gamma': .90, 'entropy': .10, 'lr': .001, 'episodes': 32, 'chunk': 128}

    def __init__(self, resume=None, jobs=None, split='train', difficulty=1.0, trace=False):
        self.profile = copy.deepcopy(resume) if resume is not None else json.loads((ROOT / 'profile.json').read_text())
        if not isinstance(self.profile.get('facts'), dict) or not self.profile['facts'] or not all(
                isinstance(v, str) for v in self.profile['facts'].values()):
            raise ValueError('resume facts must be a nonempty string mapping')
        if len(self.profile['facts']) > 5:
            raise ValueError('five resume snippet slots supported')
        self.jobs = jobs if jobs is not None else load_jobs(os.environ.get('JOBS_FILE', ROOT / 'elton_fly_job_links.txt'))
        if not self.jobs:
            raise ValueError('job list is empty')
        self.job = self.jobs[0]
        self.split, self.difficulty, self.trace_enabled = split, difficulty, trace
        self.started = False
        self.traces = []
        self.blocking_error = self.last_error = None

    @classmethod
    def from_resume(cls, path, jobs):
        return cls(json.loads(Path(path).read_text()), jobs)

    def reset(self, rng):
        self.t = 0
        self.total_actions = 0
        self.done = False
        self.completed = 0
        self.blocking_error = self.last_error = None
        self.fields, self._validator, self.layout = generate(self.profile, rng, self.split, self.difficulty)
        self.values = {}
        self.page = self.focus = 0
        self.input_mode = 'type'
        self.repeat_counts = {}
        self.paid = set()
        self.corrected = set()
        self.ever_wrong = set()
        self.paid_pages = set()
        self.mode_paid = set()
        self.errors = self.entry_attempts = self.correct_entries = 0
        self.cumulative_reward = 0.0
        self.traces = []
        self.last_caption = 'An unfamiliar application awaits'
        owned = {s.casefold() for s in self.profile.get('skills', [])}
        required = {s.casefold() for s in self.job.get('skills', [])}
        self.relevance = len(owned & required) / len(required) if required else 0.0
        self.resume_choices = [{'label': key.replace('_', ' '), 'value': value}
                               for key, value in self.profile['facts'].items()]
        rng.shuffle(self.resume_choices)
        self.started = True
        return self.obs()

    @property
    def current(self):
        return self.fields[self.focus]

    def candidates(self):
        if self.current['kind'] == 'select':
            return [{'label': v, 'value': v} for v in self.current['options']]
        return self.resume_choices

    def public_view(self):
        """All browser-visible data; no semantic answer IDs or validation output."""
        return {'fields': copy.deepcopy([f for f in self.fields if f['page'] == self.page]),
                'current': copy.deepcopy(self.current), 'values': dict(self.values),
                'candidates': copy.deepcopy(self.candidates()), 'page': self.page,
                'pages': self.layout['pages'], 'focus': self.focus, 'input_mode': self.input_mode,
                'layout': self.layout['layout'], 'resume': copy.deepcopy(self.profile)}

    def obs(self):
        return self.encode_view(self.public_view())

    @staticmethod
    def encode_view(view):
        """Encode public text/control state, also used for DOM-sourced observations."""
        field, choices = view['current'], view['candidates']
        features = list(encode(field['label']))
        for i in range(5):
            features.extend(encode(choices[i]['label'] + ' ' + choices[i]['value'])
                            if i < len(choices) else [0.] * 8)
        features.extend(encode(view['values'].get(field['id'], '')))
        features.extend([1. if field['kind'] == 'select' else .5 if field['kind'] == 'textarea' else 0.,
                         float(view['input_mode'] == 'select'), float(field['required']),
                         view['page'] / 3, view['pages'] / 3, view['focus'] / 5,
                         float(bool(view['values'].get(field['id']))), 1.])
        return np.asarray(features, dtype=np.float32)

    def mask(self):
        # Purely structural availability, never correctness-dependent.
        return [i < len(self.candidates()) for i in range(5)] + [sum(f['page'] == self.page for f in self.fields) > 1, True, True]

    def report_error(self, code):
        self.last_error = validate_error(code)
        if code in BLOCKING:
            self.blocking_error = code
            self.done = True

    def pause(self):
        """New training chunk for the SAME form. No reset or answer repair."""
        self.t, self.done = 0, False
        return self.obs()

    def step(self, action, rng):
        if self.done:
            raise ValueError('episode ended; reset or resume pending form explicitly')
        before = self.obs().tolist() if self.trace_enabled else None
        public_before = self.public_view() if self.trace_enabled else None
        self.t += 1
        self.total_actions += 1
        event, reward, entered, successful = 'invalid_action', -1.0, None, False
        field = self.current
        if isinstance(action, (int, np.integer)) and 0 <= action < len(self.actions):
            if action < 5 and self.mask()[action]:
                desired_mode = 'select' if field['kind'] == 'select' else 'type'
                if self.input_mode == desired_mode:
                    entered = self.candidates()[action]['value']
                    previous = self.values.get(field['id'])
                    self.values[field['id']] = entered
                    self.entry_attempts += 1
                    correct = self._validator.correct(field, self.values)
                    self.correct_entries += int(correct)
                    successful = correct
                    if entered == previous:
                        event, reward = 'repeated_entry', -.15 if correct else -1.15
                    elif correct:
                        event, reward = 'field_correct', -.03
                        if field['id'] in self.ever_wrong and field['id'] not in self.paid and field['id'] not in self.corrected:
                            reward = 2.0
                            self.corrected.add(field['id'])
                            self.paid.add(field['id'])
                        elif field['id'] not in self.paid:
                            reward = 1.0
                            self.paid.add(field['id'])
                    else:
                        self.ever_wrong.add(field['id'])
                        event, reward = 'field_incorrect', -1.0
            elif action == 5 and self.mask()[5]:
                indices = [i for i, f in enumerate(self.fields) if f['page'] == self.page]
                self.focus = indices[(indices.index(self.focus) + 1) % len(indices)]
                event, reward, successful = 'focus_changed', -.3, True
            elif action == 6:
                page_fields = [f for f in self.fields if f['page'] == self.page]
                if self._validator.accepted(page_fields, self.values):
                    if self.page + 1 < self.layout['pages']:
                        reward = .3 if self.page not in self.paid_pages else -.04
                        self.paid_pages.add(self.page)
                        self.page += 1
                        self.focus = next(i for i, f in enumerate(self.fields) if f['page'] == self.page)
                        event, successful = 'page_advanced', True
                    elif self._validator.accepted(self.fields, self.values):
                        self.completed, self.done = 1, True
                        event, reward, successful = 'application_completed', 15.0 + 5.0 * self.relevance, True
                else:
                    event, reward = 'validation_rejected', -1.0
            elif action == 7:
                self.input_mode = 'select' if self.input_mode == 'type' else 'type'
                desired_mode = 'select' if field['kind'] == 'select' else 'type'
                aligned = self.input_mode == desired_mode
                reward = .1 if aligned and field['id'] not in self.mode_paid else -.8
                if aligned:
                    self.mode_paid.add(field['id'])
                event, successful = 'input_mode_changed', aligned
        if not successful:
            self.errors += 1
        # Penalize revisiting the same operation/state; cheap loops must not
        # dominate learning to finish. This private accounting never selects actions.
        signature = json.dumps([field["id"], int(action), entered, self.page,
                                self.input_mode, self.values], sort_keys=True)
        repeats = self.repeat_counts.get(signature, 0)
        self.repeat_counts[signature] = repeats + 1
        repetition_penalty = min(1.5, .25 * repeats)
        reward -= repetition_penalty
        self.cumulative_reward += reward
        self.done = bool(self.done or self.t >= self.max_steps)
        self.last_caption = event.replace('_', ' ')
        info = {'event': event, 'successful': successful, 'field_id': field['id'],
                'field_label': field['label'], 'entered': entered, 'reward': reward,
                'cumulative_reward': self.cumulative_reward, 'repetition_penalty': repetition_penalty, 'completed': self.completed,
                'error': self.blocking_error or self.last_error, 'relevance': self.relevance,
                'paused': self.done and not self.completed and not self.blocking_error}
        if self.trace_enabled:
            self.traces.append({'observation': before, 'public_observation': public_before, 'action_index': int(action),
                                'action': self.actions[action] if 0 <= action < 8 else 'invalid', **info})
        return self.obs(), float(reward), self.done, info

    def caption(self, action):
        return "Simulation: " + self.last_caption

    def metrics(self):
        return {'reward': self.cumulative_reward, 'completed': self.completed,
                'field_accuracy': self._validator.accuracy(self.fields, self.values),
                'actions': self.total_actions, 'error_rate': self.errors / max(1, self.total_actions)}

    def snapshot(self):
        return {'version': VERSION, 'profile': self.profile, 'job': self.job, 'fields': self.fields,
                'expected': self._validator.expected, 'resume_choices': self.resume_choices, 'layout': self.layout, 'values': self.values,
                'page': self.page, 'focus': self.focus, 'input_mode': self.input_mode,
                'repeat_counts': self.repeat_counts, 'paid': list(self.paid), 'corrected': list(self.corrected), 'ever_wrong': list(self.ever_wrong),
                'paid_pages': list(self.paid_pages), 'mode_paid': list(self.mode_paid), 'errors': self.errors, 'entry_attempts': self.entry_attempts,
                'correct_entries': self.correct_entries, 'cumulative_reward': self.cumulative_reward,
                'total_actions': self.total_actions, 'relevance': self.relevance, 'blocking_error': self.blocking_error, 'last_error': self.last_error}

    def restore(self, state):
        if state.get('version') != VERSION:
            raise ValueError('legacy simulation state cannot be resumed with this environment version')
        if state['profile'] != self.profile:
            raise ValueError('profile changed during saved application')
        for key in ('repeat_counts', 'resume_choices', 'fields', 'layout', 'values', 'page', 'focus', 'input_mode', 'errors',
                    'entry_attempts', 'correct_entries', 'cumulative_reward', 'total_actions', 'relevance',
                    'blocking_error', 'last_error'):
            setattr(self, key, copy.deepcopy(state[key]))
        self._validator = Validator(state['expected'])
        for key in ('paid', 'corrected', 'ever_wrong', 'paid_pages', 'mode_paid'):
            setattr(self, key, set(state[key]))
        self.pause()
