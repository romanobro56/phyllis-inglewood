"""Durable, single-worker practice queue. SQLite transactions prevent lost saves.

Only the runner writes history, never the training step. Practice completions are
isolated from future real submission records by the mode column.
"""
import json
import sqlite3
from pathlib import Path
from flies.jobs.jobs import job_id


class History:
    def __init__(self, path, email, mode='practice_v6'):
        self.email = email.strip().casefold()
        self.mode = mode
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        self.db = sqlite3.connect(path)
        self.db.execute('''CREATE TABLE IF NOT EXISTS applications (
            email TEXT, mode TEXT, job_id TEXT, url TEXT, status TEXT,
            error TEXT, state TEXT, updated TEXT DEFAULT CURRENT_TIMESTAMP,
            PRIMARY KEY(email, mode, job_id))''')
        self.db.execute("""CREATE TABLE IF NOT EXISTS events (
            id INTEGER PRIMARY KEY AUTOINCREMENT, email TEXT, mode TEXT, job_id TEXT,
            url TEXT, created TEXT DEFAULT CURRENT_TIMESTAMP, detail TEXT)""")
        self.db.commit()

    def close(self):
        self.db.close()

    def get(self, job):
        row = self.db.execute('SELECT status, error, state FROM applications WHERE email=? AND mode=? AND job_id=?',
                              (self.email, self.mode, job_id(job))).fetchone()
        return None if row is None else {'status': row[0], 'error': row[1], 'state': json.loads(row[2])}

    def details(self):
        rows = self.db.execute('SELECT job_id,url,status,error,updated,state FROM applications WHERE email=? AND mode=? ORDER BY updated,job_id',
                               (self.email, self.mode))
        records = []
        for identity, url, status, error, updated, raw in rows:
            state = json.loads(raw)
            fields = state.get('fields') or []
            values = state.get('values', {})
            expected = state.get('expected', {})
            if state.get('version', 0) >= 3:
                checks = {f['id']: bool(expected.get(f['id'])) and values.get(f['id']) == expected.get(f['id']) for f in fields}
            else:
                facts = state.get('profile', {}).get('facts', {})
                checks = {f['key']: bool(facts.get(f['key'])) and values.get(f['key']) == facts.get(f['key']) for f in fields}
            events = [dict(id=e[0], timestamp=e[1], **json.loads(e[2])) for e in self.db.execute(
                'SELECT id,created,detail FROM events WHERE email=? AND mode=? AND job_id=? ORDER BY id',
                (self.email, self.mode, identity))]
            records.append({'url': url, 'mode': self.mode, 'status': status, 'error': error,
                            'updated': updated, 'values': values,
                            'field_checks': checks, 'fields': fields,
                            'events': events,
                            'log_note': 'Action logs began after this application; only saved values are available.' if not events else 'Events recorded from logging-enabled runs; earlier actions may be unavailable.'})
        return records

    def records(self):
        return [dict(zip(('url', 'status', 'error', 'updated'), row)) for row in
                self.db.execute('SELECT url,status,error,updated FROM applications WHERE email=? AND mode=? ORDER BY updated,job_id',
                                (self.email, self.mode))]

    def next_job(self, jobs):
        # Restore pending work even if the file was reordered or the link removed.
        row = self.db.execute('SELECT state FROM applications WHERE email=? AND mode=? AND status=? ORDER BY updated,job_id LIMIT 1',
                              (self.email, self.mode, 'in_progress')).fetchone()
        if row:
            return json.loads(row[0])['job']
        for job in jobs:
            if self.get(job) is None:
                return job
        return None

    def save(self, env, event=None):
        if self.mode == 'live':
            raise ValueError('simulation cannot write live application history; a verified live adapter is required')
        state = env.snapshot()
        status = 'completed' if env.completed else 'blocked' if env.blocking_error and env.done else 'in_progress'
        with self.db:
            previous = self.get(env.job)
            if previous and previous['status'] in ('completed', 'blocked'):
                raise ValueError('refusing to overwrite terminal job history')
            self.db.execute('''INSERT INTO applications(email,mode,job_id,url,status,error,state)
                VALUES(?,?,?,?,?,?,?) ON CONFLICT(email,mode,job_id) DO UPDATE SET
                status=excluded.status,error=excluded.error,state=excluded.state,updated=CURRENT_TIMESTAMP''',
                (self.email, self.mode, job_id(env.job), env.job.get('url', ''), status,
                 env.blocking_error or env.last_error, json.dumps(state)))
            if event is not None:
                self.db.execute('INSERT INTO events(email,mode,job_id,url,detail) VALUES(?,?,?,?,?)',
                                (self.email, self.mode, job_id(env.job), env.job.get('url', ''), json.dumps(event)))

    def restore(self, env):
        saved = self.get(env.job)
        if not saved or saved['status'] != 'in_progress':
            return
        env.restore(saved['state'])
