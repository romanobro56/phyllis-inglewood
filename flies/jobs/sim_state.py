"""Tenant-isolated simulated account/application state; separate from model weights.

The vault contains only a protected master key. Synthetic passwords are
resolved via opaque handles, never returned to policy observations or loggers.
No real credentials are accepted by this simulation.
"""
import base64
import copy
import hashlib
import hmac
import json
import os
from pathlib import Path
import secrets
import tempfile


def atomic_json(path, payload, private=True):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + '.tmp')
    fd = os.open(temporary, os.O_WRONLY | os.O_CREAT | os.O_TRUNC, 0o600)
    with os.fdopen(fd, 'w') as f:
        json.dump(payload, f, indent=2)
        f.flush()
        os.fsync(f.fileno())
    os.replace(temporary, path)
    if private:
        os.chmod(path, 0o600)


def private_vault_path(scope):
    """Keep credential material off permissionless Windows-mounted repo paths."""
    root=Path(os.environ.get('ELTON_SECRET_DIR', str(Path(tempfile.gettempdir())/'elton-simulation-secrets')))
    root.mkdir(parents=True,exist_ok=True,mode=0o700)
    os.chmod(root,0o700)
    if root.stat().st_mode & 0o077:
        raise ValueError('ELTON_SECRET_DIR must support private filesystem permissions')
    name=hashlib.sha256(str(Path(scope).resolve()).encode()).hexdigest()[:24]
    return root/(name+'.json')


class SecretVault:
    def __init__(self, key=None):
        self.key = key or secrets.token_bytes(32)

    @classmethod
    def open(cls, path):
        path = Path(path)
        if path.exists():
            if path.stat().st_mode & 0o077:
                raise ValueError('simulation vault must have mode 0600')
            return cls(base64.b64decode(json.loads(path.read_text())['key']))
        vault = cls()
        atomic_json(path, {'key': base64.b64encode(vault.key).decode()})
        if path.stat().st_mode & 0o077:
            raise ValueError('vault filesystem does not enforce mode 0600; use a private Linux filesystem')
        return vault

    def resolve(self, tenant, ref):
        digest = hmac.new(self.key, f'{tenant}:{ref}'.encode(), hashlib.sha256).hexdigest()
        if ref == 'password_weak':
            return digest[:4]
        return 'Aa9!' + digest[:20]


def password_hash(password, salt):
    # Test accounts only. A future live adapter must use an OS credential store.
    return hashlib.pbkdf2_hmac('sha256', password.encode(), salt.encode(), 1000).hex()


class SimulationState:
    def __init__(self, accounts=None, applications=None, drafts=None, vault_id=None):
        self.vault_id=vault_id
        self.accounts = copy.deepcopy(accounts or {})
        self.applications = copy.deepcopy(applications or {})
        self.drafts = copy.deepcopy(drafts or {})

    def bind_vault(self,vault):
        identity=hashlib.sha256(vault.key).hexdigest()
        if self.vault_id and self.vault_id!=identity:
            raise ValueError('Simulation account state belongs to a different credential vault')
        self.vault_id=identity

    @staticmethod
    def identity(tenant, email):
        return tenant + '|' + email.casefold().strip()

    @staticmethod
    def application_id(tenant, email, job):
        return SimulationState.identity(tenant, email) + '|' + job

    def account(self, tenant, email):
        return self.accounts.get(self.identity(tenant, email))

    def register(self, tenant, email, password, verified=True):
        key = self.identity(tenant, email)
        if key in self.accounts:
            return False
        salt = hashlib.sha256(key.encode()).hexdigest()[:16]
        self.accounts[key] = {'salt': salt, 'hash': password_hash(password, salt), 'verified': verified}
        return True

    def login(self, tenant, email, password):
        record = self.account(tenant, email)
        return bool(record and record['verified'] and hmac.compare_digest(record['hash'], password_hash(password, record['salt'])))

    def submitted(self, tenant, email, job):
        return self.application_id(tenant, email, job) in self.applications

    def snapshot(self):
        return copy.deepcopy({'vault_id': self.vault_id, 'accounts': self.accounts, 'applications': self.applications, 'drafts': self.drafts})

    def save(self, path):
        atomic_json(path, self.snapshot())

    @classmethod
    def load(cls, path, required=False):
        path = Path(path)
        if required and not path.is_file():
            raise FileNotFoundError('Missing required simulated account/application state: '+str(path))
        return cls(**json.loads(path.read_text())) if path.exists() else cls()
