"""Environment-independent FlyBrain selection and schema-checked persistence."""
from pathlib import Path
import hashlib
import json
import torch
import numpy as np
from core.brain import FlyBrain, save_brain, load_brain
from flies.jobs.contract import SCHEMA, ACTIONS


def digest(brain):
    h = hashlib.sha256()
    for name, parameter in brain.named_parameters():
        h.update(name.encode())
        h.update(parameter.detach().cpu().numpy().tobytes())
    return h.hexdigest()


def save_model(brain, path, task):
    save_brain(brain, Path(path), task)
    payload = torch.load(path, map_location='cpu', weights_only=False)
    payload['observation_schema'] = SCHEMA
    torch.save(payload, path)
    Path(path).chmod(0o600)


def checkpoint_path(path):
    path = Path(path)
    if path.is_dir():
        pointer = json.loads((path/'latest.json').read_text())
        if pointer['schema'] != SCHEMA:
            raise ValueError('incompatible checkpoint manifest')
        return path / pointer['directory'] / 'brain.pt'
    return path


def load_model(path, task, device='cpu'):
    requested=Path(path)
    expected_hash=json.loads((requested/'latest.json').read_text()).get('model_hash') if requested.is_dir() else None
    path = checkpoint_path(path)
    payload = torch.load(path, map_location='cpu', weights_only=False)
    if payload.get('observation_schema') != SCHEMA:
        raise ValueError('Checkpoint has an incompatible observation/action schema. No random fallback is permitted.')
    brain = load_brain(path, task, device)
    if expected_hash and digest(brain)!=expected_hash:
        raise ValueError('checkpoint manifest/model hash mismatch')
    if brain.synthetic:
        raise ValueError('synthetic connectome is not accepted')
    return brain


def probabilities(brain, observations, masks, policy='trained', device='cpu'):
    masks = np.asarray(masks, bool)
    if policy == 'random':
        return masks / masks.sum(1, keepdims=True)
    if brain is None:
        raise ValueError('neural policy requires an explicit brain')
    with torch.no_grad():
        logits, _ = brain(torch.as_tensor(np.asarray(observations), device=device),
                          torch.as_tensor(masks, device=device), lesion=policy == 'lesioned')
        return logits.softmax(-1).cpu().numpy().astype(float)
