"""Atomic feature-local artifacts and strict FlyBrain checkpoint compatibility."""
import hashlib
import json
import os
from pathlib import Path
import tempfile

import torch
from core.brain import FlyBrain
from flies.chess.chess_env import ACTIONS, OBS_DIM, ENCODING, ENCODINGS

ROOT = Path(__file__).resolve().parent
FIXED = ("edge_index", "t_perm", "t_index", "sign", "input_idx", "groups")


def output_path(path):
    path = Path(path).resolve()
    if ROOT not in path.parents:
        raise ValueError("Chess artifacts must stay inside flies/chess/")
    path.parent.mkdir(parents=True, exist_ok=True)
    return path


def atomic_write(path, value, binary=False):
    path = output_path(path)
    fd, tmp = tempfile.mkstemp(prefix="." + path.name, dir=path.parent)
    try:
        with os.fdopen(fd, "wb" if binary else "w") as f:
            if binary:
                torch.save(value, f)
            else:
                f.write(value)
            f.flush()
            os.fsync(f.fileno())
        os.replace(tmp, path)
    finally:
        if os.path.exists(tmp):
            os.unlink(tmp)


def write_json(path, data):
    atomic_write(path, json.dumps(data, indent=2, allow_nan=False))


def prepare_run(directory, resume=None):
    """Keep an accidental fresh launch from overwriting a valid experiment."""
    directory = output_path(Path(directory) / '.run').parent
    markers = ('resume.pt', 'training.json', 'history.json', 'summary.json', 'evaluation.json', 'checkpoint_0000_0.pt', 'initial_0.pt', 'brain_0.pt')
    if not resume and (any((directory/name).exists() for name in markers)
                       or any(directory.glob('game_*.json'))):
        raise FileExistsError('Output already contains a run; use --resume or a fresh output directory: '+str(directory))
    return directory


def file_hash(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def state_hash(brain):
    h = hashlib.sha256()
    for k, v in sorted(brain.state_dict().items()):
        h.update(k.encode())
        h.update(v.detach().cpu().contiguous().numpy().tobytes())
    return h.hexdigest()


def brain_payload(brain):
    return dict(state_dict=brain.state_dict(), preset=brain.preset, fly="chess", obs_dim=OBS_DIM,
                actions=ACTIONS, graph_hash=brain.graph_hash, encoding=getattr(brain, "encoding", ENCODING),
                policy_temperature=getattr(brain, "policy_temperature", 1.0),
                synthetic=brain.synthetic, format_version=1)


def validate_and_load(brain, ck):
    if (ck.get("fly") != "chess" or ck.get("obs_dim") != OBS_DIM or ck.get("actions") != ACTIONS
            or ck.get("graph_hash") != brain.graph_hash or ck.get("preset") != brain.preset
            or ck.get("encoding", ENCODING) != getattr(brain, "encoding", ENCODING)):
        raise ValueError("Incompatible chess checkpoint: graph, encoding, preset, or action dimensions")
    state = ck["state_dict"]
    own = brain.state_dict()
    for key in FIXED:
        # core now rebuilds the wiring from data/<preset>.npz instead of saving it (non-persistent buffers);
        # whenever a checkpoint or the brain does carry it, it must still match exactly
        mine = own[key] if key in own else getattr(brain, key, None)
        if key in own and key not in state:
            raise ValueError(f"Checkpoint changed fixed biological architecture: {key}")
        if key in state and (mine is None or not torch.equal(state[key].cpu(), mine.cpu())):
            raise ValueError(f"Checkpoint changed fixed biological architecture: {key}")
    state = {k: v for k, v in state.items() if k in own}
    for value in state.values():
        if value.is_floating_point() and not torch.isfinite(value).all():
            raise ValueError("Nonfinite checkpoint")
    temperature = ck.get("policy_temperature", 1.0)
    if not valid_temperature(temperature):
        raise ValueError("Invalid checkpoint policy temperature")
    brain.load_state_dict(state, strict=True)
    brain.policy_temperature = temperature
    return brain


def load_policy(path=None, size="tiny", device="cpu", encoding=None):
    # Only load trusted local checkpoints: torch optimizer/RNG records use pickle.
    ck = torch.load(path, map_location=device, weights_only=False) if path else None
    if ck and ck["preset"] != size:
        raise ValueError("--size must match checkpoint preset")
    brain = FlyBrain(size, OBS_DIM, 8, "chess").to(device)
    brain.encoding = encoding or (ck.get("encoding", ENCODING) if ck else ENCODING)
    if brain.encoding not in ENCODINGS:
        raise ValueError("Unknown checkpoint observation encoding")
    brain.policy_temperature = ck.get("policy_temperature", 1.0) if ck else 1.0
    if not valid_temperature(brain.policy_temperature):
        raise ValueError("Invalid checkpoint policy temperature")
    if ck:
        validate_and_load(brain, ck)
        brain.checkpoint_training_status=ck.get('training_status')
    return brain


def valid_temperature(value):
    import math
    return isinstance(value, (float, int)) and math.isfinite(value) and value > 0


def identity(brain, checkpoint=None, lesion=False):
    status=getattr(brain,'checkpoint_training_status',None)
    label='LESIONED' if lesion else 'ENGINEERING-PROBE' if status=='engineering-probe-only' else 'TRAINED' if checkpoint else 'UNTRAINED'
    return dict(label=label,training_status=status,
                preset=brain.preset, synthetic=brain.synthetic, neurons=brain.n,
                graph_hash=brain.graph_hash, state_sha256=state_hash(brain),
                checkpoint_sha256=file_hash(checkpoint) if checkpoint else None,
                checkpoint=str(checkpoint) if checkpoint else None,
                encoding=getattr(brain, "encoding", ENCODING),
                policy_temperature=getattr(brain, "policy_temperature", 1.0),
                lesion=lesion, action_groups=brain.describe_groups(ACTIONS))
