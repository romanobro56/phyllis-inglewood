"""FlyBrain: the fruit-fly connectome as a trainable recurrent network.

Fixed by the fly (never trained):
    - which neurons connect to which (FlyWire v783, >= 5 synapses)
    - the sign of every connection (excitatory / inhibitory, from the neurotransmitter)
    - where input enters (sensory neurons) and where actions are read (descending neurons)

Trained per fly:
    - the strength of each existing connection (log scale, sign kept)
    - per-neuron bias and leak
    - the input projection from the task's observation onto sensory neurons

There is NO learned output layer: an action's score is the mean firing rate of
its fixed group of real descending neurons. lesion=True zeroes the connectome.
"""
import hashlib
from functools import lru_cache
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

ROOT = Path(__file__).resolve().parents[1]

STEPS = 8           # brain steps per decision
GROUP_SIZE = 16     # descending neurons per action
TEMPERATURE = 0.05  # firing-rate difference -> action preference
MAX_RATE = 10.0
IN_GAIN = 3.0


class _ConnectomeMatmul(torch.autograd.Function):
    """out = W @ r for the fixed sparse wiring, with a gradient for W's values that never
    materialises an N x N matrix (torch's built-in sparse backward does, which needs 72 GB)."""

    @staticmethod
    def forward(ctx, w, r, edge_index, t_index, t_perm, n):
        A = torch.sparse_coo_tensor(edge_index, w, (n, n), is_coalesced=True, check_invariants=False)
        ctx.save_for_backward(w, r, edge_index, t_index, t_perm)
        ctx.n = n
        return torch.sparse.mm(A, r)

    @staticmethod
    def backward(ctx, grad_out):
        w, r, edge_index, t_index, t_perm = ctx.saved_tensors
        n = ctx.n
        grad_w = grad_r = None
        if ctx.needs_input_grad[0]:
            post, pre = edge_index
            grad_w = torch.empty_like(w)
            chunk = max(1, 2 ** 25 // max(1, r.shape[1]))     # bound memory to ~128 MB per chunk
            for s in range(0, w.numel(), chunk):
                e = slice(s, s + chunk)
                grad_w[e] = (grad_out[post[e]] * r[pre[e]]).sum(1)
        if ctx.needs_input_grad[1]:
            At = torch.sparse_coo_tensor(t_index, w[t_perm], (n, n), is_coalesced=True, check_invariants=False)
            grad_r = torch.sparse.mm(At, grad_out)
        return grad_w, grad_r, None, None, None, None


def pick_device(name="auto"):
    if name != "auto":
        return torch.device(name)
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


@lru_cache(maxsize=4)
def load_graph(preset):
    path = ROOT / "data" / f"{preset}.npz"
    if not path.exists():
        hint = "python -m core.graph --size synthetic" if preset == "synthetic" else \
               f"python scripts/download_data.py && python -m core.graph --size {preset}"
        raise SystemExit(f"Missing data/{preset}.npz. Run: {hint}")
    z = np.load(path, allow_pickle=False)
    return {k: z[k] for k in z.files}


def action_groups(fly_name, n_actions, output_idx, group_size=GROUP_SIZE):
    """Fixed, disjoint groups of descending neurons, one per action, seeded by the fly's name."""
    seed = int(hashlib.sha256(fly_name.encode()).hexdigest()[:8], 16)
    rng = np.random.default_rng(seed)
    g = min(group_size, len(output_idx) // n_actions)
    chosen = rng.choice(output_idx, size=n_actions * g, replace=False)
    return chosen.reshape(n_actions, g)


class FlyBrain(nn.Module):
    def __init__(self, preset, obs_dim, n_actions, fly_name):
        super().__init__()
        g = load_graph(preset)
        self.preset, self.fly_name = preset, fly_name
        self.n = int(g["n"])
        self.graph_hash = str(g["graph_hash"])
        self.synthetic = bool(g["synthetic"])
        self.root_id = g["root_id"]

        pre = torch.as_tensor(g["pre"], dtype=torch.long)
        post = torch.as_tensor(g["post"], dtype=torch.long)
        order = torch.argsort(post * self.n + pre)          # coalesced order for sparse mm
        post, pre = post[order], pre[order]
        self.register_buffer("edge_index", torch.stack([post, pre]))
        t_perm = torch.argsort(pre * self.n + post)          # same edges, transposed and coalesced
        self.register_buffer("t_perm", t_perm)
        self.register_buffer("t_index", torch.stack([pre[t_perm], post[t_perm]]))
        self.register_buffer("sign", torch.as_tensor(g["sign"], dtype=torch.float32)[order])
        count = torch.as_tensor(g["count"], dtype=torch.float32)[order]

        # initial strength ~ log(1 + synapses), scaled so the average neuron receives ~1 unit of input
        w0 = torch.log1p(count)
        in_sum = torch.zeros(self.n).index_add_(0, self.edge_index[0], w0)
        scale = 1.0 / in_sum[in_sum > 0].mean()
        self.log_w = nn.Parameter(torch.log(w0 * scale))
        self.bias = nn.Parameter(torch.zeros(self.n))
        self.leak_logit = nn.Parameter(torch.zeros(self.n))  # sigmoid(0) = 0.5

        self.register_buffer("input_idx", torch.as_tensor(g["input_idx"], dtype=torch.long))
        self.w_in = nn.Parameter(torch.randn(obs_dim, len(g["input_idx"])) / np.sqrt(obs_dim))

        groups = action_groups(fly_name, n_actions, g["output_idx"])
        self.register_buffer("groups", torch.as_tensor(groups, dtype=torch.long))
        self.n_actions = n_actions

    def weights(self, lesion=False):
        w = self.sign * torch.exp(self.log_w)
        return torch.zeros_like(w) if lesion else w

    def forward(self, obs, mask=None, lesion=False, record=False):
        """obs: (B, obs_dim) -> logits (B, n_actions), info dict.

        record=True also returns every neuron's rate at every step: info["rates"] (STEPS, B, N).
        """
        B = obs.shape[0]
        w = self.weights(lesion)
        inject = torch.zeros(self.n, B, device=obs.device)
        inject[self.input_idx] = IN_GAIN * (obs @ self.w_in).T
        leak = torch.sigmoid(self.leak_logit)[:, None]
        bias = self.bias[:, None]

        r = torch.zeros(self.n, B, device=obs.device)
        acc, rates = 0.0, []
        for t in range(STEPS):
            drive = _ConnectomeMatmul.apply(w, r, self.edge_index, self.t_index, self.t_perm, self.n) \
                + bias + inject
            r = (1 - leak) * r + leak * torch.clamp(F.relu(drive), max=MAX_RATE)
            if t >= STEPS // 2:
                acc = acc + r
            if record:
                rates.append(r.detach().T)
        out = acc / (STEPS - STEPS // 2)                      # (N, B), late-window average
        K, G = self.groups.shape
        group_rate = out[self.groups.reshape(-1)].reshape(K, G, B).mean(1).T  # (B, K)
        logits = group_rate / TEMPERATURE
        if mask is not None:
            logits = logits.masked_fill(~mask, -1e9)
        info = {"group_rate": group_rate.detach(), "mean_rate": out.mean().item()}
        if record:
            info["rates"] = torch.stack(rates)                # (STEPS, B, N)
        return logits, info

    def describe_groups(self, action_names):
        """Action name -> list of FlyWire root IDs (and graph indices) driving it."""
        return {name: {"index": self.groups[i].tolist(),
                       "root_id": [str(self.root_id[j]) for j in self.groups[i].tolist()]}
                for i, name in enumerate(action_names)}


def save_brain(brain, path, task_cls):
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save({"state_dict": brain.state_dict(), "preset": brain.preset, "fly": brain.fly_name,
                "obs_dim": task_cls.obs_dim, "actions": list(task_cls.actions),
                "graph_hash": brain.graph_hash}, path)


def load_brain(path, task_cls, device="cpu"):
    ck = torch.load(path, map_location=device, weights_only=False)
    if list(ck["actions"]) != list(task_cls.actions) or ck["obs_dim"] != task_cls.obs_dim:
        raise ValueError(f"{path}: actions/obs_dim changed since training; retrain this fly")
    brain = FlyBrain(ck["preset"], ck["obs_dim"], len(ck["actions"]), ck["fly"]).to(device)
    if brain.graph_hash != ck["graph_hash"]:
        raise ValueError(f"{path}: trained on a different data/{ck['preset']}.npz; retrain or copy that file")
    brain.load_state_dict(ck["state_dict"])
    return brain
