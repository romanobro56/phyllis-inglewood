import sys, time, json, base64, math
import numpy as np
_argv = sys.argv[:]; sys.argv = ['x']
from honest_prep import *
sys.argv = _argv
S = np.load('reach783.npy'); mask = np.zeros(conn.n, bool); mask[S] = True
pre_of = np.repeat(np.arange(conn.n), np.diff(conn.indptr))
def exc_from(m):
    k = m[pre_of] & (conn.w > 0); tot = np.zeros(conn.n); np.add.at(tot, conn.post[k], conn.w[k]); return tot
tot = exc_from(mask)
def sub(R):
    remap = np.full(conn.n, -1, np.int64); remap[R] = np.arange(R.size)
    keep = (remap[pre_of] >= 0) & (remap[conn.post] >= 0)
    indptr, post, w = fd.to_csr(remap[pre_of[keep]], remap[conn.post[keep]].astype(np.int32), conn.w[keep], R.size)
    return fd.Connectome(conn.ids[R], indptr, post, w), remap
for thr in (10, 15, 20, 30, 50):
    R = np.flatnonzero(mask | (tot >= thr)); remap = np.full(conn.n, -1); remap[R] = 1
    print(f'margin >= {thr:>2} synapses: {R.size:>6} neurons, {int(((remap[pre_of] >= 0) & (remap[conn.post] >= 0)).sum()):>8} connections')
thr = 15

def random_stim(seed, seconds, groups, top=300.0):
    """Each channel switches on and off at random (20-150 ms), random rates; half the time one antenna is favoured."""
    rng = np.random.default_rng(seed); keys = list(groups); nseg = int(seconds / 0.01); tab = np.zeros((nseg, len(keys)))
    for j, k in enumerate(keys):
        i = 0
        while i < nseg:
            ln = int(rng.integers(2, 15)); on = rng.random() < 0.6
            tab[i:i + ln, j] = rng.uniform(40, top) * (0.3 if (rng.random() < 0.25) else 1.0) if on else 0.0; i += ln
    return (lambda t: {k: float(tab[min(nseg - 1, int(t / 0.01)), j]) for j, k in enumerate(keys)}), tab

if __name__ == '__main__' and len(sys.argv) > 1 and sys.argv[1] == 'cover':
    union = mask.copy(); t0 = time.time()
    for s in range(8):
        fn, _ = random_stim(100 + s, 1.0, GROUPS, top=300 if s % 2 else 200)
        c = run(fn, 1.0, seed=50 + s); f = c > 0; new = int((f & ~union).sum()); union |= f
        print(f'random {s}: fired {int(f.sum())}, new {new}, union {int(union.sum())} ({time.time() - t0:.0f}s)', flush=True)
    np.save('reach783b.npy', np.flatnonzero(union))

if __name__ == '__main__' and len(sys.argv) > 1 and sys.argv[1] == 'leak':
    S2 = np.load('reach783b.npy'); m2 = np.zeros(conn.n, bool); m2[S2] = True; tot2 = exc_from(m2); t0 = time.time()
    full = []
    for s in range(3):
        fn, _ = random_stim(900 + s, 1.0, GROUPS, top=280); full.append(run(fn, 1.0, seed=70 + s))
        print(f'held-out {s}: full model fired {int((full[-1] > 0).sum())}, outside reach set {int(((full[-1] > 0) & ~m2).sum())} ({time.time() - t0:.0f}s)', flush=True)
    for thr in (30, 20, 15):
        R = np.flatnonzero(m2 | (tot2 >= thr)); c, remap = sub(R); g = {k: remap[v] for k, v in GROUPS.items()}
        for s in range(3):
            fn, _ = random_stim(900 + s, 1.0, GROUPS, top=280); r = run(fn, 1.0, seed=70 + s, groups=g, c=c)
            same = np.array_equal(r, full[s][R]) and full[s].sum() == r.sum()
            print(f'margin {thr}: {R.size} neurons; held-out {s} identical to full model: {same} (spikes {int(r.sum())} vs {int(full[s].sum())}; neurons differing {int((r != full[s][R]).sum())})', flush=True)
