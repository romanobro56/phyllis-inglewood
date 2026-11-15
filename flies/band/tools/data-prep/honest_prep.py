"""Honest-ear prep on FlyWire v783: Johnston's organ groups by antenna side and class, and what they can reach."""
import json, sys, time
import numpy as np, pandas as pd
sys.path.insert(0, '/home/claude/work')
import fly_drums as fd
conn = fd.load_connectome('/home/claude/work/data783', '783')
A = pd.read_csv('/home/claude/ext/annot/repo/supplemental_files/Supplemental_file1_neuron_annotations.tsv', sep='\t', low_memory=False)
idx_of = {int(f): i for i, f in enumerate(conn.ids)}
A = A[A.root_id.isin(idx_of)].copy(); A['idx'] = A.root_id.map(idx_of)
ct = A.cell_type.astype(str)
GROUPS = {}
for side, s in (('left', 'L'), ('right', 'R')):
    for cls in 'ABCDE':
        m = A[(ct.str.startswith('JO-' + cls)) & (A.side == side)]
        GROUPS[cls + s] = np.sort(m.idx.values)
DN = A[A.super_class == 'descending']
sup = np.full(conn.n, '', object); sup[A.idx.values] = A.super_class.values
name = np.full(conn.n, '', object); name[A.idx.values] = np.where(A.cell_type.notna(), A.cell_type, A.hemibrain_type).astype(str)
hb = np.full(conn.n, '', object); hb[A.idx.values] = A.hemibrain_type.astype(str).values
sideof = np.full(conn.n, '', object); sideof[A.idx.values] = A.side.astype(str).values

def run(rates_fn, seconds, seed=1, groups=GROUPS, c=conn, count=True):
    b = fd.FlyBrain(c, seed=seed)
    for k, ids in groups.items(): b.channels[k] = ids; b.rfc[ids] = 0; b.is_driven[ids] = True; b._activate(ids)
    n = int(seconds / fd.DT); cnt = np.zeros(c.n, np.int32)
    for i in range(n):
        s = b.step(rates_fn(i * fd.DT))
        if s.size: cnt[s] += 1
    return cnt

if __name__ == '__main__':
    print({k: len(v) for k, v in GROUPS.items()}, '| DNs in model:', len(DN))
    t = time.time(); c = run(lambda t: {k: 150.0 for k in ('AL', 'AR', 'BL', 'BR')}, 0.5)
    print(f'0.5 s of A+B at 150 Hz: {time.time() - t:.0f}s wall; fired {np.count_nonzero(c)}; DNs fired {np.count_nonzero(c[DN.idx.values])}')
