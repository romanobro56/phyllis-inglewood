import sys, time, json
import numpy as np
from collections import Counter
sys.argv = ['x']
from honest_prep import *
ALL = list(GROUPS)
def const(keys, hz): return lambda t: {k: hz for k in keys}
def pulses(keys, hz, period=0.035, duty=0.4): return lambda t: {k: (hz if (t % period) < duty * period else 0.0) for k in keys}
def antiphase(hz, f=40):   # C/D on the forward half-cycle, E on the backward half, A/B always
    def fn(t):
        ph = (t * f) % 1 < 0.5
        d = {k: hz for k in ALL if k[0] in 'AB'}
        d.update({k: (hz if ph else 0.0) for k in ALL if k[0] in 'CD'}); d.update({k: (0.0 if ph else hz) for k in ALL if k[0] == 'E'}); return d
    return fn
conds = [('A+B 150', const([k for k in ALL if k[0] in 'AB'], 150), 1.0), ('A only 250', const(['AL', 'AR'], 250), 1.0), ('B only 250', const(['BL', 'BR'], 250), 1.0),
         ('C+D only 250', const([k for k in ALL if k[0] in 'CD'], 250), 1.0), ('E only 250', const(['EL', 'ER'], 250), 1.0),
         ('all 250', const(ALL, 250), 1.0), ('all 100', const(ALL, 100), 1.0), ('left ear all 250', const([k for k in ALL if k[1] == 'L'], 250), 1.0), ('right ear all 250', const([k for k in ALL if k[1] == 'R'], 250), 1.0),
         ('A+B pulses 35ms', pulses([k for k in ALL if k[0] in 'AB'], 300), 1.0), ('all pulses 35ms', pulses(ALL, 300), 1.0), ('all pulses 120ms', pulses(ALL, 300, 0.12, 0.3), 1.0), ('antiphase 40Hz', antiphase(250), 1.0)]
union = np.zeros(conn.n, bool); res = {}; t0 = time.time()
for nm, fn, sec in conds:
    c = run(fn, sec, seed=len(res) + 1); res[nm] = c; f = c > 0; new = (f & ~union).sum(); union |= f
    print(f'{nm:<20} fired {f.sum():>5} (DN {np.count_nonzero(c[DN.idx.values]):>3})  new {new:>4}  union {union.sum():>5}  ({time.time() - t0:.0f}s)', flush=True)
np.save('reach783.npy', np.flatnonzero(union)); np.savez('reach783_counts.npz', **{k.replace(' ', '_').replace('+', 'p'): v for k, v in res.items()})
print('union by super_class:', dict(Counter(sup[union])))
dn = DN.idx.values; fdn = dn[union[dn]]
print('DNs reached:', len(fdn), 'by side', dict(Counter(sideof[fdn])))
c = res['all 250']; top = fdn[np.argsort(-c[fdn])][:40]
print('top DNs under "all 250" (name side Hz):', ' '.join(f'{name[i]}{sideof[i][0].upper()}:{c[i]}' for i in top))
for pat in ['vpoEN', 'vpoDN', 'DNp01', 'pC1', 'AMMC-A1', 'pC2', 'aPN1']:
    ii = [i for i in range(conn.n) if pat in name[i] or pat in hb[i]]
    print(f'{pat}: {len(ii)} neurons, reached {int(union[ii].sum())}; max Hz per condition', {k: int(v[ii].max()) for k, v in res.items() if v[ii].max() > 0})
# laterality: left-ear drive -> left vs right DN spikes
for nm in ['left ear all 250', 'right ear all 250']:
    c = res[nm]; print(nm, '-> DN spikes left', int(c[dn[sideof[dn] == 'left']].sum()), 'right', int(c[dn[sideof[dn] == 'right']].sum()))
