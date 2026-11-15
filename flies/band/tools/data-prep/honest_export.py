import sys, json, base64, math
import numpy as np
_a = sys.argv[:]; sys.argv = ['x']
from honest_reduce import *
S2 = np.load('reach783b.npy'); m2 = np.zeros(conn.n, bool); m2[S2] = True
R = np.flatnonzero(m2 | (exc_from(m2) >= 30)); c, remap = sub(R)
b64 = lambda a: base64.b64encode(np.ascontiguousarray(a).tobytes()).decode()
assert c.n < 65536 and np.abs(c.w).max() < 32767, (c.n, np.abs(c.w).max())
P = fd.PARAMS
dn = DN.idx.values; dn = dn[remap[dn] >= 0]
gf = np.array([i for i in dn if name[i] == 'DNp01'])
net = {'n': int(c.n), 'indptr': b64(c.indptr.astype('<i4')), 'post': b64(c.post.astype('<u2')), 'w': b64(c.w.astype('<i2')),
       'groups': {k: b64(remap[v].astype('<i4')) for k, v in GROUPS.items()},
       'dnL': b64(remap[dn[sideof[dn] == 'left']].astype('<i4')), 'dnR': b64(remap[dn[sideof[dn] == 'right']].astype('<i4')),
       'gf': [int(remap[i]) for i in gf], 'gfSide': [str(sideof[i]) for i in gf],
       'consts': {'v_0': P['v_0'], 'v_th': P['v_th'], 'v_rst': P['v_rst'], 'w_syn': P['w_syn'], 'dt': fd.DT, 'kv': math.exp(-fd.DT / P['t_mbr']), 'kg': math.exp(-fd.DT / P['tau']),
                  'kvg': P['tau'] / (P['tau'] - P['t_mbr']) * (math.exp(-fd.DT / P['tau']) - math.exp(-fd.DT / P['t_mbr'])),
                  'delay': int(round(P['t_dly'] / fd.DT)), 'rfc': int(round(P['t_rfc'] / fd.DT)), 'kick': P['w_syn'] * P['f_poi']},
       'meta': {'n_full': int(conn.n), 'syn_full': int(conn.post.size), 'n_net': int(c.n), 'syn_net': int(c.post.size), 'n_reach': int(m2.sum()),
                'n_dn_full': int(len(DN)), 'n_dn_reach': int(m2[DN.idx.values].sum()), 'jo': {k: int(len(v)) for k, v in GROUPS.items()}}}
json.dump(net, open('honest_net.json', 'w'))
# positions for drawing
VOX = np.array([4.0, 4.0, 40.0]) / 1000.0
Apos = A.dropna(subset=['pos_x']); Pall = np.full((conn.n, 3), np.nan); Pall[Apos.idx.values] = Apos[['pos_x', 'pos_y', 'pos_z']].values * VOX
old = json.load(open('page2_data.json'))
allpos = Apos[['pos_x', 'pos_y', 'pos_z']].values * VOX
import pandas as pd
full = pd.read_csv('/home/claude/ext/annot/repo/supplemental_files/Supplemental_file1_neuron_annotations.tsv', sep='\t', usecols=['pos_x', 'pos_y', 'pos_z'], low_memory=False).dropna().values * VOX
centre = np.array([(full[:, k].min() + full[:, k].max()) / 2 for k in range(3)])
Pr = Pall[R]; miss = np.isnan(Pr[:, 0]); Pr[miss] = np.nanmean(Pall, axis=0)
role = np.zeros(c.n, 'u1')
for k, v in GROUPS.items(): role[remap[v]] = 'ABCDE'.index(k[0]) + 1
role[remap[dn]] = 6; role[remap[gf]] = 7
side = np.array([0 if sideof[i] == 'left' else 1 if sideof[i] == 'right' else 2 for i in R], 'u1')
page = {'bg': old['bg'], 'pos': b64(np.clip(np.round((Pr - centre) * 20), -32767, 32767).astype('<i2')), 'role': b64(role), 'side': b64(side), 'reach': b64(m2[R].astype('u1')), 'meta': net['meta']}
json.dump(page, open('honest_page.json', 'w'))
print('net MB', round(len(json.dumps(net)) / 1e6, 2), 'page MB', round(len(json.dumps(page)) / 1e6, 2), net['meta'], 'GF', net['gf'], net['gfSide'], 'missing pos', int(miss.sum()))
# left/right in image space: mean x of left vs right JO
for k in ('AL', 'AR', 'EL', 'ER'): print(k, 'mean x (um, centred)', round(float(np.nanmean(Pall[GROUPS[k], 0] - centre[0])), 1))
