#!/usr/bin/env python3
"""
fly_drums.py - a simulated fruit-fly brain plays MIDI drums.

What it does
------------
1. Loads the FlyWire whole-brain connectome used by Shiu et al. (Nature 2024),
   straight from a clone of https://github.com/philshiu/Drosophila_brain_model
2. Runs the same leaky integrate-and-fire model as that repo's model.py
   (a numpy port, dt = 0.1 ms, same constants), so you don't need Brian2.
3. You "play" rhythms into two sensory channels of the fly, step by step:
      antenna : Johnston's-organ mechanosensory neurons (touching the antenna
                makes a real fly groom it with its front legs)
      sugar   : sugar-tasting neurons on the proboscis (makes a fly extend
                its proboscis to feed)
   Every step in a pattern is a short burst of Poisson stimulation to those neurons.
4. Each drum is a neuron (or a pair) downstream in the brain. Every time that
   neuron starts a burst of spikes, the drum is hit, harder when it fires harder.
   The default kit uses neurons named in the paper (proboscis motor neuron MN9,
   antennal-grooming neurons aBN1/DN1/DN2) plus the strongest other responders.
5. Writes a General MIDI drum track (.mid). Optional: a WAV preview with built-in
   synth drums, and live playback to a MIDI port (e.g. your DAW / drum machine).

Quick start
-----------
    git clone https://github.com/philshiu/Drosophila_brain_model
    pip install numpy pandas pyarrow              # add: mido python-rtmidi  for --port
    python fly_drums.py --data Drosophila_brain_model --out fly.mid --wav fly.wav

Patterns: one character per 16th note. 'X' = strong stimulus, 'x' = soft, '.' = rest.
Patterns loop, so a 12-step pattern against a 16-step one gives you a polyrhythm.
    python fly_drums.py --antenna "X...x...X...x.x." --sugar "X..x..X...X..x.." --bpm 100

Swap neurons in and out of the kit with --kit (JSON file, see --help) and use
--list-responders to see which neurons answer your rhythm. Play live into a DAW or
drum machine with --port "IAC" --loop  (--port list shows your MIDI outputs).

The first run builds a small cache next to the connectome (~1 min). After that,
simulating runs at very roughly 5-10x slower than real time on a laptop CPU, so a
4-bar groove takes well under a minute. Every run is different: the stimulation is
random (Poisson), like a real drummer who never plays it exactly the same twice.
"""
import argparse
import json
import math
import struct
import time
import wave
from pathlib import Path

import numpy as np

# ----------------------------------------------------------------------------
# Neuron IDs (FlyWire v630, as used in the Shiu et al. repo's figures.ipynb)
# ----------------------------------------------------------------------------
SUGAR_R = [
    720575940624963786, 720575940630233916, 720575940637568838, 720575940638202345, 720575940617000768,
    720575940630797113, 720575940632889389, 720575940621754367, 720575940621502051, 720575940640649691,
    720575940639332736, 720575940616885538, 720575940639198653, 720575940620900446, 720575940617937543,
    720575940632425919, 720575940633143833, 720575940612670570, 720575940628853239, 720575940629176663,
    720575940611875570,
]
JON_CE = [
    720575940619341105, 720575940630122015, 720575940611061526, 720575940615848788, 720575940628444667,
    720575940627941431, 720575940632449619, 720575940650244342, 720575940631866508, 720575940638681845,
    720575940628978450, 720575940609522461, 720575940621442224, 720575940602506208, 720575940629022149,
    720575940627109991, 720575940630020111, 720575940615986459, 720575940618684481, 720575940620382889,
    720575940630080071, 720575940626565455, 720575940630319671, 720575940602720940, 720575940630564179,
    720575940637632419, 720575940615809349, 720575940626042149, 720575940637054835, 720575940602132509,
    720575940614188149, 720575940616951124, 720575940628101126, 720575940629055721, 720575940616589878,
    720575940622449388, 720575940614427195, 720575940625797617, 720575940638664437, 720575940618467195,
    720575940621729757, 720575940613971485, 720575940627585688, 720575940629650997, 720575940630059847,
    720575940608742409, 720575940614351477, 720575940633153375, 720575940622937528, 720575940604753437,
    720575940611783464, 720575940618599872, 720575940609541917, 720575940637410869, 720575940630070343,
    720575940621397417, 720575940614035485, 720575940610018266, 720575940626307902, 720575940634634606,
    720575940614060829, 720575940624799290, 720575940641921421, 720575940623298559, 720575940625559358,
    720575940629138959, 720575940621625597, 720575940625962568, 720575940632767383, 720575940624915230,
]
JON_F = [
    720575940606239243, 720575940626956777, 720575940604973746, 720575940622222856, 720575940642517284,
    720575940629719404, 720575940616613022, 720575940604299454, 720575940615473186, 720575940622217992,
    720575940606800341, 720575940629267498, 720575940637366335, 720575940624224408, 720575940609543197,
    720575940633364179, 720575940629502009, 720575940606431189, 720575940625733960, 720575940638529525,
    720575940617524053, 720575940628935564, 720575940624308355, 720575940631170346, 720575940627704375,
    720575940625885512, 720575940614929245, 720575940647493241, 720575940618888368, 720575940625087546,
    720575940606657493, 720575940617273560, 720575940640591861, 720575940639410035, 720575940621532413,
    720575940627523584, 720575940621521917, 720575940621097398, 720575940625915338, 720575940606222428,
    720575940627868471, 720575940622179497, 720575940608297774, 720575940614026269, 720575940613012959,
    720575940628100614, 720575940606611401, 720575940628649465, 720575940610008217, 720575940623791152,
    720575940625571240, 720575940634923621, 720575940609530653, 720575940635968745, 720575940625703434,
    720575940613105311, 720575940629386819, 720575940623077389, 720575940625763015, 720575940628359017,
]
JON_D_m = [
    720575940630834171, 720575940622892988, 720575940621289537, 720575940641395163, 720575940616064546,
    720575940628978409, 720575940652566177, 720575940627493096, 720575940619085397, 720575940635545310,
    720575940645728803, 720575940629141775, 720575940626557995, 720575940631098338, 720575940639904475,
    720575940635067034,
]
JON_ALL = JON_CE + JON_F + JON_D_m
INPUTS = {'antenna': JON_ALL, 'sugar': SUGAR_R}

NAMED = {
    'MN9_L': 720575940660219265,   # proboscis motor neuron (feeding)
    'MN9_R': 720575940645521262,
    'aBN1': 720575940630907434,    # antennal grooming brain neuron
    'DN1': 720575940616185531,     # antennal grooming descending neurons
    'DN2': 720575940629806974,
}

# General MIDI percussion (channel 10)
GM = {
    'kick': 36, 'rim': 37, 'snare': 38, 'clap': 39, 'closed_hat': 42, 'low_tom': 45,
    'pedal_hat': 44, 'open_hat': 46, 'mid_tom': 47, 'high_tom': 50, 'crash': 49,
    'ride': 51, 'tambourine': 54, 'cowbell': 56, 'shaker': 70, 'conga': 63,
}

# Default kit: drum -> neuron name(s). Empty slots are filled automatically.
DEFAULT_KIT = {
    'kick': ['MN9_L', 'MN9_R'],
    'snare': ['DN1'],
    'closed_hat': ['aBN1'],
    'clap': ['DN2'],
    'shaker': [], 'low_tom': [], 'rim': [],
}

# Model constants from model.py (Shiu et al. 2024)
PARAMS = dict(
    v_0=-52.0, v_rst=-52.0, v_th=-45.0,    # mV
    t_mbr=20e-3, tau=5e-3,                 # s
    t_rfc=2.2e-3, t_dly=1.8e-3,            # s
    w_syn=0.275, f_poi=250,                # mV per synapse, Poisson kick factor
)
DT = 1e-4  # s (Brian2 default clock)


# ----------------------------------------------------------------------------
# Connectome loading
# ----------------------------------------------------------------------------
class Connectome:
    def __init__(self, ids, indptr, post, w):
        self.ids = ids                      # FlyWire ID of each model index
        self.n = len(ids)
        self.indptr, self.post, self.w = indptr, post, w
        self.index = {int(f): i for i, f in enumerate(ids)}

    def idx(self, flyids):
        missing = [f for f in flyids if int(f) not in self.index]
        if missing:
            raise KeyError(f'{len(missing)} FlyWire IDs are not in this connectome version '
                           f'(e.g. {missing[0]}). IDs differ between v630 and v783.')
        return np.array([self.index[int(f)] for f in flyids], dtype=np.int64)


def load_connectome(data_dir, version='630'):
    data_dir = Path(data_dir)
    if version == '630':
        comp, con = '2023_03_23_completeness_630_final.csv', '2023_03_23_connectivity_630_final.parquet'
    else:
        comp, con = 'Completeness_783.csv', 'Connectivity_783.parquet'
    cache = data_dir / f'.fly_drums_cache_{version}.npz'
    if cache.is_file():
        z = np.load(cache)
        return Connectome(z['ids'], z['indptr'], z['post'], z['w'])

    import pandas as pd
    print(f'Building connectome cache from {con} (one-time, ~1 min)...', flush=True)
    ids = pd.read_csv(data_dir / comp, index_col=0).index.values.astype(np.int64)
    df = pd.read_parquet(data_dir / con,
                         columns=['Presynaptic_Index', 'Postsynaptic_Index', 'Excitatory x Connectivity'])
    pre = df['Presynaptic_Index'].values.astype(np.int64)
    post = df['Postsynaptic_Index'].values.astype(np.int32)
    w = df['Excitatory x Connectivity'].values.astype(np.int32)
    indptr, post, w = to_csr(pre, post, w, len(ids))
    try:
        np.savez(cache, ids=ids, indptr=indptr, post=post, w=w)
    except OSError:
        pass
    return Connectome(ids, indptr, post, w)


def to_csr(pre, post, w, n):
    order = np.argsort(pre, kind='stable')
    counts = np.bincount(pre, minlength=n)
    indptr = np.zeros(n + 1, dtype=np.int64)
    np.cumsum(counts, out=indptr[1:])
    return indptr, post[order].astype(np.int32), w[order].astype(np.int32)


# ----------------------------------------------------------------------------
# The brain: numpy port of model.py's leaky integrate-and-fire network
# ----------------------------------------------------------------------------
class FlyBrain:
    """Same equations, constants and update order as Brian2 runs them in model.py:
         dv/dt = (v_0 - v + g) / t_mbr   (unless refractory)
         dg/dt = -g / tau                (unless refractory)
       spike when v > v_th -> v = v_rst, g = 0; each synapse adds w_syn * count to the
       target's g after t_dly. Stimulated neurons get Poisson kicks of w_syn*f_poi on v.
       Only neurons that have been perturbed are integrated (the rest sit exactly at rest).
    """

    def __init__(self, conn, seed=None, params=PARAMS):
        p = params
        self.c = conn
        self.n = conn.n
        self.w = conn.w.astype(np.float64) * p['w_syn']
        self.v0, self.vth, self.vrst = p['v_0'], p['v_th'], p['v_rst']
        self.kv = math.exp(-DT / p['t_mbr'])
        self.kg = math.exp(-DT / p['tau'])
        self.kvg = p['tau'] / (p['tau'] - p['t_mbr']) * (self.kg - self.kv)
        self.delay = int(round(p['t_dly'] / DT))
        self.kick = p['w_syn'] * p['f_poi']
        self.v = np.full(self.n, self.v0)
        self.g = np.zeros(self.n)
        self.last = np.full(self.n, -10 ** 9, dtype=np.int64)
        self.rfc = np.full(self.n, int(round(p['t_rfc'] / DT)), dtype=np.int64)
        self.is_active = np.zeros(self.n, dtype=bool)
        self.is_driven = np.zeros(self.n, dtype=bool)
        self.active = np.empty(0, dtype=np.int64)
        self.queue = [np.empty(0, dtype=np.int64)] * self.delay
        self.channels = {}
        self.i = 0
        self.rng = np.random.default_rng(seed)

    def add_channel(self, name, flyids):
        idx = self.c.idx(flyids)
        self.channels[name] = idx
        self.rfc[idx] = 0             # model.py: no refractory period for Poisson targets
        self.is_driven[idx] = True
        self._activate(idx)

    def _activate(self, idx):
        new = idx[~self.is_active[idx]]
        if new.size:
            new = np.unique(new)
            self.is_active[new] = True
            self.active = np.concatenate([self.active, new])

    def _deliver(self, pre):
        s, e = self.c.indptr[pre], self.c.indptr[pre + 1]
        if pre.size == 1:
            sl = slice(s[0], e[0])
            posts, ws = self.c.post[sl], self.w[sl]
        else:
            ln = e - s
            off = np.repeat(s - np.cumsum(ln) + ln, ln)
            k = np.arange(ln.sum()) + off
            posts, ws = self.c.post[k], self.w[k]
        np.add.at(self.g, posts, ws)
        self._activate(posts.astype(np.int64))

    def _prune(self):
        a = self.active
        keep = (np.abs(self.g[a]) > 1e-7) | (np.abs(self.v[a] - self.v0) > 1e-7) | self.is_driven[a] \
            | (self.i - self.last[a] < self.rfc[a] + 1)
        drop = a[~keep]
        self.v[drop] = self.v0
        self.g[drop] = 0.0
        self.is_active[drop] = False
        self.active = a[keep]

    def step(self, rates):
        """Advance 0.1 ms. rates: {channel: Hz}. Returns indices of neurons that spiked."""
        i = self.i
        a = self.active
        # 1) integrate (exact solution of the linear ODEs), skipping refractory neurons
        upd = a[(i - self.last[a]) >= self.rfc[a]]
        v, g = self.v[upd], self.g[upd]
        v = self.v0 + (v - self.v0) * self.kv + g * self.kvg
        self.v[upd] = v
        self.g[upd] = g * self.kg
        # 2) threshold
        spk = upd[v > self.vth]
        self.last[spk] = i
        # 3) synapses: deliver spikes from t - t_dly, queue this step's spikes
        slot = i % self.delay
        due = self.queue[slot]
        if due.size:
            self._deliver(due)
        self.queue[slot] = spk
        #    Poisson stimulation
        for name, r in rates.items():
            if r > 0:
                idx = self.channels[name]
                hit = idx[self.rng.random(idx.size) < r * DT]
                self.v[hit] += self.kick
        # 4) reset
        self.v[spk] = self.vrst
        self.g[spk] = 0.0
        self.i += 1
        if i % 500 == 0:
            self._prune()
        return spk


# ----------------------------------------------------------------------------
# Rhythm in, spikes out
# ----------------------------------------------------------------------------
def parse_pattern(p):
    p = ''.join(ch for ch in p if ch in 'Xx.-')
    if not p:
        raise ValueError('empty pattern')
    return p.replace('-', '.')


def perform(brain, patterns, bpm, bars, stim_ms, soft_hz, strong_hz, record=None, tail_s=0.3,
            progress=True):
    """Play the patterns into the brain. Returns {neuron index: spike times (s)} for `record`
    (array of indices) or for every neuron that spiked if record is None."""
    step_s = 60.0 / bpm / 4
    n_steps = bars * 16
    total = int(round((n_steps * step_s + tail_s) / DT))
    stim_n = int(round(stim_ms * 1e-3 / DT))
    step_n = step_s / DT
    rec_mask = None
    if record is not None:
        rec_mask = np.zeros(brain.n, dtype=bool)
        rec_mask[record] = True
    times, who = [], []
    t0 = time.time()
    for k in range(total):
        s = int(k // step_n)
        within = k - int(round(s * step_n))
        rates = {}
        for ch, pat in patterns.items():
            r = 0.0
            if s < n_steps and within < stim_n:
                c = pat[s % len(pat)]
                r = strong_hz if c == 'X' else soft_hz if c == 'x' else 0.0
            rates[ch] = r
        spk = brain.step(rates)
        if spk.size:
            if rec_mask is not None:
                spk = spk[rec_mask[spk]]
            if spk.size:
                times.append(np.full(spk.size, k * DT))
                who.append(spk)
        if progress and k % 5000 == 0 and k:
            el = time.time() - t0
            print(f'\r  simulating {k * DT:5.1f}/{total * DT:.1f} s brain time '
                  f'({el / (k * DT):.1f}x slower than real time)', end='', flush=True)
    if progress:
        print()
    if not times:
        return {}
    t = np.concatenate(times)
    w = np.concatenate(who)
    order = np.argsort(w, kind='stable')
    t, w = t[order], w[order]
    u, starts = np.unique(w, return_index=True)
    return {int(n): t[a:b] for n, a, b in zip(u, starts, list(starts[1:]) + [len(w)])}


# ----------------------------------------------------------------------------
# Spikes -> drum hits
# ----------------------------------------------------------------------------
def burst_onsets(t, gap=0.04, window=0.02):
    """Spike train -> (onset times, spikes in the first `window` s of each burst)."""
    if len(t) == 0:
        return np.empty(0), np.empty(0, dtype=int)
    t = np.sort(t)
    on = t[np.r_[True, np.diff(t) > gap]]
    sizes = np.searchsorted(t, on + window, side='right') - np.searchsorted(t, on, side='left')
    return on, sizes


def resolve_neurons(conn, items):
    idx = []
    for it in items:
        if isinstance(it, str) and it in NAMED:
            idx.append(conn.index[NAMED[it]])
        else:
            idx.extend(conn.idx([int(it)]).tolist())
    return idx


def neuron_label(conn, i):
    f = int(conn.ids[i])
    for k, v in NAMED.items():
        if v == f:
            return k
    return str(f)


def pick_auto_voices(conn, spikes, taken, n_slots, exclude, step_s, n_events, min_hits=3):
    """Choose strong responders whose hit patterns differ from the voices already in the kit,
    preferring selective neurons (ones that answer some steps, not every one)."""
    def hitset(on):
        return set(np.round(on / step_s).astype(int).tolist())

    chosen_sets = []
    for idxs in taken:
        on = np.concatenate([burst_onsets(spikes.get(i, np.empty(0)))[0] for i in idxs]) if idxs else np.empty(0)
        if on.size:
            lat = np.median(on % step_s)
            chosen_sets.append(hitset(on - lat))
    cands = []
    for n, t in spikes.items():
        if n in exclude:
            continue
        on, sizes = burst_onsets(t)
        if on.size < min_hits:
            continue
        lat = np.median(on % step_s)
        frac = on.size / max(1, n_events)
        selectivity = 1.0 if frac <= 0.5 else 0.3
        cands.append((float(sizes.mean()) * np.sqrt(on.size) * selectivity, n, hitset(on - lat)))
    cands.sort(key=lambda c: -c[0])
    picks = []
    for score, n, hs in cands:
        if len(picks) == n_slots:
            break
        if all(len(hs ^ s) / max(1, len(hs | s)) >= 0.4 for s in chosen_sets):
            picks.append(n)
            chosen_sets.append(hs)
    return picks


def build_kit(conn, kit_spec, spikes, exclude, step_s, n_events):
    """kit_spec: {drum: [neuron names / FlyWire IDs] or [] for auto}. Returns [(drum, note, [idx])]."""
    kit, auto_slots = [], []
    for drum, items in kit_spec.items():
        note = GM[drum] if drum in GM else int(drum)
        if items:
            kit.append([drum, note, resolve_neurons(conn, items)])
        else:
            auto_slots.append((drum, note))
    if auto_slots:
        used = set(exclude) | {i for _, _, idxs in kit for i in idxs}
        picks = pick_auto_voices(conn, spikes, [k[2] for k in kit], len(auto_slots), used, step_s, n_events)
        # denser responders go to the drums listed first
        picks.sort(key=lambda n: -burst_onsets(spikes[n])[0].size)
        for (drum, note), n in zip(auto_slots, picks):
            kit.append([drum, note, [n]])
    return kit


def kit_hits(kit, spikes, step_s, timing='tight', gap=0.04):
    """Return sorted list of (time_s, midi_note, velocity) and a per-drum summary."""
    hits, summary = [], []
    for drum, note, idxs in kit:
        t = np.concatenate([spikes.get(i, np.empty(0)) for i in idxs]) if idxs else np.empty(0)
        on, sizes = burst_onsets(t, gap=gap)
        if on.size == 0:
            summary.append((drum, note, idxs, 0, None))
            continue
        lat = float(np.median(on % step_s))
        if timing in ('tight', 'grid'):
            on = on - lat
        if timing == 'grid':
            on = np.round(on / (step_s / 2)) * (step_s / 2)   # 32nd-note grid
        on = np.maximum(on, 0.0)
        vel = np.clip(127 * np.sqrt(sizes / sizes.max()), 45, 127)   # harder firing = harder hit
        last = -1.0
        for tt, vv in zip(on, vel):
            if tt - last >= gap * 0.75:
                hits.append((float(tt), note, int(round(vv))))
                last = tt
        summary.append((drum, note, idxs, on.size, lat * 1000))
    hits.sort()
    return hits, summary


# ----------------------------------------------------------------------------
# Output: MIDI file, WAV preview, live MIDI port
# ----------------------------------------------------------------------------
def _vlq(n):
    out = [n & 0x7F]
    n >>= 7
    while n:
        out.append((n & 0x7F) | 0x80)
        n >>= 7
    return bytes(reversed(out))


def write_midi(path, hits, bpm, note_len=0.06, name='fly brain drums'):
    tpq = 960
    ticks_per_s = tpq * bpm / 60.0
    ev = []
    for t, note, vel in hits:
        on = int(round(t * ticks_per_s))
        ev.append((on, 1, bytes([0x99, note, vel])))
        ev.append((on + int(note_len * ticks_per_s), 0, bytes([0x89, note, 0])))
    ev.sort(key=lambda e: (e[0], e[1]))
    us = int(round(60e6 / bpm))
    trk = bytearray()
    trk += b'\x00\xff\x03' + _vlq(len(name)) + name.encode()
    trk += b'\x00\xff\x51\x03' + us.to_bytes(3, 'big')
    trk += b'\x00\xff\x58\x04\x04\x02\x18\x08'
    last = 0
    for tick, _, msg in ev:
        trk += _vlq(tick - last) + msg
        last = tick
    trk += b'\x00\xff\x2f\x00'
    with open(path, 'wb') as f:
        f.write(b'MThd' + struct.pack('>IHHH', 6, 0, 1, tpq))
        f.write(b'MTrk' + struct.pack('>I', len(trk)) + trk)


def _drum_sound(note, sr, rng):
    def env(d, n):
        return np.exp(-np.arange(n) / (d * sr))

    def noise(n):
        return rng.standard_normal(n)

    def hp(x):
        return np.diff(x, prepend=0.0)

    n = int(0.6 * sr)
    t = np.arange(n) / sr
    if note == 36:      # kick
        f = 45 + 80 * np.exp(-t / 0.04)
        return np.sin(2 * np.pi * np.cumsum(f) / sr) * env(0.18, n) + 0.3 * noise(n) * env(0.003, n)
    if note == 38:      # snare
        return 0.6 * hp(noise(n)) * env(0.09, n) + 0.5 * np.sin(2 * np.pi * 185 * t) * env(0.05, n)
    if note == 39:      # clap
        x = np.zeros(n)
        for k in range(3):
            s = int(k * 0.011 * sr)
            x[s:] += hp(noise(n - s)) * env(0.008 if k < 2 else 0.12, n - s)
        return 0.7 * x
    if note in (42, 44, 46):   # hats
        d = {42: 0.025, 44: 0.05, 46: 0.25}[note]
        return 0.35 * hp(hp(noise(n))) * env(d, n)
    if note in (45, 47, 50, 63):  # toms / conga
        f0 = {45: 95, 47: 135, 50: 185, 63: 260}[note]
        f = f0 * (1 + 0.5 * np.exp(-t / 0.03))
        return np.sin(2 * np.pi * np.cumsum(f) / sr) * env(0.22 if note != 63 else 0.12, n)
    if note in (49, 51):        # crash / ride
        m = sum(np.sign(np.sin(2 * np.pi * fr * t)) for fr in (342, 513, 787, 1021, 1477))
        d = 0.5 if note == 49 else 0.35
        return 0.12 * hp(m + 2 * noise(n)) * env(d, n)
    if note in (54, 70):        # tambourine / shaker
        a = np.minimum(1, t / 0.006)
        return 0.3 * hp(hp(noise(n))) * a * env(0.05, n)
    if note == 56:              # cowbell
        x = np.sign(np.sin(2 * np.pi * 540 * t)) + np.sign(np.sin(2 * np.pi * 800 * t))
        return 0.15 * x * env(0.12, n)
    if note == 37:              # rim
        return 0.6 * np.sin(2 * np.pi * 1700 * t) * env(0.008, n)
    return 0.4 * hp(noise(n)) * env(0.03, n)


def write_wav(path, hits, sr=44100, seed=0):
    rng = np.random.default_rng(seed)
    if not hits:
        return
    length = int((hits[-1][0] + 1.0) * sr)
    out = np.zeros((length, 2))
    cache = {}
    notes = sorted({h[1] for h in hits})
    pan = {nt: (k / max(1, len(notes) - 1) - 0.5) * 0.7 for k, nt in enumerate(notes)}
    for t, note, vel in hits:
        if note not in cache:
            cache[note] = _drum_sound(note, sr, rng)
        s = cache[note] * (vel / 127.0) ** 1.5
        i = int(t * sr)
        e = min(length, i + s.size)
        p = pan[note] if note != 36 else 0.0
        out[i:e, 0] += s[:e - i] * (1 - p)
        out[i:e, 1] += s[:e - i] * (1 + p)
    out /= max(1e-9, np.abs(out).max()) / 0.9
    with wave.open(str(path), 'wb') as w:
        w.setnchannels(2)
        w.setsampwidth(2)
        w.setframerate(sr)
        w.writeframes((out * 32767).astype('<i2').tobytes())


def play_to_port(port_name, hits, loop=False, note_len=0.06):
    import mido   # pip install mido python-rtmidi
    if port_name in ('', '?', 'list'):
        print('MIDI outputs:', mido.get_output_names())
        return
    names = mido.get_output_names()
    match = [n for n in names if port_name.lower() in n.lower()]
    if not match:
        raise SystemExit(f'No MIDI output matching {port_name!r}. Available: {names}')
    with mido.open_output(match[0]) as out:
        print(f'Playing to {match[0]} (Ctrl-C to stop)')
        events = []
        for t, note, vel in hits:
            events.append((t, mido.Message('note_on', channel=9, note=note, velocity=vel)))
            events.append((t + note_len, mido.Message('note_off', channel=9, note=note, velocity=0)))
        events.sort(key=lambda e: e[0])
        span = hits[-1][0] + 0.5 if hits else 0
        try:
            while True:
                start = time.perf_counter()
                for t, msg in events:
                    d = start + t - time.perf_counter()
                    if d > 0:
                        time.sleep(d)
                    out.send(msg)
                if not loop:
                    break
                time.sleep(max(0.0, start + span - time.perf_counter()))
        except KeyboardInterrupt:
            pass


# ----------------------------------------------------------------------------
# Main
# ----------------------------------------------------------------------------
def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--data', default='.', help='folder of the Drosophila_brain_model clone (default: .)')
    ap.add_argument('--version', default='630', choices=['630', '783'],
                    help='FlyWire data version; the built-in neuron IDs are v630 (default)')
    ap.add_argument('--antenna', default='x.x.X.x.x.x.X.x.', help='antenna-touch pattern (16 steps per bar)')
    ap.add_argument('--sugar', default='X.....X...X..x..', help='sugar-taste pattern')
    ap.add_argument('--bpm', type=float, default=110)
    ap.add_argument('--bars', type=int, default=4)
    ap.add_argument('--stim-ms', type=float, default=40, help='length of each stimulus pulse (ms)')
    ap.add_argument('--soft-hz', type=float, default=100, help="stimulation rate for 'x' steps")
    ap.add_argument('--strong-hz', type=float, default=220, help="stimulation rate for 'X' steps")
    ap.add_argument('--kit', help='JSON file: {"kick": ["MN9_L"], "snare": [7205759...], "low_tom": []} '
                                  '(drum = GM name or MIDI note number; [] = pick a neuron automatically)')
    ap.add_argument('--timing', default='tight', choices=['raw', 'tight', 'grid'],
                    help="raw = the brain's own latencies; tight = each drum's average latency removed, "
                         "jitter kept (default); grid = snapped to 32nd notes")
    ap.add_argument('--seed', type=int, default=None, help='random seed (the fly never plays it the same twice)')
    ap.add_argument('--out', default='fly_drums.mid', help='MIDI file to write')
    ap.add_argument('--wav', help='also render a WAV preview with built-in synth drums')
    ap.add_argument('--port', help="play live to a MIDI output whose name contains this text ('list' to list)")
    ap.add_argument('--loop', action='store_true', help='with --port: loop until Ctrl-C')
    ap.add_argument('--list-responders', type=int, metavar='N', default=0,
                    help='print the N neurons that responded most, to help you build a --kit')
    ap.add_argument('--spikes', help='save every recorded spike to this .npz (for your own analysis)')
    a = ap.parse_args(argv)

    if a.port in ('list', '?'):
        play_to_port('list', [])
        return
    patterns = {'antenna': parse_pattern(a.antenna), 'sugar': parse_pattern(a.sugar)}
    kit_spec = DEFAULT_KIT
    if a.kit:
        kit_spec = json.loads(Path(a.kit).read_text())

    conn = load_connectome(a.data, a.version)
    brain = FlyBrain(conn, seed=a.seed)
    for ch, ids in INPUTS.items():
        brain.add_channel(ch, ids)
    step_s = 60.0 / a.bpm / 4
    print(f'Fly brain: {conn.n:,} neurons, {conn.post.size:,} connections. '
          f'Playing {a.bars} bars at {a.bpm:g} bpm into its antennae and taste neurons...')
    spikes = perform(brain, patterns, a.bpm, a.bars, a.stim_ms, a.soft_hz, a.strong_hz)
    inputs = set(np.concatenate([brain.channels[c] for c in brain.channels]).tolist())

    n_events = sum(any(p[s % len(p)] != '.' for p in patterns.values()) for s in range(a.bars * 16))
    kit = build_kit(conn, kit_spec, spikes, inputs, step_s, n_events)
    hits, summary = kit_hits(kit, spikes, step_s, a.timing)

    print('\nThe band:')
    for drum, note, idxs, n, lat in summary:
        who = ' + '.join(neuron_label(conn, i) for i in idxs) or '(no responder found)'
        lat_s = f'{lat:5.1f} ms' if lat is not None else '   -   '
        print(f'  {drum:<11} (note {note:>2})  <- {who:<42} {n:>3} hits, latency {lat_s}')
    if a.list_responders:
        print('\nTop responders (FlyWire ID, bursts, spikes, median latency ms):')
        rows = []
        for n, t in spikes.items():
            if n in inputs:
                continue
            on, _ = burst_onsets(t)
            rows.append((t.size, on.size, float(np.median(on % step_s)) * 1000, neuron_label(conn, n)))
        for sp, nb, lat, lab in sorted(rows, reverse=True)[:a.list_responders]:
            print(f'  {lab:<20} {nb:>4} bursts {sp:>5} spikes  {lat:5.1f} ms')
    if a.spikes:
        np.savez_compressed(a.spikes, flywire_id=np.array([conn.ids[n] for n in spikes for _ in spikes[n]]),
                            t=np.concatenate(list(spikes.values())) if spikes else np.empty(0))

    write_midi(a.out, hits, a.bpm)
    print(f'\nWrote {len(hits)} hits to {a.out}')
    if a.wav:
        write_wav(a.wav, hits)
        print(f'Wrote audio preview to {a.wav}')
    if a.port:
        play_to_port(a.port, hits, loop=a.loop)


if __name__ == '__main__':
    main()
