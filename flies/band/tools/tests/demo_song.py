"""An original 16-bar demo song (A minor, 96 bpm) with known notes, for testing the song ear."""
import json
import sys

import numpy as np

sys.path.insert(0, __import__('os').path.join(__import__('os').path.dirname(__file__), '..', 'data-prep'))
import fly_drums as fd

BPM, SR = 96, 44100
STEP = 60 / BPM / 4
CHORDS = ['Am', 'F', 'C', 'G', 'Am', 'F', 'Dm', 'E'] * 2
TRIAD = {'Am': (57, [0, 3, 7]), 'F': (53, [0, 4, 7]), 'C': (48, [0, 4, 7]), 'G': (55, [0, 4, 7]), 'Dm': (50, [0, 3, 7]), 'E': (52, [0, 4, 7])}
BASSROOT = {'Am': 45, 'F': 41, 'C': 48, 'G': 43, 'Dm': 50, 'E': 40}
KICK, SNARE, HAT = 'X.....X.X.......', '....X.......X...', 'x.x.x.x.x.x.x.x.'
FILL = '....X.......XXXX'
BASSPAT = [0, 0, None, 0, 12, 0, None, 7]
MEL = [[69, None, 72, None, 76, 74, 72, None], [69, None, None, None, 72, None, 69, 67], [67, None, 72, None, 76, None, 79, 76],
       [74, None, None, None, 71, None, 74, None], [76, None, 76, 74, 72, None, 69, None], [72, None, 69, None, 65, None, 69, None],
       [74, None, 77, None, 76, 74, 72, None], [71, None, 68, None, 71, None, 76, None]]


def hz(m): return 440 * 2 ** ((m - 69) / 12)


def tone(f, dur, kind, sr=SR):
    n = int(dur * sr); t = np.arange(n) / sr
    if kind == 'saw':
        ph = (f * t) % 1.0; x = 2 * ph - 1
    elif kind == 'tri':
        vib = 1 + 0.004 * np.sin(2 * np.pi * 5.2 * t)
        ph = np.cumsum(f * vib / sr) % 1.0; x = 2 * np.abs(2 * ph - 1) - 1
    else:
        x = np.sin(2 * np.pi * f * t)
    a = np.minimum(1, t / 0.01) * np.exp(-t / (dur * 0.9 + 0.05))
    return x * a


def lowpass(x, fc, sr=SR):
    a = np.exp(-2 * np.pi * fc / sr); y = np.zeros_like(x); acc = 0.0
    for i in range(len(x)):
        acc = (1 - a) * x[i] + a * acc; y[i] = acc
    return y


def render():
    bars = len(CHORDS); n = int((bars * 16 * STEP + 1.5) * SR)
    out = np.zeros(n); rng = np.random.default_rng(0)
    truth = {'bpm': BPM, 'bars': bars, 'kick': [], 'snare': [], 'hat': [], 'bass': [], 'mel': [], 'chords': []}
    drum_cache = {nt: fd._drum_sound(nt, SR, rng) for nt in (36, 38, 42)}
    for b, ch in enumerate(CHORDS):
        snare = FILL if b % 8 == 7 else SNARE
        for s in range(16):
            slot = b * 16 + s; t = slot * STEP; i = int(t * SR)
            for pat, nt, name in ((KICK, 36, 'kick'), (snare, 38, 'snare'), (HAT, 42, 'hat')):
                if pat[s] != '.':
                    v = 1.0 if pat[s] == 'X' else 0.7
                    d = drum_cache[nt] * v * 0.8; e = min(n, i + d.size); out[i:e] += d[:e - i]; truth[name].append(slot)
        # bass, 8ths
        for e8 in range(8):
            off = BASSPAT[e8]; t = (b * 16 + e8 * 2) * STEP
            if off is None: truth['bass'].append(None); continue
            m = BASSROOT[ch] + off; x = lowpass(tone(hz(m), STEP * 1.8, 'saw'), 500) * 0.55
            i = int(t * SR); e = min(n, i + x.size); out[i:e] += x[:e - i]; truth['bass'].append(m)
        # chord pad, whole bar
        root, iv = TRIAD[ch]; t = b * 16 * STEP; i = int(t * SR)
        x = sum(lowpass(tone(hz(root + d), 16 * STEP, 'saw'), 1200) for d in iv) * 0.09
        e = min(n, i + x.size); out[i:e] += x[:e - i]
        truth['chords'] += [ch] * 4
        # melody, 8ths
        for e8, m in enumerate(MEL[b % 8]):
            if m is None: truth['mel'].append(None); continue
            t = (b * 16 + e8 * 2) * STEP; x = tone(hz(m), STEP * 1.9, 'tri') * 0.45
            i = int(t * SR); e = min(n, i + x.size); out[i:e] += x[:e - i]; truth['mel'].append(m)
    out /= np.abs(out).max() / 0.9
    return out, truth


if __name__ == '__main__':
    x, truth = render()
    from scipy.signal import resample_poly
    x22 = resample_poly(x, 1, 2)
    x22.astype('<f8').tofile('/tmp/demo22.f64')
    json.dump(truth, open('/tmp/demo_truth.json', 'w'))
    import wave
    with wave.open('/tmp/demo_song.wav', 'wb') as w:
        w.setnchannels(1); w.setsampwidth(2); w.setframerate(SR); w.writeframes((x * 32767).astype('<i2').tobytes())
    print('demo song', round(len(x) / SR, 1), 's; slots', len(truth['chords']) * 4)
