"""Test song with a voice-like lead: vibrato, glides between notes, formants, breathy consonants. Stems kept."""
import sys, wave
import numpy as np
from scipy.signal import resample_poly, butter, sosfilt
HERE = __import__('os').path.dirname(__import__('os').path.abspath(__file__))
sys.path[:0] = [HERE, __import__('os').path.join(HERE, '..', 'data-prep')]   # demo_song.py is here, fly_drums.py (drum sounds) in data-prep
import demo_song as ds, fly_drums as fd
SR = ds.SR
TAU = float(sys.argv[1]) if len(sys.argv) > 1 else 0
TAG = sys.argv[2] if len(sys.argv) > 2 else 'sv'

def voice(notes, n):
    """notes: list of (start_sample, midi, dur_samples). One continuous singer: pitch glides, vibrato, formants."""
    f = np.zeros(n); amp = np.zeros(n); rng = np.random.default_rng(3); cons = np.zeros(n)
    for (i, m, d) in notes:
        e = min(n, i + d); f[i:e] = ds.hz(m)
        a = np.ones(e - i); k = int(0.03 * SR); a[:k] = np.linspace(0, 1, k); r = int(0.06 * SR); a[-r:] *= np.linspace(1, 0, r); amp[i:e] = np.maximum(amp[i:e], a)
        if rng.random() < 0.45:   # a consonant: short hiss at the start of the note
            c = int(0.05 * SR); s0 = max(0, i - c // 2); cons[s0:s0 + c] += rng.standard_normal(min(c, n - s0)) * np.hanning(c)[:min(c, n - s0)] * 0.5
    last = 220.0
    for i in range(n):   # hold the last pitch through gaps
        if f[i] > 0: last = f[i]
        else: f[i] = last
    g = np.exp(-1 / (0.025 * SR)); fs = np.empty(n); acc = f[0]
    for i in range(n): acc = acc * g + f[i] * (1 - g); fs[i] = acc
    t = np.arange(n) / SR; depth = 0.022 * np.clip(np.cumsum(amp) * 0 + 1, 0, 1)   # about +-38 cents
    fs = fs * (1 + depth * np.sin(2 * np.pi * 5.6 * t + 0.4 * np.sin(2 * np.pi * 0.7 * t)))
    ph = 2 * np.pi * np.cumsum(fs) / SR; x = np.zeros(n)
    for h in range(1, 26):
        fh = fs * h; gain = (np.exp(-((fh - 650) / 220) ** 2) + 0.7 * np.exp(-((fh - 1150) / 280) ** 2) + 0.45 * np.exp(-((fh - 2700) / 450) ** 2) + 0.12 / h)
        x += gain * np.sin(h * ph) * (fh < SR * 0.45)
    x *= amp * (1 + 0.08 * np.sin(2 * np.pi * 4.1 * t))
    x += sosfilt(butter(4, 3500, 'high', fs=SR, output='sos'), cons)
    return x / np.abs(x).max()

def render():
    bars = len(ds.CHORDS); n = int((bars * 16 * ds.STEP + 1.5) * SR)
    st = {k: np.zeros((2, n)) for k in ('drums', 'bass', 'vocal', 'keys')}
    rng = np.random.default_rng(0); dc = {nt: fd._drum_sound(nt, SR, rng) for nt in (36, 38, 42)}; pan = {36: (0.5, 0.5), 38: (0.55, 0.45), 42: (0.3, 0.7)}
    if TAU: t = np.arange(int(0.6 * SR)) / SR; dc[36] = np.sin(2 * np.pi * np.cumsum(50 + 90 * np.exp(-t / 0.03)) / SR) * np.exp(-t / TAU) + 0.3 * rng.standard_normal(t.size) * np.exp(-t / 0.003)   # a tighter kick
    def put(part, x, i, gl=0.5, gr=0.5, delay_r=0):
        e = min(n, i + x.size); st[part][0, i:e] += gl * 2 * x[:e - i]
        j = i + delay_r; e = min(n, j + x.size)
        if j < n: st[part][1, j:e] += gr * 2 * x[:e - j]
    notes = []
    for b, ch in enumerate(ds.CHORDS):
        snare = ds.FILL if b % 8 == 7 else ds.SNARE
        for s in range(16):
            i = int((b * 16 + s) * ds.STEP * SR)
            for pat, nt in ((ds.KICK, 36), (snare, 38), (ds.HAT, 42)):
                if pat[s] != '.': put('drums', dc[nt] * (1.0 if pat[s] == 'X' else 0.7) * 0.8, i, *pan[nt])
        for e8 in range(8):
            off = ds.BASSPAT[e8]
            if off is None: continue
            put('bass', ds.lowpass(ds.tone(ds.hz(ds.BASSROOT[ch] + off), ds.STEP * 1.8, 'saw'), 500) * 0.55, int((b * 16 + e8 * 2) * ds.STEP * SR))
        root, iv = ds.TRIAD[ch]; i = int(b * 16 * ds.STEP * SR)
        for d, (gl, gr) in zip(iv, ((0.75, 0.25), (0.5, 0.5), (0.25, 0.75))):
            for side, cents in ((0, -7), (1, 7)):
                x = ds.lowpass(ds.tone(ds.hz(root + d + cents / 100), 16 * ds.STEP, 'saw'), 1200) * 0.09
                if side == 0: put('keys', x, i, gl, 0)
                else: put('keys', x, i, 0, gr, delay_r=int(0.011 * SR))
        mel = ds.MEL[b % 8]
        for e8, m in enumerate(mel):
            if m is None: continue
            ln = 1
            while e8 + ln < 8 and mel[e8 + ln] is None and ln < 3: ln += 1
            notes.append((int((b * 16 + e8 * 2) * ds.STEP * SR), m, int(ln * 2 * ds.STEP * SR * 0.95)))
    v = voice(notes, n) * 0.42; st['vocal'][0] += v; st['vocal'][1] += v
    mix = sum(st.values()); pk = np.abs(mix).max() / 0.9
    return mix / pk, {k: v / pk for k, v in st.items()}

if __name__ == '__main__':
    mix, stems = render(); r = lambda x: resample_poly(x, 1, 2).astype('<f4')
    r(mix[0]).tofile(f'/tmp/{TAG}_L.f32'); r(mix[1]).tofile(f'/tmp/{TAG}_R.f32')
    for k, v in stems.items(): r(v[0]).tofile(f'/tmp/{TAG}_{k}_L.f32'); r(v[1]).tofile(f'/tmp/{TAG}_{k}_R.f32')
    with wave.open(f'/tmp/demo_{TAG}.wav', 'wb') as w:
        w.setnchannels(2); w.setsampwidth(2); w.setframerate(SR); w.writeframes((mix.T * 32767).astype('<i2').tobytes())
    print('voice demo', round(mix.shape[1] / SR, 1), 's; stem rms', {k: round(float(np.sqrt((v ** 2).mean())), 3) for k, v in stems.items()})
