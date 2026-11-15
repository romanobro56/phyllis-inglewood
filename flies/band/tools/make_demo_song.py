"""Render the demo song the world's band plays: tools/tests/demo_song.py (an original 16-bar piece in A minor,
96 bpm) mixed in stereo, so the four-way split has a stereo picture to work with: drums, bass and melody in
the centre, the chord pad spread wide.

    python flies/band/tools/make_demo_song.py          # -> flies/band/demo-song.mp3 (needs ffmpeg)
"""
import subprocess
import sys
import wave
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE / "data-prep"))
sys.path.insert(0, str(HERE / "tests"))
import fly_drums as fd  # noqa: E402
import demo_song as ds  # noqa: E402


def render():
    SR, STEP = ds.SR, ds.STEP
    bars = len(ds.CHORDS); n = int((bars * 16 * STEP + 1.5) * SR)
    out = np.zeros((n, 2)); rng = np.random.default_rng(0)

    def add(x, i, pan):                       # pan -1 (left) .. 1 (right), constant power
        g = np.array([np.cos((pan + 1) * np.pi / 4), np.sin((pan + 1) * np.pi / 4)]) * np.sqrt(2)
        e = min(n, i + x.size); out[i:e] += x[:e - i, None] * g

    drums = {nt: fd._drum_sound(nt, SR, rng) for nt in (36, 38, 42)}
    for b, ch in enumerate(ds.CHORDS):
        snare = ds.FILL if b % 8 == 7 else ds.SNARE
        for s in range(16):
            i = int((b * 16 + s) * STEP * SR)
            for pat, nt, pan in ((ds.KICK, 36, 0), (snare, 38, 0.05), (ds.HAT, 42, 0.3)):
                if pat[s] != '.':
                    add(drums[nt] * (1.0 if pat[s] == 'X' else 0.7) * 0.8, i, pan)
        for e8 in range(8):
            off = ds.BASSPAT[e8]
            if off is not None:
                m = ds.BASSROOT[ch] + off
                add(ds.lowpass(ds.tone(ds.hz(m), STEP * 1.8, 'saw'), 500) * 0.55, int((b * 16 + e8 * 2) * STEP * SR), 0)
        root, iv = ds.TRIAD[ch]
        for d, pan in zip(iv, (-0.85, 0.85, -0.5)):   # the pad, wide
            add(ds.lowpass(ds.tone(ds.hz(root + d), 16 * STEP, 'saw'), 1200) * 0.11, int(b * 16 * STEP * SR), pan)
        for e8, m in enumerate(ds.MEL[b % 8]):
            if m is not None:
                add(ds.tone(ds.hz(m), STEP * 1.9, 'tri') * 0.45, int((b * 16 + e8 * 2) * STEP * SR), 0)
    return out / (np.abs(out).max() / 0.9), SR


if __name__ == '__main__':
    x, sr = render()
    wav = HERE.parent / 'demo-song.wav'
    with wave.open(str(wav), 'wb') as w:
        w.setnchannels(2); w.setsampwidth(2); w.setframerate(sr); w.writeframes((x * 32767).astype('<i2').tobytes())
    mp3 = HERE.parent / 'demo-song.mp3'
    subprocess.run(['ffmpeg', '-loglevel', 'error', '-y', '-i', str(wav), '-b:a', '160k', str(mp3)], check=True)
    wav.unlink()
    print(f'{mp3}: {len(x) / sr:.1f} s')
