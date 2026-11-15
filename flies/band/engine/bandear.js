// honest.js - a line of flies passing a sound along, with as little invented as possible.
//
// Ear: each antenna is one mechanical resonator (Goepfert & Robert): its best frequency slides from ~200 Hz for
// faint sound to ~800 Hz for loud sound, and faint sound is amplified. Its movement stretches five classes of
// Johnston's organ neurons: A (fast vibration) and B (slow vibration) hear; C/D (pushed forward) and E (pushed
// back) feel wind and gravity and only join in when the antenna swings more than ~200 nm.
// Brain: the FlyWire wiring with the Shiu et al. neuron model, both antennae.
// Song: the only way the brain talks to the body is its descending neurons. The wing circuit they drive is
// not in this data set, so a small stand-in sings real fly song (pulses 35 ms apart, a ~150 Hz hum) as hard as
// the descending neurons on that side are firing. Nothing is fitted to the song.
(function (root) {
'use strict';
const isNode = typeof module !== 'undefined' && module.exports && typeof require !== 'undefined';
const FS = isNode ? require('./flysim.js') : root.FlySim;
const SR = 22050, STEP = 0.0005, SUB = 5;          // stimulus step (s) and brain steps (0.1 ms) per stimulus step
const NCH = 14;                                     // per antenna: A+, A-, B+, B-, C, D, E
const clamp = (v, a, b) => v < a ? a : v > b ? b : v;

function loadNet(J) {
  const b = FS.b64, net = { n: J.n, indptr: b(J.indptr, Int32Array), post: b(J.post, Uint16Array), c: J.consts, meta: J.meta, gf: J.gf, gfSide: J.gfSide, groups: {} };
  const wc = b(J.w, Int16Array); net.w = new Float64Array(wc.length); for (let i = 0; i < wc.length; i++) net.w[i] = wc[i] * J.consts.w_syn;
  for (const k in J.groups) net.groups[k] = b(J.groups[k], Int32Array);
  net.dnL = b(J.dnL, Int32Array); net.dnR = b(J.dnR, Int32Array);
  net.kind = new Uint8Array(net.n);                 // 1 hearing-organ neuron, 2 left descending, 3 right descending
  for (const k in net.groups) for (const i of net.groups[k]) net.kind[i] = 1;
  for (const i of net.dnL) net.kind[i] = 2; for (const i of net.dnR) net.kind[i] = 3;
  return net;
}
// The same brain with its wiring scrambled: every connection keeps its strength and its sender but lands on a
// random neuron. Used to ask whether the fly's actual wiring matters.
function scramble(net, seed) {
  const r = FS.mulberry32(seed), post = Uint16Array.from(net.post);
  for (let i = post.length - 1; i > 0; i--) { const j = Math.floor(r() * (i + 1)), t = post[i]; post[i] = post[j]; post[j] = t; }
  return Object.assign({}, net, { post });
}

// ---------------------------------------------------------------- choosing what to listen to
function pickWindow(L, R, seconds) {   // the loudest stretch of the song
  const hop = Math.round(SR / 2), nb = Math.floor(L.length / hop), e = new Float64Array(nb + 1), W = Math.min(nb, Math.round(seconds * 2));
  for (let b = 0; b < nb; b++) { let s = 0; for (let i = b * hop; i < (b + 1) * hop; i += 4) s += L[i] * L[i] + R[i] * R[i]; e[b + 1] = e[b] + s; }
  let best = 0, bs = -1; for (let b = 0; b + W <= nb; b++) { const v = e[b + W] - e[b]; if (v > bs) { bs = v; best = b; } }
  return { start: best * hop, len: Math.min(L.length - best * hop, W * hop) };
}

// ---------------------------------------------------------------- the ear
const EAR = {
  loud: 1.5,                 // mm/s air movement in the sound's loud moments: roughly 90 dB, or a singing fly a few mm away
  bfHi: 800, bfLo: 200, slope: 11, Q: 1.24, ampMax: 20,   // best frequency 800 Hz at >= 1 mm/s, falling 11 Hz per dB; up to x10 boost when faint
  splitHz: 100,              // class B follows movement below ~100 Hz, class A above
  thr: 5, half: 80, rMax: 260,          // nm of movement to start firing / reach half rate; top firing rate (Hz)
  ceThr: 200, ceHalf: 300, ceMax: 200,  // classes C, D, E only respond to swings beyond ~200 nm
};
function loudness(L, R) {   // 95th percentile of 50 ms loudness, both channels
  const w = Math.round(SR * 0.05), v = [];
  for (let o = 0; o + w <= L.length; o += w) { let s = 0; for (let i = o; i < o + w; i++) s += L[i] * L[i] + R[i] * R[i]; v.push(Math.sqrt(s / (2 * w))); }
  v.sort((a, b) => a - b); return v.length ? v[Math.floor(0.95 * (v.length - 1))] || 1e-9 : 1e-9;
}
// One antenna. x: sound at 22050 Hz; V0: mm/s per unit. Returns its movement (nm) and the firing rates it causes.
function antenna(x, V0) {
  const n = x.length, disp = new Float32Array(n), nS = Math.ceil(n / SR / STEP), rates = new Float32Array(nS * 7), cnt = new Uint16Array(nS);
  const E = EAR, aLev = 1 - Math.exp(-1 / (0.03 * SR)), leak = 1 - 2 * Math.PI * 5 / SR, k = 1 / E.Q;
  const lp = f => 1 - Math.exp(-2 * Math.PI * f / SR), aB = lp(E.splitHz), aS = lp(150);
  let lev2 = 1e-6, g = 0, a1 = 0, a2 = 0, a3 = 0, gain = 1, ic1 = 0, ic2 = 0, pos = 0, b1 = 0, b2 = 0, xs = 0, bfSum = 0;
  const fire = u => { u = (u - E.thr) / E.half; return u > 0 ? E.rMax * u / (1 + u) : 0; };
  const fireCE = u => { u = (u - E.ceThr) / E.ceHalf; return u > 0 ? E.ceMax * u / (1 + u) : 0; };
  for (let i = 0; i < n; i++) {
    const v = x[i] * V0; lev2 += aLev * (v * v - lev2);
    if ((i & 31) === 0) {   // the antenna retunes itself to how loud it is
      const dB = 10 * Math.log10(lev2 + 1e-9), f0 = clamp(E.bfHi + E.slope * dB, E.bfLo, E.bfHi);
      g = Math.tan(Math.PI * f0 / SR); a1 = 1 / (1 + g * (g + k)); a2 = g * a1; a3 = g * a2; gain = Math.pow(10, clamp(-dB * E.ampMax / 55, 0, E.ampMax) / 20); bfSum += f0;
    }
    const v3 = v - ic2, v1 = a1 * ic1 + a2 * v3, v2 = ic2 + a2 * ic1 + a3 * v3; ic1 = 2 * v1 - ic1; ic2 = 2 * v2 - ic2;
    pos = pos * leak + k * v1 * gain * 1e6 / SR;        // antenna velocity (mm/s) summed into movement (nm)
    disp[i] = pos;
    b1 += aB * (pos - b1); b2 += aB * (b1 - b2); xs += aS * (pos - xs);
    const xA = pos - b2, xB = b2, s = Math.min(nS - 1, Math.floor(i / SR / STEP)), o = s * 7;
    rates[o] += fire(xA); rates[o + 1] += fire(-xA); rates[o + 2] += fire(xB); rates[o + 3] += fire(-xB);
    const f = fireCE(xs); rates[o + 4] += f; rates[o + 5] += f; rates[o + 6] += fireCE(-xs); cnt[s]++;
  }
  for (let s = 0; s < nS; s++) if (cnt[s]) for (let c = 0; c < 7; c++) rates[s * 7 + c] /= cnt[s];
  return { disp, rates, nS, bf: bfSum / Math.ceil(n / 32) };
}
// Both antennae; the left channel moves the left antenna. opt.level: the sound is played so that its loud moments
// move the air at that many mm/s. opt.v0: a fixed mm/s per unit instead (used for a fly's song, whose loudness is real).
function hear(L, R, opt = {}) {
  const V0 = opt.v0 || (opt.level || EAR.loud) / loudness(L, R), a = antenna(L, V0), b = antenna(R, V0), nS = a.nS, rates = new Float32Array(nS * NCH);
  for (let s = 0; s < nS; s++) for (let c = 0; c < 7; c++) { rates[s * NCH + c] = a.rates[s * 7 + c]; rates[s * NCH + 7 + c] = b.rates[s * 7 + c]; }
  return { nS, rates, dispL: a.disp, dispR: b.disp, bf: (a.bf + b.bf) / 2 };
}

// ---------------------------------------------------------------- the brain
function Grow(Type, cap = 1 << 16) { this.a = new Type(cap); this.n = 0; this.T = Type; }
Grow.prototype.push = function (v) { if (this.n === this.a.length) { const b = new this.T(this.n * 2); b.set(this.a); this.a = b; } this.a[this.n++] = v; };
Grow.prototype.done = function () { return this.a.slice(0, this.n); };
function channels(net, seed) {   // within classes A and B, half the neurons fire on each direction of swing
  const r = FS.mulberry32(seed), out = [];
  const halves = arr => { const a = Array.from(arr); for (let i = a.length - 1; i > 0; i--) { const j = Math.floor(r() * (i + 1)); [a[i], a[j]] = [a[j], a[i]]; } const h = a.length >> 1; return [Int32Array.from(a.slice(0, h)), Int32Array.from(a.slice(h))]; };
  for (const s of ['L', 'R']) out.push(...halves(net.groups['A' + s]), ...halves(net.groups['B' + s]), net.groups['C' + s], net.groups['D' + s], net.groups['E' + s]);
  return out;
}
// Run one brain on what its antennae felt. Returns what its descending neurons did (spikes per ms, each side),
// its giant-fibre spikes, which neurons fired, and a thinned list of spikes for drawing.
function runBrain(net, stim, seed, onProgress) {
  const br = new FS.Brain(net, seed); for (const g of channels(net, seed + 17)) br.addChannel(g);
  const nMs = Math.ceil(stim.nS * STEP * 1000), dL = new Float32Array(nMs), dR = new Float32Array(nMs), fired = new Uint32Array(net.n), rates = new Float64Array(NCH);
  const kind = net.kind, isGF = new Uint8Array(net.n); for (const i of net.gf) isGF[i] = 1;
  const sT = new Grow(Uint32Array), sN = new Grow(Uint16Array), gf = new Grow(Uint32Array), bT = new Grow(Uint32Array), bN = new Grow(Uint16Array);
  let total = 0, deep = 0, thin = 0;
  for (let s = 0; s < stim.nS; s++) {
    for (let c = 0; c < NCH; c++) rates[c] = stim.rates[s * NCH + c];
    for (let q = 0; q < SUB; q++) {
      const k = s * SUB + q, sp = br.step(rates);
      for (let z = 0; z < sp.length; z++) {
        const j = sp[z], kd = kind[j]; fired[j]++; total++;
        if (kd !== 1) { deep++; bT.push(k); bN.push(j); }
        if (kd >= 2) { const ms = Math.min(nMs - 1, Math.floor(k / 10)); if (kd === 2) dL[ms]++; else dR[ms]++; if (isGF[j]) gf.push(k); }
        if (kd !== 1 ? (thin++ % 3 === 0 || kd >= 2) : (thin++ % 12 === 0)) { sT.push(k); sN.push(j); }
      }
    }
    if (onProgress && s % 2000 === 0) onProgress(s / stim.nS);
  }
  let nFired = 0, nDN = 0; for (let i = 0; i < net.n; i++) if (fired[i]) { nFired++; if (kind[i] >= 2) nDN++; }
  return { dL, dR, nMs, gf: gf.done(), t: sT.done(), n: sN.done(), bT: bT.done(), bN: bN.done(), total, deep, nFired, nDN };
}

// ---------------------------------------------------------------- the wing
// Stand-in for the wing circuit in the body, which is not in this data set. One wing sings at a time, on the
// side whose descending neurons are busier. It sings as loud as those neurons are firing. When they are busier
// than they have been lately it makes pulse song (the loud mode: clicks 35 ms apart), otherwise the quieter hum,
// and a short hum follows each pulse train, as in real flies. WING.full is the descending activity (spikes per ms)
// that gives a full-strength wing beat; WING.reach is how hard a full-strength song moves the air at the next
// fly's antenna (mm/s), a few millimetres away.
const WING = { tau: 20, slow: 1500, floor: 0.02, up: 2.5, full: 0.6, reach: 4, ipi: 35, pulseHz: 280, sigma: 0.0022, sineHz: 150, rebound: 120, margin: 1.3, hold: 40, humGain: 0.35 };
function sing(dL, dR, nMs) {
  const W = WING, n = Math.round(nMs / 1000 * SR), out = [new Float32Array(n), new Float32Array(n)], a = 1 - Math.exp(-1 / W.tau), aS = 1 - Math.exp(-1 / W.slow);
  const pulses = [], hum = new Float32Array(nMs), side = new Uint8Array(nMs), drive = new Float32Array(nMs);
  let sL = 0, sR = 0, cur = 0, want = 0, last = -1e9, lastAmp = 0, hA = 0, mean = 0, started = false;
  for (let t = 0; t < nMs; t++) {
    sL += a * (dL[t] - sL); sR += a * (dR[t] - sR);
    const mine = cur ? sR : sL, other = cur ? sL : sR;
    want = other > W.margin * mine && other > W.floor ? want + 1 : 0;
    if (want >= W.hold && t - last >= W.ipi) { cur = 1 - cur; want = 0; }
    const d = cur ? sR : sL; drive[t] = d; side[t] = cur;
    if (!started && d > W.floor) { mean = d; started = true; }
    mean += aS * (d - mean);
    let target = 0;
    if (d > W.floor) {
      if (d >= W.up * mean) { if (t - last >= W.ipi) { lastAmp = Math.min(1, d / W.full); pulses.push([t, cur, lastAmp]); last = t; } }
      else target = W.humGain * Math.min(1, d / W.full);
    }
    if (t - last > W.ipi * 0.6 && t - last < W.rebound) target = Math.max(target, W.humGain * lastAmp * (1 - (t - last) / W.rebound));
    hA += 0.25 * (target - hA); hum[t] = hA;
  }
  const pan = (ch, s) => ch === s ? 1 : 0.5, half = Math.ceil(4 * W.sigma * SR);
  for (const [t, s, amp] of pulses) { const c = Math.round(t / 1000 * SR); for (let i = -half; i <= half; i++) { const p = c + i; if (p < 0 || p >= n) continue; const tt = i / SR, v = amp * Math.exp(-tt * tt / (2 * W.sigma * W.sigma)) * Math.sin(2 * Math.PI * W.pulseHz * tt + 0.6); out[0][p] += v * pan(0, s); out[1][p] += v * pan(1, s); } }
  let ph = 0; const dph = 2 * Math.PI * W.sineHz / SR;
  for (let i = 0; i < n; i++) { const tm = i / SR * 1000, t0 = Math.min(nMs - 1, Math.floor(tm)), t1 = Math.min(nMs - 1, t0 + 1), f = tm - t0, am = hum[t0] * (1 - f) + hum[t1] * f; ph += dph; if (am > 1e-5) { const v = am * Math.sin(ph), s = side[t0]; out[0][i] += v * pan(0, s); out[1][i] += v * pan(1, s); } }
  // out is the song at its real loudness (1 = the wing at full swing). For a person to listen to, it is turned
  // up to a comfortable level, but never by more than x6, so a fly that barely sang still sounds like it barely sang.
  const mag = []; for (let i = 0; i < n; i += 7) mag.push(Math.max(Math.abs(out[0][i]), Math.abs(out[1][i]))); mag.sort((x, y) => x - y);
  const top = mag[Math.floor(0.999 * (mag.length - 1))] || 0, gain = top > 1e-6 ? Math.min(6, 0.7 / top) : 1;
  return { L: out[0], R: out[1], gain, pulses, hum, side, drive, silent: top <= 1e-4 };
}

// ---------------------------------------------------------------- one fly: hears, its brain runs, it sings
function envelope(a, b, nMs, win = 20) {   // loudness per ms of a stereo signal at 22050 Hz, smoothed
  const e = new Float32Array(nMs), per = SR / 1000;
  for (let t = 0; t < nMs; t++) { let s = 0, c = 0; for (let i = Math.floor(t * per); i < Math.floor((t + 1) * per) && i < a.length; i++) { s += a[i] * a[i] + b[i] * b[i]; c++; } e[t] = c ? Math.sqrt(s / (2 * c)) : 0; }
  const o = new Float32Array(nMs); let acc = 0; for (let t = 0; t < nMs; t++) { acc += e[t]; if (t >= win) acc -= e[t - win]; o[t] = acc / Math.min(t + 1, win); }
  return o;
}
function corrLag(a, b, maxLag) {   // best correlation of b against a, with b allowed to run up to maxLag ms late
  let best = -2, bl = 0; const n = Math.min(a.length, b.length);
  for (let lag = 0; lag <= maxLag; lag += 2) {
    let sa = 0, sb = 0, saa = 0, sbb = 0, sab = 0, m = 0;
    for (let t = 0; t + lag < n; t += 2) { const x = a[t], y = b[t + lag]; sa += x; sb += y; saa += x * x; sbb += y * y; sab += x * y; m++; }
    const va = saa - sa * sa / m, vb = sbb - sb * sb / m, r = va > 0 && vb > 0 ? (sab - sa * sb / m) / Math.sqrt(va * vb) : 0;
    if (r > best) { best = r; bl = lag; }
  }
  return { r: best, lag: bl };
}
function fly(net, L, R, seed, onProgress, opt) {
  const ear = hear(L, R, opt), b = runBrain(net, ear, seed, onProgress), song = sing(b.dL, b.dR, b.nMs);
  return { ear, brain: b, song };
}

// ---------------------------------------------------------------- rebuilding the record from a fly's song
// A fly's song says two things: when it clicked, and how loud it hummed. Here the record supplies the sound and
// the song supplies when and how loud. The record is split into its hits and its held sounds; a hit is let
// through only where the fly clicked, and the held sounds are made as loud as the fly's hum says.
function fft(re, im) {
  const n = re.length;
  for (let i = 1, j = 0; i < n; i++) { let bit = n >> 1; for (; j & bit; bit >>= 1) j ^= bit; j ^= bit; if (i < j) { let t = re[i]; re[i] = re[j]; re[j] = t; t = im[i]; im[i] = im[j]; im[j] = t; } }
  for (let len = 2; len <= n; len <<= 1) {
    const ang = -2 * Math.PI / len, wr = Math.cos(ang), wi = Math.sin(ang), h = len >> 1;
    for (let i = 0; i < n; i += len) { let cr = 1, ci = 0; for (let k = 0; k < h; k++) { const a = i + k, b = a + h, vr = re[b] * cr - im[b] * ci, vi = re[b] * ci + im[b] * cr; re[b] = re[a] - vr; im[b] = im[a] - vi; re[a] += vr; im[a] += vi; const t = cr * wr - ci * wi; ci = cr * wi + ci * wr; cr = t; } }
  }
}
function runMedian(a, w, rank = 0.5) {   // sliding median (or another rank: 0.3 = the value 30% of the way up), window w (odd), shrinking at the ends
  const n = a.length, h = w >> 1, out = new Float32Array(n), s = [];
  const at = v => { let lo = 0, hi = s.length; while (lo < hi) { const m = (lo + hi) >> 1; if (s[m] < v) lo = m + 1; else hi = m; } return lo; };
  for (let i = 0; i <= Math.min(h, n - 1); i++) s.splice(at(a[i]), 0, a[i]);
  for (let i = 0; i < n; i++) { out[i] = s[Math.min(s.length - 1, Math.floor(rank * s.length))]; if (i + h + 1 < n) { const v = a[i + h + 1]; s.splice(at(v), 0, v); } if (i - h >= 0) s.splice(at(a[i - h]), 1); }
  return out;
}
// Split a stereo record into hits (sudden, broad in pitch) and held sounds (steady in pitch) by median filtering
// its spectrogram along time and along frequency. The two parts add back up to the record.
function stems(L, R) {
  const N = 1024, hop = 256, nb = N / 2 + 1, len = L.length, n = Math.max(1, Math.ceil((len - N) / hop) + 1), w = Float64Array.from({ length: N }, (_, i) => 0.5 - 0.5 * Math.cos(2 * Math.PI * i / N));
  const re = new Float64Array(N), im = new Float64Array(N);
  const spec = x => { const RE = new Float32Array(n * nb), IM = new Float32Array(n * nb); for (let f = 0; f < n; f++) { const o = f * hop; for (let i = 0; i < N; i++) { re[i] = (x[o + i] || 0) * w[i]; im[i] = 0; } fft(re, im); for (let k = 0; k < nb; k++) { RE[f * nb + k] = re[k]; IM[f * nb + k] = im[k]; } } return { RE, IM }; };
  const A = spec(L), B = spec(R), mag = new Float32Array(n * nb), mask = new Float32Array(n * nb), col = new Float32Array(n);
  for (let q = 0; q < n * nb; q++) mag[q] = Math.hypot(A.RE[q] + B.RE[q], A.IM[q] + B.IM[q]) / 2;
  for (let k = 0; k < nb; k++) { for (let f = 0; f < n; f++) col[f] = mag[f * nb + k]; const m = runMedian(col, 17); for (let f = 0; f < n; f++) mask[f * nb + k] = m[f]; }   // steady part
  for (let f = 0; f < n; f++) { const p = runMedian(mag.subarray(f * nb, (f + 1) * nb), 17); for (let k = 0; k < nb; k++) { const h2 = mask[f * nb + k] ** 2, p2 = p[k] * p[k]; mask[f * nb + k] = p2 / (h2 + p2 + 1e-12); } }
  const synth = (S, hits) => { const out = new Float32Array(len); for (let f = 0; f < n; f++) { const o = f * hop; for (let k = 0; k < nb; k++) { const m = hits ? mask[f * nb + k] : 1 - mask[f * nb + k]; re[k] = S.RE[f * nb + k] * m; im[k] = -S.IM[f * nb + k] * m; } for (let k = 1; k < N / 2; k++) { re[N - k] = re[k]; im[N - k] = -im[k]; } fft(re, im); for (let i = 0; i < N && o + i < len; i++) out[o + i] += re[i] / N * w[i] / 1.5; } return out; };
  const st = { hitL: synth(A, true), hitR: synth(B, true), heldL: synth(A, false), heldR: synth(B, false), nMs: Math.ceil(len / SR * 1000) };
  st.heldEnv = envelope(st.heldL, st.heldR, st.nMs, 50);
  const sorted = Float32Array.from(st.heldEnv).sort(); st.heldMid = sorted[sorted.length >> 1] || 1e-6;
  return st;
}
// What a song says about how hard its singer was driven, ms by ms (0..1): the hum's loudness, or the click's.
function decodeSong(song) {
  const nMs = song.hum.length, d = new Float32Array(nMs), o = new Float32Array(nMs);
  for (let t = 0; t < nMs; t++) d[t] = song.hum[t] / WING.humGain;
  for (const [t, , amp] of song.pulses) for (let q = t; q < Math.min(nMs, t + WING.ipi); q++) d[q] = Math.max(d[q], amp);
  let acc = 0; for (let t = 0; t < nMs; t++) { acc += d[t]; if (t >= 12) acc -= d[t - 12]; o[Math.max(0, t - 6)] = acc / Math.min(t + 1, 12); }
  return o;
}
function rebuild(st, song, lagMs) {   // lagMs: how late this fly's song runs behind the record
  const nMs = st.nMs, len = st.heldL.length, d = decodeSong(song), gate = new Float32Array(nMs), outL = new Float32Array(len), outR = new Float32Array(len);
  const live = Float32Array.from(d).filter(v => v > 0.02).sort(), ref = live.length ? live[live.length >> 1] : 0;
  const amps = song.pulses.map(p => p[2]).sort((a, b) => a - b), pRef = amps.length ? amps[amps.length >> 1] : 1;
  for (const [t, , amp] of song.pulses) { const a = clamp(amp / pRef, 0.5, 1.6); for (let q = Math.max(0, t - 25); q < Math.min(nMs, t + 70); q++) { const e = Math.min(1, (q - (t - 25)) / 8, (t + 70 - q) / 12); gate[q] = Math.max(gate[q], a * e); } }
  const floor = 0.3 * st.heldMid, at = (a, x) => { const i = Math.floor(x), f = x - i; return i < 0 ? a[0] : i + 1 >= a.length ? a[a.length - 1] : a[i] * (1 - f) + a[i + 1] * f; };
  if (ref > 0) for (let i = 0; i < len; i++) {
    const tm = i / SR * 1000, tf = tm + lagMs, held = Math.min(4, at(d, tf) / ref) * st.heldMid / Math.max(floor, at(st.heldEnv, tm)), g = at(gate, tf);
    outL[i] = clamp(st.heldL[i] * held + st.hitL[i] * g, -0.98, 0.98); outR[i] = clamp(st.heldR[i] * held + st.hitR[i] * g, -0.98, 0.98);
  } else for (let i = 0; i < len; i++) { const g = at(gate, i / SR * 1000 + lagMs); outL[i] = st.hitL[i] * g; outR[i] = st.hitR[i] * g; }
  return { L: outL, R: outR };
}

// ---------------------------------------------------------------- a band: the record in four parts
// Ordinary signal processing, not the fly: hits (drums) by median filtering the spectrogram; of the held sound,
// bass is what lies below ~220 Hz, vocals what sits in the centre of the stereo picture in the voice range, and
// keys whatever is left. In a mono file "vocals" is simply the held sound in the voice range.
const PARTS = ['drums', 'bass', 'vocals', 'keys'];
function split4old(L, R) {
  const N = 1024, hop = 256, nb = N / 2 + 1, BIN = SR / N, len = L.length, n = Math.max(1, Math.ceil((len - N) / hop) + 1), w = Float64Array.from({ length: N }, (_, i) => 0.5 - 0.5 * Math.cos(2 * Math.PI * i / N));
  const re = new Float64Array(N), im = new Float64Array(N), cells = n * nb;
  const spec = x => { const RE = new Float32Array(cells), IM = new Float32Array(cells); for (let f = 0; f < n; f++) { const o = f * hop; for (let i = 0; i < N; i++) { re[i] = (x[o + i] || 0) * w[i]; im[i] = 0; } fft(re, im); for (let k = 0; k < nb; k++) { RE[f * nb + k] = re[k]; IM[f * nb + k] = im[k]; } } return { RE, IM }; };
  const A = spec(L), B = spec(R), mag = new Float32Array(cells), mE = new Float32Array(cells), sE = new Float32Array(cells), H = new Float32Array(cells), col = new Float32Array(n);
  let me = 0, se = 0;
  for (let q = 0; q < cells; q++) { const mr = (A.RE[q] + B.RE[q]) / 2, mi = (A.IM[q] + B.IM[q]) / 2, sr = (A.RE[q] - B.RE[q]) / 2, si = (A.IM[q] - B.IM[q]) / 2; mE[q] = mr * mr + mi * mi; sE[q] = sr * sr + si * si; mag[q] = Math.sqrt(mE[q]); me += mE[q]; se += sE[q]; }
  const mono = se < 1e-3 * me;
  for (let k = 0; k < nb; k++) { for (let f = 0; f < n; f++) col[f] = mag[f * nb + k]; const m = runMedian(col, 17); for (let f = 0; f < n; f++) H[f * nb + k] = m[f]; }
  const M = PARTS.map(() => new Float32Array(cells)), lp = new Float32Array(nb), bp = new Float32Array(nb), vr = new Float32Array(nb);
  for (let k = 0; k < nb; k++) { const fz = k * BIN; lp[k] = clamp((260 - fz) / 100, 0, 1); bp[k] = clamp((fz - 150) / 100, 0, 1) * clamp((6000 - fz) / 2500, 0, 1); vr[k] = clamp((fz - 300) / 200, 0, 1) * clamp((3400 - fz) / 800, 0, 1); }
  for (let f = 0; f < n; f++) {
    const p = runMedian(mag.subarray(f * nb, (f + 1) * nb), 17);
    for (let k = 0; k < nb; k++) {
      const q = f * nb + k, h2 = H[q] * H[q], p2 = p[k] * p[k], pm = p2 / (h2 + p2 + 1e-12), hm = 1 - pm; let cm;
      if (mono) cm = vr[k]; else { let ms = 0, ss = 0; for (let d = -2; d <= 2; d++) { const g = f + d; if (g >= 0 && g < n) { ms += mE[g * nb + k]; ss += sE[g * nb + k]; } } cm = Math.pow(clamp((ms / (ms + ss + 1e-12) - 0.6) / 0.35, 0, 1), 1.5); }
      const bass = hm * lp[k], voc = hm * (1 - lp[k]) * bp[k] * cm;
      M[0][q] = pm; M[1][q] = bass; M[2][q] = voc; M[3][q] = Math.max(0, hm - bass - voc);
    }
  }
  const synth = (S, m) => { const out = new Float32Array(len); for (let f = 0; f < n; f++) { const o = f * hop; for (let k = 0; k < nb; k++) { re[k] = S.RE[f * nb + k] * m[f * nb + k]; im[k] = -S.IM[f * nb + k] * m[f * nb + k]; } for (let k = 1; k < N / 2; k++) { re[N - k] = re[k]; im[N - k] = -im[k]; } fft(re, im); for (let i = 0; i < N && o + i < len; i++) out[o + i] += re[i] / N * w[i] / 1.5; } return out; };
  const apply = (l, r) => { const a = spec(l), b = spec(r); return M.map(m => ({ L: synth(a, m), R: synth(b, m) })); };
  return { mono, parts: M.map(m => ({ L: synth(A, m), R: synth(B, m) })), apply };
}

// The split, in two passes (median-filter separation of held sound from hits, run at two window lengths).
// Pass 1 looks through long windows, which see pitch finely: whatever holds its pitch for about half a second
// is held sound, cleanly lifted out from between the drum hits. Pass 2 looks at what is left through short
// windows, which see time finely: there a voice still changes smoothly from one instant to the next while a
// drum hit does not, so singing that wobbled too much for pass 1 is kept out of the drums. Held and smooth
// sound is then shared out: low and in the centre of the stereo picture is bass, centred in the voice range
// is vocals, the rest is keys. Down low, where kick and bass live together, a pass in between decides, with
// windows fine enough in pitch to tell their notes apart: sound that keeps its level for a few tenths of a
// second is bass, sound that dies away is kick drum.
const TUNE = { n1: 4096, t1: 0.42, f1: 500, beta1: 2, nL: 2048, tLow: 0.3, lowRank: 0.4, lowBeta: 1.3, n2: 512, t2: 0.2, f2: 500, beta: 2, betaSide: 2, hiss: 5000, xLo: 200, xHi: 320 };
const odd = v => { const k = Math.max(3, Math.round(v)); return k % 2 ? k : k + 1; };
const hannCache = {};
const hannW = N => hannCache[N] || (hannCache[N] = Float64Array.from({ length: N }, (_, i) => 0.5 - 0.5 * Math.cos(2 * Math.PI * i / N)));
function spectra(x, N) {
  const hop = N / 4, nb = N / 2 + 1, n = Math.max(1, Math.ceil((x.length - N) / hop) + 1), RE = new Float32Array(n * nb), IM = new Float32Array(n * nb), w = hannW(N), re = new Float64Array(N), im = new Float64Array(N);
  for (let f = 0; f < n; f++) { const o = f * hop; for (let i = 0; i < N; i++) { re[i] = (x[o + i] || 0) * w[i]; im[i] = 0; } fft(re, im); for (let k = 0; k < nb; k++) { RE[f * nb + k] = re[k]; IM[f * nb + k] = im[k]; } }
  return { RE, IM, n, nb, N, hop };
}
function resynth(S, mask, len) {
  const { N, hop, nb, n } = S, w = hannW(N), re = new Float64Array(N), im = new Float64Array(N), out = new Float32Array(len);
  for (let f = 0; f < n; f++) { const o = f * hop; for (let k = 0; k < nb; k++) { const m = mask[f * nb + k]; re[k] = S.RE[f * nb + k] * m; im[k] = -S.IM[f * nb + k] * m; } for (let k = 1; k < N / 2; k++) { re[N - k] = re[k]; im[N - k] = -im[k]; } fft(re, im); for (let i = 0; i < N && o + i < len; i++) out[o + i] += re[i] / N * w[i] / 1.5; }
  return out;
}
// One pass over a stereo sound: its spectra, how steady each cell is in time (Ht) and how sudden (Pf), and
// how central it sits in the stereo picture (0 off to a side .. 1 dead centre).
function look(l, r, N, wtOf, wf, W, rankOf = () => 0.5) {
  const A = spectra(l, N), B = spectra(r, N), n = A.n, nb = A.nb, cells = n * nb, mag = new Float32Array(cells), mE = new Float32Array(cells), sE = new Float32Array(cells);
  let me = 0, se = 0;
  for (let q = 0; q < cells; q++) { const mr = (A.RE[q] + B.RE[q]) / 2, mi = (A.IM[q] + B.IM[q]) / 2, sr = (A.RE[q] - B.RE[q]) / 2, si = (A.IM[q] - B.IM[q]) / 2; mE[q] = mr * mr + mi * mi; sE[q] = sr * sr + si * si; mag[q] = Math.sqrt(mE[q]); me += mE[q]; se += sE[q]; }
  const Ht = new Float32Array(cells), Pf = new Float32Array(cells), centre = new Float32Array(cells), col = new Float32Array(n), cM = new Float64Array(n + 1), cS = new Float64Array(n + 1);
  for (let k = 0; k < nb; k++) {
    for (let f = 0; f < n; f++) { col[f] = mag[f * nb + k]; cM[f + 1] = cM[f] + mE[f * nb + k]; cS[f + 1] = cS[f] + sE[f * nb + k]; }
    const m = runMedian(col, wtOf(k), rankOf(k));
    for (let f = 0; f < n; f++) { Ht[f * nb + k] = m[f]; const a = Math.max(0, f - W), b = Math.min(n, f + W + 1), ms = cM[b] - cM[a], ss = cS[b] - cS[a]; centre[f * nb + k] = ms / (ms + ss + 1e-12); }
  }
  for (let f = 0; f < n; f++) Pf.set(runMedian(mag.subarray(f * nb, (f + 1) * nb), wf), f * nb);
  return { A, B, n, nb, cells, mag, Ht, Pf, centre, mono: se < 1e-3 * me };
}
function split4(L, R) {
  const T = TUNE, len = L.length, sub = (a, b) => { const o = new Float32Array(a.length); for (let i = 0; i < o.length; i++) o[i] = a[i] - b[i]; return o; }, add = (a, b) => { const o = new Float32Array(a.length); for (let i = 0; i < o.length; i++) o[i] = a[i] + b[i]; return o; };
  const share = (fz, c, mono) => {   // of held or smooth sound at this pitch and stereo position: how much is bass, how much vocals
    const lo = clamp((T.xHi - fz) / (T.xHi - T.xLo), 0, 1), range = clamp((fz - 150) / 100, 0, 1) * clamp((6000 - fz) / 2500, 0, 1);
    return [lo * (mono ? 1 : clamp((c - 0.5) / 0.3, 0, 1)), (1 - lo) * range * (mono ? 1 : Math.pow(clamp((c - 0.6) / 0.35, 0, 1), 1.5))];
  };
  const pair = (S, D, m) => [resynth(S, m, len), resynth(D, m, len)];
  // pass 1: long windows, held sound above the kick-and-bass range
  const bin1 = SR / T.n1, hop1 = T.n1 / 4 / SR, P1 = look(L, R, T.n1, () => odd(T.t1 / hop1), odd(T.f1 / bin1), 1), mono = P1.mono;
  const held = new Float32Array(P1.cells), heldBass = new Float32Array(P1.cells), heldVoc = new Float32Array(P1.cells);
  for (let f = 0; f < P1.n; f++) for (let k = 0; k < P1.nb; k++) { const q = f * P1.nb + k, h2 = P1.Ht[q] ** 2, p2 = P1.Pf[q] ** 2, m = h2 / (h2 + T.beta1 * T.beta1 * p2 + 1e-12) * clamp((k * bin1 - 180) / 100, 0, 1), [b, v] = share(k * bin1, P1.centre[q], mono); held[q] = m; heldBass[q] = m * b; heldVoc[q] = m * v; }
  // pass L: medium windows, the low range. A cell that keeps its level is bass, one that dies away is kick.
  const r1 = pair(P1.A, P1.B, held), rest1 = [sub(L, r1[0]), sub(R, r1[1])];
  const binL = SR / T.nL, hopL = T.nL / 4 / SR, PL = look(rest1[0], rest1[1], T.nL, () => odd(T.tLow / hopL), 3, 2, () => T.lowRank);
  const lowAll = new Float32Array(PL.cells), lowDrum = new Float32Array(PL.cells), lowBass = new Float32Array(PL.cells);
  for (let f = 0; f < PL.n; f++) for (let k = 0; k < PL.nb; k++) { const fz = k * binL, lowB = clamp((300 - fz) / 100, 0, 1); if (!lowB) break; const q = f * PL.nb + k, kick = clamp(1 - (T.lowBeta * PL.Ht[q] / (PL.mag[q] + 1e-9)) ** 2, 0, 1), [b] = share(fz, PL.centre[q], mono); lowAll[q] = lowB; lowDrum[q] = lowB * kick; lowBass[q] = lowB * (1 - kick) * b; }
  // pass 2: short windows on the rest, drum hits against sound that changes smoothly. It is stricter about
  // calling something a drum in the stereo centre, where a voice could be, than out at the sides. Up above
  // about 6 kHz whatever was not held in pass 1 is taken to be cymbals.
  const rL = pair(PL.A, PL.B, lowAll), rest2 = [sub(rest1[0], rL[0]), sub(rest1[1], rL[1])];
  const bin2 = SR / T.n2, hop2 = T.n2 / 4 / SR, P2 = look(rest2[0], rest2[1], T.n2, () => odd(T.t2 / hop2), odd(T.f2 / bin2), 6);
  const mDrum = new Float32Array(P2.cells), mBass = new Float32Array(P2.cells), mVoc = new Float32Array(P2.cells);
  for (let f = 0; f < P2.n; f++) for (let k = 0; k < P2.nb; k++) { const q = f * P2.nb + k, h = P2.Ht[q], p = P2.Pf[q], be = mono ? T.beta : T.betaSide + (T.beta - T.betaSide) * clamp((P2.centre[q] - 0.5) / 0.3, 0, 1), md = Math.max(p * p / (p * p + be * be * h * h + 1e-12), clamp((k * bin2 - T.hiss) / 2000, 0, 1)), [b, v] = share(k * bin2, P2.centre[q], mono); mDrum[q] = md; mBass[q] = (1 - md) * b; mVoc[q] = (1 - md) * v; }
  // the same three passes applied to any sound (also used to measure how much of each true instrument lands in each part)
  const apply = (l, r, S1, D1, SL, DL, S2, D2) => {
    S1 = S1 || spectra(l, T.n1); D1 = D1 || spectra(r, T.n1);
    const h = pair(S1, D1, held), a1 = [sub(l, h[0]), sub(r, h[1])]; SL = SL || spectra(a1[0], T.nL); DL = DL || spectra(a1[1], T.nL);
    const la = pair(SL, DL, lowAll), a2 = [sub(a1[0], la[0]), sub(a1[1], la[1])]; S2 = S2 || spectra(a2[0], T.n2); D2 = D2 || spectra(a2[1], T.n2);
    const hb = pair(S1, D1, heldBass), hv = pair(S1, D1, heldVoc), ld = pair(SL, DL, lowDrum), lb = pair(SL, DL, lowBass), d2 = pair(S2, D2, mDrum), b2 = pair(S2, D2, mBass), v2 = pair(S2, D2, mVoc);
    const ch = c => { const x = c ? r : l, drums = add(ld[c], d2[c]), bass = add(add(hb[c], lb[c]), b2[c]), voc = add(hv[c], v2[c]); return [drums, bass, voc, sub(sub(sub(x, drums), bass), voc)]; }, o = [ch(0), ch(1)];
    return PARTS.map((_, p) => ({ L: o[0][p], R: o[1][p] }));
  };
  return { mono, parts: apply(L, R, P1.A, P1.B, PL.A, PL.B, P2.A, P2.B), apply: (l, r) => apply(l, r) };
}

// ---------------------------------------------------------------- recreating a part from the fly's brain
// The part is cut into equal slots on the beat. For each slot we take what the fly's brain did while it heard
// that slot: how often each of its own neurons (not the hearing-organ neurons) fired, in three steps across the
// slot. A slot is then replaced by a DIFFERENT slot of the same part, at least a beat away: the one whose brain
// activity was most alike. Nothing is fitted to the music; the choices are the fly's own confusions.
function brainStates(b, grid, sub = 3, latency = 0.008) {
  const cols = new Map(), nS = grid.n, per = grid.slot / sub;
  for (let q = 0; q < b.bN.length; q++) if (!cols.has(b.bN[q])) cols.set(b.bN[q], cols.size);
  const D = cols.size * sub, F = new Float32Array(nS * D);
  for (let q = 0; q < b.bT.length; q++) { const t = b.bT[q] * 1e-4 - latency - grid.t0, sl = Math.floor(t / grid.slot); if (sl < 0 || sl >= nS) continue; const sb = Math.min(sub - 1, Math.floor((t - sl * grid.slot) / per)); F[sl * D + cols.get(b.bN[q]) * sub + sb]++; }
  for (let i = 0; i < F.length; i++) F[i] = Math.sqrt(F[i]);
  return { F, D, nS, neurons: cols.size };
}
function choose(S, seed, apart = 4) {   // for each slot, the most alike other slot
  const { F, D, nS } = S, src = new Int32Array(nS), r = FS.mulberry32(seed), nrm = new Float64Array(nS);
  for (let a = 0; a < nS; a++) { let s = 0; for (let d = 0; d < D; d++) s += F[a * D + d] * F[a * D + d]; nrm[a] = s; }
  for (let a = 0; a < nS; a++) {
    let best = -1, bd = Infinity;
    for (let c = 0; c < nS; c++) { if (Math.abs(c - a) < apart) continue; let dot = 0; const oa = a * D, oc = c * D; for (let d = 0; d < D; d++) dot += F[oa + d] * F[oc + d]; const dist = nrm[a] + nrm[c] - 2 * dot + 1e-3 * r(); if (dist < bd) { bd = dist; best = c; } }
    src[a] = best < 0 ? a : best;
  }
  return src;
}
function assemble(part, grid, src) {   // lay the chosen slots end to end, with short cross-fades
  const len = part.L.length, L = new Float32Array(len), R = new Float32Array(len), sl = grid.slot * SR, fade = Math.round(0.006 * SR);
  for (let a = 0; a < grid.n; a++) {
    const o = Math.round((grid.t0 + a * grid.slot) * SR), so = Math.round((grid.t0 + src[a] * grid.slot) * SR), m = Math.round((grid.t0 + (a + 1) * grid.slot) * SR) - o;
    for (let i = -fade; i < m + fade; i++) { const p = o + i, q = so + i; if (p < 0 || p >= len || q < 0 || q >= len) continue; const g = i < fade ? (i + fade) / (2 * fade) : i >= m - fade ? (m + fade - i) / (2 * fade) : 1; L[p] += part.L[q] * g; R[p] += part.R[q] * g; }
  }
  return { L, R };
}
// How alike two slots sound: loudness in 16 bands from 60 Hz to 8 kHz, in decibels.
function slotSpectra(part, grid) {
  const N = 2048, K = 16, nS = grid.n, E = new Float32Array(nS * K), re = new Float64Array(N), im = new Float64Array(N), edges = Array.from({ length: K + 1 }, (_, i) => 60 * Math.pow(8000 / 60, i / K));
  for (let a = 0; a < nS; a++) {
    const o = Math.round((grid.t0 + a * grid.slot) * SR), m = Math.max(1, Math.round(grid.slot * SR)), hops = Math.max(1, Math.ceil(m / (N / 2)));
    for (let h = 0; h < hops; h++) { for (let i = 0; i < N; i++) { const p = o + h * (N / 2) + i, wv = 0.5 - 0.5 * Math.cos(2 * Math.PI * i / N); re[i] = p < part.L.length && i + h * (N / 2) < m + N / 2 ? (part.L[p] + part.R[p]) / 2 * wv : 0; im[i] = 0; } fft(re, im);
      for (let k = 0; k < K; k++) { const lo = Math.max(1, Math.round(edges[k] / SR * N)), hi = Math.max(lo + 1, Math.round(edges[k + 1] / SR * N)); let s = 0; for (let j = lo; j < hi && j < N / 2; j++) s += re[j] * re[j] + im[j] * im[j]; E[a * K + k] += s / hops; } }
  }
  let top = 1e-12; for (const v of E) top = Math.max(top, v);
  for (let i = 0; i < E.length; i++) E[i] = Math.max(-60, 10 * Math.log10(E[i] / top + 1e-9));
  return { E, K, nS };
}
// 0 = no better than picking another slot at random, 1 = every chosen slot sounds exactly like the one it replaced.
function matchScore(Sp, src, apart = 4) {
  const { E, K, nS } = Sp, dist = (a, c) => { let s = 0; for (let k = 0; k < K; k++) s += Math.abs(E[a * K + k] - E[c * K + k]); return s / K; };
  let ch = 0, rnd = 0, cnt = 0, same = 0;
  for (let a = 0; a < nS; a++) { ch += dist(a, src[a]); let s = 0, c = 0; for (let b = 0; b < nS; b++) if (Math.abs(b - a) >= apart) { s += dist(a, b); c++; } if (c) { rnd += s / c; cnt++; } }
  return cnt && rnd > 0 ? clamp(1 - ch / rnd, 0, 1) : 0;
}

const api = { SR, STEP, EAR, WING, PARTS, loadNet, scramble, pickWindow, loudness, antenna, hear, channels, runBrain, sing, envelope, corrLag, fly, stems, decodeSong, rebuild, split4, split4old, TUNE, brainStates, choose, assemble, slotSpectra, matchScore };
if (isNode) module.exports = api; else root.FlyEar = api;
})(typeof self !== 'undefined' ? self : this);
