// songear.js - listen to a full song: tempo and beat, pick its liveliest minute, and turn that into
// drum hits, a bass line, a melody and chords on a 16th-note grid. Runs in a Web Worker or in Node.
(function (root) {
'use strict';
const isNode = typeof module !== 'undefined' && module.exports && typeof require !== 'undefined';
const FS = isNode ? require('./flysim.js') : root.FlySim;

// ---------------------------------------------------------------- FFT
const tw = {};
function twiddles(n) { if (tw[n]) return tw[n]; const c = new Float64Array(n / 2), s = new Float64Array(n / 2); for (let k = 0; k < n / 2; k++) { c[k] = Math.cos(-2 * Math.PI * k / n); s[k] = Math.sin(-2 * Math.PI * k / n); } return (tw[n] = [c, s]); }
function fft(re, im) {
  const n = re.length, [C, S] = twiddles(n);
  for (let i = 1, j = 0; i < n; i++) { let bit = n >> 1; for (; j & bit; bit >>= 1) j ^= bit; j ^= bit; if (i < j) { let t = re[i]; re[i] = re[j]; re[j] = t; t = im[i]; im[i] = im[j]; im[j] = t; } }
  for (let len = 2; len <= n; len <<= 1) {
    const half = len >> 1, stride = n / len;
    for (let i = 0; i < n; i += len) for (let k = 0; k < half; k++) {
      const a = i + k, b = a + half, wr = C[k * stride], wi = S[k * stride];
      const vr = re[b] * wr - im[b] * wi, vi = re[b] * wi + im[b] * wr;
      re[b] = re[a] - vr; im[b] = im[a] - vi; re[a] += vr; im[a] += vi;
    }
  }
}
function stft(x, nfft, hop, onFrame) {
  const win = new Float64Array(nfft); for (let i = 0; i < nfft; i++) win[i] = 0.5 - 0.5 * Math.cos(2 * Math.PI * i / nfft);
  const re = new Float64Array(nfft), im = new Float64Array(nfft), mag = new Float64Array(nfft / 2 + 1);
  const nfr = Math.max(1, Math.floor((x.length - nfft) / hop) + 1);
  for (let f = 0; f < nfr; f++) {
    const o = f * hop; for (let i = 0; i < nfft; i++) { re[i] = (x[o + i] || 0) * win[i]; im[i] = 0; }
    fft(re, im); for (let k = 0; k <= nfft / 2; k++) mag[k] = Math.hypot(re[k], im[k]);
    onFrame(f, mag);
  }
  return nfr;
}
const hz = m => 440 * Math.pow(2, (m - 69) / 12);
const pct = (arr, p) => { const s = Float64Array.from(arr).sort(); if (!s.length) return 0; const r = (s.length - 1) * p / 100, lo = Math.floor(r); return s[lo] + (s[Math.ceil(r)] - s[lo]) * (r - lo); };

// ---------------------------------------------------------------- features
const BASS_LO = 28, BASS_HI = 55, MEL_LO = 57, MEL_HI = 88;
function analyze(x, sr) {
  // fine STFT for drums and timing
  const nf = 1024, hf = 256, bin = sr / nf, B = (a, b) => [Math.max(1, Math.round(a / bin)), Math.min(nf / 2, Math.round(b / bin))];
  const [l0, l1] = B(40, 120), [m0, m1] = B(1000, 4000), [h0, h1] = B(6000, Math.min(11000, sr / 2 - 1));
  const nFine = Math.max(1, Math.floor((x.length - nf) / hf) + 1);
  const low = new Float32Array(nFine), mid = new Float32Array(nFine), high = new Float32Array(nFine), flux = new Float32Array(nFine), rms = new Float32Array(nFine);
  let prev = new Float64Array(nf / 2 + 1);
  stft(x, nf, hf, (f, mag) => {
    let a = 0, b = 0, c = 0, fl = 0, e = 0;
    for (let k = l0; k <= l1; k++) a += mag[k] * mag[k];
    for (let k = m0; k <= m1; k++) b += mag[k] * mag[k];
    for (let k = h0; k <= h1; k++) c += mag[k] * mag[k];
    for (let k = 1; k <= nf / 2; k++) { const lm = Math.log1p(10 * mag[k]); fl += Math.max(0, lm - prev[k]); prev[k] = lm; e += mag[k] * mag[k]; }
    low[f] = Math.sqrt(a); mid[f] = Math.sqrt(b); high[f] = Math.sqrt(c); flux[f] = fl; rms[f] = Math.sqrt(e);
  });
  // coarse STFT for pitch: bass and melody salience, chroma
  const nc = 8192, hc = 1024, binc = sr / nc, nCoarse = Math.max(1, Math.floor((x.length - nc) / hc) + 1);
  const nb = BASS_HI - BASS_LO + 1, nm = MEL_HI - MEL_LO + 1;
  const bassSal = new Float32Array(nCoarse * nb), melSal = null, chroma = new Float32Array(nCoarse * 12);
  const pk = (mag, f) => { const k = Math.round(f / binc); if (k < 1 || k >= nc / 2) return 0; return Math.max(mag[k - 1], mag[k], mag[k + 1]); };
  const cbin0 = Math.round(80 / binc), cbin1 = Math.round(2000 / binc), pcOf = new Int8Array(nc / 2 + 1).fill(-1);
  for (let k = cbin0; k <= cbin1; k++) { const m = 69 + 12 * Math.log2(k * binc / 440); pcOf[k] = ((Math.round(m) % 12) + 12) % 12; }
  stft(x, nc, hc, (f, mag) => {
    for (let m = BASS_LO; m <= BASS_HI; m++) { let s = 0; for (let h = 1; h <= 5; h++) s += Math.pow(0.75, h - 1) * pk(mag, hz(m) * h); bassSal[f * nb + m - BASS_LO] = s; }
    for (let k = cbin0; k <= cbin1; k++) chroma[f * 12 + pcOf[k]] += Math.log1p(mag[k]);
  });
  // medium STFT for the melody: whitened spectrum, harmonic salience
  const nmd = 2048, hmd = 256, binm = sr / nmd, nMed = Math.max(1, Math.floor((x.length - nmd) / hmd) + 1), melS = new Float32Array(nMed * nm);
  const wh = new Float64Array(nmd / 2 + 1), lg = new Float64Array(nmd / 2 + 1), MW = [1, 0.6, 0.4, 0.3];
  const k0 = Math.round(180 / binm), k1 = Math.min(nmd / 2 - 8, Math.round(5000 / binm)), R = 8;
  stft(x, nmd, hmd, (f, mag) => {
    for (let k = Math.max(0, k0 - R); k <= k1 + R; k++) lg[k] = Math.log1p(mag[k] || 0);
    for (let k = k0; k <= k1; k++) { let run = 0; for (let q = k - R; q <= k + R; q++) run += lg[q]; wh[k] = Math.max(0, lg[k] - run / (2 * R + 1)); }
    const pw = fr => { const k = fr / binm, k_ = Math.floor(k); if (k_ < k0 || k_ + 1 > k1) return 0; return Math.max(wh[k_], wh[k_ + 1]); };
    for (let m = MEL_LO; m <= MEL_HI; m++) { let s = 0; for (let h = 1; h <= 4; h++) s += MW[h - 1] * pw(hz(m) * h); melS[f * nm + m - MEL_LO] = s; }
  });
  return { sr, fine: { hop: hf, nfft: nf, low, mid, high, flux, rms, n: nFine }, coarse: { hop: hc, nfft: nc, bassSal, melSal, chroma, n: nCoarse, nb, nm },
           med: { hop: hmd, nfft: nmd, melS, n: nMed }, duration: x.length / sr };
}

// ---------------------------------------------------------------- tempo, beats, bars
function beats(F) {
  const fps = F.sr / F.fine.hop, o = F.fine.flux, n = o.length, w = Math.round(fps * 0.5);
  const env = new Float64Array(n); let acc = 0;
  for (let i = 0; i < n; i++) { acc += o[i]; if (i >= w) acc -= o[i - w]; const mean = acc / Math.min(i + 1, w); env[i] = Math.max(0, o[i] - mean); }
  let best = 0, bestLag = fps / 2, ac = [];
  const lo = Math.floor(fps * 60 / 180), hi = Math.ceil(fps * 60 / 60);
  for (let L = lo; L <= hi; L++) { let s = 0; for (let i = 0; i + L < n; i++) s += env[i] * env[i + L]; ac[L] = s; }
  for (let L = lo + 1; L < hi; L++) {
    const bpm = 60 * fps / L, wgt = Math.exp(-0.5 * Math.pow(Math.log2(bpm / 110) / 0.7, 2)), v = ac[L] * wgt;
    if (v > best && ac[L] >= ac[L - 1] && ac[L] >= ac[L + 1]) { best = v; bestLag = L; }
  }
  const a = ac[bestLag - 1] || 0, b = ac[bestLag], c = ac[bestLag + 1] || 0, den = a - 2 * b + c;
  const P = bestLag + (den ? 0.5 * (a - c) / den : 0);
  let bestPhase = 0, bestScore = -1;
  for (let ph = 0; ph < P; ph += 0.25) { let s = 0; for (let t = ph; t < n; t += P) s += env[Math.round(t)] || 0; if (s > bestScore) { bestScore = s; bestPhase = ph; } }
  const t0 = F.fine.nfft / 2 / F.sr, beatT = [];
  for (let t = bestPhase; t < n; t += P) beatT.push(t / fps + t0);
  // downbeat: which of the four beat positions carries the most kick
  const lowFlux = i => Math.max(0, (F.fine.low[i] || 0) - (F.fine.low[i - 1] || 0));
  let down = 0, dScore = -1;
  for (let d = 0; d < 4; d++) { let s = 0; for (let i = d; i < beatT.length; i += 4) { const f = Math.round((beatT[i] - t0) * fps); for (let q = f - 2; q <= f + 3; q++) s += lowFlux(q); } if (s > dScore) { dScore = s; down = d; } }
  return { bpm: 60 * fps / P, beat: P / fps, beats: beatT, down };
}
function pickExcerpt(F, T, seconds = 60) {
  const bars = Math.floor((T.beats.length - T.down) / 4);
  const W = Math.max(1, Math.min(bars, Math.round(seconds / (4 * T.beat))));
  const fps = F.sr / F.fine.hop, cum = new Float64Array(F.fine.n + 1);
  for (let i = 0; i < F.fine.n; i++) cum[i + 1] = cum[i] + F.fine.rms[i];
  let best = 0, bestE = -1;
  for (let b = 0; b + W <= bars; b++) {
    const s = Math.round(T.beats[T.down + 4 * b] * fps), e = Math.round((T.beats[T.down + 4 * (b + W) - 1] + T.beat) * fps);
    const E = cum[Math.min(F.fine.n, e)] - cum[Math.max(0, s)]; if (E > bestE) { bestE = E; best = b; }
  }
  const start = T.beats[T.down + 4 * best];
  return { startBar: best, bars: W, start, beat: T.beat, slot: T.beat / 4, end: start + W * 4 * T.beat, totalBars: bars };
}

// ---------------------------------------------------------------- transcription
const TEMPL = []; for (let r = 0; r < 12; r++) { TEMPL.push({ root: r, minor: false, pcs: [r, (r + 4) % 12, (r + 7) % 12] }); TEMPL.push({ root: r, minor: true, pcs: [r, (r + 3) % 12, (r + 7) % 12] }); }
function transcribe(F, X) {
  const fps = F.sr / F.fine.hop, t0 = F.fine.nfft / 2 / F.sr, nS = X.bars * 16, S = X.slot;
  const fi = t => Math.round((t - t0) * fps);
  const diffMax = (arr, a, b) => { let m = 0; for (let i = Math.max(1, a); i <= Math.min(arr.length - 1, b); i++) m = Math.max(m, arr[i] - arr[i - 1]); return m; };
  const kv = new Float64Array(nS), mv = new Float64Array(nS), hv = new Float64Array(nS);
  for (let s = 0; s < nS; s++) {
    const t = X.start + s * S, a = fi(t - 0.03), b = fi(t + 0.04);
    kv[s] = diffMax(F.fine.low, a, b); mv[s] = diffMax(F.fine.mid, a, b); hv[s] = diffMax(F.fine.high, a, b);
  }
  const nk = pct(kv, 95) || 1, nm = pct(mv, 95) || 1, nh = pct(hv, 95) || 1, nh2 = pct(hv, 75) || 1;
  const drums = { kick: new Uint8Array(nS), snare: new Uint8Array(nS), hat: new Uint8Array(nS), strength: { kick: new Float32Array(nS), snare: new Float32Array(nS), hat: new Float32Array(nS) } };
  for (let s = 0; s < nS; s++) {
    const k = kv[s] / nk, m = mv[s] / nm, h = hv[s] / nh;
    if (k > 0.45) { drums.kick[s] = 1; drums.strength.kick[s] = Math.min(1, k); }
    const sn = Math.min(m, h); if (sn > 0.5) { drums.snare[s] = 1; drums.strength.snare[s] = Math.min(1, sn); }
    const h2 = hv[s] / nh2; if (h2 > 0.55 && !(drums.snare[s] && h < 1.4 * m)) { drums.hat[s] = 1; drums.strength.hat[s] = Math.min(1, 0.4 + 0.6 * h); }
  }
  // pitched parts on 8ths, chords on beats, from the coarse frames
  const cfps = F.sr / F.coarse.hop, ct0 = F.coarse.nfft / 2 / F.sr, C = F.coarse;
  const coarseRange = (t1, t2) => { let a = Math.round((t1 - ct0) * cfps), b = Math.round((t2 - ct0) * cfps); a = Math.max(0, Math.min(C.n - 1, a)); b = Math.max(a, Math.min(C.n - 1, b)); return [a, b]; };
  const n8 = X.bars * 8, bass = new Int16Array(n8).fill(-1), mel = new Int16Array(n8).fill(-1), bassE = new Float64Array(n8), melPk = new Float64Array(n8), melE = new Float64Array(n8);
  const bassBest = new Int16Array(n8), melBest = new Int16Array(n8);
  for (let e = 0; e < n8; e++) {
    const t = X.start + e * 2 * S, [a, b] = coarseRange(t + 0.02, t + 2 * S - 0.02);
    const sb = new Float64Array(C.nb);
    for (let f = a; f <= b; f++) for (let q = 0; q < C.nb; q++) sb[q] += C.bassSal[f * C.nb + q];
    let ib = 0; for (let q = 1; q < C.nb; q++) if (sb[q] > sb[ib]) ib = q;
    if (ib >= 12 && sb[ib - 12] >= 0.95 * sb[ib]) ib -= 12;   // a strong sub-octave means the real note is lower
    // melody: score each note by how sharply it starts in this 8th (pads hold still, tunes move)
    const Md = F.med, mfps = F.sr / Md.hop, mt0 = Md.nfft / 2 / F.sr;
    const ma = Math.max(1, Math.round((t - 0.03 - mt0) * mfps)), mb = Math.min(Md.n - 1, Math.round((t + 2 * S * 0.6 - mt0) * mfps));
    const inc = new Float64Array(C.nm);
    for (let f = ma; f <= mb; f++) for (let q = 0; q < C.nm; q++) inc[q] += Math.max(0, Md.melS[f * C.nm + q] - Md.melS[(f - 1) * C.nm + q]);
    let im = 0, msum = 0; for (let q = 0; q < C.nm; q++) { msum += inc[q]; if (inc[q] > inc[im]) im = q; }
    bassBest[e] = BASS_LO + ib; melBest[e] = MEL_LO + im;
    bassE[e] = sb[ib] / (b - a + 1); melPk[e] = inc[im] / (msum / C.nm + 1e-9); melE[e] = inc[im];
  }
  const bThr = 0.35 * pct(bassE, 80), mThr = 0.6 * pct(melE, 85);
  for (let e = 0; e < n8; e++) { if (bassE[e] > bThr) bass[e] = bassBest[e]; if (melE[e] > mThr && melPk[e] > 3) mel[e] = melBest[e]; }
  const nB = X.bars * 4, chords = new Int16Array(nB).fill(-1);
  for (let q = 0; q < nB; q++) {
    const t = X.start + q * X.beat, [a, b] = coarseRange(t, t + X.beat);
    const v = new Float64Array(12); for (let f = a; f <= b; f++) for (let p = 0; p < 12; p++) v[p] += C.chroma[f * 12 + p];
    const mean = v.reduce((s, z) => s + z, 0) / 12; for (let p = 0; p < 12; p++) v[p] = Math.max(0, v[p] - mean);
    let best = -1, bs = 0; TEMPL.forEach((T, i) => { const s = T.pcs.reduce((z, p) => z + v[p], 0) - 0.5 * v.reduce((z, x, p) => z + (T.pcs.includes(p) ? 0 : x), 0) / 3; if (s > bs) { bs = s; best = i; } });
    chords[q] = best;
  }
  // smooth chords: majority over each half bar
  for (let q = 0; q + 1 < nB; q += 2) { if (chords[q] !== chords[q + 1]) { const keep = chords[q] >= 0 ? chords[q] : chords[q + 1]; chords[q] = chords[q + 1] = keep; } }
  return { drums, bass, mel, chords, n16: nS, n8, nBeats: nB };
}
const chordName = i => i < 0 ? '–' : ['C', 'C#', 'D', 'Eb', 'E', 'F', 'F#', 'G', 'Ab', 'A', 'Bb', 'B'][TEMPL[i].root] + (TEMPL[i].minor ? 'm' : '');

const api = { fft, analyze, beats, pickExcerpt, transcribe, TEMPL, chordName, hz };

// ---------------------------------------------------------------- the fly hears the notes
// The ear sends each sound to its own group of hearing neurons: JO-B is split into 14 groups (kick, 12 bass
// note names, snare), JO-A into 13 (hi-hat, 12 note names for melody and chords).
function earGroups(net) {
  const split = (arr, k, seed) => { const r = FS.mulberry32(seed), a = Array.from(arr); for (let i = a.length - 1; i > 0; i--) { const j = Math.floor(r() * (i + 1)); [a[i], a[j]] = [a[j], a[i]]; } return Array.from({ length: k }, (_, g) => Int32Array.from(a.filter((_, i) => i % k === g))); };
  return { B: split(net.chB, 14, 3), A: split(net.chA, 13, 4) };
}
function stimulus(R, X) {
  const NC = 27, nMs = Math.ceil((X.bars * 4 * X.beat + 0.4) * 1000), rates = new Float32Array(NC * nMs);
  const put = (ch, t0, dur, hzv) => { const a = Math.max(0, Math.round(t0 * 1000)), b = Math.min(nMs, Math.round((t0 + dur) * 1000)); for (let i = a; i < b; i++) { const q = i * NC + ch; if (rates[q] < hzv) rates[q] = hzv; } };
  const S = X.slot, B = c => c, A = c => 14 + c;
  for (let s = 0; s < R.n16; s++) {
    const t = s * S, st = R.drums.strength;
    if (R.drums.kick[s]) put(B(0), t, 0.03, 230 * (0.45 + 0.55 * st.kick[s]));
    if (R.drums.snare[s]) put(B(13), t, 0.03, 220 * (0.45 + 0.55 * st.snare[s]));
    if (R.drums.hat[s]) put(A(0), t, 0.03, 200 * (0.45 + 0.55 * st.hat[s]));
  }
  for (let e = 0; e < R.n8; e++) {
    const t = e * 2 * S;
    if (R.bass[e] >= 0) put(B(1 + R.bass[e] % 12), t, 2 * S * 0.85, 220);
    if (R.mel[e] >= 0) put(A(1 + R.mel[e] % 12), t, 2 * S * 0.85, 200);
  }
  for (let q = 0; q < R.nBeats; q++) if (R.chords[q] >= 0) for (const pc of TEMPL[R.chords[q]].pcs) put(A(1 + pc), q * X.beat, X.beat, 90);
  return { nMs, rates, NC };
}
function runFly(net, groups, stim, seed, onProgress) {
  const br = new FS.Brain(net, seed);
  for (const g of groups.B) br.addChannel(g);
  for (const g of groups.A) br.addChannel(g);
  const dt = net.c.dt, n = Math.round(stim.nMs / 1000 / dt), NC = stim.NC, rates = new Float64Array(NC), sT = [], sN = [];
  for (let k = 0; k < n; k++) {
    const ms = Math.min(stim.nMs - 1, Math.floor(k * dt * 1000));
    for (let c = 0; c < NC; c++) rates[c] = stim.rates[ms * NC + c];
    for (const j of br.step(rates)) { sT.push(k); sN.push(j); }
    if (onProgress && k % 20000 === 0) onProgress(k / n);
  }
  return { t: Uint32Array.from(sT), n: Uint16Array.from(sN) };
}

// ---------------------------------------------------------------- reading the fly's version out
// Which deeper neurons fired in each 16th slot (inputs excluded).
function slotFeatures(spikes, X, net, n16) {
  const isIn = new Uint8Array(net.n); for (const i of net.chA) isIn[i] = 1; for (const i of net.chB) isIn[i] = 1;
  const dt = net.c.dt, sets = Array.from({ length: n16 }, () => new Set());
  for (let q = 0; q < spikes.t.length; q++) {
    const j = spikes.n[q]; if (isIn[j]) continue;
    const s = Math.floor((spikes.t[q] * dt - 0.008) / X.slot);
    if (s >= 0 && s < n16) sets[s].add(j);
  }
  return sets.map(z => Int32Array.from(z));
}
const unitSets = (feats, per) => { const out = []; for (let u = 0; u * per < feats.length; u++) { const z = new Set(); for (let s = u * per; s < Math.min(feats.length, (u + 1) * per); s++) for (const j of feats[s]) z.add(j); out.push(Int32Array.from(z)); } return out; };
function nbTrain(unitsList, targetsList, nClass) {   // Bernoulli naive Bayes over "fired in this unit"
  const cnt = new Map(), N = new Float64Array(nClass); let total = 0;
  unitsList.forEach((units, r) => units.forEach((js, u) => { const c = targetsList[r][u]; N[c]++; total++; for (const j of js) { let a = cnt.get(j); if (!a) cnt.set(j, a = new Float64Array(nClass)); a[c]++; } }));
  const model = { nClass, logPrior: Array.from(N, v => Math.log((v + 1) / (total + nClass))), w: new Map(), base: new Float64Array(nClass) };
  for (const [j, a] of cnt) {
    const fired = a.reduce((s, v) => s + v, 0); if (fired < 3) continue;
    const lp = new Float64Array(nClass);
    for (let c = 0; c < nClass; c++) { const p = (a[c] + 0.5) / (N[c] + 1); model.base[c] += Math.log(1 - p); lp[c] = Math.log(p) - Math.log(1 - p); }
    model.w.set(j, lp);
  }
  return model;
}
function nbDecode(model, units) {
  return units.map(js => { const sc = Float64Array.from(model.logPrior); for (let c = 0; c < model.nClass; c++) sc[c] += model.base[c]; for (const j of js) { const lp = model.w.get(j); if (lp) for (let c = 0; c < model.nClass; c++) sc[c] += lp[c]; } let b = 0; for (let c = 1; c < model.nClass; c++) if (sc[c] > sc[b]) b = c; return b; });
}
function train(featsList, R) {
  const model = { drums: {} };
  for (const d of ['kick', 'snare', 'hat']) {   // best single neuron, plus a second if their union matches better
    const tgt = R.drums[d], nT = tgt.reduce((s, v) => s + v, 0) * featsList.length, cnt = new Map(), hit = new Map();
    featsList.forEach(f => f.forEach((js, s) => { for (const j of js) { cnt.set(j, (cnt.get(j) || 0) + 1); if (tgt[s]) hit.set(j, (hit.get(j) || 0) + 1); } }));
    const f1 = j => 2 * (hit.get(j) || 0) / (cnt.get(j) + nT);
    const ranked = [...cnt.keys()].sort((a, b) => f1(b) - f1(a)).slice(0, 12);
    let best = [ranked[0]], bestF = ranked.length ? f1(ranked[0]) : 0;
    for (const j2 of ranked.slice(1)) {
      let c = 0, h = 0; featsList.forEach(f => f.forEach((js, s) => { const a = js.includes(best[0]) || js.includes(j2); if (a) { c++; if (tgt[s]) h++; } }));
      const F = 2 * h / (c + nT); if (F > bestF + 0.02) { bestF = F; best = [best[0], j2]; break; }
    }
    model.drums[d] = best.filter(x => x !== undefined);
  }
  const u8 = featsList.map(f => unitSets(f, 2)), u4 = featsList.map(f => unitSets(f, 4));
  const pcT = arr => Array.from(arr, m => m < 0 ? 12 : m % 12);
  model.bass = nbTrain(u8, featsList.map(() => pcT(R.bass)), 13);
  model.mel = nbTrain(u8, featsList.map(() => pcT(R.mel)), 13);
  model.chords = nbTrain(u4, featsList.map(() => Array.from(R.chords, c => c < 0 ? 24 : c)), 25);
  return model;
}
function decode(model, feats, R) {
  const out = { drums: {} };
  for (const d of ['kick', 'snare', 'hat']) out.drums[d] = Uint8Array.from(feats, js => model.drums[d].some(j => js.includes(j)) ? 1 : 0);
  const u8 = unitSets(feats, 2), u4 = unitSets(feats, 4);
  out.bass = Int16Array.from(nbDecode(model.bass, u8), c => c === 12 ? -1 : c);
  out.mel = Int16Array.from(nbDecode(model.mel, u8), c => c === 12 ? -1 : c);
  out.chords = Int16Array.from(nbDecode(model.chords, u4), c => c === 24 ? -1 : c);
  return out;
}
function score(dec, R) {
  const res = {};
  const df = d => { let tp = 0, a = 0, b = 0; for (let s = 0; s < R.n16; s++) { if (dec.drums[d][s]) a++; if (R.drums[d][s]) b++; if (dec.drums[d][s] && R.drums[d][s]) tp++; } return a + b ? 2 * tp / (a + b) : 1; };
  res.drums = (df('kick') + df('snare') + df('hat')) / 3;
  const acc = (got, want) => { let ok = 0, n = 0; for (let i = 0; i < want.length; i++) { const w = want[i] < 0 ? -1 : want[i] % 12, g = got[i]; if (w < 0 && g < 0) continue; n++; if (w === g) ok++; } return n ? ok / n : 1; };
  res.bass = acc(dec.bass, R.bass); res.mel = acc(dec.mel, R.mel);
  let ok = 0, n = 0; for (let q = 0; q < R.nBeats; q++) { if (R.chords[q] < 0 && dec.chords[q] < 0) continue; n++; if (R.chords[q] === dec.chords[q]) ok++; } res.chords = n ? ok / n : 1;
  res.overall = (res.drums + res.bass + res.mel + res.chords) / 4;
  return res;
}
Object.assign(api, { earGroups, stimulus, runFly, slotFeatures, train, decode, score });

// ---------------------------------------------------------------- practice: player neurons whose input synapses learn
// One "player" neuron per drum and per bass / melody note name. After each round, every synapse onto a player
// changes by eta * (times it was active when the player should have fired - times it was active when the
// player fired wrongly). Only synapses that exist in the connectome change; none are added.
const CLASSES = [['kick', 'drum'], ['snare', 'drum'], ['hat', 'drum'], ...Array.from({ length: 12 }, (_, i) => ['bass' + i, 'bass']), ...Array.from({ length: 12 }, (_, i) => ['mel' + i, 'mel'])];
function classUnits(R, name) {   // target per unit for one class
  if (name === 'kick' || name === 'snare' || name === 'hat') return { per: 1, tgt: Uint8Array.from(R.drums[name]) };
  const part = name.startsWith('bass') ? R.bass : R.mel, pc = +name.replace(/\D/g, '');
  return { per: 2, tgt: Uint8Array.from(part, m => m >= 0 && m % 12 === pc ? 1 : 0) };
}
function choosePlayers(feats, R) {
  const used = new Set(), players = [];
  for (const [name, part] of CLASSES) {
    const { per, tgt } = classUnits(R, name), units = unitSets(feats, per), nT = tgt.reduce((a, b) => a + b, 0);
    if (!nT) { players.push({ name, part, idx: -1 }); continue; }
    const cnt = new Map(), hit = new Map();
    units.forEach((js, u) => { for (const j of js) { cnt.set(j, (cnt.get(j) || 0) + 1); if (tgt[u]) hit.set(j, (hit.get(j) || 0) + 1); } });
    let best = -1, bf = -1;
    for (const [j, c] of cnt) { if (used.has(j)) continue; const f = 2 * (hit.get(j) || 0) / (c + nT); if (f > bf) { bf = f; best = j; } }
    used.add(best); players.push({ name, part, idx: best, f1: bf });
  }
  return players;
}
function playersDecode(players, feats, R) {
  const out = { drums: {}, bass: new Int16Array(R.n8).fill(-1), mel: new Int16Array(R.n8).fill(-1) };
  const u8 = unitSets(feats, 2);
  for (const p of players) {
    if (p.part === 'drum') out.drums[p.name] = Uint8Array.from(feats, js => p.idx >= 0 && js.includes(p.idx) ? 1 : 0);
  }
  for (const part of ['bass', 'mel']) {
    const ps = players.filter(p => p.part === part && p.idx >= 0);
    u8.forEach((js, u) => { let best = null; for (const p of ps) if (js.includes(p.idx) && (!best || p.f1 > best.f1)) best = p; if (best) out[part][u] = +best.name.replace(/\D/g, ''); });
  }
  return out;
}
function makePlastic(net, players) {   // remember the original weights and where each player's input synapses are
  if (!net.w0) net.w0 = Float64Array.from(net.w);
  net.w.set(net.w0);
  const want = new Map(players.filter(p => p.idx >= 0).map(p => [p.idx, p]));
  const syn = new Map(); players.forEach(p => syn.set(p.idx, []));
  for (let pre = 0; pre < net.n; pre++) for (let k = net.indptr[pre]; k < net.indptr[pre + 1]; k++) { const q = net.post[k]; if (want.has(q)) syn.get(q).push([k, pre]); }
  return syn;
}
function practise(net, players, syn, spikes, feats, R, X, eta = 0.08, etaDrum = 0.012) {
  const dt = net.c.dt, n16 = R.n16, preActive = new Map();   // pre neuron -> per-16th-slot activity (just before / during)
  const pres = new Set(); for (const list of syn.values()) for (const [, pre] of list) pres.add(pre);
  for (const pre of pres) preActive.set(pre, new Uint8Array(n16));
  for (let q = 0; q < spikes.t.length; q++) { const a = preActive.get(spikes.n[q]); if (!a) continue; const s = Math.floor((spikes.t[q] * dt + 0.002) / X.slot); if (s >= 0 && s < n16) a[s] = 1; }
  let changed = 0;
  for (const p of players) {
    if (p.idx < 0) continue;
    const { per, tgt } = classUnits(R, p.name), units = unitSets(feats, per);
    const err = units.map((js, u) => (tgt[u] ? 1 : 0) - (js.includes(p.idx) ? 1 : 0));
    for (const [k, pre] of syn.get(p.idx)) {
      const a = preActive.get(pre); let g = 0;
      for (let u = 0; u < err.length; u++) if (err[u]) { let act = 0; for (let s = u * per; s < Math.min(n16, (u + 1) * per); s++) act |= a[s]; if (act) g += err[u]; }
      if (!g) continue;
      const w0 = net.w0[k], lim = Math.max(3, 3 * Math.abs(w0));
      net.w[k] = Math.max(-lim, Math.min(lim, net.w[k] + (p.part === "drum" ? etaDrum : eta) * g)); changed++;
    }
  }
  return changed;
}
Object.assign(api, { CLASSES, choosePlayers, playersDecode, makePlastic, practise });

// ---------------------------------------------------------------- recruiting: more players per drum / note each round
function rankCandidates(featsList, R, perClass = 5) {
  const out = {};
  for (const [name, part] of CLASSES) {
    const { per, tgt } = classUnits(R, name), nT = tgt.reduce((a, b) => a + b, 0) * featsList.length;
    if (!nT) { out[name] = []; continue; }
    const cnt = new Map(), hit = new Map();
    for (const f of featsList) unitSets(f, per).forEach((js, u) => { for (const j of js) { cnt.set(j, (cnt.get(j) || 0) + 1); if (tgt[u]) hit.set(j, (hit.get(j) || 0) + 1); } });
    out[name] = [...cnt.entries()].map(([j, c]) => ({ idx: j, f1: 2 * (hit.get(j) || 0) / (c + nT) })).sort((a, b) => b.f1 - a.f1).slice(0, perClass);
  }
  return out;
}
function recruit(ranked, k) {   // up to k players per class, each neuron used once
  const used = new Set(), players = [];
  for (const [name, part] of CLASSES) { let n = 0; for (const c of ranked[name] || []) { if (n >= k) break; if (used.has(c.idx) || c.f1 <= 0) continue; used.add(c.idx); players.push({ name, part, idx: c.idx, f1: c.f1 }); n++; } }
  return players;
}
function ensembleDecode(players, feats, R) {
  const out = { drums: {}, bass: new Int16Array(R.n8).fill(-1), mel: new Int16Array(R.n8).fill(-1) };
  const by = {}; for (const p of players) (by[p.name] = by[p.name] || []).push(p);
  for (const d of ['kick', 'snare', 'hat']) {
    const ps = by[d] || [], tot = ps.reduce((a, p) => a + p.f1, 0);
    out.drums[d] = Uint8Array.from(feats, js => { let w = 0; for (const p of ps) if (js.includes(p.idx)) w += p.f1; return tot && w >= 0.5 * tot ? 1 : 0; });
  }
  const u8 = unitSets(feats, 2);
  for (const part of ['bass', 'mel']) u8.forEach((js, u) => {
    let best = -1, bw = 0;
    for (let pc = 0; pc < 12; pc++) { const ps = by[part + pc] || []; let w = 0; for (const p of ps) if (js.includes(p.idx)) w += p.f1; const tot = ps.reduce((a, p) => a + p.f1, 0); if (tot && w / tot > bw) { bw = w / tot; best = pc; } }
    if (bw >= 0.34) out[part][u] = best;
  });
  return out;
}
Object.assign(api, { rankCandidates, recruit, ensembleDecode });


if (isNode) module.exports = api; else root.SongEar = api;
})(typeof self !== 'undefined' ? self : this);

// ---------------------------------------------------------------- the whole lesson, for the worker
(function (root) {
const isNode = typeof module !== 'undefined' && module.exports && typeof require !== 'undefined';
const SE = isNode ? module.exports : root.SongEar;
// notes for playback: drums on 16ths; bass and melody on 8ths in the octave the ear heard most for that note name
function versionEvents(dec, R, X) {
  const oct = arr => { const c = {}; for (const m of arr) if (m >= 0) { const k = m % 12; c[k] = c[k] || {}; c[k][m] = (c[k][m] || 0) + 1; } const o = {}; for (const k in c) o[k] = +Object.entries(c[k]).sort((a, b) => b[1] - a[1])[0][0]; return o; };
  const bo = oct(R.bass), mo = oct(R.mel), ev = { drums: [], bass: [], mel: [], chords: [] }, S = X.slot;
  for (const d of ['kick', 'snare', 'hat']) (dec.drums[d] || []).forEach((v, s) => { if (v) ev.drums.push([s * S, d, d === 'hat' ? 80 : 110]); });
  const notes = (arr, octs, base, out) => { for (let e = 0; e < arr.length; e++) { const pc = arr[e]; if (pc < 0) continue; let len = 1; while (e + len < arr.length && arr[e + len] === pc && len < 4) len++; out.push([e * 2 * S, octs[pc] !== undefined ? octs[pc] : base + pc, len * 2 * S * 0.9, 0.85]); e += len - 1; } };
  notes(dec.bass, bo, 36, ev.bass); notes(dec.mel, mo, 60, ev.mel);
  for (let q = 0; q < dec.chords.length; q++) { const c = dec.chords[q]; if (c < 0) continue; let len = 1; while (q + len < dec.chords.length && dec.chords[q + len] === c) len++; const T = SE.TEMPL[c]; ev.chords.push([q * X.beat, (T.root + 3) % 12 + 57, T.minor, len * X.beat]); q += len - 1; }
  for (const k in ev) ev[k].sort((a, b) => a[0] - b[0]);
  return ev;
}
function lesson(net, x, sr, rounds, post, seedBase = 1000) {
  const F = SE.analyze(x, sr), T = SE.beats(F), X = SE.pickExcerpt(F, T, 60), R = SE.transcribe(F, X);
  const heardDec = { drums: R.drums, bass: Int16Array.from(R.bass, m => m < 0 ? -1 : m % 12), mel: Int16Array.from(R.mel, m => m < 0 ? -1 : m % 12), chords: R.chords };
  post('heard', { X, bpm: T.bpm, duration: F.duration, events: versionEvents(heardDec, R, X), chords: Array.from(R.chords, SE.chordName) });
  const G = SE.earGroups(net), stim = SE.stimulus(R, X);
  if (!net.w0) net.w0 = Float64Array.from(net.w);
  net.w.set(net.w0);
  const sp0 = SE.runFly(net, G, stim, seedBase), f0 = SE.slotFeatures(sp0, X, net, R.n16), feats = [f0];
  const pub = ps => ps.map(p => ({ name: p.name, part: p.part, idx: p.idx }));
  post('round', { r: 0, t: sp0.t, n: sp0.n, players: pub(SE.recruit(SE.rankCandidates(feats, R), 3)) });
  for (let r = 1; r <= rounds; r++) {
    // practice round r: recruit up to r players per drum / note (max 3), play, score, then let their synapses learn
    const players = SE.recruit(SE.rankCandidates(feats, R), Math.min(r, 3));
    const syn = new Map(); for (const p of players) syn.set(p.idx, []);
    for (let pre = 0; pre < net.n; pre++) for (let q = net.indptr[pre]; q < net.indptr[pre + 1]; q++) { const L = syn.get(net.post[q]); if (L) L.push([q, pre]); }
    const sp = SE.runFly(net, G, stim, seedBase + r), ft = SE.slotFeatures(sp, X, net, R.n16);
    const dec = SE.ensembleDecode(players, ft, R);
    dec.chords = SE.decode(SE.train(feats, R), ft, R).chords;
    const sc = SE.score(dec, R);
    post('round', { r, t: sp.t, n: sp.n, score: sc, events: versionEvents(dec, R, X), players: pub(players) });
    SE.practise(net, players, syn, sp, ft, R, X);
    feats.push(ft);
  }
  net.w.set(net.w0);   // leave the brain as the connectome says
}
Object.assign(SE, { versionEvents, lesson });
})(typeof self !== 'undefined' ? self : this);
