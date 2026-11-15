// flysim.js - the Shiu et al. 2024 leaky integrate-and-fire fly brain, restricted to the neurons the
// ears can reach (exact for hearing input), plus the ear (drum-hit detector) and drum read-out.
// Runs in a browser Web Worker or in Node. Mirrors fly_drums.FlyBrain step for step.
(function (root) {
'use strict';

function mulberry32(seed) {
  let a = seed >>> 0;
  return function () {
    a = (a + 0x6D2B79F5) | 0;
    let t = Math.imul(a ^ (a >>> 15), 1 | a);
    t = (t + Math.imul(t ^ (t >>> 7), 61 | t)) ^ t;
    return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
  };
}

function b64(s, Type) {
  let u8;
  if (typeof Buffer !== 'undefined') { const b = Buffer.from(s, 'base64'); u8 = new Uint8Array(b.buffer.slice(b.byteOffset, b.byteOffset + b.length)); }
  else { const bin = atob(s); u8 = new Uint8Array(bin.length); for (let i = 0; i < bin.length; i++) u8[i] = bin.charCodeAt(i); }
  return new Type(u8.buffer);
}

function loadNet(J) {
  const net = { n: J.n, indptr: b64(J.indptr, Int32Array), post: b64(J.post, Uint16Array), wcount: b64(J.w, Int16Array),
    chA: b64(J.chA, Int32Array), chB: b64(J.chB, Int32Array), c: J.consts, kit: J.kit, earMap: J.earMap, pulse: J.pulse, filters: J.filters };
  net.w = new Float64Array(net.wcount.length);
  for (let i = 0; i < net.w.length; i++) net.w[i] = net.wcount[i] * J.consts.w_syn;
  return net;
}

// ---------------------------------------------------------------- the brain
function Brain(net, seed) {
  const n = net.n, C = net.c;
  this.net = net; this.n = n;
  this.v0 = C.v_0; this.vth = C.v_th; this.vrst = C.v_rst;
  this.kv = C.kv; this.kg = C.kg; this.kvg = C.kvg; this.delay = C.delay; this.kick = C.kick; this.dt = C.dt;
  this.v = new Float64Array(n).fill(C.v_0); this.g = new Float64Array(n);
  this.last = new Float64Array(n).fill(-1e9);
  this.rfc = new Int32Array(n).fill(C.rfc);
  this.isActive = new Uint8Array(n); this.isDriven = new Uint8Array(n);
  this.active = []; this.queue = []; for (let k = 0; k < this.delay; k++) this.queue.push([]);
  this.channels = []; this.i = 0; this.rand = mulberry32(seed);
}
Brain.prototype.addChannel = function (idx) {
  for (const i of idx) { this.rfc[i] = 0; this.isDriven[i] = 1; }
  this._activate(Array.from(idx));
  this.channels.push(idx);
};
Brain.prototype._activate = function (idx) {
  const fresh = [];
  for (const i of idx) if (!this.isActive[i]) fresh.push(i);
  if (!fresh.length) return;
  fresh.sort((x, y) => x - y);
  let prev = -1;
  for (const i of fresh) if (i !== prev) { this.isActive[i] = 1; this.active.push(i); prev = i; }
};
Brain.prototype._deliver = function (pre) {
  const net = this.net, g = this.g, posts = [];
  for (const p of pre) {
    const e = net.indptr[p + 1];
    for (let k = net.indptr[p]; k < e; k++) { const q = net.post[k]; g[q] += net.w[k]; posts.push(q); }
  }
  this._activate(posts);
};
Brain.prototype._prune = function () {
  const a = this.active, keep = [];
  for (const i of a) {
    if (Math.abs(this.g[i]) > 1e-7 || Math.abs(this.v[i] - this.v0) > 1e-7 || this.isDriven[i] || (this.i - this.last[i] < this.rfc[i] + 1)) keep.push(i);
    else { this.v[i] = this.v0; this.g[i] = 0; this.isActive[i] = 0; }
  }
  this.active = keep;
};
// rates: array of Hz, one per channel (in addChannel order). Returns spiking neuron indices.
Brain.prototype.step = function (rates) {
  const i = this.i, a = this.active, v = this.v, g = this.g, last = this.last, rfc = this.rfc, spk = [];
  const v0 = this.v0, kv = this.kv, kg = this.kg, kvg = this.kvg, vth = this.vth;
  for (let k = 0; k < a.length; k++) {
    const j = a[k];
    if (i - last[j] >= rfc[j]) {
      const nv = v0 + (v[j] - v0) * kv + g[j] * kvg;
      v[j] = nv; g[j] = g[j] * kg;
      if (nv > vth) spk.push(j);
    }
  }
  for (const j of spk) last[j] = i;
  const slot = i % this.delay, due = this.queue[slot];
  if (due.length) this._deliver(due);
  this.queue[slot] = spk;
  for (let c = 0; c < this.channels.length; c++) {
    const r = rates[c];
    if (r > 0) {
      const idx = this.channels[c], p = r * this.dt;
      for (let k = 0; k < idx.length; k++) if (this.rand() < p) v[idx[k]] += this.kick;
    }
  }
  for (const j of spk) { v[j] = this.vrst; g[j] = 0; }
  this.i++;
  if (i % 500 === 0) this._prune();
  return spk;
};

// ---------------------------------------------------------------- hearing: hits -> stimulation -> spikes
// hits: [[time_s, drum, strength 0..1]], drum in kick/snare/hat (+ low_tom -> like kick, shaker -> like hat)
function earRates(net, hits, dur) {
  const dt = net.c.dt, n = Math.round(dur / dt), A = new Float64Array(n), B = new Float64Array(n), w = Math.round(net.pulse / dt);
  for (const [t, drum, s] of hits) {
    const m = net.earMap[drum] || net.earMap.snare, i0 = Math.round(t / dt);
    for (const [ch, hz] of Object.entries(m)) {
      const arr = ch === 'joA' ? A : B, val = hz * (0.45 + 0.55 * s);
      for (let i = i0; i < Math.min(n, i0 + w); i++) if (arr[i] < val) arr[i] = val;
    }
  }
  return [A, B];
}
// Run a listening session. Returns {spikes: [stepIndex, neuron] pairs flattened, trains: per-neuron times}
function listen(net, hits, dur, seed, onProgress) {
  const br = new Brain(net, seed); br.addChannel(net.chA); br.addChannel(net.chB);
  const [A, B] = earRates(net, hits, dur), n = A.length, sT = [], sN = [], rates = [0, 0];
  for (let k = 0; k < n; k++) {
    rates[0] = A[k]; rates[1] = B[k];
    const s = br.step(rates);
    for (const j of s) { sT.push(k); sN.push(j); }
    if (onProgress && k % 5000 === 0) onProgress(k / n);
  }
  return { t: Uint32Array.from(sT), n: Uint16Array.from(sN) };
}
// Drum read-out: bursts of each kit neuron -> hits, shifted earlier by its reaction time
function burstOnsets(times, gap = 0.04, win = 0.02) {
  const on = [], size = [];
  for (let k = 0; k < times.length; k++) {
    if (k === 0 || times[k] - times[k - 1] > gap) {
      let c = 0; for (let j = k; j < times.length && times[j] <= times[k] + win + 1e-12; j++) c++;
      on.push(times[k]); size.push(c);
    }
  }
  return [on, size];
}
function playBack(net, spikes) {
  const dt = net.c.dt, byNeuron = {};
  for (const kname in net.kit) byNeuron[net.kit[kname].idx] = [];
  for (let k = 0; k < spikes.t.length; k++) { const arr = byNeuron[spikes.n[k]]; if (arr) arr.push(spikes.t[k] * dt); }
  const out = [];
  for (const [drum, kk] of Object.entries(net.kit)) {
    const [on, size] = burstOnsets(byNeuron[kk.idx]);
    if (!on.length) continue;
    const mx = Math.max(...size), shift = kk.extra ? 0.02 : kk.latency;
    on.forEach((t, j) => out.push([Math.max(0, t - shift), drum, Math.round(Math.min(127, Math.max(45, 127 * Math.sqrt(size[j] / mx))))]));
  }
  out.sort((a, b) => a[0] - b[0]);
  return out;
}

// ---------------------------------------------------------------- the ear: audio -> hits
// Filters: scipy Butterworth SOS + steady-state initial conditions, applied forward and backward like sosfiltfilt.
function sosfilt(sos, zi0, x, x0) {
  const y = new Float64Array(x.length), ns = sos.length;
  const z = sos.map((s, k) => [zi0[k][0] * x0, zi0[k][1] * x0]);
  for (let i = 0; i < x.length; i++) {
    let xi = x[i];
    for (let k = 0; k < ns; k++) {
      const [b0, b1, b2, , a1, a2] = sos[k], zz = z[k];
      const yi = b0 * xi + zz[0];
      zz[0] = b1 * xi - a1 * yi + zz[1];
      zz[1] = b2 * xi - a2 * yi;
      xi = yi;
    }
    y[i] = xi;
  }
  return y;
}
function sosfiltfilt(F, x) {
  const pad = Math.min(F.padlen, x.length - 1), n = x.length;
  const ext = new Float64Array(n + 2 * pad);
  for (let i = 0; i < pad; i++) { ext[i] = 2 * x[0] - x[pad - i]; ext[n + pad + i] = 2 * x[n - 1] - x[n - 2 - i]; }
  ext.set(x, pad);
  let y = sosfilt(F.sos, F.zi, ext, ext[0]);
  y.reverse();
  y = sosfilt(F.sos, F.zi, y, y[0]);
  y.reverse();
  return y.subarray(pad, pad + n);
}
function movingRms(sig, w) {
  const n = sig.length, pre = new Float64Array(n + 1);
  for (let i = 0; i < n; i++) pre[i + 1] = pre[i] + sig[i] * sig[i];
  const out = new Float64Array(n), lo = Math.floor(w / 2), hi = Math.floor((w - 1) / 2);
  for (let i = 0; i < n; i++) { const a = Math.max(0, i - lo), b = Math.min(n, i + hi + 1); out[i] = Math.sqrt(Math.max(0, pre[b] - pre[a]) / w); }
  return out;
}
function percentile(arr, p) {
  const s = Float64Array.from(arr).sort(), r = (s.length - 1) * p / 100, lo = Math.floor(r), hi = Math.ceil(r);
  return s[lo] + (s[hi] - s[lo]) * (r - lo);
}
// x: Float64Array mono at 44100 Hz. FILT: {kick, snare, hat} filter specs exported from scipy.
function detectHits(x, sr, FILT) {
  const env = (k, ms) => movingRms(sosfiltfilt(FILT[k], x), Math.max(1, Math.floor(sr * ms / 1000)));
  const hop = Math.floor(sr * 0.002), fs = sr / hop, ms = m => Math.round(m * fs / 1000);
  const E = {};
  for (const [k, m] of [['kick', 8], ['snare', 4], ['hat', 2]]) { const full = env(k, m), sub = new Float64Array(Math.ceil(full.length / hop)); for (let i = 0; i < sub.length; i++) sub[i] = full[i * hop]; E[k] = sub; }
  const N = E.kick.length, nov = new Float64Array(N);
  for (const k of ['kick', 'snare', 'hat']) {
    const v = E[k], sc = percentile(v, 99.5) + 1e-9;
    for (let i = 0; i < N; i++) {
      let m = -Infinity; for (let r = 2; r < 6; r++) m = Math.max(m, v[(i - r + N) % N]);
      const d = Math.max(0, v[i] - m) / sc; if (d > nov[i]) nov[i] = d;
    }
  }
  const ons = []; let last = -1e9;
  for (let p = 0; p < N; p++) {
    if (nov[p] > 0.12 && nov[p] >= nov[(p - 1 + N) % N] && nov[p] >= nov[(p + 1) % N] && p - last >= ms(30)) { ons.push(p); last = p; }
  }
  const mean = (arr, a, b) => { a = Math.max(0, a); b = Math.min(arr.length, b); if (b <= a) return NaN; let s = 0; for (let i = a; i < b; i++) s += arr[i]; return s / (b - a); };
  const maxv = (arr, a, b) => { let m = -Infinity; for (let i = Math.max(0, a); i < Math.min(arr.length, b); i++) m = Math.max(m, arr[i]); return m; };
  const feats = ons.map(p => {
    const pa = Math.max(0, p - ms(25)), pb = Math.max(1, p - ms(4));
    return { kick: mean(E.kick, p, p + ms(50)) - mean(E.kick, pa, pb), snare: mean(E.snare, p + ms(14), p + ms(50)) - mean(E.snare, pa, pb),
             hat: maxv(E.hat, p, p + ms(14)) - mean(E.hat, pa, pb) };
  });
  if (!feats.length) return [];
  const ref = {}; for (const k of ['kick', 'snare', 'hat']) ref[k] = percentile(feats.map(f => f[k]), 95) + 1e-9;
  const norm = feats.map(f => ({ kick: f.kick / ref.kick, snare: f.snare / ref.snare, hat: f.hat / ref.hat }));
  const sn = norm.filter(q => q.snare > 0.3).map(q => q.hat / q.snare);
  const snareHat = sn.length >= 3 ? percentile(sn, 25) : 1.0;
  const out = [];
  ons.forEach((p, j) => {
    const q = norm[j], t = Math.max(0, (p - ms(4)) / fs);
    if (q.kick > 0.3) out.push([t, 'kick', Math.min(1, q.kick)]);
    const isSnare = q.snare > 0.3 || (q.snare > 0.14 && q.hat < 2.5 * snareHat * q.snare);
    if (isSnare) out.push([t, 'snare', Math.min(1, q.snare)]);
    if (q.hat > 0.3 && (!isSnare || q.hat > 1.5 * snareHat * q.snare)) out.push([t, 'hat', Math.min(1, q.hat)]);
  });
  out.sort((a, b) => a[0] - b[0]);
  return out;
}

const api = { mulberry32, b64, loadNet, Brain, earRates, listen, burstOnsets, playBack, detectHits, sosfiltfilt };
if (typeof module !== 'undefined' && module.exports) module.exports = api; else root.FlySim = api;
})(typeof self !== 'undefined' ? self : this);

// ---------------------------------------------------------------- loop pedal (appended)
(function (root) {
const FS = (typeof module !== 'undefined' && module.exports) ? module.exports : root.FlySim;
// One continuous brain. Cycle 0 hears the clip; every later cycle hears the fly's own drums from
// the cycle before, one loop later. Returns per-cycle {spikes (relative step, neuron), fly hits}.
function LoopPedal(net, L, seed, gain) {
  this.net = net; this.L = L; this.gain = gain; this.k = 0; this.dt = net.c.dt;
  this.br = new FS.Brain(net, seed); this.br.addChannel(net.chA); this.br.addChannel(net.chB);
  this.steps = Math.round(L / this.dt);
  this.pending = [];   // ear hits scheduled in absolute time
}
LoopPedal.prototype.feed = function (hits, cycleStart, strengthScale) {
  for (const [t, d, s] of hits) if (t >= 0 && t < this.L) this.pending.push([cycleStart + t, d, Math.min(1, s * strengthScale)]);
};
LoopPedal.prototype.runCycle = function () {
  const k = this.k, t0 = k * this.L, n = this.steps, dt = this.dt, net = this.net;
  const inCycle = this.pending.filter(h => h[0] >= t0 - net.pulse && h[0] < t0 + this.L).map(h => [h[0] - t0, h[1], h[2]]);
  // rates for this cycle (pulses that started late last cycle carry over)
  const [A, B] = FS.earRates(net, inCycle.map(([t, d, s]) => [Math.max(0, t), d, s]), this.L);
  for (const arr of [A, B]) for (let i = 0; i < arr.length; i++) arr[i] *= this.gain;
  const sT = [], sN = [], rates = [0, 0];
  for (let i = 0; i < n; i++) { rates[0] = A[i]; rates[1] = B[i]; for (const j of this.br.step(rates)) { sT.push(i); sN.push(j); } }
  const spikes = { t: Uint32Array.from(sT), n: Uint16Array.from(sN) };
  const fly = FS.playBack(net, spikes).filter(h => h[0] < this.L);
  this.k++;
  return { spikes, fly, heard: inCycle };
};
FS.LoopPedal = LoopPedal;
})(typeof self !== 'undefined' ? self : this);
