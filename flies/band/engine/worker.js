// worker.js - one simulated fly per worker.
// Messages in:  init {net}            the network file's text
//               prep {L, R, seconds}  a whole song at 22050 Hz: find the beat, pick an excerpt, split it into four parts
//               fly  {p, L, R, ...}   one part: antennae -> brain -> song, and the recreation
// Messages out: ready, prep, progress, fly, error
if (typeof FlyEar === 'undefined') importScripts('flysim.js', 'songear.js', 'bandear.js');
let net = null, scr = null;
function send(d) { const tr = []; for (const v of Object.values(d)) if (ArrayBuffer.isView(v) && !tr.includes(v.buffer)) tr.push(v.buffer); postMessage(d, tr); }
function down(a, k) { const o = new Float32Array(Math.ceil(a.length / k)); for (let i = 0; i < o.length; i++) { let s = 0, c = 0; for (let j = i * k; j < Math.min(a.length, (i + 1) * k); j++) { s += a[j]; c++; } o[i] = c ? s / c : 0; } return o; }
function swing(x) { const per = 220.5, n = Math.ceil(x.length / per), o = new Float32Array(n); for (let i = 0; i < n; i++) { let s = 0, c = 0; for (let j = Math.floor(i * per); j < Math.min(x.length, Math.floor((i + 1) * per)); j++) { s += x[j] * x[j]; c++; } o[i] = c ? Math.sqrt(s / c) : 0; } return o; }
function asSound(x, top) { const o = new Float32Array(x.length); for (let i = 0; i < x.length; i++) o[i] = Math.max(-0.98, Math.min(0.98, 0.6 * x[i] / top)); return o; }
function prep(m) {
  const H = FlyEar, { L, R } = m, mid = new Float32Array(L.length); for (let i = 0; i < L.length; i++) mid[i] = (L[i] + R[i]) / 2;
  const F = SongEar.analyze(mid, H.SR), Tb = SongEar.beats(F), X = SongEar.pickExcerpt(F, Tb, m.seconds);
  if (!(X.bars >= 1)) throw new Error('That clip is too short to find a beat in.');
  const a = Math.max(0, Math.round(X.start * H.SR)), b = Math.min(L.length, Math.round(X.end * H.SR) + 256), l = L.slice(a, b), r = R.slice(a, b);
  const sp = H.split4(l, r), mixLoud = H.loudness(l, r), nMs = Math.ceil(l.length / H.SR * 1000);
  const out = { type: 'prep', session: m.session, start: a, len: b - a, bpm: Tb.bpm, bars: X.bars, startBar: X.startBar, slot: X.slot, n: X.bars * 16, mono: sp.mono, nMs, v0: [] };
  sp.parts.forEach((part, p) => { out.v0.push(Math.min(H.EAR.loud / H.loudness(part.L, part.R), 4 * H.EAR.loud / mixLoud)); out['L' + p] = part.L; out['R' + p] = part.R; out['env' + p] = down(H.envelope(part.L, part.R, nMs), 10); });
  send(out);
}
function flyJob(m) {
  const H = FlyEar, brain = m.scrambled ? (scr || (scr = H.scramble(net, 1))) : net, part = { L: m.L, R: m.R }, grid = { t0: 0, slot: m.slot, n: m.n };
  const f = H.fly(brain, part.L, part.R, m.seed, v => postMessage({ type: 'progress', session: m.session, key: m.key, p: m.p, v }), { v0: m.v0 }), b = f.brain, s = f.song;
  const src = H.choose(H.brainStates(b, grid), m.seed + 3), re = H.assemble(part, grid, src), match = H.matchScore(H.slotSpectra(part, grid), src);
  const mag = []; for (let i = 0; i < f.ear.dispL.length; i += 5) mag.push(Math.max(Math.abs(f.ear.dispL[i]), Math.abs(f.ear.dispR[i]))); mag.sort((x, y) => x - y); const top = Math.max(1, mag[Math.floor(0.999 * (mag.length - 1))] || 1);
  const cl = v => Math.max(-0.98, Math.min(0.98, v * s.gain));
  send({ type: 'fly', session: m.session, key: m.key, p: m.p, antL: asSound(f.ear.dispL, top), antR: asSound(f.ear.dispR, top), songL: Float32Array.from(s.L, cl), songR: Float32Array.from(s.R, cl), reL: re.L, reR: re.R,
    pulses: Float32Array.from(s.pulses.flat()), hum: down(s.hum, 5), side: s.side.filter((_, i) => i % 5 === 0), swL: swing(f.ear.dispL), swR: swing(f.ear.dispR), envRe: down(H.envelope(re.L, re.R, b.nMs), 10), t: b.t, n: b.n, src,
    stats: { nFired: b.nFired, nDN: b.nDN, gf: b.gf.length, bf: f.ear.bf, match, silent: s.silent, clicks: s.pulses.length } });
}
onmessage = e => {
  const m = e.data;
  try {
    if (m.type === 'init') { net = FlyEar.loadNet(JSON.parse(m.net)); postMessage({ type: 'ready' }); }
    else if (m.type === 'prep') prep(m);
    else if (m.type === 'fly') flyJob(m);
  } catch (err) { postMessage({ type: 'error', session: m.session, message: String((err && err.message) || err) }); }
};
