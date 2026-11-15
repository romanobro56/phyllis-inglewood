const H = require('../../engine/bandear.js'), SE = require('../../engine/songear.js'), fs = require('fs');
const net0 = H.loadNet(JSON.parse(fs.readFileSync(__dirname + '/../../data/network.json')));
const rd = p => { const b = fs.readFileSync(p); return new Float32Array(b.buffer.slice(b.byteOffset, b.byteOffset + b.length)); };
function wav(path, chans, sr) { const n = chans[0].length, C = chans.length, buf = Buffer.alloc(44 + n * 2 * C); buf.write('RIFF', 0); buf.writeUInt32LE(36 + n * 2 * C, 4); buf.write('WAVEfmt ', 8); buf.writeUInt32LE(16, 16); buf.writeUInt16LE(1, 20); buf.writeUInt16LE(C, 22); buf.writeUInt32LE(sr, 24); buf.writeUInt32LE(sr * 2 * C, 28); buf.writeUInt16LE(2 * C, 32); buf.writeUInt16LE(16, 34); buf.write('data', 36); buf.writeUInt32LE(n * 2 * C, 40); for (let i = 0; i < n; i++) for (let c = 0; c < C; c++) buf.writeInt16LE(Math.max(-32767, Math.min(32767, Math.round(chans[c][i] * 32767))), 44 + 2 * (i * C + c)); fs.writeFileSync(path, buf); }
const arg = k => { const i = process.argv.indexOf('--' + k); return i > 0 ? process.argv[i + 1] : null; };
const net = arg('scramble') ? H.scramble(net0, +arg('scramble')) : net0, tag = arg('tag') || 'real';
const TAG = process.env.TAG || 'st', Lf = rd(`/tmp/${TAG}_L.f32`), Rf = rd(`/tmp/${TAG}_R.f32`), mid = Float32Array.from(Lf, (v, i) => (v + Rf[i]) / 2);
let t = Date.now(); const lap = s => { const d = ((Date.now() - t) / 1000).toFixed(1); t = Date.now(); return `${s} ${d}s`; };
const F = SE.analyze(mid, H.SR), Tb = SE.beats(F), X = SE.pickExcerpt(F, Tb, 30);
const a = Math.max(0, Math.round(X.start * H.SR)), b = Math.min(Lf.length, Math.round(X.end * H.SR) + 256), L = Lf.slice(a, b), R = Rf.slice(a, b);
const grid = { t0: 0, slot: X.slot, n: X.bars * 16 };
console.log(`${tag}: ${Tb.bpm.toFixed(0)} bpm, bars ${X.startBar + 1}-${X.startBar + X.bars}, ${grid.n} slots of ${(1000 * X.slot).toFixed(0)} ms | ${lap('analysis')}`);
const sp = H.split4(L, R); console.log(lap('split'), 'mono', sp.mono);
const mixLoud = H.loudness(L, R), mix = [new Float32Array(L.length), new Float32Array(L.length)];
sp.parts.forEach((part, p) => {
  const pl = H.loudness(part.L, part.R), v0 = Math.min(H.EAR.loud / pl, 4 * H.EAR.loud / mixLoud);
  const f = H.fly(net, part.L, part.R, 1000 + 97 * p, null, { v0 }), br = f.brain, S = H.brainStates(br, grid), src = H.choose(S, 5 + p), out = H.assemble(part, grid, src), Sp = H.slotSpectra(part, grid);
  const rnd = Int32Array.from(src, (_, i) => { let c; do c = Math.floor(Math.random() * grid.n); while (Math.abs(c - i) < 4); return c; });
  const samePos = src.filter((s, i) => (s - i) % 16 === 0).length, sameHalf = src.filter((s, i) => (s - i) % 8 === 0).length;
  console.log(`${H.PARTS[p].padEnd(6)} played x${(v0 * mixLoud / H.EAR.loud).toFixed(1)} | neurons fired ${br.nFired}, outputs ${br.nDN}, brain spikes ${br.deep} (${S.neurons} cells) | song: ${f.song.pulses.length} clicks${f.song.silent ? ' SILENT' : ''} | match ${(100 * H.matchScore(Sp, src)).toFixed(0)}% (random choice ${(100 * H.matchScore(Sp, rnd)).toFixed(0)}%) | same place in another bar ${(100 * samePos / grid.n).toFixed(0)}%, same half-bar place ${(100 * sameHalf / grid.n).toFixed(0)}% (chance 6% / 12%) | ${lap('')}`);
  for (let i = 0; i < L.length; i++) { mix[0][i] += out.L[i]; mix[1][i] += out.R[i]; }
  wav(`/tmp/be_${tag}_${H.PARTS[p]}.wav`, [out.L, out.R], H.SR);
});
wav(`/tmp/be_${tag}_band.wav`, mix, H.SR);
const e0 = H.envelope(L, R, Math.ceil(L.length / H.SR * 1000)), e1 = H.envelope(mix[0], mix[1], Math.ceil(L.length / H.SR * 1000));
console.log(`band recreation loudness vs record r=${H.corrLag(e0, e1, 10).r.toFixed(2)}`);
