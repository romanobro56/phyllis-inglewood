const H = require('../../engine/bandear.js'), fs = require('fs');
const rd = p => { const b = fs.readFileSync(p); return new Float32Array(b.buffer.slice(b.byteOffset, b.byteOffset + b.length)); };
const a = Math.round(10 * H.SR), b = Math.round(40 * H.SR), cut = x => x.slice(a, b);
const TAG = process.env.TAG || 'sv', L = cut(rd(`/tmp/${TAG}_L.f32`)), R = cut(rd(`/tmp/${TAG}_R.f32`)), names = ['drums', 'bass', 'vocal', 'keys'];
const truth = names.map(k => ({ L: cut(rd(`/tmp/${TAG}_${k}_L.f32`)), R: cut(rd(`/tmp/${TAG}_${k}_R.f32`)) }));
const en = p => { let s = 0; for (let i = 2048; i < p.L.length - 2048; i++) s += p.L[i] * p.L[i] + p.R[i] * p.R[i]; return s; };
const sdr = (t, e) => { let s = 0, d = 0; for (let i = 2048; i < t.L.length - 2048; i++) { s += t.L[i] ** 2 + t.R[i] ** 2; d += (t.L[i] - e.L[i]) ** 2 + (t.R[i] - e.R[i]) ** 2; } return 10 * Math.log10(s / d); };
function report(tag, sp, ms) {
  console.log(`\n${tag}  (${(ms / 1000).toFixed(1)} s, mono ${sp.mono})`);
  const land = truth.map(t => sp.apply(t.L, t.R).map(en)), tot = land.map(r => r.reduce((x, y) => x + y, 0));
  console.log('  where each true instrument lands (% of its energy):     ' + names.map(n => n.padStart(7)).join(''));
  truth.forEach((_, q) => console.log(`    true ${names[q].padEnd(6)} ->` + ' '.repeat(40) + land[q].map(v => (100 * v / tot[q]).toFixed(0).padStart(7)).join('')));
  console.log('  what each part is made of (% of the part):');
  sp.parts.forEach((_, p) => { const col = truth.map((_, q) => land[q][p]), s = col.reduce((x, y) => x + y, 0); console.log(`    ${names[p].padEnd(6)} part: ` + col.map((v, q) => `${names[q]} ${(100 * v / s).toFixed(0)}%`).join(', ') + `   | SDR ${sdr(truth[p], sp.parts[p]).toFixed(1)} dB`); });
  let s = 0, d = 0; for (let i = 0; i < L.length; i++) { const e = sp.parts.reduce((x, p) => x + p.L[i], 0); s += L[i] ** 2; d += (L[i] - e) ** 2; } console.log(`  parts add back up to the song: ${(10 * Math.log10(s / (d + 1e-20))).toFixed(0)} dB`);
}
if (process.env.MONO) { for (let i = 0; i < L.length; i++) { L[i] = R[i] = (L[i] + R[i]) / 2; } for (const p of truth) for (let i = 0; i < L.length; i++) p.L[i] = p.R[i] = (p.L[i] + p.R[i]) / 2; }
const which = process.argv[2] || 'both';
if (which !== 'new') { const t = Date.now(), sp = H.split4old(L, R); report('OLD split', sp, Date.now() - t); }
if (process.argv[3]) Object.assign(H.TUNE, JSON.parse(process.argv[3]));
if (which !== 'old') { const t = Date.now(), sp = H.split4(L, R); report('NEW split ' + JSON.stringify(H.TUNE), sp, Date.now() - t); }
