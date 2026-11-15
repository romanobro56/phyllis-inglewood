// app.js - the page: controls, sound, and drawing. No modelling happens here; that is all in engine/.
(async () => {
'use strict';
// The data and the engine live in separate files. The single-file build (tools/build_standalone.py) puts them
// in window.FLYBAND_INLINE instead, so that one .html file works on its own.
const INL = window.FLYBAND_INLINE || null;
let P, NET_TEXT;
try {
  P = INL ? INL.drawing : await (await fetch('data/drawing.json')).json();
  NET_TEXT = INL ? INL.network : await (await fetch('data/network.json')).text();
} catch (e) {
  document.getElementById('status').innerHTML = '<strong>Could not load the data files</strong>Serve this folder over HTTP (see README.md); browsers block these files when a page is opened straight from disk.';
  return;
}
const NF = 4, SR = 22050, DT = 1e-4, EXCERPT = 30;
const PARTS = [{ name: 'Drums', color: '#ff7a5c' }, { name: 'Bass', color: '#ffbf47' }, { name: 'Vocals', color: '#c6f36b' }, { name: 'Keys', color: '#6cc9ff' }];
const reduceMotion = window.matchMedia('(prefers-reduced-motion: reduce)').matches;
const COL = { 1: '#38d8ff', 2: '#ffa63d', 3: '#7fe08a', 4: '#7fe08a', 5: '#ff7ac8', 6: '#ffe27a', 7: '#ffffff', 0: '#9aa8d0' }, SONG = '#d9ccff';
function b64(s, Type) { const bin = atob(s), u8 = new Uint8Array(bin.length); for (let i = 0; i < bin.length; i++) u8[i] = bin.charCodeAt(i); return new Type(u8.buffer); }
const bg = b64(P.bg, Int16Array), pos = b64(P.pos, Int16Array), role = b64(P.role, Uint8Array), reach = b64(P.reach, Uint8Array);
const NA = role.length, NBG = bg.length / 3, M = P.meta;
const fmt = n => n.toLocaleString('en-US');
document.getElementById('note').innerHTML = `
<p><b>Real:</b> the wiring of one fly brain (FlyWire, a female) with a published neuron model, copied four times. Each antenna is the single swinging feather it is: it retunes itself between about 200 and 800&nbsp;Hz depending on loudness, and nothing higher gets in. Its movement drives the five real classes of antennal neurons on each side. Sound can reach ${fmt(M.n_reach)} of the model's ${fmt(M.n_full)} neurons, among them ${M.n_dn_reach} of its ${fmt(M.n_dn_full)} output neurons.</p>
<p><b>Not the fly:</b> splitting your song into drums, bass, vocals and keys is ordinary signal processing. Each fly is then played one part, loud.</p>
<p><b>Their songs:</b> each fly's output neurons drive a stand-in wing (the real wing circuit lies outside this data set) that sings real fly song: clicks 35&nbsp;ms apart and a 150&nbsp;Hz hum. No sound from your record is in it.</p>
<p><b>Their recreation:</b> each part is cut into sixteenth-note slots. Every slot is replaced by a <i>different</i> slot of the same part, at least a beat away: the one during which that fly's brain (not its ears) behaved most alike. Nothing plays back in order and nothing is fitted to the music. Where it sounds right, the fly's brain can tell those moments apart; where it sounds wrong, it cannot. The percentage under each fly says how much better than chance its choices sound.</p>
<p><b>Check it:</b> switch the wiring to scrambled: the same neurons and connection strengths, but every connection lands on a random neuron.</p>`;

// ---------- four workers, one brain each; the first also splits the song ----------
const workerURL = INL ? URL.createObjectURL(new Blob([INL.engine], { type: 'text/javascript' })) : 'engine/worker.js';
const WK = [];
function makeWorker(i) { if (WK[i]) WK[i].w.terminate(); const w = new Worker(workerURL); WK[i] = { w, ready: false, busy: 0 }; w.onmessage = e => onMsg(i, e.data); w.postMessage({ type: 'init', net: NET_TEXT }); }
for (let i = 0; i < NF; i++) makeWorker(i);
const whenReady = i => new Promise(res => { const iv = setInterval(() => { if (WK[i].ready) { clearInterval(iv); res(); } }, 50); });

// ---------- audio: per fly, four things to hear, then mute/solo ----------
const KINDS = ['rec', 'ant', 'song', 're'];
let actx = null, master = null, analyser = null, tdBuf = null, bus = null, live = [];
function ensureAudio() {
  if (actx) return actx;
  const AC = window.AudioContext || window.webkitAudioContext; if (!AC) return null;
  actx = new AC(); master = actx.createGain(); master.gain.value = 0.9;
  analyser = actx.createAnalyser(); analyser.fftSize = 1024; tdBuf = new Float32Array(1024);
  master.connect(analyser); analyser.connect(actx.destination);
  bus = PARTS.map((_, i) => {
    const out = actx.createGain(); out.gain.value = 1; out.connect(master); const o = { out };
    for (const k of KINDS) {
      const g = actx.createGain(); g.gain.value = 0;
      if (k === 'song' && actx.createStereoPanner) { const pan = actx.createStereoPanner(); pan.pan.value = (i - 1.5) * 0.4; g.connect(pan); pan.connect(out); } else g.connect(out);   // the four singers stand left to right
      o[k] = g;
    }
    return o;
  });
  return actx;
}
function silence() { for (const n of live) { try { n.stop(); } catch (e) {} } live = []; }
let hear = 'rec', userChose = false;
const mute = [false, false, false, false], solo = [false, false, false, false];
const LEVEL = { rec: 1, ant: 0.5, song: 0.5, re: 1 };
function applyMix() {
  if (!actx) return; const t = actx.currentTime, set = (g, v) => g.gain.setTargetAtTime(v, t, 0.03), on = mode === 'play', anySolo = solo.some(Boolean);
  bus.forEach((b, i) => { for (const k of KINDS) set(b[k], on && hear === k ? LEVEL[k] : 0); set(b.out, !mute[i] && (!anySolo || solo[i]) ? 1 : 0); });
}
const stereo = (l, r) => { const b = actx.createBuffer(2, l.length, SR); b.copyToChannel(l, 0); b.copyToChannel(r, 1); return b; };

// ---------- state ----------
// mode: idle | loading (splitting the song) | play | done (kept, not playing)
let song = null, session = 0, seedBase = 1, prep = null, scrambled = 0, mode = 'idle', sceneStart = 0, cache = new Map(), cur = null;
const dur = () => prep ? prep.len / SR : 1;
function setStatus(big, small) { const html = `<strong>${big}</strong>${small || ''}`, el = document.getElementById('status'); if (el.dataset.h !== html) { el.innerHTML = html; el.dataset.h = html; } }
const have = p => !!(cur && cur.flies[p]);
const nReady = () => cur ? cur.flies.filter(Boolean).length : 0;
function setButtons() {
  const playing = mode === 'play', kept = !!prep, wait = mode === 'loading';
  document.getElementById('go-icon').setAttribute('d', wait || playing ? 'M2.5 2.5h9v9h-9z' : 'M3 1.5v11l9.5-5.5z');
  document.getElementById('go-tx').textContent = wait ? 'Cancel' : playing ? 'Stop' : kept ? 'Play' : 'Start the band';
  document.getElementById('go').disabled = !wait && !kept && !song;
  document.querySelectorAll('[data-hear]').forEach(b => { b.disabled = !prep || (b.dataset.hear !== 'rec' && nReady() === 0); b.setAttribute('aria-pressed', b.dataset.hear === hear); });
  document.querySelectorAll('[data-scr]').forEach(b => b.setAttribute('aria-pressed', +b.dataset.scr === scrambled));
}
function idleStatus() { if (!song) setStatus('Upload a song', `MP3, WAV or M4A · four flies each get one part of its liveliest ${EXCERPT} seconds`); else setStatus(song.name, 'ready: drums, bass, vocals and keys, one fly each'); }
function stop() {   // forget everything about this run
  session++; silence(); for (let i = 0; i < NF; i++) if (WK[i].busy > 0) makeWorker(i);
  cache = new Map(); cur = null; prep = null; mode = 'idle'; hear = 'rec'; userChose = false;
  for (const h of heads) h.last.fill(-1e9);
  scoreDirty = true; setButtons(); applyMix(); updateChips();
}
document.getElementById('upload').onclick = () => document.getElementById('file').click();
document.getElementById('file').onchange = e => { const f = e.target.files[0]; e.target.value = ''; if (!f) return; stop(); song = { name: f.name.replace(/\.[^.]+$/, '').slice(0, 40), file: f }; markSrc(); idleStatus(); };
function markSrc() { const up = document.getElementById('upload'); up.textContent = song ? `${song.name.slice(0, 22)}${song.name.length > 22 ? '…' : ''}` : 'Upload a song'; up.title = song ? 'Choose a different song' : ''; setButtons(); }
async function resample(buf, sr, from, d) {
  const off = new OfflineAudioContext(2, Math.max(1, Math.ceil(d * sr)), sr), src = off.createBufferSource();
  src.buffer = buf; src.connect(off.destination); src.start(0, from, d);
  const r = await off.startRendering(); return [r.getChannelData(0).slice(), r.getChannelData(1).slice()];
}
async function start() {
  if (!song) return;
  if (!ensureAudio()) { setStatus('No audio', 'This browser has no Web Audio.'); return; }
  if (actx.state === 'suspended') await actx.resume();
  stop(); mode = 'loading'; setButtons(); setStatus('Reading your song', 'decoding the file');
  const my = session; seedBase = 1000 + Math.floor(Math.random() * 1e6);
  let L, R;
  try { if (!song.buffer) song.buffer = await actx.decodeAudioData(await song.file.arrayBuffer()); [L, R] = await resample(song.buffer, SR, 0, Math.min(song.buffer.duration, 420)); }
  catch (e) { if (my === session) { stop(); setStatus('Could not read that file', 'Try an MP3, WAV or M4A.'); } return; }
  if (my !== session) return;
  setStatus('Splitting your song', 'into drums, bass, vocals and keys');
  await whenReady(0); if (my !== session) return;
  WK[0].busy++; WK[0].w.postMessage({ type: 'prep', session: my, L, R, seconds: EXCERPT }, [L.buffer, R.buffer]);
}
// The flies for the current wiring: use what is kept, or set all four listening.
function request() {
  if (!prep) return;
  const key = String(scrambled), was = mode === 'play' ? actx.currentTime - sceneStart : -1;
  if (!cache.has(key)) {
    cache.set(key, { key, flies: new Array(NF).fill(null), prog: new Float32Array(NF) });
    PARTS.forEach((_, p) => { const L = prep.parts[p].L.slice(), R = prep.parts[p].R.slice(), my = session; WK[p].busy++;
      whenReady(p).then(() => { if (my === session) WK[p].w.postMessage({ type: 'fly', session: my, key, p, L, R, v0: prep.v0[p], slot: prep.slot, n: prep.n, seed: seedBase + 97 * p, scrambled: !!scrambled }, [L.buffer, R.buffer]); }); });
  }
  cur = cache.get(key);
  for (const h of heads) h.last.fill(-1e9);
  scoreDirty = true; updateChips();
  if (was >= 0) playFrom(Math.min(was, dur() - 1)); else setButtons();
}
function onMsg(i, m) {
  if (m.type === 'ready') { WK[i].ready = true; return; }
  if (m.session !== session) return;
  if (m.type === 'error') { stop(); setStatus('Something went wrong', m.message); return; }
  if (m.type === 'prep') {
    WK[0].busy--;
    prep = { start: m.start, len: m.len, bpm: m.bpm, bars: m.bars, startBar: m.startBar, slot: m.slot, n: m.n, mono: m.mono, nMs: m.nMs, v0: m.v0,
      parts: PARTS.map((_, p) => ({ L: m['L' + p], R: m['R' + p], env: m['env' + p], buf: stereo(m['L' + p], m['R' + p]) })) };
    request(); playFrom(0); return;
  }
  const res = cache.get(m.key); if (!res) return;
  if (m.type === 'progress') { res.prog[m.p] = m.v; return; }
  if (m.type === 'fly') {
    WK[i].busy--;
    const f = { ant: stereo(m.antL, m.antR), song: stereo(m.songL, m.songR), re: stereo(m.reL, m.reR), pulses: m.pulses, hum: m.hum, side: m.side, swL: m.swL, swR: m.swR, envRe: m.envRe, t: m.t, n: m.n, src: m.src, stats: m.stats, ptr: 0, pp: 0, flash: 0, flashT: -1 };
    res.flies[m.p] = f; res.prog[m.p] = 1;
    if (res !== cur) return;
    scoreDirty = true; updateChips();
    if (mode === 'play') { const local = actx.currentTime - sceneStart; if (local < dur() - 0.3) for (const k of ['ant', 'song', 're']) source(f[k], bus[m.p][k], actx.currentTime + 0.05, Math.max(0, local + 0.05)); }
    if (nReady() === NF && !userChose) { hear = 're'; scoreDirty = true; applyMix(); }   // first time round: slide from the record into the recreation
    setButtons();
  }
}
function source(buf, dst, at, off, d) { const n = actx.createBufferSource(); n.buffer = buf; n.connect(dst); if (d === undefined) n.start(at, off); else n.start(at, off, d); live.push(n); }
function playFrom(from) {
  if (!prep) return;
  silence(); mode = 'play'; from = Math.max(0, Math.min(from, dur() - 0.2)); sceneStart = actx.currentTime + 0.12 - from;
  const at = sceneStart + from;
  PARTS.forEach((_, p) => { source(prep.parts[p].buf, bus[p].rec, at, from); const f = cur && cur.flies[p]; if (f) for (const k of ['ant', 'song', 're']) source(f[k], bus[p][k], at, from); });
  for (const h of heads) h.last.fill(-1e9);
  if (cur) for (const f of cur.flies) if (f) { const lim = from / DT; let lo = 0, hi = f.t.length; while (lo < hi) { const md = (lo + hi) >> 1; if (f.t[md] < lim) lo = md + 1; else hi = md; } f.ptr = lo; f.pp = 0; while (f.pp < f.pulses.length && f.pulses[f.pp] < from * 1000) f.pp += 3; f.flash = 0; }
  applyMix(); setButtons();
}
function halt() { silence(); mode = 'done'; applyMix(); setButtons(); }
function playStatus() {
  const pre = scrambled ? 'Scrambled wiring · ' : '', left = NF - nReady(), wait = left ? ` · ${left === NF ? 'the flies are' : left + (left === 1 ? ' fly is' : ' flies are')} still listening` : '';
  if (hear === 'rec') setStatus('Your song', `${Math.round(prep.bpm)} bpm · bars ${prep.startBar + 1}–${prep.startBar + prep.bars} · one part played to each fly${prep.mono ? ' · mono file: “vocals” is just the voice range' : ''}${wait}`);
  else if (hear === 'ant') setStatus('What their antennae feel', `${pre}each part as its fly's antennae move to it · the highs never arrive${wait}`);
  else if (hear === 'song') setStatus('The flies sing along', `${pre}each one's output neurons drive its wing · no sound from your record${wait}`);
  else setStatus("The flies' recreation", `${pre}every moment swapped for another one its brain found most alike${wait}`);
}
document.querySelectorAll('[data-hear]').forEach(b => b.onclick = () => { hear = b.dataset.hear; userChose = true; if (mode !== 'play' && prep) playFrom(0); else { applyMix(); setButtons(); } scoreDirty = true; });
document.querySelectorAll('[data-scr]').forEach(b => b.onclick = () => { scrambled = +b.dataset.scr; setButtons(); request(); });
document.getElementById('go').onclick = () => {
  if (mode === 'loading') { stop(); idleStatus(); }
  else if (mode === 'play') halt();
  else if (prep) playFrom(0);
  else start();
};
document.addEventListener('keydown', e => { if (e.code === 'Space' && document.activeElement.tagName !== 'BUTTON') { e.preventDefault(); document.getElementById('go').click(); } });

// ---------- the strip: each part's loudness over time ----------
const sc = document.getElementById('sc'), sg = sc.getContext('2d'), cap = document.getElementById('cap');
let scoreDirty = true, scoreImg = null;
sc.onclick = e => { if (!prep) return; const r = sc.getBoundingClientRect(); playFrom((e.clientX - r.left) / r.width * dur()); };
function drawScore() {
  const dpr = Math.min(window.devicePixelRatio || 1, 2), w = sc.clientWidth, h = sc.clientHeight;
  sc.width = Math.round(w * dpr); sc.height = Math.round(h * dpr);
  scoreImg = document.createElement('canvas'); scoreImg.width = sc.width; scoreImg.height = sc.height;
  const c = scoreImg.getContext('2d'), W2 = sc.width, H2 = sc.height, rh = H2 / NF;
  c.fillStyle = '#0a0d14'; c.fillRect(0, 0, W2, H2);
  c.strokeStyle = 'rgba(255,255,255,0.05)'; for (let r = 1; r < NF; r++) { c.beginPath(); c.moveTo(0, r * rh); c.lineTo(W2, r * rh); c.stroke(); }
  cap.textContent = !prep ? '' : hear === 'rec' ? 'each part as played · click to jump' : `faint: the part as played · bright: ${hear === 'ant' ? "how far its fly's antennae swing" : hear === 'song' ? "its fly's song (ticks: clicks, band: hum)" : "its fly's recreation"} · click to jump`;
  if (!prep) return;
  const top = a => { let m = 1e-9; for (const v of a) m = Math.max(m, v); return m; };
  const path = (row, arr, t, fill, color, alpha) => {
    const y0 = (row + 1) * rh - 1, hgt = rh - 3, n = arr.length; c.beginPath(); if (fill) c.moveTo(0, y0);
    for (let x = 0; x < W2; x++) { const y = y0 - Math.min(1, arr[Math.min(n - 1, Math.floor(x / W2 * n))] / t) * hgt; if (x || fill) c.lineTo(x, y); else c.moveTo(x, y); }
    c.globalAlpha = alpha; if (fill) { c.lineTo(W2, y0); c.closePath(); c.fillStyle = color; c.fill(); } else { c.strokeStyle = color; c.lineWidth = 1.2 * dpr; c.stroke(); } c.globalAlpha = 1;
  };
  PARTS.forEach((pt, p) => {
    const env = prep.parts[p].env, t0 = top(env), f = cur && cur.flies[p];
    path(p, env, t0, true, pt.color, hear === 'rec' ? 0.6 : 0.2);
    if (!f || hear === 'rec') return;
    if (hear === 're') path(p, f.envRe, t0, false, pt.color, 0.95);
    else if (hear === 'ant') { const sw = Float32Array.from(f.swL, (v, i) => (v + f.swR[i]) / 2); path(p, sw, top(sw), false, pt.color, 0.95); }
    else { path(p, f.hum, 0.36, true, pt.color, 0.55); c.fillStyle = '#ffffff'; const y0 = (p + 1) * rh - 1; for (let q = 0; q < f.pulses.length; q += 3) { const hh = Math.max(2, (rh - 3) * Math.min(1, f.pulses[q + 2])); c.globalAlpha = 0.9; c.fillRect(Math.round(f.pulses[q] / prep.nMs * W2), y0 - hh, Math.max(1, dpr), hh); } c.globalAlpha = 1; }
  });
}

// ---------- heads ----------
const stage = document.getElementById('stage'), cv = document.getElementById('cv'), g = cv.getContext('2d');
const heads = Array.from({ length: NF }, () => ({ sx: new Float32Array(NA), sy: new Float32Array(NA), last: new Float64Array(NA).fill(-1e9), cx: 0, cy: 0, u: 100 }));
let W = 0, H = 0, dpr = 1, layer = null, layerDirty = true;
const sprites = {};
function sprite(color) {
  if (sprites[color]) return sprites[color];
  const s = document.createElement('canvas'); s.width = s.height = 64; const c = s.getContext('2d'), gr = c.createRadialGradient(32, 32, 0, 32, 32, 32);
  gr.addColorStop(0, '#ffffff'); gr.addColorStop(0.12, color); gr.addColorStop(0.4, color + '66'); gr.addColorStop(1, color + '00');
  c.fillStyle = gr; c.fillRect(0, 0, 64, 64); return (sprites[color] = s);
}
function ell(c, x, y, rx, ry, rot = 0) { c.beginPath(); c.ellipse(x, y, rx, ry, rot, 0, Math.PI * 2); }
const chipsEl = document.getElementById('chips'), chipEls = PARTS.map((pt, i) => {
  const d = document.createElement('div'); d.className = 'chip'; d.style.setProperty('--c', pt.color);
  d.innerHTML = `<span class="hd"><b>${pt.name}</b><button aria-pressed="false" title="Mute ${pt.name}">M</button><button aria-pressed="false" title="Solo ${pt.name}">S</button></span><small></small><small></small>`;
  const [mb, sb] = d.querySelectorAll('button');
  mb.onclick = () => { mute[i] = !mute[i]; mb.setAttribute('aria-pressed', mute[i]); applyMix(); };
  sb.onclick = () => { solo[i] = !solo[i]; sb.setAttribute('aria-pressed', solo[i]); applyMix(); };
  chipsEl.appendChild(d); return d;
});
function updateChips() {
  chipEls.forEach((d, i) => {
    const [a, b] = d.querySelectorAll('small'), f = cur && cur.flies[i];
    if (f) { const s = f.stats; a.innerHTML = `<em>${fmt(s.nFired)}</em> neurons fired · <em>${s.nDN}</em> outputs`; b.innerHTML = `tells moments apart <em>${Math.round(100 * s.match)}%</em>`; b.title = 'How much more alike than chance the swapped-in moments sound to the ones they replace. 0% is chance, 100% is identical.'; }
    else { a.textContent = !cur ? `hears the ${PARTS[i].name.toLowerCase()}` : cur.prog[i] > 0 ? `listening ${Math.round(100 * cur.prog[i])}%` : 'getting ready'; b.textContent = ''; }
  });
}
const scoreEl = document.getElementById('score'), legendEl = document.getElementById('legend');
function layout() {
  const r = stage.getBoundingClientRect(); W = r.width; H = r.height; dpr = Math.min(window.devicePixelRatio || 1, 2);
  cv.width = Math.round(W * dpr); cv.height = Math.round(H * dpr); scoreDirty = true; layerDirty = true;
  legendEl.style.bottom = `${scoreEl.offsetHeight + 16}px`;
  const top = W < 980 ? 118 : 104, bottom = H - scoreEl.offsetHeight - 16 - legendEl.offsetHeight - 10, chipH = 58, usable = W - 40;
  const u = Math.max(60, Math.min(usable / NF / 1.3, (bottom - top - chipH) / 0.98)), slack = Math.max(0, bottom - top - chipH - 0.98 * u);
  heads.forEach((h, i) => {
    h.u = u; h.cx = 20 + usable * (i + 0.5) / NF; h.cy = top + slack / 2 + 0.36 * u;
    const k = 0.40 * u / 400; for (let j = 0; j < NA; j++) { h.sx[j] = h.cx + pos[j * 3] / 20 * k; h.sy[j] = h.cy + 0.02 * u + pos[j * 3 + 1] / 20 * k; }
    chipEls[i].style.left = `${h.cx}px`; chipEls[i].style.top = `${h.cy + 0.64 * u}px`;
  });
}
function drawLayer() {
  layer = document.createElement('canvas'); layer.width = cv.width; layer.height = cv.height;
  const b = layer.getContext('2d'); b.setTransform(dpr, 0, 0, dpr, 0, 0);
  for (const h of heads) {
    const { cx, cy, u } = h;
    ell(b, cx, cy, 0.5 * u, 0.34 * u); b.fillStyle = '#080b12'; b.fill(); b.strokeStyle = '#273145'; b.lineWidth = 1.2; b.stroke();
    b.fillStyle = '#1b2232'; [[0, -0.305], [-0.022, -0.28], [0.022, -0.28]].forEach(([x, y]) => { ell(b, cx + x * u, cy + y * u, 0.008 * u, 0.008 * u); b.fill(); });
    b.strokeStyle = '#273145'; b.fillStyle = '#080b12';
    b.beginPath(); b.moveTo(cx - 0.032 * u, cy + 0.325 * u); b.lineTo(cx - 0.026 * u, cy + 0.37 * u); b.lineTo(cx + 0.026 * u, cy + 0.37 * u); b.lineTo(cx + 0.032 * u, cy + 0.325 * u); b.fill(); b.stroke();
    b.fillStyle = 'rgba(120,145,200,0.13)'; const k = 0.40 * u / 400, stepBg = u > 260 ? 1 : u > 150 ? 2 : 4, dot = Math.max(0.8, Math.min(1.25, u / 300));
    for (let i = 0; i < NBG; i += stepBg) b.fillRect(cx + bg[i * 3] / 20 * k, cy + 0.02 * u + bg[i * 3 + 1] / 20 * k, dot, dot);
    for (const s of [-1, 1]) {
      const ex = cx + s * 0.305 * u, ey = cy - 0.005 * u, rx = 0.19 * u, ry = 0.285 * u, rot = s * 0.12;
      b.save(); ell(b, ex, ey, rx, ry, rot); b.clip(); b.fillStyle = 'rgba(150,32,44,0.08)'; b.fillRect(ex - rx * 1.3, ey - ry * 1.2, rx * 2.6, ry * 2.4); b.restore();
      ell(b, ex, ey, rx, ry, rot); b.strokeStyle = 'rgba(214,80,92,0.30)'; b.lineWidth = 1.2; b.stroke();
    }
    b.globalCompositeOperation = 'lighter'; const sc2 = u / 900;   // every neuron sound can reach, faintly
    for (let i = 0; i < NA; i++) { if (!reach[i]) continue; const big = role[i] >= 6, rad = (big ? 9 : 6) * sc2; b.globalAlpha = big ? 0.2 : 0.07; b.drawImage(sprite(COL[role[i]]), h.sx[i] - rad, h.sy[i] - rad, rad * 2, rad * 2); }
    b.globalAlpha = 1; b.globalCompositeOperation = 'source-over';
  }
}
function drawAntenna(h, s, angle, glow) {
  const { cx, cy, u } = h, ax = cx + s * 0.05 * u, ay = cy - 0.205 * u, px = cx + s * 0.062 * u, py = cy - 0.162 * u;
  g.lineWidth = 1.1; g.strokeStyle = '#5d6984'; g.fillStyle = 'rgba(12,16,25,0.55)';
  ell(g, ax, ay, 0.016 * u, 0.014 * u); g.fill(); g.stroke();
  if (glow > 0.01) { const rr = (0.05 + 0.06 * glow) * u; g.save(); g.globalCompositeOperation = 'lighter'; g.globalAlpha = Math.min(1, glow); g.drawImage(sprite(COL[1]), px - rr, py - rr, rr * 2, rr * 2); g.restore(); }
  ell(g, px, py, 0.024 * u, 0.027 * u); g.fill(); g.stroke();
  g.save(); g.translate(px, py + 0.022 * u); g.rotate(s * angle);
  ell(g, s * 0.006 * u, 0.045 * u, 0.025 * u, 0.048 * u, s * 0.12); g.fill(); g.stroke();
  g.strokeStyle = '#8a95ad'; g.lineWidth = 1.1; g.beginPath(); g.moveTo(s * 0.022 * u, 0.015 * u); g.quadraticCurveTo(s * 0.11 * u, -0.01 * u, s * 0.19 * u, -0.1 * u); g.stroke();
  g.restore();
}
// A fly sings with one wing held out; the other stays folded.
function drawWing(h, s, out, amp, now) {
  const { cx, cy, u } = h, len = (out ? 0.38 : 0.24) * u, bx = cx + s * 0.1 * u, by = cy + 0.37 * u;
  const ang = (out ? 0.3 : 1.25) + (out && !reduceMotion ? Math.min(0.14, amp * 0.2) * Math.sin(now / 1000 * 2 * Math.PI * 24) : 0), k = out ? 0.45 + 0.55 * Math.min(1, amp * 1.5) : 0.3;
  g.save(); g.translate(bx, by); g.scale(s, 1); g.rotate(ang);
  g.fillStyle = `rgba(200,205,245,${0.1 * k})`; g.strokeStyle = `rgba(200,205,245,${0.7 * k})`; g.lineWidth = 1;
  g.beginPath(); g.moveTo(0, 0); g.bezierCurveTo(len * 0.3, -0.1 * u, len * 0.85, -0.09 * u, len, -0.01 * u); g.bezierCurveTo(len * 0.8, 0.05 * u, len * 0.3, 0.045 * u, 0, 0); g.fill(); g.stroke();
  g.strokeStyle = `rgba(200,205,245,${0.35 * k})`; for (const [y1, y2] of [[-0.045, -0.035], [-0.01, -0.01], [0.022, 0.018]]) { g.beginPath(); g.moveTo(len * 0.06, 0); g.quadraticCurveTo(len * 0.5, y1 * u, len * 0.93, y2 * u); g.stroke(); }
  g.restore();
  if (out && amp > 0.03) { const tx = bx + s * Math.cos(0.3) * len * 0.7, ty = by + Math.sin(0.3) * len * 0.7, rr = (0.06 + 0.1 * Math.min(1, amp)) * u; g.save(); g.globalCompositeOperation = 'lighter'; g.globalAlpha = Math.min(0.9, amp * 1.4); g.drawImage(sprite(SONG), tx - rr, ty - rr, rr * 2, rr * 2); g.restore(); }
}

// ---------- the frame ----------
let lastChip = 0;
function frame(now) {
  if (!W || stage.clientWidth !== Math.round(W) || stage.clientHeight !== Math.round(H)) layout();
  if (layerDirty) { drawLayer(); layerDirty = false; }
  if (scoreDirty) { drawScore(); scoreDirty = false; }
  g.setTransform(dpr, 0, 0, dpr, 0, 0); g.clearRect(0, 0, W, H); g.drawImage(layer, 0, 0, W, H);
  const clock = actx ? actx.currentTime : 0, playing = mode === 'play', local = playing ? clock - sceneStart : -1, D = dur();
  if (playing && local > D + 0.25) halt();
  if (mode === 'play') playStatus();
  else if (mode === 'done' && prep) setStatus(song.name, `kept · ${scrambled ? 'scrambled wiring · ' : ''}listen to your song, the antennae, the flies' songs or their recreation`);
  if (cur && nReady() < NF && now - lastChip > 300) { lastChip = now; updateChips(); }
  g.save(); g.globalCompositeOperation = 'lighter';
  heads.forEach((h, i) => {
    const f = cur && cur.flies[i];
    if (f && local >= 0) { const lim = local / DT; while (f.ptr < f.t.length && f.t[f.ptr] <= lim) { h.last[f.n[f.ptr]] = sceneStart + f.t[f.ptr] * DT; f.ptr++; } }
    if (!f || !playing) return;
    const sc2 = h.u / 900;
    for (let j = 0; j < NA; j++) {
      const age = clock - h.last[j]; if (age < 0 || age > 0.5) continue;
      const I = Math.exp(-age / 0.1), rl = role[j], rad = (rl >= 6 ? 15 : rl ? 6 : 8) * sc2 * (1 + 1.4 * I);
      g.globalAlpha = Math.min(1, I) * (rl && rl < 6 ? 0.65 : 1); g.drawImage(sprite(COL[rl]), h.sx[j] - rad, h.sy[j] - rad, rad * 2, rad * 2);
    }
  });
  g.restore();
  heads.forEach((h, i) => {
    const f = cur && cur.flies[i], on = f && playing && local >= 0;
    const q = on ? Math.min(f.swL.length - 1, Math.floor(local * 100)) : 0, aL = on ? Math.min(1.2, f.swL[q] / 250) : 0, aR = on ? Math.min(1.2, f.swR[q] / 250) : 0;
    const wob = t => reduceMotion ? 0 : Math.min(0.24, t * 0.3) * Math.sin(now / 1000 * 2 * Math.PI * 11);
    drawAntenna(h, -1, wob(aL), aL); drawAntenna(h, 1, wob(aR), aR);
    let amp = 0, sd = 1;
    if (on) {
      const ms = local * 1000; while (f.pp < f.pulses.length && f.pulses[f.pp] <= ms) { f.flash = f.pulses[f.pp + 2]; f.flashT = f.pulses[f.pp] / 1000; f.pp += 3; }
      const z = Math.min(f.hum.length - 1, Math.floor(local * 200)); sd = f.side[z]; amp = Math.max(f.hum[z] * 2.4, f.flash * Math.exp(-(local - f.flashT) / 0.04));
    }
    drawWing(h, -1, on && sd === 0, amp, now); drawWing(h, 1, on && sd === 1, amp, now);
  });
  if (scoreImg) {
    sg.setTransform(1, 0, 0, 1, 0, 0); sg.clearRect(0, 0, sc.width, sc.height); sg.drawImage(scoreImg, 0, 0);
    if (local >= 0 && playing) { const x = Math.round(Math.min(1, local / D) * sc.width) + 0.5; sg.strokeStyle = '#e6ebf3'; sg.lineWidth = 1.5 * dpr; sg.beginPath(); sg.moveTo(x, 0); sg.lineTo(x, sc.height); sg.stroke(); }
  }
  requestAnimationFrame(frame);
}
markSrc(); idleStatus(); updateChips();
requestAnimationFrame(frame);
})();
