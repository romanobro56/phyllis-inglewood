// The Fly Band, in the shared world: four players (drums, bass, vocals, keys) and a conductor.
//
// This station is a browser exhibit (no task.py): every fly's brain is the band's own spiking simulation
// (engine/, 8,161 FlyWire neurons, leaky integrate-and-fire, see WRITEUP.md), run in background threads of
// the viewer's page. The song is split into four parts (signal processing, not the fly); each player hears one
// part, the conductor hears the whole mix. What you see move is driven by each fly's own output neurons:
// its wing (the stand-in wing of WRITEUP.md, "Their songs") and, for the conductor, the baton. The glow over
// each head is that brain's spikes as they happen. Instruments only bounce to the part as played.
import { Fly, box, cylinder } from "/web/lib/fly.js";

const BASE = new URL("./", import.meta.url).href;          // /flies/band/
const SR = 22050, DT = 1e-4, EXCERPT = 30;
const PARTS = [
  { name: "Drums", color: 0xff7a5c, css: "#e0613f" },
  { name: "Bass", color: 0xffbf47, css: "#c98a12" },
  { name: "Vocals", color: 0xc6f36b, css: "#6c9a1c" },
  { name: "Keys", color: 0x6cc9ff, css: "#2b8fcc" },
];
const CONDUCTOR = 4, NF = 5;                               // worker/fly index 4 hears the whole mix
const KINDS = ["rec", "ant", "song", "re"];
const LEVEL = { rec: 1, ant: 0.5, song: 0.5, re: 1 };
const HEAR = { rec: "The song", ant: "Their antennae", song: "Their songs", re: "Their recreation" };

export default {
  title: "The Inglewoods",
  blurb: "Four flies each hear one part of a song, a fifth conducts. Their own spiking brains run in your browser.",
  playTimeout: 0,

  build({ THREE, group, fly, setCaption }) {
    this.THREE = THREE;
    this.setCaption = setCaption || (() => {});
    this.focused = false;
    const stage = box(4.6, 0.16, 2.9, 0x2a2230, [0, 0.08, -0.15]);
    group.add(stage);
    group.add(box(4.6, 0.02, 0.06, 0xd97757, [0, 0.17, 1.28]));            // stage lip
    const FLOOR = 0.17;

    // four players in a shallow arc, each with its instrument; the conductor in front, facing them
    this.players = PARTS.map((pt, i) => {
      const x = (i - 1.5) * 1.05, z = -0.75 + Math.abs(i - 1.5) * 0.18;
      const f = new Fly(pt.color === 0xc6f36b ? 0x7a6a3a : 0x8a6a3a);
      f.root.position.set(x, FLOOR, z + 0.35);
      f.root.rotation.y = -x * 0.12;                                       // facing the audience (+z), angled in
      group.add(f.root);
      const inst = new THREE.Group();
      inst.position.set(x, FLOOR, z);
      group.add(inst);
      const glow = new THREE.Mesh(new THREE.SphereGeometry(0.1, 16, 12),
        new THREE.MeshBasicMaterial({ color: pt.color, transparent: true, opacity: 0.25 }));
      glow.position.set(0, 0.95, 0);
      f.root.add(glow);
      return { fly: f, inst, glow, parts: this.instrument(THREE, i, inst), heat: 0 };
    });

    fly.root.position.set(0, FLOOR + 0.18, 0.75);
    fly.root.rotation.y = Math.PI;                                         // the conductor faces the band
    group.add(box(0.7, 0.18, 0.55, 0x3a2c22, [0, FLOOR + 0.09, 0.75]));      // podium
    const baton = new THREE.Group();
    baton.position.set(0.12, 0.45, 0.12);
    const stick = cylinder(0.008, 0.42, 0xf4efe6, [0, 0.21, 0]);
    baton.add(stick);
    fly.body.add(baton);
    const cglow = new THREE.Mesh(new THREE.SphereGeometry(0.1, 16, 12),
      new THREE.MeshBasicMaterial({ color: 0xffffff, transparent: true, opacity: 0.25 }));
    cglow.position.set(0, 0.95, 0);
    fly.root.add(cglow);
    this.conductor = { fly, baton, glow: cglow, heat: 0 };

    this.lights = [-1.6, 1.6].map((x) => {
      const l = new THREE.SpotLight(0xffe2c4, 6, 7, 0.55, 0.6);
      l.position.set(x, 3.2, 1.8);
      l.target.position.set(x * 0.3, 0, -0.6);
      group.add(l, l.target);
      return l;
    });

    this.state = { song: null, prep: null, cache: new Map(), cur: null, mode: "idle", hear: "rec", userChose: false,
                   scrambled: 0, session: 0, seedBase: 1, sceneStart: 0, live: [] };
    this.panel();
    const loop = () => { this.frame(); requestAnimationFrame(loop); };
    requestAnimationFrame(loop);
    this.setCaption("waiting for an audience · visit to start the band");
  },

  // --- instruments (only bounce to the part as played: that is the song, not the fly) ------------------
  instrument(THREE, i, g) {
    const mat = (c, m = 0.2, r = 0.5) => new THREE.MeshStandardMaterial({ color: c, metalness: m, roughness: r });
    const add = (geo, m, x, y, z, rx = 0, rz = 0) => { const o = new THREE.Mesh(geo, m); o.position.set(x, y, z); o.rotation.set(rx, 0, rz); o.castShadow = true; g.add(o); return o; };
    if (i === 0) {                                                         // drum kit
      const kick = add(new THREE.CylinderGeometry(0.26, 0.26, 0.24, 24), mat(0xb33a2a), 0, 0.27, -0.15, Math.PI / 2);
      const snare = add(new THREE.CylinderGeometry(0.15, 0.15, 0.09, 20), mat(0xdddddd, 0.6, 0.3), 0.36, 0.45, 0.1);
      add(new THREE.CylinderGeometry(0.01, 0.01, 0.42, 6), mat(0x888888, 0.8), 0.36, 0.2, 0.1);
      add(new THREE.CylinderGeometry(0.012, 0.012, 0.8, 6), mat(0x888888, 0.8), -0.4, 0.4, -0.05);
      const cym = add(new THREE.CylinderGeometry(0.2, 0.2, 0.012, 24), mat(0xd9b44a, 0.9, 0.25), -0.4, 0.81, -0.05);
      return { pulse: [kick, snare], cym };
    }
    if (i === 1) {                                                         // bass guitar on a stand
      const body = add(new THREE.BoxGeometry(0.34, 0.42, 0.07), mat(0x1c3f7a, 0.3, 0.35), 0.38, 0.38, 0.05, 0, 0.25);
      add(new THREE.BoxGeometry(0.06, 0.62, 0.04), mat(0x6b4423), 0.24, 0.85, 0.05, 0, 0.25);
      const strings = add(new THREE.BoxGeometry(0.04, 0.95, 0.005), mat(0xeeeeee, 0.9), 0.3, 0.68, 0.095, 0, 0.25);
      add(new THREE.BoxGeometry(0.42, 0.34, 0.3), mat(0x222222), -0.42, 0.17, -0.1);   // amp
      return { pulse: [body], strings };
    }
    if (i === 2) {                                                         // microphone
      add(new THREE.CylinderGeometry(0.012, 0.012, 1.0, 8), mat(0x777777, 0.8), 0, 0.5, 0.65);
      add(new THREE.CylinderGeometry(0.14, 0.14, 0.02, 16), mat(0x333333, 0.6), 0, 0.01, 0.65);
      const mic = add(new THREE.SphereGeometry(0.06, 14, 10), mat(0x999999, 0.9, 0.3), 0, 1.04, 0.62);
      const glowMat = new THREE.MeshBasicMaterial({ color: 0xc6f36b, transparent: true, opacity: 0 });
      const halo = add(new THREE.SphereGeometry(0.11, 14, 10), glowMat, 0, 1.04, 0.62);
      return { pulse: [mic], halo };
    }
    const top = add(new THREE.BoxGeometry(0.95, 0.06, 0.3), mat(0x1a1a1a), 0, 0.62, 0.2);   // keyboard
    for (const s of [-1, 1]) add(new THREE.BoxGeometry(0.03, 0.6, 0.03), mat(0x555555, 0.7), s * 0.4, 0.3, 0.2, 0, s * 0.35);
    const keys = [];
    for (let k = 0; k < 14; k++) keys.push(add(new THREE.BoxGeometry(0.058, 0.02, 0.16), mat(0xf4f1ea, 0, 0.6), -0.39 + k * 0.06, 0.66, 0.26));
    return { pulse: [top], keys };
  },

  // --- the little control panel shown while the band is being visited ---------------------------------
  panel() {
    const el = document.createElement("div");
    el.id = "band-panel";
    el.innerHTML = `
      <style>
        #band-panel { position: fixed; left: 50%; bottom: 92px; transform: translateX(-50%) translateY(8px); z-index: 4;
          width: min(760px, 92vw); padding: 12px 16px 10px; border-radius: 14px; background: rgba(255,246,242,.93);
          border: 1px solid rgba(43,26,20,.12); box-shadow: 0 6px 30px rgba(120,50,30,.16); font-size: 13px; color: #2b1a14;
          opacity: 0; pointer-events: none; transition: opacity .3s, transform .3s; }
        #band-panel.on { opacity: 1; pointer-events: auto; transform: translateX(-50%); }
        #band-panel .top { display: flex; align-items: center; gap: 10px; flex-wrap: wrap; }
        #band-panel b.t { font-size: 15px; }
        #band-panel .st { color: #7a4f42; flex: 1; min-width: 180px; }
        #band-panel button { font: inherit; border: 1px solid rgba(43,26,20,.25); background: transparent; border-radius: 999px;
          padding: 4px 11px; cursor: pointer; color: #2b1a14; }
        #band-panel button[aria-pressed="true"] { background: #d97757; border-color: #d97757; color: #fff; }
        #band-panel button:disabled { opacity: .4; cursor: default; }
        #band-panel .row { display: flex; gap: 6px; margin-top: 8px; flex-wrap: wrap; align-items: center; }
        #band-panel .lab { font: 10px ui-monospace, Menlo, monospace; letter-spacing: .12em; color: #a8432a; margin-right: 2px; }
        #band-panel .chips { display: grid; grid-template-columns: repeat(5, 1fr); gap: 6px; margin-top: 9px; }
        #band-panel .chip { font-size: 11px; line-height: 1.35; color: #7a4f42; border-top: 3px solid var(--c); padding-top: 4px; }
        #band-panel .chip b { color: #2b1a14; display: block; font-size: 12px; }
        #band-panel a { color: #a8432a; }
      </style>
      <div class="top"><b class="t">The Inglewoods</b><span class="st"></span>
        <button data-go>Play</button><button data-up>Upload a song</button></div>
      <div class="row"><span class="lab">LISTEN TO</span>${KINDS.map((k) => `<button data-hear="${k}">${HEAR[k]}</button>`).join("")}
        <span class="lab" style="margin-left:10px">WIRING</span><button data-scr="0">Real</button><button data-scr="1">Scrambled</button></div>
      <div class="chips">${[...PARTS, { name: "Conductor", css: "#8a8a8a" }].map((p) => `<div class="chip" style="--c:${p.css}"><b>${p.name}</b><span></span></div>`).join("")}</div>
      <div class="row" style="justify-content:space-between;color:#7a4f42;font-size:11px">
        <span>Brains: 8,161 FlyWire neurons each, spiking in your browser. The split into parts is signal processing, not the fly.</span>
        <a href="${BASE}index.html" target="_blank" rel="noopener">full band app</a></div>
      <input type="file" accept="audio/*" hidden>`;
    document.body.appendChild(el);
    this.el = el;
    this.ui = { st: el.querySelector(".st"), go: el.querySelector("[data-go]"), file: el.querySelector("input"),
                chips: [...el.querySelectorAll(".chip span")] };
    el.querySelector("[data-up]").onclick = () => this.ui.file.click();
    this.ui.file.onchange = (e) => {
      const f = e.target.files[0]; e.target.value = "";
      if (f) this.load({ name: f.name.replace(/\.[^.]+$/, "").slice(0, 40), file: f });
    };
    this.ui.go.onclick = () => {
      const s = this.state;
      if (s.mode === "play") this.halt();
      else if (s.prep) this.playFrom(0);
      else this.start();
    };
    el.querySelectorAll("[data-hear]").forEach((b) => (b.onclick = () => {
      this.state.hear = b.dataset.hear; this.state.userChose = true;
      if (this.state.mode !== "play" && this.state.prep) this.playFrom(0); else this.applyMix();
      this.buttons();
    }));
    el.querySelectorAll("[data-scr]").forEach((b) => (b.onclick = () => { this.state.scrambled = +b.dataset.scr; this.request(); this.buttons(); }));
    this.buttons();
  },

  status(text) { this.ui.st.textContent = text; this.setCaption(text); },

  buttons() {
    const s = this.state, ready = s.cur ? s.cur.flies.filter(Boolean).length : 0;
    this.ui.go.textContent = s.mode === "play" ? "Stop" : s.mode === "loading" ? "Working..." : "Play";
    this.ui.go.disabled = s.mode === "loading";
    this.el.querySelectorAll("[data-hear]").forEach((b) => {
      b.setAttribute("aria-pressed", b.dataset.hear === s.hear);
      b.disabled = !s.prep || (b.dataset.hear !== "rec" && ready === 0);
    });
    this.el.querySelectorAll("[data-scr]").forEach((b) => b.setAttribute("aria-pressed", +b.dataset.scr === s.scrambled));
    this.ui.chips.forEach((c, i) => {
      const f = s.cur?.flies[i];
      c.textContent = f ? `${f.stats.nFired.toLocaleString()} neurons fired · ${f.stats.nDN} outputs` +
        (i < 4 ? ` · tells moments apart ${Math.round(100 * f.stats.match)}%` : "")
        : !s.cur ? (i < 4 ? `hears the ${PARTS[i].name.toLowerCase()}` : "hears the whole song")
        : s.cur.prog[i] > 0 ? `listening ${Math.round(100 * s.cur.prog[i])}%` : "getting ready";
    });
  },

  // --- visiting: start the band the first time, stop the sound when you leave -----------------------------
  focus(on) {
    this.focused = on;
    this.el.classList.toggle("on", on);
    if (!on) { if (this.state.mode === "play") this.halt(); return; }
    this.audio();
    if (this.actx.state === "suspended") this.actx.resume();
    if (this.state.prep) this.playFrom(0);
    else if (this.state.mode === "idle") this.load({ name: "Demo song (an original piece, A minor, 96 bpm)", url: BASE + "demo-song.mp3" });
  },

  // --- the engine: one band worker per fly (engine/worker.js, unchanged) ---------------------------------
  workers() {
    if (this.WK) return this.WK;
    this.net = this.net || fetch(BASE + "data/network.json").then((r) => { if (!r.ok) throw new Error("band network missing"); return r.text(); });
    this.WK = Array.from({ length: NF }, (_, i) => this.makeWorker(i));
    return this.WK;
  },
  makeWorker(i) {
    const w = new Worker(BASE + "engine/worker.js");
    const wk = { w, ready: null, busy: 0 };
    wk.ready = new Promise((res) => { wk.res = res; });
    w.onmessage = (e) => this.onMsg(i, e.data);
    this.net.then((text) => w.postMessage({ type: "init", net: text }));
    return wk;
  },
  reset() {                                                                // forget this run, kill busy workers
    const s = this.state;
    s.session++; this.silence();
    if (this.WK) this.WK.forEach((wk, i) => { if (wk.busy > 0) { wk.w.terminate(); this.WK[i] = this.makeWorker(i); } });
    s.cache = new Map(); s.cur = null; s.prep = null; s.mode = "idle"; s.hear = "rec"; s.userChose = false;
    this.players.forEach((pl) => delete pl.top);                          // per-song loudness scale
    this.buttons(); this.applyMix();
  },

  audio() {
    if (this.actx) return this.actx;
    const actx = (this.actx = new (window.AudioContext || window.webkitAudioContext)());
    const master = actx.createGain(); master.gain.value = 0.9; master.connect(actx.destination);
    this.bus = Array.from({ length: NF }, (_, i) => {
      const out = actx.createGain(); out.connect(master);
      const o = { out };
      for (const k of KINDS) {
        const g = actx.createGain(); g.gain.value = 0;
        const pan = actx.createStereoPanner ? actx.createStereoPanner() : null;
        if (pan) { pan.pan.value = i < 4 ? (i - 1.5) * 0.35 : 0; g.connect(pan); pan.connect(out); } else g.connect(out);
        o[k] = g;
      }
      return o;
    });
    return actx;
  },
  stereo(l, r) { const b = this.actx.createBuffer(2, l.length, SR); b.copyToChannel(l, 0); b.copyToChannel(r, 1); return b; },
  silence() { for (const n of this.state.live) { try { n.stop(); } catch (e) { /* already stopped */ } } this.state.live = []; },
  applyMix() {
    if (!this.actx) return;
    const s = this.state, t = this.actx.currentTime, on = s.mode === "play";
    this.bus.forEach((b, i) => {
      for (const k of KINDS) {
        // the conductor hears the whole song: only its own wing song is worth hearing on top of the band
        const v = on && s.hear === k && (i < 4 || k === "song") ? LEVEL[k] * (i < 4 ? 1 : 0.6) : 0;
        b[k].gain.setTargetAtTime(v, t, 0.03);
      }
    });
  },

  async load(song) {
    this.reset();
    const s = this.state, my = s.session;
    s.song = song; s.mode = "loading"; this.buttons();
    this.status(`Reading ${song.name}`);
    const actx = this.audio();
    let L, R;
    try {
      const bytes = song.file ? await song.file.arrayBuffer() : await (await fetch(song.url)).arrayBuffer();
      const buf = await actx.decodeAudioData(bytes);
      const d = Math.min(buf.duration, 420), off = new OfflineAudioContext(2, Math.ceil(d * SR), SR), src = off.createBufferSource();
      src.buffer = buf; src.connect(off.destination); src.start(0, 0, d);
      const r = await off.startRendering();
      L = r.getChannelData(0).slice(); R = r.getChannelData(1).slice();
    } catch (e) {
      if (my === s.session) { this.reset(); this.status("Could not read that song. Try an MP3, WAV or M4A."); }
      return;
    }
    if (my !== s.session) return;
    this.status("Splitting the song into drums, bass, vocals and keys (signal processing, not the fly)");
    const WK = this.workers();
    await WK[0].ready;
    if (my !== s.session) return;
    WK[0].busy++;
    WK[0].w.postMessage({ type: "prep", session: my, L, R, seconds: EXCERPT }, [L.buffer, R.buffer]);
  },
  start() { if (this.state.song) this.load(this.state.song); },

  request() {                                                              // set the five brains listening
    const s = this.state;
    if (!s.prep) return;
    const key = String(s.scrambled), was = s.mode === "play" ? this.actx.currentTime - s.sceneStart : -1;
    if (!s.cache.has(key)) {
      s.cache.set(key, { key, flies: new Array(NF).fill(null), prog: new Float32Array(NF) });
      for (let p = 0; p < NF; p++) {
        const src = p < 4 ? s.prep.parts[p] : s.prep.mix, L = src.L.slice(), R = src.R.slice(), my = s.session, wk = this.WK[p];
        wk.busy++;
        wk.ready.then(() => {
          if (my === s.session) wk.w.postMessage({ type: "fly", session: my, key, p, L, R, v0: p < 4 ? s.prep.v0[p] : undefined,
            slot: s.prep.slot, n: s.prep.n, seed: s.seedBase + 97 * p, scrambled: !!s.scrambled }, [L.buffer, R.buffer]);
        });
      }
    }
    s.cur = s.cache.get(key);
    this.buttons();
    if (was >= 0) this.playFrom(Math.min(was, this.dur() - 1));
  },

  onMsg(i, m) {
    const s = this.state;
    if (m.type === "ready") { this.WK[i].res(); return; }
    if (m.session !== s.session) return;
    if (m.type === "error") { this.reset(); this.status(`The band hit a problem: ${m.message}`); return; }
    if (m.type === "prep") {
      this.WK[0].busy--;
      const parts = PARTS.map((_, p) => ({ L: m["L" + p], R: m["R" + p], env: m["env" + p], buf: this.stereo(m["L" + p], m["R" + p]) }));
      const mixL = new Float32Array(m.len), mixR = new Float32Array(m.len);
      for (const pt of parts) for (let k = 0; k < m.len; k++) { mixL[k] += pt.L[k] || 0; mixR[k] += pt.R[k] || 0; }
      s.prep = { len: m.len, bpm: m.bpm, bars: m.bars, startBar: m.startBar, slot: m.slot, n: m.n, mono: m.mono, nMs: m.nMs, v0: m.v0,
                 parts, mix: { L: mixL, R: mixR } };
      s.seedBase = 1000 + Math.floor(Math.random() * 1e6);
      s.mode = "ready";
      this.request();
      if (this.focused) this.playFrom(0);
      return;
    }
    const res = s.cache.get(m.key);
    if (!res) return;
    if (m.type === "progress") { res.prog[m.p] = m.v; return; }
    if (m.type === "fly") {
      this.WK[i].busy--;
      res.flies[m.p] = { ant: this.stereo(m.antL, m.antR), song: this.stereo(m.songL, m.songR), re: this.stereo(m.reL, m.reR),
        pulses: m.pulses, hum: m.hum, side: m.side, t: m.t, n: m.n, stats: m.stats, ptr: 0, pp: 0, flash: 0, flashT: -1 };
      res.prog[m.p] = 1;
      if (res !== s.cur) return;
      if (s.mode === "play") {
        const local = this.actx.currentTime - s.sceneStart, f = res.flies[m.p];
        if (local < this.dur() - 0.3) for (const k of ["ant", "song", "re"]) this.source(f[k], this.bus[m.p][k], this.actx.currentTime + 0.05, Math.max(0, local + 0.05));
        this.seek(f, local);
      }
      if (res.flies.slice(0, 4).every(Boolean) && !s.userChose) { s.hear = "re"; this.applyMix(); }
      this.buttons();
    }
  },

  dur() { return this.state.prep ? this.state.prep.len / SR : 1; },
  source(buf, dst, at, off) { const n = this.actx.createBufferSource(); n.buffer = buf; n.connect(dst); n.start(at, off); this.state.live.push(n); },
  seek(f, from) {                                                          // spike and pulse pointers to `from` seconds
    const lim = from / DT; let lo = 0, hi = f.t.length;
    while (lo < hi) { const md = (lo + hi) >> 1; if (f.t[md] < lim) lo = md + 1; else hi = md; }
    f.ptr = lo; f.pp = 0;
    while (f.pp < f.pulses.length && f.pulses[f.pp] < from * 1000) f.pp += 3;
    f.flash = 0;
  },
  playFrom(from) {
    const s = this.state;
    if (!s.prep || !this.actx) return;
    this.silence();
    s.mode = "play";
    from = Math.max(0, Math.min(from, this.dur() - 0.2));
    s.sceneStart = this.actx.currentTime + 0.12 - from;
    const at = s.sceneStart + from;
    s.prep.parts.forEach((pt, p) => this.source(pt.buf, this.bus[p].rec, at, from));
    if (s.cur) s.cur.flies.forEach((f, p) => { if (f) { for (const k of ["ant", "song", "re"]) this.source(f[k], this.bus[p][k], at, from); this.seek(f, from); } });
    this.applyMix(); this.buttons();
  },
  halt() { this.silence(); this.state.mode = this.state.prep ? "ready" : "idle"; this.applyMix(); this.buttons(); },

  // --- every frame: brains -> wings, baton and glow; the part as played -> instruments ---------------------
  frame() {
    const s = this.state, now = performance.now();
    const playing = s.mode === "play" && this.actx, local = playing ? this.actx.currentTime - s.sceneStart : -1;
    if (playing && local > this.dur() + 0.25) this.playFrom(0);           // loop the excerpt while you watch
    if (playing && now - (this.lastStatus || 0) > 400) {
      this.lastStatus = now;
      const ready = s.cur ? s.cur.flies.filter(Boolean).length : 0;
      const p = s.prep, wait = ready < NF ? ` · ${NF - ready} of the 5 brains still listening` : "";
      this.status(`${HEAR[s.hear]}${s.scrambled ? " (scrambled wiring)" : ""} · ${s.song.name} · ${Math.round(p.bpm)} bpm, bars ${p.startBar + 1}-${p.startBar + p.bars}${wait}`);
      if (ready < NF) this.buttons();
    }
    const flies = [...this.players.map((pl) => pl), this.conductor];
    flies.forEach((pl, i) => {
      const f = s.cur?.flies[i], on = f && playing && local >= 0;
      // brain glow: spikes this frame, all neurons of this fly's brain
      let spikes = 0;
      if (on) { const lim = local / DT; while (f.ptr < f.t.length && f.t[f.ptr] <= lim) { spikes++; f.ptr++; } }
      pl.heat = pl.heat * 0.85 + Math.min(1, spikes / 120) * 0.6;
      pl.glow.material.opacity = 0.18 + Math.min(0.8, pl.heat);
      pl.glow.scale.setScalar(1 + 1.2 * Math.min(1, pl.heat));
      // wing drive from the output neurons (the stand-in wing: clicks and hum)
      let amp = 0, side = 1;
      if (on) {
        const ms = local * 1000;
        while (f.pp < f.pulses.length && f.pulses[f.pp] <= ms) { f.flash = f.pulses[f.pp + 2]; f.flashT = f.pulses[f.pp] / 1000; f.pp += 3; }
        const z = Math.min(f.hum.length - 1, Math.floor(local * 200));
        side = f.side[z]; amp = Math.max(f.hum[z] * 2.4, f.flash * Math.exp(-(local - f.flashT) / 0.04));
      }
      for (const { pivot, s: sd } of pl.fly.wings) {
        const out = on && (sd < 0 ? side === 0 : side === 1);
        pivot.rotation.z = sd * (out ? 0.25 + Math.min(1, amp * 1.6) * (0.6 + 0.4 * Math.sin(now / 1000 * 2 * Math.PI * 24)) : 0);
      }
      if (i === CONDUCTOR) {
        const target = on ? -0.6 + Math.min(1.4, amp * 3) : -0.3 + 0.05 * Math.sin(now / 900);
        pl.baton.rotation.x += (target - pl.baton.rotation.x) * 0.35;
        pl.baton.rotation.z = on ? 0.4 * Math.sin(local * Math.PI * (s.prep.bpm / 60)) * Math.min(1, amp * 4) : 0;
        return;
      }
      // the instrument, from the part as played (the record, not the fly)
      const env = s.prep?.parts[i]?.env, e = playing && env ? Math.min(1, (env[Math.min(env.length - 1, Math.floor(local * 100))] || 0) / (pl.top ||= Math.max(1e-6, ...env))) : 0;
      const P = pl.parts;
      P.pulse.forEach((o) => o.scale.setScalar(1 + 0.06 * e));
      if (P.cym) P.cym.rotation.z = 0.25 * e * Math.sin(now / 40);
      if (P.strings) P.strings.position.x = 0.3 + 0.004 * e * Math.sin(now / 8);
      if (P.halo) P.halo.material.opacity = 0.5 * e;
      if (P.keys) P.keys.forEach((k, j) => (k.position.y = 0.66 - 0.012 * e * (Math.sin(now / 90 + j * 1.7) > 0.3 ? 1 : 0)));
      pl.fly.body.position.y = 0.02 * e * Math.abs(Math.sin(now / 160 + i));
    });
    if (s.prep === null) this.players.forEach((pl, i) => { pl.fly.body.position.y = 0.01 * Math.abs(Math.sin(now / 700 + i)); });
  },
};
