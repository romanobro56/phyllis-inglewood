// The GTA V character switch: cut up out of the scene in steps (the building from above, then the map at three
// heights), hold until the next scene is loaded, pan across, and cut back down in steps into the next scene.
// The map phase is plain DOM images on top of the canvas; the 3D scene isn't drawn while they cover it.
//   web/assets/map/map.json   {"image": "map.jpg", "pins": {"<scene>": [x, y]}}   x, y from 0 to 1 on the image
//   web/scenes/<scene>/aerial.jpg   the building seen from above (optional)
import { tween, wait } from "./fly.js";

const STEP = 600;                 // ms between cuts
const ZOOMS = [10, 4.5, 2];       // map heights, close to far (1 = the whole map just covers the screen)
const ease = (t) => (t < 0.5 ? 4 * t * t * t : 1 - Math.pow(-2 * t + 2, 3) / 2);

export class Switcher {
  constructor(canvas) {
    this.canvas = canvas;
    this.el = document.getElementById("switch");
    this.mapEl = this.el.querySelector(".map");
    this.aerialEl = this.el.querySelector(".aerial");
    this.flashEl = document.getElementById("flash");
    this.map = null;
    this.covering = false;        // true while the map hides the 3D view (the world skips drawing then)
    this.sound = new Sound();
    this.view = { x: 0.5, y: 0.5, z: 1 };
    this.drift = 0;
    fetch("/web/assets/map/map.json").then((r) => (r.ok ? r.json() : null)).then(async (m) => {
      if (!m || !(await loads(`/web/assets/map/${m.image}`))) return;
      this.mapEl.src = `/web/assets/map/${m.image}`;
      this.map = m;
    }).catch(() => {});
    const tick = () => { if (this.covering) { this.drift += 1; this.place(); } requestAnimationFrame(tick); };
    tick();
  }

  pin(scene) { return this.map?.pins?.[scene] || [0.5, 0.5]; }

  // Show the map so that (x, y) is in the middle, z times bigger than the screen, never showing its edges.
  place(v = this.view) {
    this.view = v;
    const W = innerWidth, H = innerHeight, B = Math.max(W, H);
    const z = v.z * (1 + this.drift * 0.00012);                        // a slow push while it's held
    const s = B * z;
    const tx = Math.min(0, Math.max(W - s, W / 2 - v.x * s));
    const ty = Math.min(0, Math.max(H - s, H / 2 - v.y * s));
    this.mapEl.style.transform = `translate(${tx}px, ${ty}px) scale(${z})`;
    this.mapEl.style.width = this.mapEl.style.height = `${B}px`;
  }

  async flash(strength = 1, ms = 260) {
    this.flashEl.style.transition = "none";
    this.flashEl.style.opacity = strength;
    await wait(40);
    this.flashEl.style.transition = `opacity ${ms}ms ease-out`;
    this.flashEl.style.opacity = 0;
  }

  grade(sat, ms = 0) {             // the colour drains out on the way up and comes back on the way down
    for (const e of [this.canvas, this.el]) {
      e.style.transition = ms ? `filter ${ms}ms ease` : "none";
      e.style.filter = sat >= 1 ? "" : `saturate(${sat}) contrast(1.06) brightness(${0.92 + 0.08 * sat})`;
    }
  }

  async cutTo(layer, v, sat) {
    this.sound.whoosh();
    this.flash(0.35, 220);
    this.aerialEl.style.opacity = layer === "aerial" ? 1 : 0;
    this.mapEl.style.opacity = layer === "map" ? 1 : 0;
    this.el.style.opacity = 1;
    this.covering = true;
    this.drift = 0;
    if (v) this.place(v);
    this.grade(sat);
    await wait(STEP);
  }

  // Up and out of `from` (a scene name). Resolves at the top, with the 3D view hidden.
  async out(from, aerial) {
    this.sound.hit();
    this.flash(0.9, 300);
    this.grade(0.6, 200);
    await wait(320);
    if (aerial && (await loads(aerial))) {
      this.aerialEl.src = aerial;
      await this.cutTo("aerial", null, 0.55);
    }
    if (!this.map) {                                                    // no map yet: just drop to black
      this.el.style.opacity = 1;
      this.aerialEl.style.opacity = this.mapEl.style.opacity = 0;
      this.covering = true;
      return;
    }
    const [x, y] = this.pin(from);
    for (const [i, z] of ZOOMS.entries()) await this.cutTo("map", { x, y, z }, [0.45, 0.32, 0.22][i]);
  }

  // At the top: wait for `ready`, then pan to `to`.
  async across(to, ready) {
    const t0 = performance.now();
    this.sound.wind(true);
    await ready;
    await wait(Math.max(0, 300 - (performance.now() - t0)));
    if (!this.map) return;
    const a = { ...this.view }, [x, y] = this.pin(to);
    if (Math.hypot(x - a.x, y - a.y) < 1e-3) return;
    await tween(1200, (t) => { const e = ease(t); this.place({ x: a.x + (x - a.x) * e, y: a.y + (y - a.y) * e, z: a.z }); });
  }

  // Back down into `to`. `reveal()` is called just before the last cut, so the scene is drawn under the map.
  async in(to, aerial, reveal) {
    this.sound.wind(false);
    if (this.map) {
      const [x, y] = this.pin(to);
      for (const [i, z] of ZOOMS.slice(0, -1).reverse().entries()) await this.cutTo("map", { x, y, z }, [0.32, 0.45][i]);
    }
    if (aerial && (await loads(aerial))) {
      this.aerialEl.src = aerial;
      await this.cutTo("aerial", null, 0.55);
    }
    this.covering = false;                                              // draw a few frames underneath first
    reveal?.();
    await wait(250);
    this.sound.hit();
    this.flash(0.85, 380);
    this.el.style.opacity = 0;
    this.grade(0.6);
    await wait(30);
    this.grade(1, 700);
  }
}

const cache = new Map();
function loads(url) {
  if (!cache.has(url)) cache.set(url, new Promise((res) => { const i = new Image(); i.onload = () => res(true); i.onerror = () => res(false); i.src = url; }));
  return cache.get(url);
}

// Synthesised hits and whooshes (Web Audio, no files). Silent until the page has had a click or key press.
class Sound {
  constructor() {
    this.ctx = null;
    const start = () => { this.ctx ||= new AudioContext(); this.ctx.resume(); };
    addEventListener("pointerdown", start);
    addEventListener("keydown", start);
  }
  noise(sec) {
    const b = this.ctx.createBuffer(1, this.ctx.sampleRate * sec, this.ctx.sampleRate), d = b.getChannelData(0);
    for (let i = 0; i < d.length; i++) d[i] = Math.random() * 2 - 1;
    const s = this.ctx.createBufferSource();
    s.buffer = b;
    return s;
  }
  hit() {
    const c = this.ctx;
    if (!c || c.state !== "running") return;
    const t = c.currentTime, o = c.createOscillator(), g = c.createGain();
    o.frequency.setValueAtTime(110, t);
    o.frequency.exponentialRampToValueAtTime(38, t + 0.5);
    g.gain.setValueAtTime(0.0001, t);
    g.gain.exponentialRampToValueAtTime(0.5, t + 0.01);
    g.gain.exponentialRampToValueAtTime(0.0001, t + 0.9);
    o.connect(g).connect(c.destination);
    o.start(t);
    o.stop(t + 1);
  }
  whoosh(sec = 0.55) {
    const c = this.ctx;
    if (!c || c.state !== "running") return;
    const t = c.currentTime, n = this.noise(sec), f = c.createBiquadFilter(), g = c.createGain();
    f.type = "bandpass";
    f.Q.value = 1.2;
    f.frequency.setValueAtTime(300, t);
    f.frequency.exponentialRampToValueAtTime(2400, t + sec * 0.6);
    g.gain.setValueAtTime(0.0001, t);
    g.gain.exponentialRampToValueAtTime(0.22, t + sec * 0.35);
    g.gain.exponentialRampToValueAtTime(0.0001, t + sec);
    n.connect(f).connect(g).connect(c.destination);
    n.start(t);
  }
  wind(on) {
    const c = this.ctx;
    if (!c || c.state !== "running") return;
    if (on && !this.bed) {
      const n = this.noise(4), f = c.createBiquadFilter(), g = c.createGain();
      n.loop = true;
      f.type = "lowpass";
      f.frequency.value = 500;
      g.gain.setValueAtTime(0.0001, c.currentTime);
      g.gain.exponentialRampToValueAtTime(0.06, c.currentTime + 0.6);
      n.connect(f).connect(g).connect(c.destination);
      n.start();
      this.bed = { n, g };
    } else if (!on && this.bed) {
      const { n, g } = this.bed, t = c.currentTime;
      g.gain.setValueAtTime(g.gain.value, t);
      g.gain.exponentialRampToValueAtTime(0.0001, t + 0.5);
      n.stop(t + 0.6);
      this.bed = null;
    }
  }
}
