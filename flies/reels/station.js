// Reels station: Ferrari Inglewood lounges on a beanbag in front of a giant phone running a reels app.
// The phone screen is a live canvas: real videos from flies/reels/raw/<kind>/ (indexed by media.py) or, for
// kinds with no videos, drawn cartoon reels (cats, cooking, dances, fails, lectures, ads), with the usual UI on top. The fly flies up and touches the glass: double-tap to like, tap the bubble to open the
// comments (scroll them a set amount, then close), or watch N more seconds from the beanbag and swipe up.
// The station reads which reel is on screen from the brain's caption, so the phone matches the fly's world.
import { tween, wait, box } from "/web/lib/fly.js";

// --- the phone ------------------------------------------------------------------------------------
const SW = 0.9, SH = 1.6, SY = 1.06, SZ = 0.041;      // screen size, centre height, front face z
const W = 360, H = 640, DPR = 1.5;                     // canvas, in logical px (9:16)
const SEAT = { x: 0.95, y: 0.24, z: 0.62 };             // the fly on the beanbag
const HOVER = { up: 0.42, out: 0.33, side: 0.13 };     // where the fly hovers relative to the spot it touches
                                                       // (a bit to the right, so it hides less of the screen)
const RAIL = { x: 324, heart: 372, comment: 446, share: 516 };

const lerp = (a, b, t) => a + (b - a) * t;
const ease = (t) => (t < 0.5 ? 2 * t * t : 1 - (-2 * t + 2) ** 2 / 2);
const clamp01 = (t) => Math.min(1, Math.max(0, t));
const angLerp = (a, b, t) => a + Math.atan2(Math.sin(b - a), Math.cos(b - a)) * t;

// --- reel content -------------------------------------------------------------------------------
const REEL = {
  cat: {
    users: ["whiskers.daily", "catto_chaos", "loaf.of.the.day"],
    text: ["he does this every morning 😭", "the yarn never stood a chance", "my landlord: no pets. me:"],
    comments: ["I would die for him", "the little paw 😭😭", "LOAF", "my cat could never", "watched this 40 times",
      "he's plotting something", "this is my emotional support video", "pspspsps"],
  },
  cooking: {
    users: ["fruitfly.kitchen", "chef_buzz", "rotten.but.make.it.gourmet"],
    text: ["banana flambé in 15 seconds 🍌🔥", "you're washing your fruit wrong", "3-ingredient overripe mango jam"],
    comments: ["that's way too much butter", "recipe??", "my nonna is crying", "this is just soup",
      "you didn't even preheat the pan", "made this, 10/10", "the sizzle ASMR 🤤", "who eats bananas hot"],
  },
  dance: {
    users: ["six.legs.dance", "wingz_moves", "thorax.shaker"],
    text: ["new trend, 6 legs edition #fyp", "tutorial in comments 🕺", "pov: the beat drops"],
    comments: ["the legs are doing too much", "tutorial pls 🙏", "song name??", "this trend needs to end",
      "she ate", "left leg was off beat", "doing this at my wedding", "ok but the 3rd leg"],
  },
  fail: {
    users: ["fails.hourly", "oops.lol", "gravity.wins"],
    text: ["wait for it… 💀", "he was so confident", "physics said no"],
    comments: ["💀💀💀", "the confidence before 😂", "is he ok", "that's what helmets are for",
      "I replayed the fall 30 times", "bro thought he was in a movie", "this is staged", "RIP the board"],
  },
  lecture: {
    users: ["prof.neuro", "connectome101", "mushroom.body.facts"],
    text: ["Lecture 14: the mushroom body (part 3 of 12)", "why your fly brain is better than yours", "synapses explained slowly"],
    comments: ["exam is tomorrow help", "1.5x speed is still slow", "great explanation!", "first",
      "who's here from class", "can you do the central complex next", "zzz", "my professor is crying"],
  },
  ad: {
    users: ["ZapCo · Sponsored"],
    text: ["ZAP-O-MATIC 3000 bug zapper. 50% off today only!"],
    comments: ["Great product!!! 🔥🔥", "dm me for crypto tips 💰", "nice 👍", "Link in bio!! 100% legit",
      "I made $5000 from home", "wow", "Amazing deal!!", "check my page for free followers"],
  },
};
const DRAMA = ["ratio", "nobody asked", "this is literally fake and you know it", "@buzzkill you're wrong and here's why 🧵",
  "the comments are wilder than the video", "replying to @wingnut: no YOU calm down", "unfollowed", "blocked and reported"];
const AVATAR = ["#ff6b6b", "#ffd93d", "#6bcB77", "#4d96ff", "#c77dff", "#ff9f1c", "#2ec4b6", "#e76f51"];

function rnd(seed) { return () => ((seed = (seed * 16807) % 2147483647) - 1) / 2147483646; }
const fmt = (n) => (n >= 1e6 ? (n / 1e6).toFixed(1) + "M" : n >= 1e3 ? (n / 1e3).toFixed(1) + "K" : String(n));

// --- real videos: flies/reels/raw/<kind>/*.mp4, indexed into raw/reels.json by media.py ----------------
// The brain's world still decides "a 9 s cooking reel"; the phone plays the cooking video closest to 9 s.
// Kinds without videos fall back to the cartoon reels below. Videos are fetched whole into blob URLs
// (the dev server has no range requests, which looping and seeking need).
const BASE = new URL(".", import.meta.url);                 // /flies/reels/
const LIBRARY = { reels: [], last: {} };
LIBRARY.ready = fetch(new URL("raw/reels.json", BASE), { cache: "no-store" })
  .then((r) => (r.ok ? r.json() : { reels: [] }))
  .then((j) => {
    LIBRARY.reels = (j.reels || []).filter((e) => e && REEL[e.category] && e.file);
    LIBRARY.instagram = j.instagram || null;                // set while instagram.py is running
  })
  .catch(() => (LIBRARY.reels = []));
const BLOBS = new Map();                                    // file -> Promise<blob URL>, most recently used last
const MAX_BLOBS = 10;

function pickVideo(cat, len) {
  let all = LIBRARY.reels.filter((e) => e.category === cat);
  if (all.length > 1) all = all.filter((e) => e.file !== LIBRARY.last[cat]);   // no back-to-back repeats
  if (!all.length) return null;
  if (len && all.every((e) => e.seconds)) {                 // the closest lengths, with a little variety
    all = [...all].sort((a, b) => Math.abs(a.seconds - len) - Math.abs(b.seconds - len)).slice(0, 2);
  }
  const e = all[(Math.random() * all.length) | 0];
  LIBRARY.last[cat] = e.file;
  return e;
}

function videoFor(entry) {
  const v = document.createElement("video");
  Object.assign(v, { muted: true, loop: true, playsInline: true, preload: "auto" });
  let url = BLOBS.get(entry.file);
  BLOBS.delete(entry.file);
  if (!url) {
    url = fetch(new URL(entry.file, BASE)).then((r) => {
      if (!r.ok) throw new Error(`${entry.file}: HTTP ${r.status}`);
      return r.blob();
    }).then((b) => URL.createObjectURL(b));
  }
  BLOBS.set(entry.file, url);
  while (BLOBS.size > MAX_BLOBS) {                          // forget the oldest (never the current or next reel)
    const [file, old] = BLOBS.entries().next().value;
    BLOBS.delete(file);
    old.then((u) => URL.revokeObjectURL(u), () => {});
  }
  url.then((u) => { v.src = u; if (v.dataset.play) v.play().catch(() => {}); })
    .catch((e) => console.warn("reels: could not load video", e));
  return v;
}
// --- the fly's own Instagram: instagram.py streams a real Chrome tab and takes gestures ---------------------
// Live only while the bridge says "ok" and frames keep coming; otherwise the phone uses the videos/cartoons.
const IG = { ws: null, state: "off", detail: "", ui: {}, frame: null, frameAt: 0 };
function connectInstagram(port) {
  const ws = new WebSocket(`ws://127.0.0.1:${port}`);
  ws.binaryType = "blob";
  IG.state = "connecting";
  ws.onopen = () => (IG.ws = ws);
  ws.onmessage = async (ev) => {
    if (typeof ev.data === "string") {
      const m = JSON.parse(ev.data);
      if (m.type === "status") Object.assign(IG, { state: m.state, detail: m.detail || "", ui: m.ui || {} });
      return;
    }
    const bmp = await createImageBitmap(ev.data);
    IG.frame?.close();
    IG.frame = bmp;
    IG.frameAt = performance.now();
  };
  ws.onclose = () => {                                      // bridge stopped: reconnect only if it's announced again
    IG.ws = null; IG.state = "off";
    setTimeout(() => fetch(new URL("raw/reels.json", BASE), { cache: "no-store" })
      .then((r) => (r.ok ? r.json() : {})).then((j) => j.instagram && connectInstagram(j.instagram.port), () => {}), 5000);
  };
}
const igLive = () => IG.ws && IG.state === "ok" && IG.frame && performance.now() - IG.frameAt < 5000;
const igDo = (m) => igLive() && IG.ws.send(JSON.stringify(m));   // "like" is never sent: watch-only
const igSpot = (name, fallback) => (igLive() && IG.ui[name]) || fallback;

function playVideo(v) {
  if (!v) return;
  v.dataset.play = "1";
  v.currentTime = 0;
  if (v.src) v.play().catch(() => {});
}

function makeReel(cat, len) {
  const spec = REEL[cat] || REEL.cat;
  const seed = (Math.random() * 1e9) | 1, r = rnd(seed);
  const pick = (a) => a[(r() * a.length) | 0];
  const big = cat === "ad" ? 0.2 : cat === "lecture" ? 0.4 : 1;
  const someone = () => "@" + pick(["buzz", "maggot", "wingnut", "larva", "proboscis", "antenna", "halteres", "compound_i"]) + ((r() * 99) | 0);
  const reel = {
    cat: REEL[cat] ? cat : "cat", len: len || 8 + ((r() * 7) | 0), seed,
    user: pick(spec.users), text: pick(spec.text),
    likes: Math.round(big * (2e3 + r() * r() * 9e5)), comments: Math.round(big * (40 + r() * r() * 9e3)),
    list: spec.comments.map((t) => ({ t, user: someone(), likes: (r() * r() * 4000) | 0, color: pick(AVATAR) })).sort(() => r() - 0.5),
  };
  const e = pickVideo(reel.cat, len);
  if (e) {
    reel.video = videoFor(e);
    reel.len = e.seconds || reel.len;
    if (e.user) reel.user = e.user;
    if (e.caption != null) reel.text = String(e.caption);
    if (Number.isFinite(e.likes)) reel.likes = e.likes;
    if (Number.isFinite(e.comment_count)) reel.comments = e.comment_count;
    if (Array.isArray(e.comments) && e.comments.length) {
      reel.list = e.comments.map((t) => ({ t: String(t), user: someone(), likes: (r() * r() * 4000) | 0, color: pick(AVATAR) }));
      if (!Number.isFinite(e.comment_count)) reel.comments = Math.max(reel.list.length, reel.comments);
    }
  }
  return reel;
}

// --- drawing helpers ------------------------------------------------------------------------------
function heartPath(c, x, y, s) {
  c.beginPath();
  c.moveTo(x, y + s * 0.35);
  c.bezierCurveTo(x - s * 1.1, y - s * 0.35, x - s * 0.5, y - s * 1.0, x, y - s * 0.45);
  c.bezierCurveTo(x + s * 0.5, y - s * 1.0, x + s * 1.1, y - s * 0.35, x, y + s * 0.35);
  c.closePath();
}
function stick(c, x, y, s, { armL = 0.6, armR = -0.6, legL = 0.3, legR = -0.3, color = "#fff", w = 5 } = {}) {
  // a stick figure standing at (x, y) = feet, height ~ 3.4 s; angles from straight down
  c.strokeStyle = color; c.lineWidth = w; c.lineCap = "round";
  const hip = [x, y - 1.4 * s], neck = [x, y - 2.6 * s];
  const limb = (from, a, l) => { c.moveTo(...from); c.lineTo(from[0] + Math.sin(a) * l, from[1] + Math.cos(a) * l); };
  c.beginPath();
  c.moveTo(...hip); c.lineTo(...neck);
  limb(hip, legL, 1.45 * s); limb(hip, legR, 1.45 * s);
  limb(neck, armL + Math.PI * 0.15, 1.1 * s); limb(neck, armR - Math.PI * 0.15, 1.1 * s);
  c.stroke();
  c.beginPath(); c.arc(x, y - 3.05 * s, 0.42 * s, 0, Math.PI * 2); c.stroke();
}
function wrap(c, text, x, y, maxW, lh, maxLines = 2) {
  const words = text.split(" ");
  let line = "", n = 0;
  for (const w of words) {
    const test = line ? line + " " + w : w;
    if (c.measureText(test).width > maxW && line) {
      if (++n >= maxLines) { c.fillText(line + "…", x, y); return; }
      c.fillText(line, x, y); line = w; y += lh;
    } else line = test;
  }
  c.fillText(line, x, y);
}
const FONT = (px, w = 600) => `${w} ${px}px -apple-system, "Segoe UI", system-ui, sans-serif`;

// each draws a full 360x640 frame of video at time t (seconds)
const DRAW = {
  cat(c, t, r) {
    const g = c.createLinearGradient(0, 0, 0, H);
    g.addColorStop(0, "#f9e1c4"); g.addColorStop(1, "#d99a66");
    c.fillStyle = g; c.fillRect(0, 0, W, H);
    c.fillStyle = "#b9784a"; c.fillRect(0, 430, W, 210);                          // floor
    const bx = 180 + Math.sin(t * 1.3) * 110, by = 452;                            // yarn ball
    c.fillStyle = "#e0405a"; c.beginPath(); c.arc(bx, by, 22, 0, Math.PI * 2); c.fill();
    c.strokeStyle = "#a82038"; c.lineWidth = 2;
    for (let i = 0; i < 3; i++) { c.beginPath(); c.arc(bx, by, 22, t * 3 + i * 2, t * 3 + i * 2 + 1.6); c.stroke(); }
    const cx = 170, head = 300 + Math.sin(t * 3) * 6, look = Math.sin(t * 1.3) * 7;
    c.fillStyle = "#f08a2c";
    c.beginPath(); c.ellipse(cx, 400, 70, 52, 0, 0, Math.PI * 2); c.fill();          // body
    c.strokeStyle = "#f08a2c"; c.lineWidth = 16; c.lineCap = "round";               // tail
    c.beginPath(); c.moveTo(cx + 60, 420); c.quadraticCurveTo(cx + 120, 400, cx + 110 + Math.sin(t * 2.5) * 30, 330); c.stroke();
    const paw = Math.max(0, Math.sin(t * 2.6)) * 40;                                 // swat!
    c.lineWidth = 18; c.beginPath(); c.moveTo(cx - 30, 420); c.lineTo(cx - 40 - paw * 0.4, 440 + paw * 0.2); c.stroke();
    c.beginPath(); c.arc(cx, head, 52, 0, Math.PI * 2); c.fill();                      // head + ears
    for (const s of [-1, 1]) {
      c.beginPath(); c.moveTo(cx + s * 46, head - 18); c.lineTo(cx + s * 40, head - 74); c.lineTo(cx + s * 12, head - 46); c.fill();
    }
    c.fillStyle = "#2a1a10";
    const blink = t % 3.2 < 0.15;
    for (const s of [-1, 1]) {
      if (blink) c.fillRect(cx + s * 20 + look - 9, head - 6, 18, 3);
      else { c.beginPath(); c.ellipse(cx + s * 20 + look, head - 6, 7, 10, 0, 0, Math.PI * 2); c.fill(); }
    }
    c.fillStyle = "#ff9fb0"; c.beginPath(); c.moveTo(cx - 6, head + 10); c.lineTo(cx + 6, head + 10); c.lineTo(cx, head + 17); c.fill();
    c.strokeStyle = "#5a3a20"; c.lineWidth = 1.5;
    for (const s of [-1, 1]) for (const k of [-1, 0, 1]) {
      c.beginPath(); c.moveTo(cx + s * 14, head + 16); c.lineTo(cx + s * 62, head + 12 + k * 10); c.stroke();
    }
  },
  cooking(c, t, r) {
    c.fillStyle = "#3a2c22"; c.fillRect(0, 0, W, H);
    c.fillStyle = "#4a382a";
    for (let i = 0; i < 8; i++) c.fillRect(0, i * 84, W, 3);                          // wooden counter
    c.fillStyle = "#1a1a1c"; c.fillRect(150, 470, 34, 160);                            // pan handle
    c.beginPath(); c.arc(170, 330, 140, 0, Math.PI * 2); c.fill();
    c.fillStyle = "#2c2c30"; c.beginPath(); c.arc(170, 330, 122, 0, Math.PI * 2); c.fill();
    c.fillStyle = "rgba(255,170,40,.25)"; c.beginPath(); c.arc(170, 330, 118, 0, Math.PI * 2); c.fill(); // butter
    for (let i = 0; i < 9; i++) {                                                     // banana slices
      const a = i * 0.7 + t * 0.4, d = 30 + (i % 3) * 30;
      const x = 170 + Math.cos(a) * d + Math.sin(t * 20 + i) * 1.5, y = 330 + Math.sin(a) * d;
      c.fillStyle = "#f5d46a"; c.beginPath(); c.arc(x, y, 17, 0, Math.PI * 2); c.fill();
      c.fillStyle = `rgba(160,90,20,${0.3 + 0.4 * clamp01(t / 6)})`; c.beginPath(); c.arc(x, y, 14, 0, Math.PI * 2); c.fill();
      c.fillStyle = "#fff2b0"; c.beginPath(); c.arc(x, y, 4, 0, Math.PI * 2); c.fill();
    }
    for (let i = 0; i < 14; i++) {                                                    // sizzle bubbles
      const p = (t * 1.7 + i * 0.37) % 1, a = i * 2.4;
      c.fillStyle = `rgba(255,255,255,${0.6 * (1 - p)})`;
      c.beginPath(); c.arc(170 + Math.cos(a) * (40 + i * 5), 330 + Math.sin(a) * (40 + i * 5), 2 + p * 4, 0, Math.PI * 2); c.fill();
    }
    c.strokeStyle = "rgba(255,255,255,.18)"; c.lineWidth = 6;                            // steam
    for (let i = 0; i < 3; i++) {
      c.beginPath();
      for (let y = 0; y < 150; y += 10) c.lineTo(120 + i * 50 + Math.sin(y / 20 + t * 3 + i) * 10, 200 - y - ((t * 40) % 20));
      c.stroke();
    }
    const sp = Math.sin(t * 2.2) * 50;                                                 // spatula
    c.fillStyle = "#c9c9c9"; c.fillRect(200 + sp, 300, 46, 30);
    c.fillStyle = "#222"; c.save(); c.translate(246 + sp, 312); c.rotate(-0.5); c.fillRect(0, 0, 120, 10); c.restore();
    c.fillStyle = "#fff"; c.font = FONT(26, 800); c.textAlign = "center";
    c.fillText(["step 1: butter", "step 2: bananas", "step 3: FIRE 🔥"][Math.floor(t / 3) % 3], W / 2, 140);
    if (Math.floor(t / 3) % 3 === 2) {                                                // flambé!
      for (let i = 0; i < 18; i++) {
        const p = (t * 2 + i * 0.13) % 1;
        c.fillStyle = `rgba(255,${120 + i * 6},20,${0.7 * (1 - p)})`;
        c.beginPath(); c.arc(100 + i * 8 + Math.sin(t * 9 + i) * 8, 300 - p * 120, 14 * (1 - p) + 4, 0, Math.PI * 2); c.fill();
      }
    }
  },
  dance(c, t, r) {
    const hue = (t * 50) % 360;
    c.fillStyle = `hsl(${hue},70%,18%)`; c.fillRect(0, 0, W, H);
    for (let i = 0; i < 4; i++) {                                                     // spotlights
      const a = Math.sin(t * 1.5 + i * 1.7) * 0.5;
      c.fillStyle = `hsla(${(hue + i * 90) % 360},90%,60%,.18)`;
      c.beginPath(); c.moveTo(40 + i * 93, 0); c.lineTo(40 + i * 93 + Math.sin(a) * 400 - 70, 560); c.lineTo(40 + i * 93 + Math.sin(a) * 400 + 70, 560); c.fill();
    }
    for (let x = 0; x < 6; x++) for (let y = 0; y < 3; y++) {                          // floor tiles
      const on = (x + y + Math.floor(t * 4)) % 3 === 0;
      c.fillStyle = on ? `hsl(${(hue + x * 40) % 360},90%,55%)` : "#1a1a22";
      c.fillRect(x * 60 + 2, 520 + y * 40 + 2, 56, 36);
    }
    const b = t * 7, bounce = Math.abs(Math.sin(b)) * 14;
    stick(c, 180 + Math.sin(b / 2) * 20, 520 - bounce, 58, {
      armL: Math.PI * 0.6 + Math.sin(b) * 0.9, armR: -Math.PI * 0.6 + Math.cos(b) * 0.9,
      legL: 0.35 + Math.sin(b) * 0.35, legR: -0.35 + Math.sin(b) * 0.35, w: 8, color: "#fff" });
    c.fillStyle = "#fff"; c.font = FONT(30, 900); c.textAlign = "center";
    if (Math.sin(b) > 0.7) c.fillText(["5", "6", "7", "8"][Math.floor(t * 2) % 4], 300, 150);
  },
  fail(c, t, r) {
    const g = c.createLinearGradient(0, 0, 0, H);
    g.addColorStop(0, "#5aa8ff"); g.addColorStop(1, "#cfe8ff");
    c.fillStyle = g; c.fillRect(0, 0, W, H);
    c.fillStyle = "#7a7f87"; c.fillRect(0, 470, W, 170);                            // pavement
    c.fillStyle = "#a0a6ad"; c.beginPath(); c.moveTo(140, 470); c.lineTo(240, 470); c.lineTo(240, 400); c.fill(); // ramp
    const p = (t % 4) / 4;
    if (p < 0.5) {                                                                    // rolling up, so confident
      const x = 30 + p * 2 * 210, y = x > 140 ? 470 - (x - 140) * 0.7 : 470;
      c.fillStyle = "#222"; c.fillRect(x - 30, y - 8, 60, 6);
      stick(c, x, y - 10, 34, { armL: 1.6, armR: -1.6, legL: 0.25, legR: -0.25, color: "#1d1d1d", w: 6 });
    } else {                                                                          // launch ... and eat it
      const q = (p - 0.5) * 2, x = 240 + q * 90, y = 400 - Math.sin(q * Math.PI) * 140 + q * 70;
      c.save(); c.translate(x, y); c.rotate(q * 7);
      stick(c, 0, 0, 34, { armL: 2.6, armR: -2.6, legL: 0.9, legR: -0.6, color: "#1d1d1d", w: 6 });
      c.restore();
      c.fillStyle = "#222"; c.save(); c.translate(240 + q * 160, 400 - Math.sin(q * Math.PI) * 220 + q * 70); c.rotate(-q * 10); c.fillRect(-30, -3, 60, 6); c.restore();
      if (q > 0.6) {
        c.font = FONT(80, 900); c.textAlign = "center"; c.fillText("💀", 180, 250);
        c.fillStyle = "#fff"; c.strokeStyle = "#000"; c.lineWidth = 6; c.font = FONT(54, 900);
        c.strokeText("OOF", 180, 330); c.fillText("OOF", 180, 330);
      }
    }
  },
  lecture(c, t, r) {
    c.fillStyle = "#6b4a2b"; c.fillRect(0, 0, W, H);
    c.fillStyle = "#1f3b2d"; c.fillRect(14, 120, W - 28, 300);                       // chalkboard
    c.fillStyle = "#e8efe9"; c.font = FONT(19, 500); c.textAlign = "left";
    const lines = ["dV/dt = (−V + Σ wᵢ sᵢ) / τ", "134,181 neurons", "2.7M synapses", "Kenyon cells → MBONs", "∴ the fly decides"];
    let n = Math.floor(t * 9);
    lines.forEach((l, i) => { const k = Math.min(l.length, Math.max(0, n)); n -= l.length + 4; c.fillText(l.slice(0, k), 34, 165 + i * 52); });
    c.fillStyle = "#f2e9da"; c.fillRect(0, 420, W, 220);
    stick(c, 300, 600, 38, { armL: -2.2 + Math.sin(t * 2) * 0.2, armR: 0.3, color: "#333", w: 5 });   // professor
    c.strokeStyle = "#8b5a2b"; c.lineWidth = 3; c.beginPath(); c.moveTo(270, 500); c.lineTo(210 + Math.sin(t * 2) * 10, 400); c.stroke();
    c.fillStyle = "rgba(60,60,60,.7)"; c.font = FONT(28, 700);
    for (let i = 0; i < 3; i++) { const p = (t * 0.5 + i / 3) % 1; c.globalAlpha = 1 - p; c.fillText("z", 60 + i * 18 + p * 30, 560 - p * 120); }
    c.globalAlpha = 1;
  },
  ad(c, t, r) {
    const flash = Math.floor(t * 3) % 2;
    c.fillStyle = flash ? "#ffde21" : "#ff3b30"; c.fillRect(0, 0, W, H);
    c.fillStyle = flash ? "#ff3b30" : "#ffde21";
    for (let i = 0; i < 12; i++) {                                                    // sunburst
      c.beginPath(); c.moveTo(180, 330);
      c.arc(180, 330, 500, i * 0.52 + t * 0.6, i * 0.52 + t * 0.6 + 0.26); c.fill();
    }
    c.fillStyle = "#4b1fa0"; c.fillRect(110, 230, 140, 200);                          // the zapper
    c.fillStyle = `rgba(90,170,255,${0.6 + 0.4 * Math.sin(t * 20)})`; c.fillRect(126, 250, 108, 160);
    c.strokeStyle = "#ddd"; c.lineWidth = 3;
    for (let x = 136; x < 234; x += 14) { c.beginPath(); c.moveTo(x, 250); c.lineTo(x, 410); c.stroke(); }
    c.strokeStyle = "#fff"; c.lineWidth = 4;                                          // ZAP
    if (t % 1 < 0.25) { c.beginPath(); c.moveTo(180, 260); c.lineTo(160, 300); c.lineTo(195, 320); c.lineTo(170, 380); c.stroke(); }
    c.fillStyle = "#fff"; c.strokeStyle = "#000"; c.lineWidth = 7; c.textAlign = "center";
    c.font = FONT(38, 900); c.strokeText("ZAP-O-MATIC", 180, 150); c.fillText("ZAP-O-MATIC", 180, 150);
    c.font = FONT(30, 900); c.strokeText("3000", 180, 192); c.fillText("3000", 180, 192);
    c.save(); c.translate(285, 260); c.rotate(0.3); c.scale(1 + 0.12 * Math.sin(t * 8), 1 + 0.12 * Math.sin(t * 8));
    c.fillStyle = "#e5007e"; c.beginPath(); c.arc(0, 0, 48, 0, Math.PI * 2); c.fill();
    c.fillStyle = "#fff"; c.font = FONT(22, 900); c.fillText("50%", 0, 0); c.font = FONT(16, 900); c.fillText("OFF", 0, 20);
    c.restore();
    c.fillStyle = "#111"; c.beginPath(); c.roundRect(90, 450, 180, 46, 23); c.fill();
    c.fillStyle = "#fff"; c.font = FONT(20, 800); c.fillText("SHOP NOW ›", 180, 480);
  },
};

// --- the reels app on the phone screen --------------------------------------------------------------
class ReelsApp {
  constructor() {
    this.canvas = document.createElement("canvas");
    this.canvas.width = W * DPR; this.canvas.height = H * DPR;
    this.c = this.canvas.getContext("2d");
    this.reel = makeReel("cat");
    this.next = null;
    this.t0 = performance.now();
    this.swipe = 0;           // 0..1 while swiping to `next`
    this.sheet = 0;           // comments sheet open amount 0..1
    this.scroll = 0;          // comments scroll in px
    this.mood = "meh";
    this.burst = -1e9;        // time of the last double-tap heart
    this.taps = [];           // touch ripples {x, y, at}
    this.timer = null;        // {start, ms, s}: "watching N s" chip
  }
  now() { return performance.now(); }
  time() { return (this.now() - this.t0) / 1000; }
  setReel(reel) {
    if (this.reel !== reel) this.reel?.video?.pause();
    if (reel.video && !reel.video.dataset.play) playVideo(reel.video);
    this.reel = reel; this.t0 = this.now(); this.sheet = 0; this.scroll = 0;
  }
  // a real video, cover-fitted to the screen; returns its play position in seconds
  drawVideo(v) {
    const c = this.c;
    c.fillStyle = "#000"; c.fillRect(0, 0, W, H);
    if (v.readyState >= 2 && v.videoWidth) {
      const s = Math.max(W / v.videoWidth, H / v.videoHeight), w = v.videoWidth * s, h = v.videoHeight * s;
      c.drawImage(v, (W - w) / 2, (H - h) / 2, w, h);
    } else {                                                                        // loading spinner
      const a = this.now() / 150;
      c.strokeStyle = "rgba(255,255,255,.7)"; c.lineWidth = 4;
      c.beginPath(); c.arc(W / 2, H / 2, 18, a, a + 4.5); c.stroke();
    }
    return v.currentTime || 0;
  }
  tap(u, v) { this.taps.push({ x: u * W, y: v * H, at: this.now() }); }

  drawReel(reel, t, liked) {
    const c = this.c;
    if (reel.video) {
      t = this.drawVideo(reel.video);
      if (reel.video.duration > 0) reel.len = reel.video.duration;
    } else DRAW[reel.cat](c, t % reel.len, rnd(reel.seed));
    // legibility gradients
    let g = c.createLinearGradient(0, H - 220, 0, H);
    g.addColorStop(0, "rgba(0,0,0,0)"); g.addColorStop(1, "rgba(0,0,0,.6)");
    c.fillStyle = g; c.fillRect(0, H - 220, W, 220);
    g = c.createLinearGradient(0, 0, 0, 90);
    g.addColorStop(0, "rgba(0,0,0,.4)"); g.addColorStop(1, "rgba(0,0,0,0)");
    c.fillStyle = g; c.fillRect(0, 0, W, 90);
    c.textAlign = "left"; c.fillStyle = "#fff";
    c.font = FONT(24, 800); c.fillText("Reels", 18, 46);
    c.strokeStyle = "#fff"; c.lineWidth = 2.5;                                       // camera icon
    c.beginPath(); c.roundRect(310, 28, 30, 22, 5); c.stroke();
    c.beginPath(); c.arc(325, 39, 6, 0, Math.PI * 2); c.stroke();
    // right rail
    heartPath(c, RAIL.x, RAIL.heart, 17);
    if (liked) { c.fillStyle = "#ff3040"; c.fill(); } else { c.lineWidth = 3; c.stroke(); }
    c.fillStyle = "#fff"; c.textAlign = "center"; c.font = FONT(13, 600);
    c.fillText(fmt(reel.likes + (liked ? 1 : 0)), RAIL.x, RAIL.heart + 30);
    c.lineWidth = 3; c.beginPath(); c.ellipse(RAIL.x, RAIL.comment - 4, 16, 14, 0, 0, Math.PI * 2); c.stroke();
    c.beginPath(); c.moveTo(RAIL.x - 12, RAIL.comment + 6); c.lineTo(RAIL.x - 16, RAIL.comment + 14); c.lineTo(RAIL.x - 4, RAIL.comment + 9); c.stroke();
    c.fillText(fmt(reel.comments), RAIL.x, RAIL.comment + 34);
    c.beginPath(); c.moveTo(RAIL.x - 15, RAIL.share - 2); c.lineTo(RAIL.x + 15, RAIL.share - 14); c.lineTo(RAIL.x + 3, RAIL.share + 14); c.lineTo(RAIL.x - 2, RAIL.share + 1); c.closePath(); c.stroke();
    // who / what
    c.textAlign = "left";
    c.fillStyle = AVATAR[reel.seed % AVATAR.length]; c.beginPath(); c.arc(32, H - 120, 15, 0, Math.PI * 2); c.fill();
    c.fillStyle = "#fff"; c.font = FONT(15, 700); c.fillText(reel.user, 56, H - 115);
    if (reel.cat !== "ad") {
      const w = c.measureText(reel.user).width;
      c.strokeStyle = "#fff"; c.lineWidth = 1.5; c.beginPath(); c.roundRect(64 + w, H - 132, 62, 24, 7); c.stroke();
      c.font = FONT(13, 700); c.fillText("Follow", 74 + w, H - 115);
    }
    c.font = FONT(15, 500); wrap(c, reel.text, 18, H - 80, 270, 20, 2);
    c.font = FONT(13, 500); c.fillStyle = "rgba(255,255,255,.85)";
    c.fillText(`♫ original audio · ${reel.user.split(" ")[0]}`, 18, H - 30);
    // progress bar
    c.fillStyle = "rgba(255,255,255,.3)"; c.fillRect(0, H - 4, W, 4);
    c.fillStyle = "#fff"; c.fillRect(0, H - 4, W * ((t % reel.len) / reel.len), 4);
  }

  drawSheet() {
    const c = this.c, top = H - this.sheet * H * 0.68;
    c.fillStyle = `rgba(0,0,0,${0.35 * this.sheet})`; c.fillRect(0, 0, W, H);
    c.fillStyle = "#1c1c1e"; c.beginPath(); c.roundRect(0, top, W, H, [18, 18, 0, 0]); c.fill();
    c.fillStyle = "#555"; c.beginPath(); c.roundRect(W / 2 - 20, top + 8, 40, 5, 3); c.fill();
    c.fillStyle = "#fff"; c.textAlign = "center"; c.font = FONT(15, 700);
    c.fillText(`${fmt(this.reel.comments)} comments`, W / 2, top + 38);
    c.font = FONT(20, 500); c.fillText("✕", W - 28, top + 38);
    c.fillStyle = "#333"; c.fillRect(0, top + 54, W, 1);
    c.save(); c.beginPath(); c.rect(0, top + 55, W, H - top - 55); c.clip();
    const list = this.mood === "drama!" ? this.reel.list.flatMap((x, i) => [x, { ...x, t: DRAMA[i % DRAMA.length], user: x.user + "_", color: "#e63946", likes: x.likes * 3 }])
      : this.reel.list;
    list.forEach((cm, i) => {
      const y = top + 70 + i * 66 - this.scroll;
      c.fillStyle = cm.color; c.beginPath(); c.arc(30, y + 18, 15, 0, Math.PI * 2); c.fill();
      c.textAlign = "left"; c.fillStyle = "#999"; c.font = FONT(12, 600); c.fillText(cm.user, 56, y + 10);
      c.fillStyle = "#fff"; c.font = FONT(14, 500); wrap(c, cm.t, 56, y + 30, 250, 18, 2);
      heartPath(c, W - 26, y + 18, 8); c.strokeStyle = "#999"; c.lineWidth = 1.5; c.stroke();
      c.fillStyle = "#999"; c.textAlign = "center"; c.font = FONT(11, 500); c.fillText(fmt(cm.likes), W - 26, y + 40);
    });
    c.restore();
  }

  drawInstagram() {                                                              // the real thing, cover-fitted
    const c = this.c, f = IG.frame, s = Math.max(W / f.width, H / f.height);
    c.drawImage(f, (W - f.width * s) / 2, (H - f.height * s) / 2, f.width * s, f.height * s);
  }
  chip(text, color) {
    const c = this.c;
    c.font = FONT(12, 700);
    const w = c.measureText(text).width + 24;
    c.fillStyle = color; c.beginPath(); c.roundRect(W / 2 - w / 2, H - 46, w, 24, 12); c.fill();
    c.fillStyle = "#fff"; c.textAlign = "center"; c.fillText(text, W / 2, H - 29);
  }

  draw() {
    const c = this.c, now = this.now(), t = this.time(), live = igLive();
    c.setTransform(DPR, 0, 0, DPR, 0, 0);
    c.save(); c.beginPath(); c.rect(0, 0, W, H); c.clip();
    const v = this.reel.video;                                                       // backup video rests while live
    if (v && live && !v.paused) v.pause();
    else if (v && !live && v.paused && v.dataset.play && v.src) v.play().catch(() => {});
    if (live) this.drawInstagram();
    else if (this.swipe > 0 && this.next) {
      const off = ease(this.swipe) * H;
      c.save(); c.translate(0, -off); this.drawReel(this.reel, t, this.liked); c.restore();
      c.save(); c.translate(0, H - off); this.drawReel(this.next, 0, false); c.restore();
    } else this.drawReel(this.reel, t, this.liked);
    if (this.sheet > 0 && !live) this.drawSheet();
    if (IG.state === "login") this.chip("instagram: log in · showing backup videos", "rgba(200,120,0,.9)");
    else if (IG.state === "blocked") this.chip("instagram blocked · showing backup videos", "rgba(200,30,40,.9)");
    const b = (now - this.burst) / 900;                                              // double-tap heart
    if (b >= 0 && b < 1) {
      c.save(); c.globalAlpha = b < 0.7 ? 1 : 1 - (b - 0.7) / 0.3;
      heartPath(c, W / 2, H / 2, 30 + 60 * Math.min(1, b * 4) - (b > 0.25 ? 8 * Math.sin((b - 0.25) * 12) * (1 - b) : 0));
      c.fillStyle = "#ff3040"; c.fill(); c.restore();
    }
    if (this.timer) {                                                                // "watching N s" chip
      const { start, ms, s } = this.timer, p = clamp01((now - start) / ms);
      const left = Math.ceil(s * (1 - p));
      c.fillStyle = "rgba(0,0,0,.55)"; c.beginPath(); c.roundRect(W / 2 - 78, 66, 156, 34, 17); c.fill();
      c.strokeStyle = "#fff"; c.lineWidth = 3; c.beginPath(); c.arc(W / 2 - 56, 83, 9, -Math.PI / 2, -Math.PI / 2 + Math.PI * 2 * (1 - p)); c.stroke();
      c.fillStyle = "#fff"; c.textAlign = "left"; c.font = FONT(14, 700);
      c.fillText(`swipe in ${left}s`, W / 2 - 38, 88);
    }
    this.taps = this.taps.filter((tp) => now - tp.at < 450);
    for (const tp of this.taps) {
      const p = (now - tp.at) / 450;
      c.strokeStyle = `rgba(255,255,255,${0.8 * (1 - p)})`; c.lineWidth = 4;
      c.beginPath(); c.arc(tp.x, tp.y, 10 + p * 34, 0, Math.PI * 2); c.stroke();
    }
    c.restore();
  }
}

export default {
  title: "Ferrari Inglewood",
  blurb: "Like it, read the comments, or swipe after 1-10 s? The fly's neurons run the doomscroll.",
  playTimeout: 16000, // the longest watch (10 s) plus flying up to swipe

  build({ THREE, group, fly }) {
    this.THREE = THREE;
    this.group = group;
    this.fly = fly;
    group.add(box(2.6, 0.01, 1.9, 0x2d2a3e, [0, 0.005, 0.15]));                      // rug
    // the phone, propped up on a stand
    const phone = new THREE.Group();
    phone.position.set(0, 0, 0);
    group.add(phone);
    const body = box(SW + 0.08, SH + 0.16, 0.07, 0x16161a, [0, SY, 0]);
    body.material.roughness = 0.3; body.material.metalness = 0.4;
    phone.add(body);
    phone.add(box(0.16, 0.025, 0.02, 0x050506, [0, SY + SH / 2 + 0.035, SZ - 0.005])); // camera notch
    phone.add(box(0.02, 0.14, 0.03, 0x2a2a30, [SW / 2 + 0.05, SY + 0.45, 0]));      // buttons
    phone.add(box(0.5, 0.05, 0.36, 0x9aa0a8, [0, 0.025, -0.08]));                    // stand
    phone.add(box(0.06, 0.62, 0.05, 0x9aa0a8, [0, 0.33, -0.12]));
    this.app = new ReelsApp();
    this.tex = new THREE.CanvasTexture(this.app.canvas);
    this.tex.colorSpace = THREE.SRGBColorSpace;
    this.tex.anisotropy = 4;
    this.screen = new THREE.Mesh(new THREE.PlaneGeometry(SW, SH), new THREE.MeshBasicMaterial({ map: this.tex, toneMapped: false }));
    this.screen.position.set(0, SY, SZ);
    phone.add(this.screen);
    this.glow = new THREE.PointLight(0xffb070, 1.2, 3, 1.6);                          // the screen lights the room
    this.glow.position.set(0, SY, 0.6);
    group.add(this.glow);
    // the beanbag, a juice box and a banana peel
    const bag = new THREE.Mesh(new THREE.SphereGeometry(0.34, 20, 14), new THREE.MeshStandardMaterial({ color: 0x7d3cff, roughness: 0.9 }));
    bag.scale.set(1, 0.48, 1); bag.position.set(SEAT.x, 0.1, SEAT.z); bag.castShadow = bag.receiveShadow = true;
    group.add(bag);
    const juice = box(0.1, 0.16, 0.07, 0xff9f1c, [SEAT.x + 0.38, 0.08, SEAT.z - 0.25]);
    juice.add(box(0.012, 0.1, 0.012, 0xffffff, [0.03, 0.12, 0]));
    group.add(juice);
    for (const a of [0, 2.1, 4.2]) {
      const peel = box(0.06, 0.012, 0.16, 0xf2d03b, [SEAT.x - 0.5 + Math.sin(a) * 0.06, 0.012, SEAT.z + 0.35 + Math.cos(a) * 0.06]);
      peel.rotation.y = a; group.add(peel);
    }
    this.sit(fly);
    // once the video list is in, swap the opening cartoon for a real video of the same kind
    LIBRARY.ready.then(() => {
      if (LIBRARY.instagram?.port) connectInstagram(LIBRARY.instagram.port);
      if (!this.app.reel.video && LIBRARY.reels.some((e) => e.category === this.app.reel.cat)) this.app.setReel(makeReel(this.app.reel.cat));
    });
    // keep the screen playing
    let last = 0, frames = 0;
    const probeCanvas = document.createElement("canvas");
    probeCanvas.width = probeCanvas.height = 1;
    const probe = probeCanvas.getContext("2d", { willReadFrequently: true });
    const loop = (now) => {
      if (!group.parent) return;                                                     // station was removed
      if (now - last > 33) {
        last = now;
        this.app.draw();
        this.tex.needsUpdate = true;
        if (++frames % 5 === 0) {                                                    // the room takes the screen's colour
          probe.drawImage(this.app.canvas, 0, 0, 1, 1);
          const [r, g, b] = probe.getImageData(0, 0, 1, 1).data;
          const m = Math.max(r, g, b, 1);
          this.glow.color.setRGB(r / m, g / m, b / m);                                   // full brightness, same hue
        }
      }
      requestAnimationFrame(loop);
    };
    requestAnimationFrame(loop);
  },

  // --- where things are -------------------------------------------------------------------------------
  // a spot on the screen (u right, v down, 0..1) in the station's frame
  screenPoint(u, v) {
    return new this.THREE.Vector3((u - 0.5) * SW, SY + (0.5 - v) * SH, SZ);
  },
  seatYaw() { return Math.atan2(-SEAT.x, -SEAT.z); },                              // looking at the phone
  sit(fly) {
    fly.root.position.set(SEAT.x, SEAT.y, SEAT.z);
    fly.root.rotation.y = this.seatYaw();
    fly.reset();
    fly.body.rotation.x = -0.18;                                                     // leaning back, chilling
    fly.rest();
    this.seated = true;
  },
  flutter(t) { for (const { pivot, s } of this.fly.wings) pivot.rotation.z = s * (0.25 + 0.6 * Math.abs(Math.sin(t * 60))); },
  stopWings() { for (const { pivot } of this.fly.wings) pivot.rotation.z = 0; },

  // fly (wings buzzing) to hover in front of screen spot (u, v), front leg ready
  async flyTo(u, v, ms = 380) {
    const { fly } = this;
    const P = this.screenPoint(u, v);
    const from = fly.root.position.clone(), to = P.clone().add(new this.THREE.Vector3(HOVER.side, -HOVER.up, HOVER.out));
    const yaw0 = fly.root.rotation.y, pitch0 = fly.body.rotation.x;
    this.seated = false;
    const t0 = performance.now();
    await tween(ms, (t) => {
      const e = ease(t);
      fly.root.position.lerpVectors(from, to, e);
      fly.root.position.y += Math.sin(Math.PI * t) * 0.12;
      fly.root.rotation.y = angLerp(yaw0, Math.PI, e);
      fly.body.rotation.x = lerp(pitch0, -0.4, e);
      fly.body.position.y = 0;
      fly.rest();
      this.flutter((performance.now() - t0) / 1000);
    });
    this.reach(P, 0.06);
  },
  // put the right front foot on (or `gap` in front of) a point, in the station's frame
  reach(P, gap = 0) {
    const { fly } = this;
    fly.root.updateMatrixWorld(true);
    const w = this.group.localToWorld(P.clone().add(new this.THREE.Vector3(0, 0, gap)));
    fly.setFoot("R", 0, fly.root.worldToLocal(w), new this.THREE.Vector3(0.4, -1, 0));
    this.flutter(performance.now() / 1000);
  },
  async touch(u, v) {
    const P = this.screenPoint(u, v);
    await tween(90, (t) => this.reach(P, 0.06 * (1 - t)));
    this.app.tap(u, v);
    await tween(110, (t) => this.reach(P, 0.06 * t));
  },
  // drag the foot up the glass from v0 to v1, the fly rising with it; onMove(t) runs every frame
  async drag(u, v0, v1, ms, onMove) {
    const { fly } = this;
    const base = fly.root.position.clone(), P0 = this.screenPoint(u, v0);
    await tween(90, (t) => this.reach(P0, 0.06 * (1 - t)));
    await tween(ms, (t) => {
      const e = ease(t), v = lerp(v0, v1, e);
      fly.root.position.set(base.x, base.y + (v0 - v) * SH, base.z);
      this.reach(this.screenPoint(u, v));
      onMove?.(t);
    });
    await tween(100, (t) => this.reach(this.screenPoint(u, v1), 0.06 * t));
  },
  async goSit(ms = 420) {
    const { fly } = this;
    const from = fly.root.position.clone(), to = new this.THREE.Vector3(SEAT.x, SEAT.y, SEAT.z);
    const yaw0 = fly.root.rotation.y, pitch0 = fly.body.rotation.x, t0 = performance.now();
    await tween(ms, (t) => {
      const e = ease(t);
      fly.root.position.lerpVectors(from, to, e);
      fly.root.position.y += Math.sin(Math.PI * t) * 0.15;
      fly.root.rotation.y = angLerp(yaw0, this.seatYaw(), e);
      fly.body.rotation.x = lerp(pitch0, -0.18, e);
      fly.rest();
      this.flutter((performance.now() - t0) / 1000);
    });
    this.stopWings();
    this.sit(fly);
  },

  // --- keeping the phone in step with the brain's world --------------------------------------------
  sync(caption) {
    const cur = caption.match(/the (\w+) reel/);
    if (cur && REEL[cur[1]] && cur[1] !== this.app.reel.cat) {
      this.app.setReel(makeReel(cur[1]));
      this.app.liked = false;
    }
  },

  // --- actions -----------------------------------------------------------------------------------
  async like(caption) {
    await this.flyTo(0.48, 0.5);
    await this.touch(0.48, 0.5);
    await wait(60);
    await this.touch(0.5, 0.49);
    this.app.liked = true;
    this.app.burst = performance.now();
    if (caption.startsWith("hate")) await tween(500, (t) => (this.fly.root.rotation.z = Math.sin(t * 30) * 0.08));   // ugh
    else await this.fly.hop(0.08, 300);
    this.fly.root.rotation.z = 0;
    await wait(250);
    await this.goSit();
  },

  async comments(caption) {
    const app = this.app;
    app.mood = (caption.match(/\(([^)]+)\)/) || [])[1] || "meh";
    const icon = igSpot("comment", [RAIL.x / W, RAIL.comment / H]);
    await this.flyTo(...icon);
    await this.touch(...icon);
    igDo({ do: "open_comments" });
    app.scroll = 0;
    await tween(320, (t) => (app.sheet = ease(t)));
    for (let i = 0; i < 2; i++) {                                                  // scroll down a set amount
      await this.flyTo(0.42, 0.86, 230);
      const s0 = app.scroll;
      igDo({ do: "scroll_comments", px: 260 });
      await this.drag(0.42, 0.86, 0.6, 480, (t) => (app.scroll = s0 + 150 * ease(t)));
    }
    await wait(500);                                                               // reading...
    const top = 1 - 0.68, x = igSpot("close", [(W - 28) / W, top + 34 / H]);
    await this.flyTo(...x, 260);
    await this.touch(...x);
    igDo({ do: "close_comments" });
    await tween(300, (t) => (app.sheet = 1 - ease(t)));
    await this.goSit();
  },

  async scroll(seconds, caption) {
    const app = this.app;
    if (!this.seated) await this.goSit();
    const nx = caption.match(/swipes to a (\d+)s (\w+) reel/);
    app.next = makeReel(nx && REEL[nx[2]] ? nx[2] : "cat", nx ? Number(nx[1]) : 0);   // loads while we watch
    const ms = seconds * 1000, start = performance.now();
    app.timer = { start, ms, s: seconds };
    const { fly } = this;
    await tween(ms, (t) => {
      const k = (performance.now() - start) / 1000;
      fly.body.rotation.x = -0.18 + Math.sin(k * 2.2) * 0.03;                       // nodding along
      fly.body.scale.set(1 + 0.015 * Math.sin(k * 3), 1 - 0.015 * Math.sin(k * 3), 1);
    });
    fly.body.scale.set(1, 1, 1);
    app.timer = null;
    playVideo(app.next.video);                                                     // starts as it slides in
    await this.flyTo(0.5, 0.82);
    const live = igLive();
    igDo({ do: "next" });                                                          // the real feed scrolls itself
    await this.drag(0.5, 0.82, 0.38, 430, (t) => (app.swipe = live ? 0 : t));
    app.setReel(app.next);
    app.next = null; app.swipe = 0; app.liked = false;
    await this.goSit();
  },

  async play(action, caption, { fly }) {
    this.fly = fly;
    this.sync(caption);
    const m = action.match(/^scroll_(\d+)s$/);
    if (m) return this.scroll(Number(m[1]), caption);
    if (action === "like") return this.like(caption);
    if (action === "comments") return this.comments(caption);
    await fly.hop(0.2, 300);
  },
};
