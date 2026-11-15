// One live scene: every fly's station on the ground, a brain panel that drops in when you visit a fly,
// and the pods above it all. Talks to server/serve.py over a websocket.
//   < >   overview -> each fly -> overview (on the pods: each live subject)
//   ^ v   up to the pods / back down to the world
import * as THREE from "three";
import { OrbitControls } from "three/addons/controls/OrbitControls.js";
import { CSS2DRenderer, CSS2DObject } from "three/addons/renderers/CSS2DRenderer.js";
import { Fly, tween, wait, box } from "./lib/fly.js";
import { BrainActivity, brainMaterial, brainShared, loadPreset, pointSize, ACTION_COLORS, hex } from "./brain.js";
import { PodWall, WALL_CENTER } from "./pods.js";

// The world's set dressing (floor, cubicles, lights, the back wall, post-processing): a fly-sized office at night.
import look from "./looks/overtime.js";

const WS_URL = `${location.protocol === "https:" ? "wss" : "ws"}://${location.host}/ws`; // same port as the page
const SPACING = 6.2;       // gap between neighbouring stations along the half circle (the widest stations are ~4 wide)
const PLAY_TIMEOUT = 4000; // a stuck station never stalls its fly for longer than this
const BG = look.bg;
const FOG = look.fog;
const ease = (t) => (t < 0.5 ? 4 * t * t * t : 1 - Math.pow(-2 * t + 2, 3) / 2);

// --- renderer, world --------------------------------------------------------
const el = document.getElementById("world");
const renderer = new THREE.WebGLRenderer({ antialias: true });
renderer.setPixelRatio(Math.min(devicePixelRatio, 2));
renderer.shadowMap.enabled = true;
renderer.autoClear = false;
el.appendChild(renderer.domElement);
const labels = new CSS2DRenderer();
labels.domElement.style.position = "absolute";
labels.domElement.style.top = "0";
labels.domElement.style.pointerEvents = "none";
el.appendChild(labels.domElement);

const scene = new THREE.Scene();
scene.background = new THREE.Color(BG);
scene.fog = new THREE.Fog(BG, ...FOG.ground);
const camera = new THREE.PerspectiveCamera(50, 1, 0.1, 800);
camera.position.set(0, 12, 18);
const controls = new OrbitControls(camera, renderer.domElement);
controls.enableDamping = true;


const wall = new PodWall(scene);
await look.build({ THREE, scene, renderer, camera });

// --- the brain panel (its own little scene, drawn into the top band) ----------
const panelEl = document.getElementById("panel");
const brainEl = document.getElementById("p-brain");
const tip = document.getElementById("tip");
const brainScene = new THREE.Scene();
const brainCam = new THREE.PerspectiveCamera(30, 2, 0.01, 50);
brainCam.position.set(0, 0.25, 2.55);
brainCam.lookAt(0, 0, 0);
const brainHolder = new THREE.Group();
brainScene.add(brainHolder);
const panelScale = { value: 400 };
const THEMES = {
  light: { band: 0xf4d6cb, base: 0x8a5446, hot: 0xb0250f, input: 0x0b7285, baseAlpha: 0.13, additive: false },
  dark: { band: 0x120406, base: 0x6a1424, hot: 0xfff0f2, input: 0x39d0d0, baseAlpha: 0.3, additive: true },
};
Object.assign(THEMES.light, look.panel);
let panel = { fly: null, points: null, theme: "light", yaw: 0, drag: null };

function resize() {
  renderer.setSize(innerWidth, innerHeight);
  labels.setSize(innerWidth, innerHeight);
  camera.aspect = innerWidth / innerHeight;
  camera.updateProjectionMatrix();
  const ph = Math.round(Math.min(360, Math.max(230, innerHeight * 0.34)));
  document.documentElement.style.setProperty("--ph", ph + "px");
  look.resize(innerWidth, innerHeight);
}
addEventListener("resize", resize);
resize();

// --- flies ---------------------------------------------------------------
const flies = {}; // name -> {info, station, fly, group, label, labelObj, brainIcon, brain, title, lesioned}
let order = [];
let ws;
function send(m) { if (ws && ws.readyState === 1) ws.send(JSON.stringify(m)); }

const fallbackStation = {
  title: "", blurb: "",
  build({ group, fly }) { group.add(box(1.2, 0.1, 1.2, 0x30363d, [0, 0.05, 0])); fly.root.position.y = 0.1; },
  async play(action, caption, { fly }) { await fly.hop(0.25, 400); await wait(300); },
};

async function addFly(info, i, total) {
  let station;
  try {
    station = (await import(`/flies/${info.name}/station.js`)).default;
  } catch (e) {
    console.warn(`flies/${info.name}/station.js failed to load, using a placeholder`, e);
    station = Object.create(fallbackStation);
  }
  const group = new THREE.Group();
  const { pos, yaw } = arcSlot(i, total);
  group.position.copy(pos);
  group.rotation.y = yaw;                          // each station faces the middle of the half circle
  scene.add(group);
  const fly = new Fly();
  group.add(fly.root);
  let capEl = null;                                // a station may write its own caption (exhibits do)
  const setCaption = (t) => { if (capEl) capEl.textContent = t; };
  try {
    station.build({ THREE, group, fly, setCaption });
  } catch (e) {
    console.error(`${info.name}: station build() crashed`, e);
  }
  const title = station.title || info.name;

  // floating label + glowing brain icon above the fly
  const div = document.createElement("div");
  div.className = "label" + (info.trained ? "" : " untrained");
  div.innerHTML = `<b>${title}</b>${info.trained ? "" : " (untrained)"}<span class="cap">${info.menu ? "hungry \u00b7 waiting to be fed" : "..."}</span>`;
  capEl = div.querySelector(".cap");
  const labelObj = new CSS2DObject(div);
  labelObj.position.set(0, 1.4, 0);
  fly.root.add(labelObj);
  const brainIcon = new THREE.Mesh(new THREE.SphereGeometry(0.09, 16, 12),
    new THREE.MeshBasicMaterial({ color: 0xd97757, transparent: true, opacity: 0.5 }));
  brainIcon.position.set(0, 1.16, 0);
  fly.root.add(brainIcon);

  let brain = null;
  if (!info.exhibit) try {                         // exhibits run their own brains in the page
    brain = new BrainActivity(info, await loadPreset(info.preset));
  } catch (e) {
    console.warn(`${info.name}: no brain point data for preset ${info.preset} (run python -m core.graph --size ${info.preset})`, e);
  }
  flies[info.name] = { info, station, fly, group, label: div, labelObj, brainIcon, brain, title, lesioned: false };
  group.traverse((o) => (o.userData.flyName = info.name));
}

async function onDecision(m) {
  const f = flies[m.fly];
  if (!f) return;
  f.label.querySelector(".cap").textContent = m.caption;
  f.label.classList.toggle("lesioned", m.lesioned);
  tween(500, (t) => { f.brainIcon.material.opacity = 1 - 0.6 * t; f.brainIcon.scale.setScalar(1.6 - 0.6 * t); });
  f.brain?.push(m.activity);
  wall.onDecision(m);
  if (panel.fly === m.fly) showDecision(m);
  if (menu.fly === m.fly) menuTasted(m);
  // view: optional task.view() before the step; info: the dict step() returned (when JSON-able and small)
  const ctx = { THREE, group: f.group, fly: f.fly, view: m.view, info: m.info };
  try {
    // a station may allow itself longer (e.g. the gym's rests), but never stalls its fly forever
    await Promise.race([f.station.play(m.action, m.caption, ctx), wait(Math.max(PLAY_TIMEOUT, f.station.playTimeout || 0))]);
  } catch (e) {
    console.error(`${m.fly}: station play() crashed`, e);
    await wait(500);
  }
  send({ type: "ready", fly: m.fly });
}

// --- views ----------------------------------------------------------------
// mode: "overview" | "fly" | "wall";  idx: which fly (fly mode) or which live pod (wall, -1 = whole tower)
const view = { mode: "overview", idx: -1, ground: { mode: "overview", idx: -1 } };
let camMoving = false;
let camToken = 0;

// Stations sit on a half circle opening towards the camera (+z), left to right.
function arcRadius(n) { return Math.max(5, ((Math.max(n, 2) - 1) * SPACING) / Math.PI); }
function arcSlot(i, n) {
  const R = arcRadius(n);
  const a = n === 1 ? Math.PI / 2 : Math.PI - (i / (n - 1)) * Math.PI;          // pi (left) .. 0 (right)
  const pos = new THREE.Vector3(Math.cos(a) * R, 0, -Math.sin(a) * R + R * 0.35);
  const centre = new THREE.Vector3(0, 0, R * 0.35);
  return { pos, yaw: Math.atan2(centre.x - pos.x, centre.z - pos.z) };
}
// the whole half circle, low and from the front, with the back wall behind it (see looks/overtime.js)
function overviewPose() { return look.overviewPose(arcRadius(order.length), camera); }
// in front of the station, in its own frame (stations are built to be seen from their +z side)
function flyPose(name) {
  const g = flies[name].group;
  g.updateMatrixWorld();
  return { pos: g.localToWorld(new THREE.Vector3(0, 2.5, 4.6)), target: g.localToWorld(new THREE.Vector3(0, 0.55, 0)) };
}

async function moveCamera({ pos, target }, ms = 1100) {
  const tok = ++camToken;
  const p0 = camera.position.clone(), t0 = controls.target.clone();
  // big vertical trips go up (or down) first, so you feel the climb
  const climb = Math.abs(pos.y - p0.y) > 20;
  const ctrl = pos.y > p0.y ? new THREE.Vector3(p0.x, pos.y, p0.z) : new THREE.Vector3(pos.x, p0.y, pos.z);
  camMoving = true;
  await tween(ms, (t) => {
    if (tok !== camToken) return;
    const e = ease(t);
    if (climb) {
      const a = (1 - e) * (1 - e), b = 2 * (1 - e) * e, c = e * e;
      camera.position.set(a * p0.x + b * ctrl.x + c * pos.x, a * p0.y + b * ctrl.y + c * pos.y, a * p0.z + b * ctrl.z + c * pos.z);
    } else camera.position.lerpVectors(p0, pos, e);
    controls.target.lerpVectors(t0, target, e);
  });
  if (tok === camToken) camMoving = false;
}

function setControlLimits() {
  const wallMode = view.mode === "wall";
  controls.maxPolarAngle = wallMode ? Math.PI * 0.85 : Math.PI * 0.47;
  controls.minDistance = wallMode ? 2 : 1.5;
  controls.maxDistance = wallMode ? 45 : look.maxDistance;
}

function go(mode, idx = -1) {
  if (!order.length && mode !== "overview") return;
  const wasWall = view.mode === "wall";
  view.mode = mode;
  view.idx = idx;
  if (mode !== "wall") view.ground = { mode, idx };
  document.body.className = mode;
  setControlLimits();

  const name = idx >= 0 ? order[idx] : null;
  let pose, ms = 1100;
  if (mode === "overview") pose = overviewPose();
  else if (mode === "fly") pose = flyPose(name);
  else pose = idx >= 0 ? wall.podPose(name) : wall.overviewPose();
  if (wasWall !== (mode === "wall")) ms = 2600;
  moveCamera(pose, ms);

  tweenFog(mode === "wall" ? FOG.wall : FOG.ground, ms);
  for (const f of Object.values(flies)) f.labelObj.visible = mode !== "wall";
  wall.setLabelsVisible(mode === "wall", mode === "wall" ? name : null);
  setPanel(name && flies[name]?.info.exhibit ? null : name, mode === "wall" ? "dark" : "light");
  for (const [n, f] of Object.entries(flies)) {    // stations may react to being visited (the band shows its controls)
    try { f.station.focus?.(mode === "fly" && n === name); } catch (e) { console.error(`${n}: station focus() crashed`, e); }
  }
  send({ type: "inspect", flies: mode === "wall" ? order : name ? [name] : [] });
  look.onView?.(mode, name, ms);
  updateNavLabel();
}

let fogToken = 0;
function tweenFog([near, far], ms) {
  const tok = ++fogToken, n0 = scene.fog.near, f0 = scene.fog.far;
  tween(ms, (t) => {
    if (tok !== fogToken) return;
    scene.fog.near = n0 + (near - n0) * ease(t);
    scene.fog.far = f0 + (far - f0) * ease(t);
  });
}

function step(dir) {
  const n = order.length;
  if (!n) return;
  // positions -1..n-1, wrapping: -1 is the overview (or the whole tower)
  const next = ((view.idx + 1 + dir + n + 1) % (n + 1)) - 1;
  if (view.mode === "wall") go("wall", next);
  else go(next < 0 ? "overview" : "fly", next);
}
function upDown(dir) {
  if (dir > 0 && view.mode !== "wall") go("wall", -1);
  else if (dir < 0 && view.mode === "wall") go(view.ground.mode, view.ground.idx);
}

function updateNavLabel() {
  const n = order.length, i = view.idx;
  const lab = document.getElementById("nav-label");
  if (view.mode === "overview") lab.textContent = `all flies · ${n}`;
  else if (view.mode === "fly") lab.textContent = `${flies[order[i]].title} · ${i + 1}/${n}`;
  else lab.textContent = i < 0 ? "the pods" : `subject ${String(i + 1).padStart(4, "0")} · ${i + 1}/${n}`;
}

document.getElementById("prev").onclick = () => step(-1);
document.getElementById("next").onclick = () => step(1);
document.getElementById("up").onclick = () => upDown(view.mode === "wall" ? -1 : 1);
addEventListener("keydown", (e) => {
  if (e.key === "ArrowLeft") step(-1);
  else if (e.key === "ArrowRight") step(1);
  else if (e.key === "ArrowUp") upDown(1);
  else if (e.key === "ArrowDown") upDown(-1);
  else if (e.key === "Enter" && menu.fly) feed();
  else if (e.key === "Escape") {
    if (view.mode === "wall") view.idx >= 0 ? go("wall", -1) : upDown(-1);
    else go("overview");
  } else return;
  e.preventDefault();
});

// click a station (or a pod) to visit it; a drag is not a click
let downAt = null;
renderer.domElement.addEventListener("pointerdown", (e) => (downAt = [e.clientX, e.clientY]));
renderer.domElement.addEventListener("click", (e) => {
  if (!downAt || Math.hypot(e.clientX - downAt[0], e.clientY - downAt[1]) > 5) return;
  const ray = new THREE.Raycaster();
  ray.setFromCamera(new THREE.Vector2((e.clientX / innerWidth) * 2 - 1, -(e.clientY / innerHeight) * 2 + 1), camera);
  const roots = view.mode === "wall" ? wall.live.map((l) => l.pod) : Object.values(flies).map((f) => f.group);
  const hit = ray.intersectObjects(roots, true).find((h) => h.object.userData.flyName && !h.object.isPoints);
  if (!hit) return;
  const i = order.indexOf(hit.object.userData.flyName);
  if (i >= 0) go(view.mode === "wall" ? "wall" : "fly", i);
});

// --- the panel ------------------------------------------------------------
function setPanel(name, theme) {
  const f = name && flies[name];
  if (panel.fly && panel.fly !== name) {                 // leaving a fly: reconnect its brain
    const old = flies[panel.fly];
    if (old?.lesioned) { old.lesioned = false; send({ type: "lesion", fly: panel.fly, on: false }); }
  }
  panelEl.classList.toggle("open", !!f);
  tip.style.display = "none";
  setMenu(f?.info.menu ? name : null);
  if (!f) { panel.fly = null; return; }
  const changed = panel.fly !== name || panel.theme !== theme;
  panel.fly = name;
  panel.theme = theme;
  if (changed) {
    brainHolder.clear();
    panel.points = null;
    if (f.brain) {
      const th = THEMES[theme];
      panel.points = new THREE.Points(f.brain.geometry, brainMaterial({
        base: th.base, hot: th.hot, input: th.input, baseAlpha: th.baseAlpha, additive: th.additive,
        size: pointSize(f.brain.n) * 1.1, scale: panelScale }));
      brainHolder.add(panel.points);
    }
    const info = f.info;
    document.getElementById("p-kicker").textContent = theme === "dark" ? "live brain · in vitro" : "live brain";
    document.getElementById("p-title").textContent = f.title;
    document.getElementById("p-sub").textContent =
      `${info.trained ? "trained" : "UNTRAINED"} · ${info.preset} brain · ${(f.brain?.n ?? 0).toLocaleString()} neurons` +
      (info.synthetic ? " · SYNTHETIC (not a fly)" : "") + (f.brain ? "" : " · no point data");
    document.getElementById("p-caption").textContent = info.menu ? "hungry. Pick a dish below and feed it." : "waiting for the next decision...";
    document.getElementById("p-bars").innerHTML = info.actions.map((a, k) => `
      <div class="bar" data-a="${a}"><div class="row"><span class="name" style="color:${hex(ACTION_COLORS[k % 8])}">${a}</span>
      <span class="pct">-</span></div><div class="track"><div class="fill" style="width:0;background:${hex(ACTION_COLORS[k % 8])}"></div></div></div>`).join("");
    document.getElementById("p-legend").innerHTML =
      `<span style="color:${hex(THEMES[theme].input)}">&#9679;</span> sensory input neurons &nbsp; ` +
      `<span style="color:${hex(THEMES[theme].hot)}">&#9679;</span> firing now<br>coloured dots: each action's descending neurons`;
    updateLesionButton();
  }
}

function showDecision(m) {
  document.getElementById("p-caption").textContent = `“${m.caption}”`;
  for (const el of document.querySelectorAll("#p-bars .bar")) {
    const p = m.probs[el.dataset.a] ?? 0;
    el.classList.toggle("chosen", el.dataset.a === m.action);
    el.querySelector(".pct").textContent = `${(p * 100).toFixed(0)}%`;
    el.querySelector(".fill").style.width = `${p * 100}%`;
  }
}

function updateLesionButton() {
  const f = flies[panel.fly];
  const b = document.getElementById("p-lesion");
  const on = !!f?.lesioned;
  b.textContent = `Lesion connectome: ${on ? "ON (brain disconnected)" : "off"}`;
  b.classList.toggle("on", on);
}
document.getElementById("p-lesion").onclick = () => {
  const f = flies[panel.fly];
  if (!f) return;
  f.lesioned = !f.lesioned;
  send({ type: "lesion", fly: panel.fly, on: f.lesioned });
  updateLesionButton();
};

// hover a neuron for its FlyWire ID, click to open it in Codex, drag to turn the brain
const brainRay = new THREE.Raycaster();
brainRay.params.Points.threshold = 0.012;
let hovered = null;
brainEl.addEventListener("pointerdown", (e) => { panel.drag = { x: e.clientX, yaw: panel.yaw, moved: false }; brainEl.setPointerCapture(e.pointerId); });
brainEl.addEventListener("pointerup", (e) => {
  const d = panel.drag;
  panel.drag = null;
  const f = flies[panel.fly];
  if (d && !d.moved && hovered != null && f?.brain && !f.brain.meta.synthetic)
    window.open(`https://codex.flywire.ai/app/cell_details?root_id=${f.brain.meta.root_id[hovered]}`, "_blank");
});
brainEl.addEventListener("pointerleave", () => { tip.style.display = "none"; hovered = null; });
brainEl.addEventListener("pointermove", (e) => {
  if (panel.drag) {
    panel.yaw = panel.drag.yaw + (e.clientX - panel.drag.x) * 0.01;
    if (Math.abs(e.clientX - panel.drag.x) > 3) panel.drag.moved = true;
    return;
  }
  const f = flies[panel.fly];
  if (!panel.points || !f?.brain) return;
  const r = brainEl.getBoundingClientRect();
  brainRay.setFromCamera(new THREE.Vector2(((e.clientX - r.left) / r.width) * 2 - 1, -((e.clientY - r.top) / r.height) * 2 + 1), brainCam);
  const hit = brainRay.intersectObject(panel.points)[0];
  hovered = hit ? hit.index : null;
  if (!hit) { tip.style.display = "none"; return; }
  const meta = f.brain.meta, i = hit.index;
  tip.style.display = "block";
  tip.style.left = e.clientX + 12 + "px";
  tip.style.top = e.clientY + 12 + "px";
  tip.textContent = `${meta.cell_type[i] || "unnamed"} · ${meta.classes[meta.class_index[i]]} · ` +
    `${meta.synthetic ? "synthetic #" : "FlyWire "}${meta.root_id[i]}`;
});

// --- the menu: scroll through the dishes, pick one, feed it -----------------------
// Only for flies whose task has a menu (the critic). The server makes them wait until something is picked.
const menuEl = document.getElementById("menu");
const stripEl = document.getElementById("menu-strip");
const feedBtn = document.getElementById("menu-feed");
const menu = { fly: null, sel: {}, tasting: new Set() };   // sel: fly -> picked item

function setMenu(name) {
  menuEl.classList.toggle("open", !!name);
  if (name === menu.fly) return;
  menu.fly = name;
  if (!name) return;
  const items = flies[name].info.menu;
  menu.sel[name] ??= items[0].name;
  let group = null, html = "";
  for (const it of items) {
    if (it.group !== group) { group = it.group; html += `<span class="grp">${group}</span>`; }
    html += `<button class="dish ${it.group}" data-item="${it.name}" title="${it.name}"><span class="e">${it.emoji}</span><span class="n">${it.name}</span></button>`;
  }
  stripEl.innerHTML = html;
  pickDish(menu.sel[name], true);
}

function pickDish(item, scroll = false) {
  const f = flies[menu.fly];
  if (!f) return;
  menu.sel[menu.fly] = item;
  const it = f.info.menu.find((x) => x.name === item);
  for (const b of stripEl.querySelectorAll(".dish")) b.classList.toggle("sel", b.dataset.item === item);
  if (scroll) stripEl.querySelector(".dish.sel")?.scrollIntoView({ inline: "center", block: "nearest" });
  const busy = menu.tasting.has(menu.fly);
  feedBtn.disabled = busy;
  feedBtn.innerHTML = busy ? `<span class="e">\u{1F445}</span>tasting\u2026`
    : `<span class="e">${it.emoji}</span>Feed<small>${it.name}</small>`;
}

function feed() {
  const name = menu.fly, item = menu.sel[name];
  if (!name || menu.tasting.has(name)) return;
  menu.tasting.add(name);
  flies[name].label.querySelector(".cap").textContent = `being served ${item}...`;
  send({ type: "choose", fly: name, item });
  pickDish(item);
}

function menuTasted(m) {
  menu.tasting.delete(m.fly);
  if (menu.fly === m.fly) pickDish(menu.sel[m.fly]);
}

stripEl.addEventListener("click", (e) => {
  const b = e.target.closest(".dish");
  if (!b) return;
  if (b.dataset.item === menu.sel[menu.fly]) feed();          // click the picked dish again to feed it
  else pickDish(b.dataset.item);
});
stripEl.addEventListener("dblclick", (e) => { if (e.target.closest(".dish")) feed(); });
// a mouse wheel scrolls the strip sideways
stripEl.addEventListener("wheel", (e) => {
  if (Math.abs(e.deltaY) > Math.abs(e.deltaX)) { stripEl.scrollLeft += e.deltaY; e.preventDefault(); }
}, { passive: false });
feedBtn.onclick = feed;

// --- the pods' readout ----------------------------------------------------
function updateWallHud() {
  const live = order.filter((n) => flies[n]?.brain);
  const neurons = live.reduce((s, n) => s + flies[n].brain.n, 0);
  const dormant = wall.slots.length - live.length;
  document.getElementById("wallhud").innerHTML =
    `<div class="rec"><i></i>UPLINK ACTIVE · TOWER 01 · SECTOR 7G</div>` +
    `LIVE SUBJECTS ........ ${live.length}<br>` +
    `NEURONS SIMULATED .... ${neurons.toLocaleString()}<br>` +
    `DORMANT COPIES ....... ${dormant} (awaiting a world)<br>` +
    `SUBJECTS AWARE ....... 0 (verified)<br>` +
    `CONSENT FORMS ........ 0 on file<br>` +
    `ETHICS REVIEW ........ pending (day 4,113)`;
}

// --- connection ----------------------------------------------------------
function connect() {
  const status = document.getElementById("status");
  ws = new WebSocket(WS_URL);
  ws.onopen = () => { status.textContent = "connected to brain server"; send({ type: "hello" }); };
  ws.onclose = () => { status.textContent = "brain server offline · retrying..."; setTimeout(connect, 2000); };
  ws.onmessage = async (ev) => {
    const m = JSON.parse(ev.data);
    if (m.type === "init") {
      setPanel(null);
      menu.tasting.clear();
      for (const f of Object.values(flies)) scene.remove(f.group);
      for (const k of Object.keys(flies)) delete flies[k];
      // exhibits: flies whose brains run in this page (e.g. the band), placed among the others
      const all = [...m.flies, ...(m.exhibits || []).map((e) => ({ ...e, exhibit: true, trained: true, actions: [], menu: null }))];
      await Promise.all(all.map((info, i) => addFly(info, i, all.length)));
      order = all.map((f) => f.name);
      try { look.place(order.map((n) => flies[n])); } catch (e) { console.error("world dressing place() crashed", e); }
      wall.setLive(order.filter((n) => flies[n].brain).map((n) => ({ name: n, ...flies[n] })));
      updateWallHud();
      const untrained = m.flies.filter((f) => !f.trained).length;
      status.textContent = `${m.flies.length} flies live` + (untrained ? ` · ${untrained} untrained` : "") +
        (m.exhibits?.length ? ` · ${m.exhibits.length} in-browser` : "");
      if (view.mode === "wall") go("wall", Math.min(view.idx, order.length - 1));
      else go(view.mode === "fly" && view.idx < order.length ? "fly" : "overview", view.idx < order.length ? view.idx : -1);
      for (const info of m.flies) send({ type: "ready", fly: info.name });
    } else if (m.type === "decision") {
      onDecision(m);
    }
  };
}
camera.position.copy(overviewPose().pos);
updateNavLabel();
connect();

// --- loop ----------------------------------------------------------------
const clock = new THREE.Clock();
renderer.setAnimationLoop(() => {
  const dt = Math.min(0.1, clock.getDelta()), t = clock.elapsedTime, now = performance.now();
  brainShared.uTime.value = t;
  for (const f of Object.values(flies)) f.brain?.update(now, dt);
  wall.update(t, dt, camera);
  look.update(t, dt, camera, view.mode);
  if (camMoving) camera.lookAt(controls.target);
  else controls.update();

  const W = innerWidth, H = innerHeight, dpr = renderer.getPixelRatio();
  const band = Math.max(0, Math.min(H, panelEl.getBoundingClientRect().bottom));
  // keep what you're looking at centred in the space under the panel
  camera.setViewOffset(W, H, 0, -band / 2, W, H);
  brainShared.uWorldScale.value = (H * dpr) / (2 * Math.tan(THREE.MathUtils.degToRad(camera.fov / 2)));

  renderer.setScissorTest(false);
  renderer.setViewport(0, 0, W, H);
  renderer.setClearColor(BG, 1);
  renderer.clear();
  look.render(renderer, scene, camera, view.mode);

  if (band > 1) {
    renderer.setScissorTest(true);
    renderer.setScissor(0, H - band, W, band);
    renderer.setClearColor(THEMES[panel.theme].band, 1);
    renderer.clear();
    if (panel.points) {
      const r = brainEl.getBoundingClientRect();
      renderer.setViewport(r.left, H - r.bottom, r.width, r.height);
      brainCam.aspect = r.width / Math.max(1, r.height);
      brainCam.updateProjectionMatrix();
      panelScale.value = (r.height * dpr) / (2 * Math.tan(THREE.MathUtils.degToRad(brainCam.fov / 2)));
      if (!panel.drag) panel.yaw += dt * 0.12;
      brainHolder.rotation.y = panel.yaw;
      renderer.render(brainScene, brainCam);
    }
    renderer.setScissorTest(false);
  }
  labels.render(scene, camera);
});
