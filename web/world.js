// The world: every fly lives in its own photoreal scene, a Gaussian splat made from one film still
// (web/scenes/<scene>/). One scene is loaded and drawn at a time. Hold Tab for the wheel, or use the arrow keys;
// moving to a fly in another scene plays the GTA V switch (lib/switch.js). Talks to server/serve.py.
//   ?edit   free camera; place flies and frames in a scene (see the keys in editKey below)
//   ?fly=<name>   start at that fly
import * as THREE from "three";
import { OrbitControls } from "three/addons/controls/OrbitControls.js";
import { GLTFLoader } from "three/addons/loaders/GLTFLoader.js";
import { SparkRenderer, SplatMesh } from "@sparkjsdev/spark";
import { box, wait } from "./lib/fly.js";
import { loadFlyAsset, makeFly } from "./lib/realfly.js";
import { Switcher } from "./lib/switch.js";
import { Wheel } from "./lib/wheel.js";

const WS_URL = `${location.protocol === "https:" ? "wss" : "ws"}://${location.host}/ws`;
const PLAY_TIMEOUT = 4000;
const FLY_SCALE = 1.2;           // station units -> scene units; Marble rooms come out at roughly 2.5 m a unit, so about a person
const PARALLAX = 0.035;          // how far the camera drifts with the mouse, as a fraction of its distance to the target
const params = new URLSearchParams(location.search);
const EDIT = params.has("edit");

// --- renderer ---------------------------------------------------------------
const el = document.getElementById("world");
const renderer = new THREE.WebGLRenderer({ antialias: false });   // MSAA is costly with splats and does nothing for them
renderer.setPixelRatio(Math.min(devicePixelRatio, 1.5));
renderer.toneMapping = THREE.NeutralToneMapping;
el.appendChild(renderer.domElement);
const scene = new THREE.Scene();
scene.background = new THREE.Color(0x050506);
const spark = new SparkRenderer({ renderer });
scene.add(spark);
const camera = new THREE.PerspectiveCamera(50, 1, 0.05, 500);
const hemi = new THREE.HemisphereLight(0xfff4e8, 0x3a3029, 0.6);
const key = new THREE.DirectionalLight(0xffe8d0, 1.6);
scene.add(hemi, key, key.target);
const controls = EDIT ? new OrbitControls(camera, renderer.domElement) : null;

function resize() {
  renderer.setSize(innerWidth, innerHeight);
  camera.aspect = innerWidth / innerHeight;
  camera.updateProjectionMatrix();
}
addEventListener("resize", resize);
resize();

// --- scenes -------------------------------------------------------------------
// web/scenes/index.json lists the scenes; each web/scenes/<name>/scene.json says where its camera and flies are:
//   {"camera": {"position": [x,y,z], "target": [x,y,z], "fov": 50},
//    "splat": {"file": "scene.spz", "quaternion": [1,0,0,0], "position": [0,0,0], "scale": 1},
//    "image": "clean.jpg", "aerial": "aerial.jpg", "light": {"direction": [x,y,z], "intensity": 1.6},
//    "flies": {"<fly>": {"position": [x,y,z], "yaw": 0, "scale": 2.8, "camera": {...optional}}}}
// position is where the fly stands. A fly that isn't in any scene gets a dark stage of its own.
const sets = {};                 // scene name -> {name, dir, cfg, root, splat, backdrop, ready}
const flies = {};                // fly name -> {info, station, fly, group, set, title, caption}
let order = [];
let current = null;              // name of the fly on screen
let busy = false;                // a switch is playing
let ws;
const send = (m) => ws?.readyState === 1 && ws.send(JSON.stringify(m));

async function loadSceneList() {
  const list = await fetch("/web/scenes/index.json").then((r) => (r.ok ? r.json() : { scenes: [] })).catch(() => ({ scenes: [] }));
  await Promise.all(list.scenes.map(async (name) => {
    const dir = `/web/scenes/${name}/`;
    const cfg = await fetch(dir + "scene.json").then((r) => (r.ok ? r.json() : null)).catch(() => null);
    if (cfg) sets[name] = makeSet(name, dir, cfg);
  }));
}

function makeSet(name, dir, cfg) {
  const root = new THREE.Group();
  root.name = name;
  return { name, dir, cfg, root, splat: null, backdrop: null, ready: null, env: null };
}

function stageFor(name) {                // a fly with no scene: a dark stage with a soft light from above
  return makeSet(`stage:${name}`, null, {
    stage: true, light: { direction: [0.3, 1, 0.5], intensity: 2.2 }, flies: { [name]: { position: [0, 0, 0], yaw: 0.5 } },
  });
}

function addFloor(set) {                 // for a stage, or a scene whose still isn't there yet
  const ys = Object.values(set.cfg.flies ?? {}).map((s) => s.position?.[1] ?? 0);
  const floor = new THREE.Mesh(new THREE.CircleGeometry(40, 64), new THREE.MeshStandardMaterial({ color: 0x0c0c0d, roughness: 0.95 }));
  floor.rotation.x = -Math.PI / 2;
  floor.position.y = Math.min(0, ...ys);
  set.root.add(floor);
  set.stage = true;
}

// Load a scene's splat (or, until there is one, its still as a flat backdrop). Resolves when it can be drawn.
function load(set) {
  set.ready ||= (async () => {
    const c = set.cfg;
    if (set.stage) return;
    if (c.stage) return addFloor(set);
    const file = c.splat?.file ?? "scene.spz";
    if (await exists(set.dir + file)) {
      const splat = new SplatMesh({ url: set.dir + file });
      await splat.initialized;
      splat.quaternion.fromArray(c.splat?.quaternion ?? [1, 0, 0, 0]);   // OpenCV axes (y down) -> three.js
      splat.position.fromArray(c.splat?.position ?? [0, 0, 0]);
      splat.scale.setScalar(c.splat?.scale ?? 1);
      set.splat = splat;
      set.root.add(splat);
      if (EDIT && (await exists(set.dir + "collider.glb"))) {     // Marble's collision mesh: something solid to click on
        const g = (await new GLTFLoader().loadAsync(set.dir + "collider.glb")).scene;
        g.traverse((o) => { if (o.isMesh) o.material = new THREE.MeshBasicMaterial({ color: 0x33ffcc, wireframe: true, transparent: true, opacity: 0.25 }); });
        g.quaternion.copy(splat.quaternion);                        // same axes as the splat (OpenCV, y down)
        g.position.copy(splat.position);
        g.scale.copy(splat.scale);
        g.visible = false;
        set.collider = g;
        set.root.add(g);
      }
      return;
    }
    for (const img of [c.image, "clean.jpg", "still.jpg"].filter(Boolean)) {
      if (!(await exists(set.dir + img))) continue;
      const tex = await new THREE.TextureLoader().loadAsync(set.dir + img);
      tex.colorSpace = THREE.SRGBColorSpace;
      set.backdrop = new THREE.Mesh(new THREE.PlaneGeometry(1, 1), new THREE.MeshBasicMaterial({ map: tex, toneMapped: false, depthWrite: false }));
      set.backdrop.userData.aspect = tex.image.width / tex.image.height;
      set.backdrop.renderOrder = -1;
      set.root.add(set.backdrop);
      return;
    }
    addFloor(set);
  })();
  return set.ready;
}

function unload(set) {
  if (set.splat) { set.root.remove(set.splat); set.splat.dispose(); set.splat = null; }
  if (set.backdrop) { set.root.remove(set.backdrop); set.backdrop.material.map.dispose(); set.backdrop = null; }
  if (set.env) { set.env.dispose(); set.env = null; }
  set.ready = null;
}

const existing = new Map();
function exists(url) {
  if (!existing.has(url)) existing.set(url, fetch(url, { method: "HEAD" }).then((r) => r.ok).catch(() => false));
  return existing.get(url);
}

// Put a fly's station group in its scene so the fly stands at slot.position. Station props stay invisible
// (layer 1): the scene itself has the real furniture.
function place(f) {
  const slot = f.set.cfg.flies[f.info.name] ?? {};
  const s = slot.scale ?? FLY_SCALE;
  f.group.scale.setScalar(s);
  f.group.rotation.y = slot.yaw ?? 0;
  f.group.updateMatrixWorld();
  const offset = f.fly.root.position.clone().multiplyScalar(s).applyAxisAngle(new THREE.Vector3(0, 1, 0), f.group.rotation.y);
  f.group.position.fromArray(slot.position ?? [0, 0, -3]).sub(offset);
}

function hideProps(f) {
  if (f.set.cfg.props) return;
  f.group.traverse((o) => { if (o.isMesh && !isFlyPart(o, f.fly)) o.layers.set(1); });
}
const isFlyPart = (o, fly) => { for (let p = o; p; p = p.parent) if (p === fly.root) return true; return false; };

// A soft dark patch under each fly, so it sits on the floor of the photo.
const blobTex = (() => {
  const c = document.createElement("canvas");
  c.width = c.height = 128;
  const g = c.getContext("2d"), r = g.createRadialGradient(64, 64, 0, 64, 64, 64);
  r.addColorStop(0, "rgba(0,0,0,0.75)");
  r.addColorStop(0.5, "rgba(0,0,0,0.35)");
  r.addColorStop(1, "rgba(0,0,0,0)");
  g.fillStyle = r;
  g.fillRect(0, 0, 128, 128);
  return new THREE.CanvasTexture(c);
})();
function contactShadow(fly) {
  const m = new THREE.Mesh(new THREE.PlaneGeometry(0.62, 0.8), new THREE.MeshBasicMaterial({ map: blobTex, transparent: true, depthWrite: false }));
  m.rotation.x = -Math.PI / 2;
  m.position.y = 0.004;
  m.renderOrder = 3;
  fly.root.add(m);
}

// --- flies --------------------------------------------------------------------
const fallbackStation = { title: "", async play(action, caption, { fly }) { await fly.hop(0.2, 400); await wait(300); } };

async function addFly(info) {
  let station;
  try {
    station = (await import(`/flies/${info.name}/station.js`)).default;
  } catch (e) {
    console.warn(`flies/${info.name}/station.js failed to load`, e);
    station = Object.create(fallbackStation);
  }
  const fly = makeFly();
  const group = new THREE.Group();
  group.add(fly.root);
  try { station.build?.({ THREE, group, fly }); } catch (e) { console.error(`${info.name}: station build() crashed`, e); }
  contactShadow(fly);
  const set = Object.values(sets).find((s) => s.cfg.flies?.[info.name]) ?? (sets[`stage:${info.name}`] = stageFor(info.name));
  const f = { info, station, fly, group, set, title: station.title || info.name, caption: "" };
  flies[info.name] = f;
  set.root.add(group);
  place(f);
  hideProps(f);
}

async function onDecision(m) {
  const f = flies[m.fly];
  if (!f) return;
  f.caption = m.caption;
  if (m.fly === current) subtitle(m.caption);
  if (menu.fly === m.fly) menu.done();
  try {
    await Promise.race([f.station.play(m.action, m.caption, { THREE, group: f.group, fly: f.fly }),
      wait(Math.max(PLAY_TIMEOUT, f.station.playTimeout || 0))]);
  } catch (e) {
    console.error(`${m.fly}: station play() crashed`, e);
    await wait(500);
  }
  send({ type: "ready", fly: m.fly });
}

// --- camera -------------------------------------------------------------------
const view = { pos: new THREE.Vector3(), target: new THREE.Vector3(), fov: 50, mouse: new THREE.Vector2(), smooth: new THREE.Vector2() };

function frame(f) {
  if (f.set.stage) {                     // no photo: frame the fly from the front, three quarters on
    const L = 0.6 * f.group.scale.x, p = f.fly.root.getWorldPosition(new THREE.Vector3());
    const fwd = new THREE.Vector3(0, 0, 1).applyQuaternion(f.fly.root.getWorldQuaternion(new THREE.Quaternion()));
    const dir = fwd.setY(0).normalize().applyAxisAngle(new THREE.Vector3(0, 1, 0), 0.6);
    const target = p.clone().add(new THREE.Vector3(0, 0.35 * L, 0));
    return { pos: target.clone().addScaledVector(dir, 2.6 * L).add(new THREE.Vector3(0, 0.8 * L, 0)), target, fov: 38 };
  }
  const c = f.set.cfg.flies[f.info.name]?.camera ?? f.set.cfg.camera ?? {};
  return { pos: new THREE.Vector3().fromArray(c.position ?? [0, 0, 0]), target: new THREE.Vector3().fromArray(c.target ?? [0, 0, -1]), fov: c.fov ?? 50 };
}

function setView(v) {
  view.pos.copy(v.pos);
  view.target.copy(v.target);
  view.fov = v.fov;
  if (controls) { camera.position.copy(v.pos); controls.target.copy(v.target); }
}

function fitBackdrop(set) {          // the still fills the frame from the scene's camera, far behind everything
  const b = set.backdrop;
  if (!b) return;
  const c = set.cfg.camera ?? {}, pos = new THREE.Vector3().fromArray(c.position ?? [0, 0, 0]);
  const dir = new THREE.Vector3().fromArray(c.target ?? [0, 0, -1]).sub(pos).normalize();
  const d = 60, h = 2 * d * Math.tan(THREE.MathUtils.degToRad((c.fov ?? 50) / 2)) * 1.08;
  const a = b.userData.aspect, w = Math.max(h * a, h * camera.aspect * 1.08);
  b.scale.set(w, w / a, 1);
  b.position.copy(pos).addScaledVector(dir, d);
  b.lookAt(pos);
}

addEventListener("pointermove", (e) => view.mouse.set((e.clientX / innerWidth) * 2 - 1, (e.clientY / innerHeight) * 2 - 1));

function updateCamera(dt) {
  if (controls) { controls.update(); return; }
  const f = flies[current];
  if (f?.set.stage && !busy) {          // no photo to hold still for: follow the fly around its stage
    const k = 1 - Math.exp(-dt * 1.2), aim = f.fly.root.getWorldPosition(new THREE.Vector3()).add(new THREE.Vector3(0, 0.21 * f.group.scale.x, 0));
    const offset = view.pos.clone().sub(view.target);
    view.target.lerp(aim, k);
    view.pos.copy(view.target).add(offset);
  }
  view.smooth.lerp(view.mouse, 1 - Math.exp(-dt * 2.5));
  const t = performance.now() / 1000, dist = view.pos.distanceTo(view.target), k = PARALLAX * dist;
  const fwd = view.target.clone().sub(view.pos).normalize();
  const right = new THREE.Vector3().crossVectors(fwd, camera.up).normalize(), up = new THREE.Vector3().crossVectors(right, fwd);
  camera.position.copy(view.pos)
    .addScaledVector(right, view.smooth.x * k + Math.sin(t * 0.31) * k * 0.08)        // plus a slow handheld breath
    .addScaledVector(up, -view.smooth.y * k * 0.6 + Math.sin(t * 0.23 + 1) * k * 0.06);
  camera.lookAt(view.target);
  if (camera.fov !== view.fov) { camera.fov = view.fov; camera.updateProjectionMatrix(); }
}

// --- going to a fly -------------------------------------------------------------
const switcher = new Switcher(renderer.domElement);
let shown = null;                    // the scene being drawn

function show(set) {
  if (shown === set) return;
  if (shown) scene.remove(shown.root);
  shown = set;
  scene.add(set.root);
  const L = set.cfg.light ?? {};
  key.position.fromArray(L.direction ?? [0.4, 1, 0.3]).normalize().multiplyScalar(10);
  key.intensity = L.intensity ?? 1.6;
  hemi.intensity = L.ambient ?? 0.6;
  scene.background.set(set.stage ? 0x050506 : 0x000000);
  scene.environment = set.env;
  fitBackdrop(set);
  if (set.splat && !set.env) {   // light the fly with the room: an environment map rendered from the splat
    const f = Object.values(flies).find((x) => x.set === set);
    const hide = Object.values(flies).filter((x) => x.set === set).map((x) => x.group);
    spark.renderEnvMap({ scene, worldCenter: f ? f.fly.root.getWorldPosition(new THREE.Vector3()) : view.target, hideObjects: hide })
      .then((env) => { set.env = env; if (shown === set) scene.environment = env; }).catch((e) => console.warn("env map", e));
  }
}

async function go(name, instant = false) {
  const f = flies[name];
  if (!f || busy || name === current) return;
  busy = true;
  const from = current && flies[current];
  current = name;
  menu.set(f.info.menu ? name : null);
  nameCard(null);
  try {
    if (instant || !from) {
      await load(f.set);
      show(f.set);
      setView(frame(f));
      document.getElementById("cover").classList.add("gone");
    } else if (from.set === f.set) {                                   // same room: just move the camera
      const a = { pos: view.pos.clone(), target: view.target.clone(), fov: view.fov }, b = frame(f);
      await anim(900, (e) => setView({ pos: a.pos.clone().lerp(b.pos, e), target: a.target.clone().lerp(b.target, e), fov: a.fov + (b.fov - a.fov) * e }));
    } else {
      const aerial = (s) => (s.dir ? s.dir + (s.cfg.aerial ?? "aerial.jpg") : null);
      await switcher.out(from.set.name, aerial(from.set));
      scene.remove(from.set.root);
      shown = null;
      for (const s of Object.values(sets)) if (s !== f.set && s.ready) unload(s);   // one splat in memory at a time
      await switcher.across(f.set.name, load(f.set));
      const b = frame(f), lift = b.pos.distanceTo(b.target) * 0.18;
      await switcher.in(f.set.name, aerial(f.set), () => {
        show(f.set);
        setView({ ...b, pos: b.pos.clone().add(new THREE.Vector3(0, lift, 0)) });
      });
      const top = view.pos.clone();
      anim(600, (e) => setView({ ...b, pos: top.clone().lerp(b.pos, e) }));
    }
  } finally {
    busy = false;
  }
  nameCard(f);
  subtitle(f.caption);
  wheel.current = order.indexOf(name);
  history.replaceState(null, "", `?${EDIT ? "edit&" : ""}fly=${name}`);
}

function anim(ms, fn) {
  return new Promise((res) => {
    const t0 = performance.now();
    const step = (now) => {
      const t = Math.min(1, (now - t0) / ms), e = t < 0.5 ? 4 * t * t * t : 1 - Math.pow(-2 * t + 2, 3) / 2;
      fn(e);
      t < 1 ? requestAnimationFrame(step) : res();
    };
    requestAnimationFrame(step);
  });
}

// --- overlay: the fly's name on arrival, its latest line, the wheel -----------
const nameEl = document.getElementById("name");
let nameTimer;
function nameCard(f) {
  clearTimeout(nameTimer);
  nameEl.classList.remove("on");
  if (!f) return;
  nameEl.innerHTML = `<b></b>${f.info.trained ? "" : "<i>untrained</i>"}`;
  nameEl.querySelector("b").textContent = f.title;
  requestAnimationFrame(() => nameEl.classList.add("on"));
  nameTimer = setTimeout(() => nameEl.classList.remove("on"), 3200);
}

const subEl = document.getElementById("sub");
let subTimer;
function subtitle(text) {
  clearTimeout(subTimer);
  subEl.textContent = text || "";
  subEl.classList.toggle("on", !!text);
  subTimer = setTimeout(() => subEl.classList.remove("on"), 3500);
}

let timeScale = 1;
const wheel = new Wheel({
  onPick: (i) => go(order[i]),
  onOpen: (open) => {
    timeScale = open ? 0.25 : 1;
    renderer.domElement.classList.toggle("dim", open);
    if (open) order.forEach((n, i) => wheel.portrait(i, portraitOf(n)));
  },
});

// Portraits: a crop of the fly from its own scene, taken whenever we're looking at it.
const portraits = {};
const shot = document.createElement("canvas");
shot.width = shot.height = 256;
function portraitOf(name) {
  const f = flies[name];
  return portraits[name] ?? (f.set.dir ? f.set.dir + (f.set.cfg.image ?? "still.jpg") : "");
}
function takePortrait() {
  const f = flies[current];
  if (!f || switcher.covering || busy) return;
  const p = f.fly.root.getWorldPosition(new THREE.Vector3()).add(new THREE.Vector3(0, 0.25 * (f.group.scale.x || 1), 0)).project(camera);
  if (Math.abs(p.x) > 1 || Math.abs(p.y) > 1) return;
  const c = renderer.domElement, x = ((p.x + 1) / 2) * c.width, y = ((1 - p.y) / 2) * c.height, s = c.height * 0.42;
  const g = shot.getContext("2d");
  g.drawImage(c, x - s / 2, y - s / 2, s, s, 0, 0, 256, 256);
  portraits[current] = shot.toDataURL("image/jpeg", 0.85);
}
let lastPortrait = 0;

// keys: left / right go to the previous / next fly
addEventListener("keydown", (e) => {
  if (wheel.open || EDIT && e.target !== document.body) return;
  if (e.key === "ArrowRight") go(order[(order.indexOf(current) + 1) % order.length]);
  if (e.key === "ArrowLeft") go(order[(order.indexOf(current) - 1 + order.length) % order.length]);
});

// --- the menu (flies that wait to be fed, like the critic): a row of dishes, click one to serve it ---------
const menu = {
  el: document.getElementById("menu"), fly: null, waiting: false,
  set(name) {
    this.fly = name;
    this.el.classList.toggle("on", !!name);
    if (!name) return;
    this.el.innerHTML = flies[name].info.menu.map((it) => `<button data-item="${it.name}" title="${it.name}">${it.emoji}</button>`).join("");
    this.el.classList.toggle("waiting", this.waiting);
  },
  done() { this.waiting = false; this.el.classList.remove("waiting"); },
};
menu.el.addEventListener("click", (e) => {
  const b = e.target.closest("button");
  if (!b || menu.waiting || !menu.fly) return;
  menu.waiting = true;
  menu.el.classList.add("waiting");
  send({ type: "choose", fly: menu.fly, item: b.dataset.item });
});

// --- edit mode: place the flies and the frames ----------------------------------
//   shift+click   put the fly on screen where you clicked (on the splat)
//   [ ]  turn it    - =  smaller / bigger    c  this view becomes its frame    v  this view becomes the scene's frame
//   k  show Marble's collision mesh (clicks land on it)
//   p  copy this scene's scene.json (paste it into web/scenes/<scene>/scene.json)
if (EDIT) {
  document.getElementById("edit").hidden = false;
  const ray = new THREE.Raycaster();
  renderer.domElement.addEventListener("pointerdown", (e) => {
    if (!e.shiftKey || !current) return;
    const f = flies[current];
    ray.setFromCamera(new THREE.Vector2((e.clientX / innerWidth) * 2 - 1, -(e.clientY / innerHeight) * 2 + 1), camera);
    const targets = [f.set.collider, f.set.splat, f.set.backdrop, ...f.set.root.children.filter((o) => o.isMesh)].filter(Boolean);
    const hit = ray.intersectObjects(targets, true)[0];
    if (!hit) return;
    slotOf(f).position = hit.point.toArray().map(round);
    place(f);
  });
  const slotOf = (f) => (f.set.cfg.flies[f.info.name] ??= {});
  const round = (x) => Math.round(x * 1000) / 1000;
  const camCfg = () => ({ position: camera.position.toArray().map(round), target: controls.target.toArray().map(round), fov: round(camera.fov) });
  addEventListener("keydown", async (e) => {
    const f = flies[current];
    if (!f) return;
    const slot = slotOf(f);
    if (e.key === "[" || e.key === "]") slot.yaw = round((slot.yaw ?? 0) + (e.key === "]" ? 1 : -1) * Math.PI / 12);
    else if (e.key === "-" || e.key === "=") slot.scale = round((slot.scale ?? FLY_SCALE) * (e.key === "=" ? 1.1 : 1 / 1.1));
    else if (e.key === "k" && f.set.collider) { f.set.collider.visible = !f.set.collider.visible; return; }
    else if (e.key === "c") slot.camera = camCfg();
    else if (e.key === "v") { f.set.cfg.camera = camCfg(); fitBackdrop(f.set); }
    else if (e.key === "p") {
      const json = JSON.stringify(f.set.cfg, null, 2);
      console.log(`web/scenes/${f.set.name}/scene.json\n${json}`);
      await navigator.clipboard?.writeText(json).catch(() => {});
      document.getElementById("edit").dataset.msg = `copied ${f.set.name}/scene.json`;
      return;
    } else return;
    place(f);
  });
}

// --- connection -----------------------------------------------------------------
function connect() {
  const status = document.getElementById("status");
  ws = new WebSocket(WS_URL);
  ws.onopen = () => { status.classList.remove("on"); send({ type: "hello" }); };
  ws.onclose = () => { status.classList.add("on"); setTimeout(connect, 2000); };
  ws.onmessage = async (ev) => {
    const m = JSON.parse(ev.data);
    if (m.type === "init") {
      for (const s of Object.values(sets)) for (const f of Object.values(flies)) if (f.set === s) s.root.remove(f.group);
      for (const k of Object.keys(flies)) delete flies[k];
      await Promise.all(m.flies.map(addFly));
      order = m.flies.map((f) => f.name);
      wheel.set(order.map((n) => ({ name: n, title: flies[n].title })), Math.max(0, order.indexOf(current)));
      const want = flies[params.get("fly")] ? params.get("fly") : order[0];
      const again = current;
      current = null;
      await go(flies[again] ? again : want, true);
      for (const info of m.flies) send({ type: "ready", fly: info.name });
    } else if (m.type === "decision") {
      onDecision(m);
    }
  };
}

await Promise.all([loadSceneList(), loadFlyAsset()]);
connect();

// --- loop -------------------------------------------------------------------------
const clock = new THREE.Clock();
renderer.setAnimationLoop(() => {
  const dt = Math.min(0.1, clock.getDelta());
  for (const f of Object.values(flies)) f.fly.update?.(dt * timeScale);
  if (switcher.covering) return;                                       // the map is on top: draw nothing
  updateCamera(dt);
  renderer.render(scene, camera);
  if (!wheel.open && performance.now() - lastPortrait > 4000) { lastPortrait = performance.now(); takePortrait(); }
});
