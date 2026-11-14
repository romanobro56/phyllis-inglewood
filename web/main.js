// The world: one station per fly, laid out on a grid. Talks to server/serve.py over a websocket.
import * as THREE from "three";
import { OrbitControls } from "three/addons/controls/OrbitControls.js";
import { CSS2DRenderer, CSS2DObject } from "three/addons/renderers/CSS2DRenderer.js";
import { Fly, tween, wait, box } from "./lib/fly.js";
import { Matrix } from "./matrix.js";

const WS_URL = `ws://${location.hostname || "localhost"}:8765`;
const SPACING = 6;
const PLAY_TIMEOUT = 4000; // a stuck station never stalls its fly for longer than this

// --- scene ---------------------------------------------------------------
const el = document.getElementById("world");
const renderer = new THREE.WebGLRenderer({ antialias: true });
renderer.setPixelRatio(Math.min(devicePixelRatio, 2));
renderer.shadowMap.enabled = true;
el.appendChild(renderer.domElement);
const labels = new CSS2DRenderer();
labels.domElement.style.position = "absolute";
labels.domElement.style.top = "0";
labels.domElement.style.pointerEvents = "none";
el.appendChild(labels.domElement);

const scene = new THREE.Scene();
scene.background = new THREE.Color(0x0d1117);
scene.fog = new THREE.Fog(0x0d1117, 25, 60);
const camera = new THREE.PerspectiveCamera(50, 1, 0.1, 200);
camera.position.set(0, 12, 18);
const controls = new OrbitControls(camera, renderer.domElement);
controls.enableDamping = true;
controls.maxPolarAngle = Math.PI * 0.48;

scene.add(new THREE.HemisphereLight(0xbcd7ff, 0x1a1a10, 0.9));
const sun = new THREE.DirectionalLight(0xffffff, 1.6);
sun.position.set(10, 20, 8);
sun.castShadow = true;
sun.shadow.mapSize.set(2048, 2048);
Object.assign(sun.shadow.camera, { left: -30, right: 30, top: 30, bottom: -30 });
scene.add(sun);
const ground = new THREE.Mesh(new THREE.PlaneGeometry(200, 200), new THREE.MeshStandardMaterial({ color: 0x1b222c }));
ground.rotation.x = -Math.PI / 2;
ground.receiveShadow = true;
scene.add(ground);
scene.add(new THREE.GridHelper(200, 100, 0x263040, 0x1e2632));

function resize() {
  renderer.setSize(innerWidth, innerHeight);
  labels.setSize(innerWidth, innerHeight);
  camera.aspect = innerWidth / innerHeight;
  camera.updateProjectionMatrix();
}
addEventListener("resize", resize);
resize();

// --- flies ---------------------------------------------------------------
const flies = {}; // name -> {info, station, fly, group, label, brainIcon, busy}
let ws;
const matrix = new Matrix({
  onClose: () => send({ type: "inspect", fly: null }),
  onLesion: (name, on) => send({ type: "lesion", fly: name, on }),
});

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
  const cols = Math.ceil(Math.sqrt(total));
  const group = new THREE.Group();
  group.position.set((i % cols - (cols - 1) / 2) * SPACING, 0, (Math.floor(i / cols) - (Math.ceil(total / cols) - 1) / 2) * SPACING);
  scene.add(group);
  const fly = new Fly();
  group.add(fly.root);
  try {
    station.build({ THREE, group, fly });
  } catch (e) {
    console.error(`${info.name}: station build() crashed`, e);
  }

  // floating label + glowing brain icon above the fly
  const div = document.createElement("div");
  div.className = "label" + (info.trained ? "" : " untrained");
  div.innerHTML = `<b>${station.title || info.name}</b>${info.trained ? "" : " (untrained)"}<span class="cap">...</span>`;
  const label = new CSS2DObject(div);
  label.position.set(0, 1.0, 0);
  fly.root.add(label);
  const brainIcon = new THREE.Mesh(new THREE.SphereGeometry(0.09, 16, 12),
    new THREE.MeshBasicMaterial({ color: 0x3cff7a, transparent: true, opacity: 0.5 }));
  brainIcon.position.set(0, 0.75, 0);
  brainIcon.userData.fly = fly;
  fly.root.add(brainIcon);

  flies[info.name] = { info, station, fly, group, label: div, brainIcon };
  fly.root.traverse((o) => (o.userData.flyName = info.name));
}

async function onDecision(m) {
  const f = flies[m.fly];
  if (!f) return;
  f.label.querySelector(".cap").textContent = m.caption;
  f.label.classList.toggle("lesioned", m.lesioned);
  tween(500, (t) => { f.brainIcon.material.opacity = 1 - 0.6 * t; f.brainIcon.scale.setScalar(1.6 - 0.6 * t); });
  if (matrix.fly === m.fly) matrix.show(m);
  const ctx = { THREE, group: f.group, fly: f.fly };
  try {
    await Promise.race([f.station.play(m.action, m.caption, ctx), wait(PLAY_TIMEOUT)]);
  } catch (e) {
    console.error(`${m.fly}: station play() crashed`, e);
    await wait(500);
  }
  send({ type: "ready", fly: m.fly });
}

// click a fly to enter its brain
const ray = new THREE.Raycaster();
renderer.domElement.addEventListener("click", (e) => {
  const p = new THREE.Vector2((e.clientX / innerWidth) * 2 - 1, -(e.clientY / innerHeight) * 2 + 1);
  ray.setFromCamera(p, camera);
  const hit = ray.intersectObjects(scene.children, true).find((h) => h.object.userData.flyName);
  if (!hit) return;
  const name = hit.object.userData.flyName;
  const f = flies[name];
  const target = new THREE.Vector3();
  f.fly.root.getWorldPosition(target);
  const from = camera.position.clone();
  const fromT = controls.target.clone();
  tween(700, (t) => {
    controls.target.lerpVectors(fromT, target, t);
    camera.position.lerpVectors(from, target.clone().add(new THREE.Vector3(0, 0.6, 1.2)), t);
  }).then(() => {
    send({ type: "inspect", fly: name });
    matrix.open(f.info, f.station.title || name);
    camera.position.copy(from); controls.target.copy(fromT);
  });
});

// --- connection ----------------------------------------------------------
function connect() {
  const status = document.getElementById("status");
  ws = new WebSocket(WS_URL);
  ws.onopen = () => { status.textContent = "connected to brain server"; send({ type: "hello" }); };
  ws.onclose = () => { status.textContent = "brain server offline - retrying..."; setTimeout(connect, 2000); };
  ws.onmessage = async (ev) => {
    const m = JSON.parse(ev.data);
    if (m.type === "init") {
      for (const f of Object.values(flies)) scene.remove(f.group);
      for (const k of Object.keys(flies)) delete flies[k];
      await Promise.all(m.flies.map((info, i) => addFly(info, i, m.flies.length)));
      const untrained = m.flies.filter((f) => !f.trained).length;
      status.textContent = `${m.flies.length} flies live` + (untrained ? ` (${untrained} untrained)` : "");
      for (const info of m.flies) send({ type: "ready", fly: info.name });
    } else if (m.type === "decision") {
      onDecision(m);
    }
  };
}
connect();

// --- loop ----------------------------------------------------------------
renderer.setAnimationLoop(() => {
  controls.update();
  renderer.render(scene, camera);
  labels.render(scene, camera);
});
