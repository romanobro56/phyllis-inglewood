// The realistic fly: DeepMind/Janelia's flybody fruit fly (web/assets/fly/fly.glb), moving with clips of real
// recorded flies. Same API as the cartoon Fly in fly.js (root, body, wings, walkTo, hop, buzz, face, rest,
// reset, setFoot), so every station.js works with either. Without the model file the cartoon fly is used.
import * as THREE from "three";
import { GLTFLoader } from "three/addons/loaders/GLTFLoader.js";
import { Fly, tween, wait } from "./fly.js";

const GLB = "/web/assets/fly/fly.glb";
const META = "/web/assets/fly/meta.json";
const LENGTH = 0.6;   // body length in station units, the same as the cartoon fly, so stations' distances still fit
const ease = (t) => (t < 0.5 ? 2 * t * t : 1 - Math.pow(-2 * t + 2, 2) / 2);

let asset = null;     // null: not tried yet, false: missing, else {scene, clips, meta}

export async function loadFlyAsset() {
  if (asset !== null) return asset;
  try {
    const [gltf, meta] = await Promise.all([
      new GLTFLoader().loadAsync(GLB),
      fetch(META).then((r) => (r.ok ? r.json() : {})).catch(() => ({})),
    ]);
    dress(gltf.scene);
    asset = { scene: gltf.scene, clips: gltf.animations, meta };
  } catch (e) {
    console.warn("no realistic fly model yet (web/assets/fly/fly.glb), using the cartoon fly", e);
    asset = false;
  }
  return asset;
}

export const makeFly = () => (asset ? new RealFly(asset) : new Fly());

// Physical materials, picked by the material names the model was exported with.
function dress(model) {
  const made = new Map();
  const make = (m) => {
    const n = (m.name || "").toLowerCase(), c = m.color ? m.color.clone() : new THREE.Color(0x6b4a2b);
    if (n.includes("membrane") || n.includes("wing"))     // thin film: iridescent, see-through, no depth (or it cuts the splats)
      return new THREE.MeshPhysicalMaterial({ name: m.name, color: 0xf4f7ff, transparent: true, opacity: 0.22,
        roughness: 0.12, iridescence: 1, iridescenceIOR: 1.35, iridescenceThicknessRange: [250, 650],
        side: THREE.DoubleSide, depthWrite: false });
    if (n.includes("red") || n.includes("eye"))
      return new THREE.MeshPhysicalMaterial({ name: m.name, color: c.getHex() ? c : 0x9c1b12, roughness: 0.38,
        clearcoat: 0.8, clearcoatRoughness: 0.25 });
    if (n.includes("bristle") || n.includes("black"))
      return new THREE.MeshPhysicalMaterial({ name: m.name, color: c, roughness: 0.55, clearcoat: 0.2 });
    if (n === "body" || n === "lower") c.multiply(new THREE.Color(0.62, 0.5, 0.38));   // flybody's tan is pale; real cuticle is browner
    return new THREE.MeshPhysicalMaterial({ name: m.name, color: c, roughness: 0.45, clearcoat: 0.45,  // chitin
      clearcoatRoughness: 0.35, sheen: 0.4, sheenRoughness: 0.5, sheenColor: 0xffe2b8 });
  };
  model.traverse((o) => {
    if (!o.isMesh) return;
    const m = o.material;
    if (!made.has(m)) made.set(m, make(m));
    o.material = made.get(m);
    o.castShadow = true;
    if (o.material.transparent) o.renderOrder = 2;
  });
}

class RealFly {
  constructor({ scene, clips, meta }) {
    this.root = new THREE.Group();
    this.body = new THREE.Group();
    this.root.add(this.body);
    const model = scene.clone(true);
    model.scale.setScalar(LENGTH);                                   // 1 unit = one body length; faces +z, feet on y = 0
    this.body.add(model);
    model.traverse((o) => (o.userData.fly = this));

    this.mixer = new THREE.AnimationMixer(model);
    this.actions = {};
    for (const c of clips) this.actions[c.name] = this.mixer.clipAction(c);
    const walk = this.actions.walk;
    if (walk) { walk.play(); walk.paused = true; walk.setEffectiveWeight(0); }   // stepped by hand from rest(gait)
    this.idle = this.actions.idle;
    this.idle?.play();
    this.walkPhase = null;
    this.nextGroom = 4 + Math.random() * 6;
    // how far one loop of the walk clip carries the fly, so its feet don't slide (from the recording)
    this.stride = (walk ? (meta.walk_speed_body_lengths_per_s ?? 1.5) * walk.getClip().duration : 0.5) * LENGTH;

    // Stations set wings[i].pivot.rotation; it's applied on top of whatever the clips do to the wing.
    const names = meta.wing_nodes || [];
    this.wings = [];
    model.traverse((o) => {
      if (names.length ? names.includes(o.name) : /wing/i.test(o.name) && !o.isMesh && o.children.length)
        this.wings.push({ node: o, pivot: new THREE.Object3D(), rest: o.quaternion.clone(), s: /left/i.test(o.name) ? -1 : 1 });
    });
    this.legs = { L: [], R: [] };
  }

  // Called every frame by the world.
  update(dt) {
    const { walk, groom } = this.actions;
    if (walk) {
      const on = this.walkPhase != null;
      walk.setEffectiveWeight(on ? 1 : 0);
      if (on) walk.time = this.walkPhase * walk.getClip().duration;
      this.idle?.setEffectiveWeight(on ? 0 : 1);
    }
    if (groom && this.walkPhase == null && !this.buzzing) {          // now and then, clean the front legs
      this.nextGroom -= dt;
      if (this.nextGroom < 0) {
        this.nextGroom = 8 + Math.random() * 10;
        groom.reset().setLoop(THREE.LoopOnce, 1).setEffectiveTimeScale(2).fadeIn(0.3).play();   // the recording is 4x slowed
        groom.clampWhenFinished = false;
        setTimeout(() => groom.fadeOut(0.4), groom.getClip().duration * 500 - 400);
      }
    }
    for (const w of this.wings) w.node.quaternion.copy(w.rest);
    this.mixer.update(dt);
    for (const w of this.wings) w.node.quaternion.multiply(w.pivot.quaternion);
  }

  // The flybody legs follow the recorded gait; stations' foot targets are accepted but not solved for.
  setFoot() {}
  setFootBody() {}

  rest(gait = null) { this.walkPhase = gait == null ? null : ((gait % 1) + 1) % 1; }

  reset() {
    this.body.position.set(0, 0, 0);
    this.body.rotation.set(0, 0, 0);
    this.rest();
  }

  async buzz(ms = 400) {
    const flap = this.actions.flap;
    this.buzzing = true;
    if (flap) {
      flap.reset().setEffectiveTimeScale(6).fadeIn(0.05).play();
      await wait(ms);
      flap.fadeOut(0.08);
    } else {
      await tween(ms, (t) => { for (const { pivot, s } of this.wings) pivot.rotation.z = s * 0.6 * Math.abs(Math.sin(t * 60)); });
      for (const { pivot } of this.wings) pivot.rotation.set(0, 0, 0);
    }
    this.buzzing = false;
  }

  hop(height = 0.25, ms = 350) {
    return tween(ms, (t) => (this.body.position.y = Math.sin(t * Math.PI) * height));
  }

  face(target) {
    const p = this.root.position;
    this.root.rotation.y = Math.atan2(target.x - p.x, target.z - p.z);
  }

  async walkTo(target, ms = 1200) {
    const from = this.root.position.clone();
    const to = new THREE.Vector3(target.x, from.y, target.z);
    this.face(to);
    const cycles = from.distanceTo(to) / this.stride;
    await tween(ms, (t) => {
      const e = ease(t);
      this.root.position.lerpVectors(from, to, e);
      this.rest(e * cycles);                                          // feet move with the body, so they don't slide
    });
    this.rest();
  }
}
