// The pods: every live fly brain suspended in red gel on a biomechanical tower, cabled in, above the
// cute little world they think they live in. Live subjects sit front and centre; the rest of the tower is
// dormant copies (labelled as such) and more towers fade into the haze.
import * as THREE from "three";
import { CSS2DObject } from "three/addons/renderers/CSS2DRenderer.js";
import { mergeGeometries } from "three/addons/utils/BufferGeometryUtils.js";
import { brainMaterial, pointSize } from "./brain.js";

export const WALL_CENTER = new THREE.Vector3(0, 64, -78);
const R = 5.2;                    // tower radius
const D = R + 1.35;               // pod centre distance from tower axis
const POD_R = 0.62, POD_LEN = 1.3;
const PER_RING = 14, RING_GAP = 3.4, RINGS = 34, CENTER_RING = 16;   // even ring: has a pod dead centre
const BRAIN_SCALE = 0.44;
const FRONT = Math.PI / 2;        // angle that faces the world (+z)

const metal = new THREE.MeshStandardMaterial({ color: 0x1d1517, metalness: 0.85, roughness: 0.42 });
const darkMetal = new THREE.MeshStandardMaterial({ color: 0x0e0a0b, metalness: 0.7, roughness: 0.6 });
const shellMat = new THREE.MeshPhysicalMaterial({
  color: 0xffb3b8, emissive: 0x4a0010, transparent: true, opacity: 0.26, roughness: 0.08, metalness: 0.1,
  clearcoat: 1, depthWrite: false });
const gelMat = new THREE.MeshStandardMaterial({
  color: 0x8a0f22, emissive: 0x3d0008, transparent: true, opacity: 0.62, side: THREE.BackSide, depthWrite: false });
const cableMat = new THREE.MeshStandardMaterial({ color: 0x2a1c1c, metalness: 0.4, roughness: 0.55 });
const fleshCableMat = new THREE.MeshStandardMaterial({ color: 0x7a2a33, roughness: 0.35, emissive: 0x200004 });

function tube(points, r, seg = 20) {
  return new THREE.TubeGeometry(new THREE.CatmullRomCurve3(points.map((p) => new THREE.Vector3(...p))), seg, r, 6, false);
}

// pod-local frame: pod axis is y, +z points away from the tower
function podParts() {
  const shell = new THREE.CapsuleGeometry(POD_R, POD_LEN, 8, 22);
  const gel = new THREE.CapsuleGeometry(POD_R * 0.93, POD_LEN, 6, 18);
  const capY = POD_LEN / 2 + POD_R * 0.62;
  const caps = mergeGeometries([
    new THREE.CylinderGeometry(POD_R * 0.86, POD_R * 1.04, 0.32, 18).translate(0, capY, 0),
    new THREE.CylinderGeometry(POD_R * 1.04, POD_R * 0.86, 0.32, 18).translate(0, -capY, 0),
    new THREE.TorusGeometry(POD_R * 1.01, 0.04, 6, 24).rotateX(Math.PI / 2).translate(0, 0.25, 0),
    new THREE.TorusGeometry(POD_R * 1.01, 0.04, 6, 24).rotateX(Math.PI / 2).translate(0, -0.25, 0),
    new THREE.BoxGeometry(0.5, 0.5, D - R).translate(0, capY - 0.05, -(D - R) / 2),   // arm to the tower
  ]);
  const cables = mergeGeometries([
    tube([[0.25, -0.2, -0.45], [0.35, -0.5, -0.85], [0.3, -0.9, -(D - R) - 0.1]], 0.05),
    tube([[-0.25, -0.1, -0.45], [-0.4, -0.6, -0.9], [-0.35, -1.2, -(D - R) - 0.1]], 0.06),
    tube([[0, -capY, 0], [0, -capY - 0.5, -0.4], [0.1, -capY - 0.6, -(D - R) - 0.1]], 0.08),
    tube([[0.1, capY, -0.3], [0.2, capY + 0.6, -0.7], [0, capY + 0.9, -(D - R) - 0.1]], 0.05),
  ]);
  // the plug: a fleshy cord from the back of the pod into the back of the brain
  const plug = mergeGeometries([
    tube([[0, 0.02, -0.12], [0, 0.12, -0.35], [0, 0.0, -POD_R * 0.95]], 0.045, 16),
    tube([[0.08, -0.06, -0.1], [0.14, -0.25, -0.35], [0.1, -0.3, -POD_R * 0.9]], 0.025, 16),
    tube([[-0.08, -0.06, -0.1], [-0.16, -0.22, -0.32], [-0.12, -0.32, -POD_R * 0.9]], 0.025, 16),
  ]);
  return { shell, gel, caps, cables, plug, capY };
}

export class PodWall {
  constructor(scene) {
    this.scene = scene;
    this.group = new THREE.Group();
    this.group.position.copy(WALL_CENTER);
    scene.add(this.group);
    this.parts = podParts();
    this.live = [];          // {name, fly, slot, brain, light, label, shell, bubbles}
    this.fillers = [];       // dormant brain Points
    this.fog = { near: { value: 12 }, far: { value: 80 } };
    this.slots = [];
    for (let ring = 0; ring < RINGS; ring++)
      for (let i = 0; i < PER_RING; i++) {
        let da = ((i / PER_RING) * Math.PI * 2 + (ring % 2) * (Math.PI / PER_RING)) % (Math.PI * 2);
        if (da > Math.PI) da -= Math.PI * 2;                          // angle away from the front
        this.slots.push({ ring, angle: FRONT + da, dr: ring - CENTER_RING, da });
      }
    // live subjects get the slots nearest the front of the centre ring
    this.slots.sort((a, b) => Math.abs(a.dr) * 5 + Math.abs(a.da) * 3 - (Math.abs(b.dr) * 5 + Math.abs(b.da) * 3));
    this.buildTower();
    this.buildDistant();
    this.lights();
  }

  slotMatrix(slot) {
    const y = (slot.ring - CENTER_RING) * RING_GAP;
    const p = new THREE.Vector3(Math.cos(slot.angle) * D, y, Math.sin(slot.angle) * D);
    // +z of the pod points away from the tower (Matrix4.lookAt aims +z from target to eye)
    const m = new THREE.Matrix4().lookAt(new THREE.Vector3(p.x, 0, p.z), new THREE.Vector3(0, 0, 0), new THREE.Vector3(0, 1, 0));
    m.setPosition(p);
    return m;
  }

  buildTower() {
    const h = RINGS * RING_GAP + 30;
    const tower = new THREE.Mesh(new THREE.CylinderGeometry(R, R * 1.15, h, 40, 1), metal);
    tower.position.y = h / 2 - (CENTER_RING + 3) * RING_GAP - 10;
    this.group.add(tower);
    // ribs between the rings
    const ribs = new THREE.InstancedMesh(new THREE.TorusGeometry(R + 0.12, 0.22, 8, 48).rotateX(Math.PI / 2), darkMetal, RINGS + 1);
    for (let r = 0; r <= RINGS; r++)
      ribs.setMatrixAt(r, new THREE.Matrix4().makeTranslation(0, (r - CENTER_RING - 0.5) * RING_GAP, 0));
    this.group.add(ribs);
    // spikes all over, like something grown rather than built
    const rnd = mulberry(7);
    const spikes = new THREE.InstancedMesh(new THREE.ConeGeometry(0.16, 2.2, 5).translate(0, 1.1, 0), darkMetal, 260);
    const q = new THREE.Quaternion(), up = new THREE.Vector3(0, 1, 0);
    for (let i = 0; i < 260; i++) {
      const a = rnd() * Math.PI * 2, y = (rnd() - 0.5) * RINGS * RING_GAP;
      const dir = new THREE.Vector3(Math.cos(a), 0.6 + rnd() * 0.8, Math.sin(a)).normalize();
      q.setFromUnitVectors(up, dir);
      const s = 0.4 + rnd() * 1.1;
      spikes.setMatrixAt(i, new THREE.Matrix4().compose(new THREE.Vector3(Math.cos(a) * R, y, Math.sin(a) * R), q, new THREE.Vector3(s, s * (0.6 + rnd()), s)));
    }
    this.group.add(spikes);
    // a crown of spikes on top
    const top = (RINGS - CENTER_RING) * RING_GAP + 4;
    const crown = new THREE.InstancedMesh(new THREE.ConeGeometry(0.5, 9, 6).translate(0, 4.5, 0), darkMetal, 18);
    for (let i = 0; i < 18; i++) {
      const a = (i / 18) * Math.PI * 2;
      q.setFromUnitVectors(up, new THREE.Vector3(Math.cos(a) * 0.35, 1, Math.sin(a) * 0.35).normalize());
      crown.setMatrixAt(i, new THREE.Matrix4().compose(new THREE.Vector3(Math.cos(a) * R * 0.8, top, Math.sin(a) * R * 0.8), q, new THREE.Vector3(1, 0.6 + rnd() * 0.8, 1)));
    }
    this.group.add(crown);

    // every pod's shell, gel, caps and cables (live pods get their own meshes later)
    const n = this.slots.length;
    const P = this.parts;
    this.inst = {
      shell: new THREE.InstancedMesh(P.shell, shellMat, n),
      gel: new THREE.InstancedMesh(P.gel, gelMat, n),
      caps: new THREE.InstancedMesh(P.caps, metal, n),
      cables: new THREE.InstancedMesh(P.cables, cableMat, n),
      plug: new THREE.InstancedMesh(P.plug, fleshCableMat, n),
    };
    this.slots.forEach((s, i) => {
      s.matrix = this.slotMatrix(s);
      for (const k in this.inst) this.inst[k].setMatrixAt(i, s.matrix);
    });
    this.inst.gel.renderOrder = 1;
    this.inst.shell.renderOrder = 3;
    for (const k in this.inst) this.group.add(this.inst[k]);

    // status lights on each pod: blink slowly, live ones pulse with every thought
    this.lightsMesh = new THREE.InstancedMesh(new THREE.SphereGeometry(0.06, 8, 6),
      new THREE.MeshBasicMaterial({ color: 0xffffff }), n);
    this.lightPhase = new Float32Array(n);
    this.slots.forEach((s, i) => {
      this.lightPhase[i] = rnd() * 10;
      const m = s.matrix.clone().multiply(new THREE.Matrix4().makeTranslation(0.32, P.capY - 0.05, 0.42));
      this.lightsMesh.setMatrixAt(i, m);
      this.lightsMesh.setColorAt(i, new THREE.Color(0x330000));
    });
    this.group.add(this.lightsMesh);
  }

  buildDistant() {
    // more towers, receding into the haze. Just silhouettes and glowing pods.
    const rnd = mulberry(3);
    const towers = [];
    for (let i = 0; i < 26; i++) {
      const a = -Math.PI * 0.1 + rnd() * Math.PI * 1.2;  // mostly behind, spreading to the sides
      const dist = 55 + rnd() * 130;
      towers.push(new THREE.Vector3(Math.cos(Math.PI + a) * dist, (rnd() - 0.5) * 20, -Math.abs(Math.sin(a)) * dist - 30));
    }
    const towerGeo = new THREE.CylinderGeometry(R, R * 1.2, 220, 16, 1);
    const tm = new THREE.InstancedMesh(towerGeo, metal, towers.length);
    towers.forEach((p, i) => tm.setMatrixAt(i, new THREE.Matrix4().makeTranslation(p.x, p.y, p.z)));
    this.group.add(tm);
    const glow = new THREE.InstancedMesh(new THREE.CapsuleGeometry(0.5, 1.1, 2, 8),
      new THREE.MeshStandardMaterial({ color: 0x5a0a14, emissive: 0xc0182c, emissiveIntensity: 0.9, roughness: 0.4 }),
      towers.length * 30 * 10);
    let k = 0;
    const m = new THREE.Matrix4();
    for (const p of towers)
      for (let r = 0; r < 30; r++)
        for (let j = 0; j < 10; j++) {
          const a = (j / 10) * Math.PI * 2 + r * 0.31;
          m.makeTranslation(p.x + Math.cos(a) * (R + 1), p.y + (r - 15) * 3.5, p.z + Math.sin(a) * (R + 1));
          glow.setMatrixAt(k++, m);
        }
    this.group.add(glow);
  }

  lights() {
    this.redLight = new THREE.PointLight(0xff3344, 260, 40, 1.6);
    this.redLight.position.set(0, 2, D + 9);
    this.group.add(this.redLight);
    const under = new THREE.PointLight(0xff1a2a, 120, 30, 1.8);
    under.position.set(0, -10, D + 4);
    this.group.add(under);
  }

  // (re)build the live subjects: flies = [{name, info, title, brain: BrainActivity}]
  setLive(flies) {
    for (const l of this.live) { this.group.remove(l.pod); l.label.element.remove(); }
    for (const f of this.fillers) this.group.remove(f);
    this.live = [];
    this.fillers = [];
    // reset instancing so every slot is shown, then hide those the live pods take over
    this.slots.forEach((s, i) => { for (const k in this.inst) this.inst[k].setMatrixAt(i, s.matrix); });
    const P = this.parts;
    const hidden = new THREE.Matrix4().makeScale(0, 0, 0);
    flies.forEach((f, i) => {
      const slot = this.slots[i];
      for (const k in this.inst) this.inst[k].setMatrixAt(i, hidden);
      const pod = new THREE.Group();
      pod.applyMatrix4(slot.matrix);
      const shell = new THREE.Mesh(P.shell, shellMat.clone());
      shell.renderOrder = 3;
      const gel = new THREE.Mesh(P.gel, gelMat.clone());
      gel.renderOrder = 1;
      pod.add(shell, gel, new THREE.Mesh(P.caps, metal), new THREE.Mesh(P.cables, cableMat), new THREE.Mesh(P.plug, fleshCableMat));
      const n = f.brain.n;
      const mat = brainMaterial({ base: 0x7a1828, hot: 0xfff0f2, input: 0x39d0d0, baseAlpha: 0.35, additive: true,
                                  size: pointSize(n) * BRAIN_SCALE * 1.15, fog: this.fog });
      const brain = new THREE.Points(f.brain.geometry, mat);
      brain.scale.setScalar(BRAIN_SCALE);
      brain.renderOrder = 2;
      const brainHolder = new THREE.Group();
      brainHolder.add(brain);
      pod.add(brainHolder);
      const glow = new THREE.PointLight(0xff4060, 0, 3, 2);
      pod.add(glow);
      const bubbles = makeBubbles();
      pod.add(bubbles);
      const div = document.createElement("div");
      div.className = "plaque";
      const label = new CSS2DObject(div);
      // stagger plaques above/below so neighbours don't collide
      const above = flies.length > 1 && Math.round(slot.da / (Math.PI * 2 / PER_RING)) % 2 !== 0;
      label.position.set(0, above ? P.capY + 0.75 : -P.capY - 0.55, 0.3);
      label.userData.home = label.position.clone();
      pod.add(label);
      pod.traverse((o) => (o.userData.flyName = f.name));
      this.group.add(pod);
      const sub = String(i + 1).padStart(4, "0");
      div.innerHTML = `<b>SUBJECT ${sub} &middot; ${f.title.toUpperCase()}</b>` +
        `<span class="more">${n.toLocaleString()} neurons simulated<br>${f.info.trained ? "conditioned" : "unconditioned"} &middot; consent: none on file<br>` +
        `believes the world is real</span><i class="cap">awaiting thought...</i>`;
      this.live.push({ name: f.name, fly: f, slot, pod, brainHolder, brain, glow, gel, shell, label, bubbles, lesioned: false });
    });
    for (const k in this.inst) this.inst[k].instanceMatrix.needsUpdate = true;
    this.buildFillers(flies);
  }

  // dormant copies near the live ones: same brain, nobody home (just noise)
  buildFillers(flies) {
    const src = flies.reduce((a, f) => (!a || f.brain.n < a.brain.n ? f : a), null);
    if (!src) return;
    for (let i = flies.length; i < this.slots.length; i++) {
      const s = this.slots[i];
      if (Math.abs(s.dr) > 3 || Math.abs(s.da) > 1.3) continue;
      const mat = brainMaterial({ base: 0x4a0c16, hot: 0xff8a9a, input: 0x4a0c16, baseAlpha: 0.22, additive: true,
                                  size: pointSize(src.brain.n) * BRAIN_SCALE, idle: 0.9, seed: i * 7.3, fog: this.fog });
      const p = new THREE.Points(src.brain.geometry, mat);
      p.scale.setScalar(BRAIN_SCALE * (0.92 + ((i * 37) % 10) / 60));
      p.renderOrder = 2;
      p.applyMatrix4(s.matrix);
      p.rotateY((((i * 13) % 7) - 3) * 0.06);
      p.userData.bob = i * 1.7;
      p.userData.y0 = p.position.y;
      this.group.add(p);
      this.fillers.push(p);
    }
  }

  // world-space pose for looking at the whole tower or one live pod
  overviewPose() {
    // from slightly below, looking up: the tower should feel too big
    const target = WALL_CENTER.clone().add(new THREE.Vector3(0, 3.5, 0));
    return { pos: WALL_CENTER.clone().add(new THREE.Vector3(0, -2.5, D + 19)), target };
  }

  podPose(name) {
    const l = this.live.find((x) => x.name === name);
    if (!l) return this.overviewPose();
    const c = new THREE.Vector3(), out = new THREE.Vector3(0, 0, 1);
    l.pod.getWorldPosition(c);
    out.applyQuaternion(l.pod.getWorldQuaternion(new THREE.Quaternion()));
    return { pos: c.clone().addScaledVector(out, 6.2).add(new THREE.Vector3(0, 0.3, 0)), target: c.clone().add(new THREE.Vector3(0, 0, 0)) };
  }

  onDecision(m) {
    const l = this.live.find((x) => x.name === m.fly);
    if (!l) return;
    l.label.element.querySelector(".cap").textContent = `“${m.caption}”`;
    l.label.element.classList.toggle("lesioned", m.lesioned);
    l.lesioned = m.lesioned;
  }

  setLabelsVisible(v, focus = null) {
    for (const l of this.live) {
      l.label.visible = v && (!focus || l.name === focus);
      l.label.element.classList.toggle("focus", l.name === focus);
      // focused: the plaque hangs beside the pod instead
      if (l.name === focus) l.label.position.set(POD_R + 1.25, 0.2, 0.3);
      else l.label.position.copy(l.label.userData.home);
    }
  }

  update(t, dt, camera) {
    // fog that hides the pod brains along with everything else in the haze
    const fog = this.scene.fog;
    if (fog) { this.fog.near.value = fog.near; this.fog.far.value = fog.far; }
    const near = camera.position.distanceTo(WALL_CENTER) < 70;
    for (const f of this.fillers) {
      f.visible = near;
      if (near) f.position.y = f.userData.y0 + Math.sin(t * 0.7 + f.userData.bob) * 0.04;
    }
    // status lights: slow dying-heartbeat blink; live pods flash on each thought
    const c = new THREE.Color();
    for (let i = this.live.length; i < this.slots.length; i += 1) {
      const b = Math.max(0, Math.sin(t * 1.3 + this.lightPhase[i])) ** 12;
      this.lightsMesh.setColorAt(i, c.setRGB(0.2 + b * 0.8, 0.0, 0.0));
    }
    for (let i = 0; i < this.live.length; i++) {
      const l = this.live[i], a = l.fly.brain;
      const v = l.lesioned ? 0.15 : 0.3 + a.pulse * 0.7;
      this.lightsMesh.setColorAt(i, c.setRGB(l.lesioned ? 1 : v, l.lesioned ? 0 : v * 0.15, 0));
      // the brain floats, and jerks when it thinks
      l.brainHolder.position.y = Math.sin(t * 0.9 + i) * 0.05;
      l.brainHolder.rotation.z = Math.sin(t * 0.5 + i * 2) * 0.05 + a.pulse * 0.05 * Math.sin(t * 60);
      l.brainHolder.rotation.y = Math.sin(t * 0.3 + i) * 0.25;
      l.brainHolder.scale.setScalar(1 + a.pulse * 0.06);
      l.glow.intensity = l.lesioned ? 0 : 0.6 + a.pulse * 4 + a.level * 1.5;
      l.gel.material.emissive.setRGB(0.24 + a.level * 0.25 + a.pulse * 0.2, 0.0, 0.03);
      l.gel.material.opacity = l.lesioned ? 0.85 : 0.6;
      updateBubbles(l.bubbles, dt);
    }
    this.lightsMesh.instanceColor.needsUpdate = true;
    this.redLight.intensity = 240 + Math.sin(t * 1.1) * 40 + (Math.random() < 0.01 ? -180 : 0); // a bad bulb
  }
}

function makeBubbles(n = 26) {
  const pos = new Float32Array(n * 3);
  for (let i = 0; i < n; i++) resetBubble(pos, i, true);
  const g = new THREE.BufferGeometry();
  g.setAttribute("position", new THREE.BufferAttribute(pos, 3));
  return new THREE.Points(g, new THREE.PointsMaterial({ color: 0xffc0c8, size: 0.035, transparent: true, opacity: 0.55, depthWrite: false }));
}
function resetBubble(pos, i, anywhere) {
  const a = Math.random() * Math.PI * 2, r = Math.random() * POD_R * 0.75;
  pos[i * 3] = Math.cos(a) * r;
  pos[i * 3 + 1] = anywhere ? (Math.random() - 0.5) * POD_LEN * 1.6 : -POD_LEN * 0.85;
  pos[i * 3 + 2] = Math.sin(a) * r;
}
function updateBubbles(b, dt) {
  const pos = b.geometry.attributes.position.array;
  for (let i = 0; i < pos.length / 3; i++) {
    pos[i * 3 + 1] += dt * (0.25 + (i % 5) * 0.06);
    pos[i * 3] += Math.sin(pos[i * 3 + 1] * 9 + i) * dt * 0.03;
    if (pos[i * 3 + 1] > POD_LEN * 0.85) resetBubble(pos, i, false);
  }
  b.geometry.attributes.position.needsUpdate = true;
}

function mulberry(a) {
  return () => {
    a |= 0; a = (a + 0x6d2b79f5) | 0;
    let t = Math.imul(a ^ (a >>> 15), 1 | a);
    t = (t + Math.imul(t ^ (t >>> 7), 61 | t)) ^ t;
    return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
  };
}
