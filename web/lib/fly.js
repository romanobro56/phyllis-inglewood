// Shared helpers for stations: a low-poly fly with simple animations, and a tween helper.
import * as THREE from "three";

// Animate over `ms` milliseconds; fn(t) gets t from 0 to 1. Resolves when done.
export function tween(ms, fn) {
  return new Promise((resolve) => {
    const start = performance.now();
    function frame(now) {
      const t = Math.min(1, (now - start) / ms);
      fn(t);
      if (t < 1) requestAnimationFrame(frame);
      else resolve();
    }
    requestAnimationFrame(frame);
  });
}

export const wait = (ms) => new Promise((r) => setTimeout(r, ms));
const ease = (t) => (t < 0.5 ? 2 * t * t : 1 - Math.pow(-2 * t + 2, 2) / 2);

// A cartoon fruit fly (~0.6 units long). Original design: brown body, red eyes, clear wings.
// Rigged: six two-segment legs (hip -> femur -> knee -> tibia) driven by a tiny IK solver, so stations can
// plant feet or grab things:  fly.setFoot("L", 0, point)  (0 front, 1 middle, 2 hind; point in fly.root's frame)
// or fly.setFootBody(...) with a point in the body's own frame. fly.rest() puts it back on all six feet.
const LEG = [ // per pair: hip on the thorax (right side), segment lengths, resting foot (body frame, right side)
  { hip: [0.075, 0.27, 0.07], femur: 0.13, tibia: 0.28, foot: [0.2, 0.03, 0.2] },
  { hip: [0.09, 0.26, 0.0], femur: 0.15, tibia: 0.3, foot: [0.24, 0.03, 0.0] },
  { hip: [0.075, 0.26, -0.08], femur: 0.18, tibia: 0.34, foot: [0.21, 0.03, -0.24] },
];
const _m = new THREE.Matrix4(), _v = new THREE.Vector3(), _q = new THREE.Quaternion();

export class Fly {
  constructor(color = 0x8a6a3a) {
    this.root = new THREE.Group();
    this.body = new THREE.Group();
    this.root.add(this.body);
    const mat = new THREE.MeshStandardMaterial({ color, roughness: 0.6 });
    const dark = new THREE.MeshStandardMaterial({ color: 0x3b2a18, roughness: 0.7 });
    const eye = new THREE.MeshStandardMaterial({ color: 0xc81e1e, roughness: 0.3, emissive: 0x300000 });
    const wingMat = new THREE.MeshStandardMaterial({ color: 0xdfefff, transparent: true, opacity: 0.45,
      side: THREE.DoubleSide, roughness: 0.1 });

    const thorax = new THREE.Mesh(new THREE.SphereGeometry(0.13, 16, 12), mat);
    thorax.position.set(0, 0.32, 0);
    const abdomen = new THREE.Mesh(new THREE.SphereGeometry(0.14, 16, 12), mat);
    abdomen.scale.set(0.9, 0.85, 1.4);
    abdomen.position.set(0, 0.3, -0.24);
    const head = new THREE.Mesh(new THREE.SphereGeometry(0.09, 16, 12), mat);
    head.position.set(0, 0.36, 0.17);
    this.body.add(thorax, abdomen, head);
    this.legs = { L: [], R: [] };
    for (const s of [-1, 1]) {
      const e = new THREE.Mesh(new THREE.SphereGeometry(0.06, 12, 10), eye);
      e.position.set(0.065 * s, 0.38, 0.2);
      this.body.add(e);
      const wing = new THREE.Mesh(new THREE.CircleGeometry(0.2, 16), wingMat);
      wing.scale.set(0.45, 1, 1);
      wing.rotation.set(-Math.PI / 2, 0, 0);
      const pivot = new THREE.Group();
      pivot.position.set(0.05 * s, 0.43, -0.02);
      wing.position.set(0.07 * s, 0, -0.17);
      pivot.add(wing);
      this.body.add(pivot);
      (this.wings ||= []).push({ pivot, s });
      // legs are built for the right side; the left side is the same, mirrored
      const side = new THREE.Group();
      side.scale.x = s;
      this.body.add(side);
      LEG.forEach((d, i) => {
        const hip = new THREE.Group();
        hip.position.set(...d.hip);
        const femur = new THREE.Mesh(new THREE.CylinderGeometry(0.016, 0.013, d.femur, 6).rotateZ(-Math.PI / 2).translate(d.femur / 2, 0, 0), dark);
        const knee = new THREE.Group();
        knee.position.x = d.femur;
        const joint = new THREE.Mesh(new THREE.SphereGeometry(0.017, 8, 6), dark);
        const tibia = new THREE.Mesh(new THREE.CylinderGeometry(0.012, 0.007, d.tibia, 6).rotateZ(-Math.PI / 2).translate(d.tibia / 2, 0, 0), dark);
        knee.add(joint, tibia);
        hip.add(femur, knee);
        side.add(hip);
        this.legs[s < 0 ? "L" : "R"][i] = { ...d, hip, knee, s };
      });
    }
    this.root.traverse((o) => { if (o.isMesh) { o.castShadow = true; o.userData.fly = this; } });
    this.rest();
  }

  // Put the foot of leg i (0 front, 1 middle, 2 hind) on `p`, a point in the BODY's frame.
  // `pole` (body frame) is roughly where the knee should point; default up and outward like an insect.
  setFootBody(side, i, p, pole) {
    const leg = this.legs[side][i];
    leg.target = new THREE.Vector3(p.x, p.y, p.z);                       // remembered, for blending
    const t = _v.set(p.x * leg.s, p.y, p.z).sub(leg.hip.position);       // into the leg's (right-side) frame
    const P = pole ? new THREE.Vector3(pole.x * leg.s, pole.y, pole.z) : new THREE.Vector3(0.6, 1, 0);
    const a = leg.femur, b = leg.tibia;
    const d = THREE.MathUtils.clamp(t.length(), Math.abs(a - b) + 1e-3, a + b - 1e-3);
    const dir = t.lengthSq() > 1e-9 ? t.clone().normalize() : new THREE.Vector3(1, 0, 0);
    const alpha = Math.acos(THREE.MathUtils.clamp((a * a + d * d - b * b) / (2 * a * d), -1, 1));
    const beta = Math.acos(THREE.MathUtils.clamp((a * a + b * b - d * d) / (2 * a * b), -1, 1));
    let up = P.sub(dir.clone().multiplyScalar(P.dot(dir)));
    if (up.lengthSq() < 1e-9) up = new THREE.Vector3(0, 1, 0).sub(dir.clone().multiplyScalar(dir.y));
    up.normalize();
    const X = dir.clone().multiplyScalar(Math.cos(alpha)).addScaledVector(up, Math.sin(alpha)).normalize();
    const Y = up.sub(X.clone().multiplyScalar(up.dot(X))).normalize();
    const Z = new THREE.Vector3().crossVectors(X, Y);
    leg.hip.quaternion.setFromRotationMatrix(_m.makeBasis(X, Y, Z));
    leg.knee.rotation.set(0, 0, -(Math.PI - beta));
  }

  // Same, with the target (and pole) in the ROOT's frame, i.e. the station's floor coordinates around the fly.
  setFoot(side, i, p, pole) {
    this.body.updateMatrix();
    const inv = _m.copy(this.body.matrix).invert();
    const pb = p.clone().applyMatrix4(inv);
    const pl = pole ? pole.clone().applyQuaternion(_q.copy(this.body.quaternion).invert()) : undefined;
    this.setFootBody(side, i, pb, pl);
  }

  // Stand normally on all six feet (gait: 0..1 phase of a walking step, or null).
  rest(gait = null) {
    for (const side of ["L", "R"])
      this.legs[side].forEach((leg, i) => {
        const [x, y, z] = leg.foot;
        let dy = 0, dz = 0;
        if (gait != null) {                               // tripod gait: R0 L1 R2 together, then the others
          const ph = gait * Math.PI * 2 + ((side === "R") === (i !== 1) ? 0 : Math.PI);
          dz = Math.cos(ph) * 0.07;
          dy = Math.max(0, Math.sin(ph)) * 0.06;
        }
        this.setFootBody(side, i, new THREE.Vector3(side === "L" ? -x : x, y + dy - this.body.position.y, z + dz));
      });
  }

  // Back to a plain standing fly: body level, legs on the floor.
  reset() {
    this.body.position.set(0, 0, 0);
    this.body.rotation.set(0, 0, 0);
    this.rest();
  }

  // Buzz the wings for ms milliseconds.
  buzz(ms = 400) {
    return tween(ms, (t) => {
      for (const { pivot, s } of this.wings) pivot.rotation.z = s * (0.2 + 0.6 * Math.abs(Math.sin(t * 40)));
    }).then(() => this.wings.forEach(({ pivot }) => (pivot.rotation.z = 0)));
  }

  // Small up-and-down hop (good for "did something").
  hop(height = 0.25, ms = 350) {
    return tween(ms, (t) => (this.body.position.y = Math.sin(t * Math.PI) * height));
  }

  // Turn to face a point (in the parent's coordinates).
  face(target) {
    const p = this.root.position;
    this.root.rotation.y = Math.atan2(target.x - p.x, target.z - p.z);
  }

  // Walk to a point (THREE.Vector3 in the parent's coordinates), legs stepping in a tripod gait.
  async walkTo(target, ms = 1200) {
    const from = this.root.position.clone();
    const to = new THREE.Vector3(target.x, from.y, target.z);
    this.face(to);
    const steps = Math.max(2, Math.round(from.distanceTo(to) / 0.14));
    await tween(ms, (t) => {
      this.root.position.lerpVectors(from, to, ease(t));
      this.body.position.y = Math.abs(Math.sin(t * Math.PI * steps)) * 0.025;
      this.rest(t * steps);
    });
    this.body.position.y = 0;
    this.rest();
  }
}

// Simple building blocks for stations.
export function box(w, h, d, color, pos = [0, 0, 0]) {
  const m = new THREE.Mesh(new THREE.BoxGeometry(w, h, d),
    new THREE.MeshStandardMaterial({ color, roughness: 0.7 }));
  m.position.set(...pos);
  m.castShadow = m.receiveShadow = true;
  return m;
}

export function cylinder(r, h, color, pos = [0, 0, 0], rot = [0, 0, 0]) {
  const m = new THREE.Mesh(new THREE.CylinderGeometry(r, r, h, 16),
    new THREE.MeshStandardMaterial({ color, roughness: 0.5, metalness: 0.3 }));
  m.position.set(...pos);
  m.rotation.set(...rot);
  m.castShadow = true;
  return m;
}
