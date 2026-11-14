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
      for (let i = 0; i < 3; i++) {
        const leg = new THREE.Mesh(new THREE.CylinderGeometry(0.012, 0.008, 0.3, 5), dark);
        leg.position.set(0.12 * s, 0.17, 0.1 - i * 0.1);
        leg.rotation.z = 0.5 * s;
        this.body.add(leg);
      }
    }
    this.root.traverse((o) => { if (o.isMesh) { o.castShadow = true; o.userData.fly = this; } });
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

  // Walk to a point (THREE.Vector3 in the parent's coordinates), bobbing and buzzing.
  async walkTo(target, ms = 1200) {
    const from = this.root.position.clone();
    const to = new THREE.Vector3(target.x, from.y, target.z);
    this.face(to);
    await tween(ms, (t) => {
      this.root.position.lerpVectors(from, to, ease(t));
      this.body.position.y = Math.abs(Math.sin(t * Math.PI * 6)) * 0.05;
    });
    this.body.position.y = 0;
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
