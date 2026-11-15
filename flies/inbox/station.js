// Inbox station: a desk with a laptop; each email gets archived, answered, flagged or unsubscribed.
// Inbox zero has not been going well: beer cans, bottles and cigarette butts everywhere.
import { tween, wait, box } from "/web/lib/fly.js";

export default {
  title: "Doris Inglewood",
  blurb: "Archive, reply, flag or unsubscribe: the fly triages your email.",

  build({ THREE, group, fly }) {
    group.add(box(1.6, 0.08, 0.9, 0x6b4a2b, [0, 0.7, 0]));                  // desk top
    for (const [x, z] of [[-0.7, -0.35], [0.7, -0.35], [-0.7, 0.35], [0.7, 0.35]])
      group.add(box(0.06, 0.7, 0.06, 0x4a321c, [x, 0.35, z]));
    group.add(box(0.6, 0.03, 0.4, 0x222222, [0, 0.76, 0.1]));                // laptop base
    const screen = box(0.6, 0.4, 0.02, 0x111111, [0, 0.96, -0.1]);
    screen.material.emissive = new THREE.Color(0x113355);
    group.add(screen);
    this.box = box(0.3, 0.2, 0.3, 0x2f6f3f, [-0.6, 0.84, 0.2]);              // archive box
    this.trash = box(0.2, 0.25, 0.2, 0x555555, [0.9, 0.12, 0.5]);
    group.add(this.box, this.trash);
    this.flag = box(0.02, 0.3, 0.02, 0xffffff, [0.6, 0.9, -0.2]);
    const cloth = box(0.15, 0.1, 0.01, 0xdd2222, [0.08, 0.1, 0]);
    this.flag.add(cloth);
    this.flag.visible = false;
    group.add(this.flag);
    this.mail = new THREE.Mesh(new THREE.BoxGeometry(0.22, 0.01, 0.15),
      new THREE.MeshStandardMaterial({ color: 0xf5f0e0 }));
    group.add(this.mail);
    fly.root.position.set(0, 0.74, 0.35);
    fly.root.rotation.y = Math.PI;
    this.THREE = THREE;
    litter(THREE, group);
  },

  async play(action, caption, { fly }) {
    const m = this.mail;
    m.visible = true;
    m.scale.setScalar(1);
    m.position.set(0, 1.4, 0.25);
    await tween(300, (t) => (m.position.y = 1.4 - 0.6 * t));               // email drops in
    await fly.hop(0.12, 250);
    const from = m.position.clone();
    if (action === "archive") {
      await tween(500, (t) => m.position.lerpVectors(from, this.box.position, t));
    } else if (action === "reply") {
      await tween(600, (t) => m.position.set(from.x, from.y + t * 1.5, from.z - t * 2));
    } else if (action === "flag") {
      this.flag.visible = true;
      await fly.buzz(400);
      this.flag.visible = false;
    } else {
      await tween(500, (t) => {
        m.position.lerpVectors(from, this.trash.position, t);
        m.scale.setScalar(1 - 0.8 * t);
      });
    }
    m.visible = false;
    await wait(200);
  },
};

// --- the mess ------------------------------------------------------------------
const DESK = 0.74; // desk top height
const CAN_COLORS = [0xc8102e, 0x1f5fa8, 0x2e7d32, 0xd4a017, 0xe8e8e8];

function rng(seed) {
  return () => ((seed = (seed * 16807) % 2147483647) - 1) / 2147483646;
}

function litter(THREE, group) {
  const r = rng(11);
  const mat = (color, extra = {}) => new THREE.MeshStandardMaterial({ color, roughness: 0.45, ...extra });
  const alu = mat(0xc9ccd1, { metalness: 0.8, roughness: 0.3 });
  const add = (o, x, y, z) => { o.position.set(x, y, z); o.traverse((m) => (m.castShadow = true)); group.add(o); return o; };

  // a beer can: standing, tipped over, or crushed flat
  function can(x, y, z, state = "up") {
    const g = new THREE.Group();
    const body = new THREE.Mesh(new THREE.CylinderGeometry(0.034, 0.034, 0.1, 14), mat(CAN_COLORS[(r() * CAN_COLORS.length) | 0], { metalness: 0.4 }));
    const top = new THREE.Mesh(new THREE.CylinderGeometry(0.03, 0.034, 0.012, 14), alu);
    top.position.y = 0.056;
    g.add(body, top);
    g.rotation.y = r() * Math.PI * 2;
    if (state === "tipped") { g.rotation.z = Math.PI / 2; y += 0.034; }
    else if (state === "crushed") { g.scale.set(1.25, 0.35, 0.8); g.rotation.z = (r() - 0.5) * 0.5; y += 0.02; }
    else y += 0.05;
    return add(g, x, y, z);
  }
  function bottle(x, y, z, tipped = false) {
    const g = new THREE.Group();
    const glass = mat(0x5a3410, { transparent: true, opacity: 0.85, roughness: 0.15 });
    const body = new THREE.Mesh(new THREE.CylinderGeometry(0.03, 0.03, 0.11, 12), glass);
    const neck = new THREE.Mesh(new THREE.CylinderGeometry(0.011, 0.026, 0.07, 10), glass);
    neck.position.y = 0.09;
    const label = new THREE.Mesh(new THREE.CylinderGeometry(0.031, 0.031, 0.045, 12), mat(0xefe6cf));
    g.add(body, neck, label);
    g.rotation.y = r() * Math.PI * 2;
    if (tipped) { g.rotation.z = Math.PI / 2; y += 0.03; } else y += 0.055;
    return add(g, x, y, z);
  }
  // a cigarette (or what's left of one), lying on whatever surface is at height y
  function cig(x, y, z, len = 0.03 + r() * 0.04, lit = false) {
    const g = new THREE.Group();
    const paper = new THREE.Mesh(new THREE.CylinderGeometry(0.007, 0.007, len, 6), mat(0xf4f1ea));
    paper.rotation.z = Math.PI / 2;
    paper.position.x = len / 2;
    const filter = new THREE.Mesh(new THREE.CylinderGeometry(0.0075, 0.0075, 0.02, 6), mat(0xd98a3a));
    filter.rotation.z = Math.PI / 2;
    filter.position.x = -0.01;
    const tip = new THREE.Mesh(new THREE.CylinderGeometry(0.0072, 0.0072, 0.006, 6),
      lit ? mat(0xff5a1a, { emissive: 0xff3a00, emissiveIntensity: 2 }) : mat(0x3a3a3a));
    tip.rotation.z = Math.PI / 2;
    tip.position.x = len + 0.003;
    g.add(paper, filter, tip);
    g.rotation.y = r() * Math.PI * 2;
    g.userData.tip = tip;
    return add(g, x, y + 0.008, z);
  }

  // on the desk
  can(0.48, DESK, -0.3);
  can(0.66, DESK, -0.05);
  can(0.58, DESK, 0.3, "crushed");
  can(-0.3, DESK, -0.32, "tipped");
  can(-0.72, DESK, -0.28, "crushed");
  bottle(-0.66, DESK, -0.05);
  // overflowing ashtray with one still burning
  const tray = add(new THREE.Mesh(new THREE.CylinderGeometry(0.1, 0.085, 0.03, 18),
    mat(0x36414a, { transparent: true, opacity: 0.9, roughness: 0.1 })), 0.36, DESK + 0.015, 0.3);
  add(new THREE.Mesh(new THREE.CylinderGeometry(0.08, 0.08, 0.012, 16), mat(0x77736c)), 0.36, DESK + 0.03, 0.3); // ash
  for (let i = 0; i < 9; i++) {
    const a = r() * Math.PI * 2, d = r() * 0.06;
    const c = cig(0.36 + Math.cos(a) * d, DESK + 0.03, 0.3 + Math.sin(a) * d);
    c.rotation.z = 0.2 + r() * 0.5;
  }
  const burning = cig(0.27, DESK + 0.03, 0.27, 0.07, true);
  burning.rotation.set(0, 0.6, 0.12);
  for (let i = 0; i < 4; i++) cig(0.1 + r() * 0.5, DESK, -0.4 + r() * 0.3 * (i % 2));
  // a crumpled pack
  const pack = add(new THREE.Mesh(new THREE.BoxGeometry(0.06, 0.022, 0.09), mat(0xe8e2d4)), -0.48, DESK + 0.011, 0.34);
  pack.rotation.set(0.1, 0.7, 0.05);
  add(new THREE.Mesh(new THREE.BoxGeometry(0.061, 0.023, 0.03), mat(0xb3261e)), -0.48, DESK + 0.012, 0.34).rotation.copy(pack.rotation);

  // on the floor, around and under the desk
  const floorSpot = () => {
    for (;;) {
      const x = (r() - 0.5) * 2.6, z = (r() - 0.5) * 1.9;
      if (Math.hypot(x - 0.9, z - 0.5) > 0.2) return [x, z];                       // not in the trash can
    }
  };
  for (let i = 0; i < 9; i++) { const [x, z] = floorSpot(); can(x, 0, z, i % 3 === 0 ? "up" : i % 3 === 1 ? "tipped" : "crushed"); }
  for (let i = 0; i < 3; i++) { const [x, z] = floorSpot(); bottle(x, 0, z, i > 0); }
  for (let i = 0; i < 28; i++) { const [x, z] = floorSpot(); cig(x, 0, z); }
  // one bottle in the trash can, at least
  bottle(0.9, 0.2, 0.5).rotation.z = 0.4;

  // smoke curling up from the burning one
  const puffMat = new THREE.MeshBasicMaterial({ color: 0xe8e4e0, transparent: true, opacity: 0.35, depthWrite: false });
  const puffs = [];
  const origin = new THREE.Vector3();
  for (let i = 0; i < 14; i++) {
    const p = new THREE.Mesh(new THREE.SphereGeometry(0.02, 8, 6), puffMat.clone());
    p.userData.t = i / 14;
    group.add(p);
    puffs.push(p);
  }
  let last = performance.now();
  (function smoke(now) {
    if (!group.parent) return;                                          // station was removed
    const dt = Math.min(0.1, (now - last) / 1000);
    last = now;
    burning.userData.tip.getWorldPosition(origin);
    group.worldToLocal(origin);
    for (const p of puffs) {
      const t = (p.userData.t = (p.userData.t + dt * 0.28) % 1);
      p.position.set(origin.x + Math.sin(t * 9 + p.id) * 0.04 * t, origin.y + t * 0.7, origin.z + Math.cos(t * 7) * 0.03 * t);
      p.scale.setScalar(0.5 + t * 3);
      p.material.opacity = 0.32 * (1 - t) * Math.min(1, t * 8);
    }
    requestAnimationFrame(smoke);
  })(last);
}
