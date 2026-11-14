// Inbox station: a desk with a laptop; each email gets archived, answered, flagged or unsubscribed.
import { tween, wait, box } from "/web/lib/fly.js";

export default {
  title: "Inbox fly",
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
