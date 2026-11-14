// Gym station: a bench press and a squat rack. The fly walks between them and lifts.
import { tween, wait, box, cylinder } from "/web/lib/fly.js";

const SPOTS = [{ x: -1.3, z: 0 }, { x: 1.3, z: 0 }]; // bench, squat rack

export default {
  title: "Gym fly",
  blurb: "Bench or squats? Rep or rest? Every choice is the fly's.",

  build({ THREE, group, fly }) {
    // bench
    group.add(box(1.0, 0.25, 0.35, 0x333344, [-1.3, 0.25, 0]));
    group.add(box(0.06, 0.7, 0.06, 0x999999, [-1.3, 0.35, 0.35]));
    group.add(box(0.06, 0.7, 0.06, 0x999999, [-1.3, 0.35, -0.35]));
    // squat rack
    for (const z of [-0.45, 0.45]) group.add(box(0.08, 1.6, 0.08, 0x999999, [1.3, 0.8, z]));
    // one barbell that moves with the fly
    this.bar = new THREE.Group();
    this.bar.add(cylinder(0.025, 1.1, 0xcccccc, [0, 0, 0], [Math.PI / 2, 0, 0]));
    for (const z of [-0.5, 0.5]) this.bar.add(cylinder(0.14, 0.06, 0x222222, [0, 0, z], [Math.PI / 2, 0, 0]));
    group.add(this.bar);
    this.station = 0;
    fly.root.position.set(SPOTS[0].x, 0.2, 0.0);
    fly.root.rotation.y = Math.PI / 2;
    this.placeBar(fly);
  },

  placeBar(fly) {
    const p = fly.root.position;
    this.bar.position.set(p.x, this.station === 0 ? 0.75 : 1.05, 0);
    this.barRest = this.bar.position.y;
  },

  async play(action, caption, { fly }) {
    if (action === "do_rep") {
      const failed = caption.includes("fail");
      const up = failed ? 0.15 : 0.4;
      await tween(450, (t) => (this.bar.position.y = this.barRest + Math.sin(t * Math.PI) * up));
      if (failed) await fly.buzz(500);
    } else {
      this.station = 1 - this.station;
      this.bar.visible = false;
      const s = SPOTS[this.station];
      fly.root.position.y = 0;
      await fly.walkTo({ x: s.x, z: s.z }, 1400);
      fly.root.position.y = this.station === 0 ? 0.2 : 0;
      fly.root.rotation.y = Math.PI / 2;
      this.placeBar(fly);
      this.bar.visible = true;
      await wait(150);
    }
  },
};
