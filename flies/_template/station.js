// TEMPLATE station: how this fly looks in the 3D world.
// build() runs once; play() runs after every decision and must resolve when the animation ends
// (the brain waits for it before deciding again). Aim for 1-2 seconds per play().
//
// Available from "/web/lib/fly.js": tween(ms, fn), wait(ms), box(w,h,d,color,pos), cylinder(r,h,color,pos,rot)
// The fly: fly.walkTo({x, z}, ms), fly.hop(height, ms), fly.buzz(ms), fly.face({x, z}), fly.root (a THREE.Group)
import { tween, wait, box } from "/web/lib/fly.js";

export default {
  title: "TEMPLATE_NAME fly",
  blurb: "One line for the info panel.",

  build({ THREE, group, fly }) {
    this.pad = box(1.2, 0.1, 1.2, 0x4466aa, [0, 0.05, 0]);
    group.add(this.pad);
    fly.root.position.set(0, 0.1, 0);
  },

  async play(action, caption, { fly }) {
    // one branch per action name from task.py
    if (action === "option_a") {
      await fly.hop(0.3, 400);
    } else {
      await fly.buzz(500);
    }
    await wait(300);
  },
};
