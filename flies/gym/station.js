// Gym station: Marlo Inglewood, a fruit fly who lifts. A flat bench (lies on its back, presses with four legs)
// and a squat rack (stands up on its hind legs, bar across its back). Each station has its own barbell.
// Rests between sets (1-5 min) play 20x faster than real time, with a countdown over Marlo's head.
import { CSS2DObject } from "three/addons/renderers/CSS2DRenderer.js";
import { tween, wait, box, cylinder } from "/web/lib/fly.js";

const SIM_SPEED = 20; // a 5 minute rest takes 15 s on screen

const lerp = (a, b, t) => a + (b - a) * t;
const ease = (t) => (t < 0.5 ? 2 * t * t : 1 - (-2 * t + 2) ** 2 / 2);
const SIDES = [["L", 1], ["R", -1]];

// Each lift lives in its own little area, turned so the camera (at +z) sees it from three-quarters:
// the bench from the lifter's feet, the squat from the front-side, so the plates never hide the fly.
// Bench area: pad along local x, head end at -x, thorax under the bar at local x = -0.3.
const BENCH = { at: [-1.3, 0, -0.1], turn: -1.15, x: -0.3, lock: 0.95, chest: 0.7, approach: [0.15, 0, 0.6] };
// Squat area: the fly stands at the origin facing local +x; the hooks are just behind its back.
const SQUAT = { at: [1.3, 0, 0], turn: -0.85, hookX: -0.17, hookY: 0.6, approach: [0.65, 0, 0] };

export default {
  title: "Marlo Inglewood",
  blurb: "Marlo Inglewood: bench or squats? Rep or rest, and for how long? Every choice is the fly's.",
  playTimeout: (5 * 60 * 1000) / SIM_SPEED + 5000, // the longest rest, plus getting up

  build({ THREE, group, fly }) {
    this.THREE = THREE;
    this.V = (x, y, z) => new THREE.Vector3(x, y, z);
    this.group = group;
    const area = (spec) => {
      const g = new THREE.Group();
      g.position.set(...spec.at);
      g.rotation.y = spec.turn;
      group.add(g);
      return g;
    };
    const bench = (this.benchArea = area(BENCH));
    const rack = (this.squatArea = area(SQUAT));
    bench.add(box(1.7, 0.01, 1.5, 0x3a3436, [0, 0.005, 0]));                    // rubber floor
    rack.add(box(1.4, 0.01, 1.4, 0x3a3436, [0.1, 0.005, 0]));
    // flat bench along local x, pad top at y = 0.40, uprights with J-hooks at the head end
    bench.add(box(1.0, 0.08, 0.3, 0x24242e, [-0.05, 0.36, 0]));
    for (const x of [-0.45, 0.35]) {
      bench.add(box(0.07, 0.32, 0.07, 0x8a8f99, [x, 0.16, 0]));
      bench.add(box(0.07, 0.03, 0.34, 0x8a8f99, [x, 0.015, 0]));
    }
    for (const z of [-0.36, 0.36]) {
      bench.add(box(0.05, 1.02, 0.05, 0x8a8f99, [BENCH.x, 0.51, z]));
      bench.add(box(0.1, 0.03, 0.06, 0xd0d3d8, [BENCH.x + 0.03, 0.92, z]));
    }
    // squat rack: two posts with hooks, and a pull-up bar on top
    for (const z of [-0.42, 0.42]) {
      rack.add(box(0.06, 1.5, 0.06, 0x8a8f99, [SQUAT.hookX, 0.75, z]));
      rack.add(box(0.1, 0.03, 0.07, 0xd0d3d8, [SQUAT.hookX + 0.05, SQUAT.hookY - 0.04, z]));
    }
    rack.add(cylinder(0.02, 0.9, 0x8a8f99, [SQUAT.hookX, 1.48, 0], [Math.PI / 2, 0, 0]));
    // a water bottle and a towel, for realism
    group.add(cylinder(0.05, 0.18, 0x2f7fd6, [-0.7, 0.09, 0.45]));
    group.add(box(0.3, 0.02, 0.18, 0xf2f2f2, [0.2, 0.01, -0.5]));

    this.benchBar = this.barbell();
    this.squatBar = this.barbell();
    this.rackBench();
    this.rackSquat();

    // the rest countdown, floating over Marlo
    const timer = document.createElement("div");
    timer.className = "label";
    timer.style.cssText = "font: 600 13px ui-monospace, Menlo, monospace; text-align: center";
    this.timerEl = timer;
    this.timer = new CSS2DObject(timer);
    this.timer.position.set(0, 1.75, 0);
    this.timer.visible = false;
    fly.root.add(this.timer);

    // start lying on the bench, bar unracked
    this.fly = fly;
    this.at = 0;
    this.lieDown(1);
    this.unrackBench();
    this.ready = true;
  },

  barbell() {
    const THREE = this.THREE;
    const g = new THREE.Group();
    g.add(cylinder(0.022, 1.15, 0xcfd2d6, [0, 0, 0], [0, 0, Math.PI / 2]));      // along local x
    for (const s of [-1, 1]) {
      g.add(cylinder(0.15, 0.05, 0x1b1b1f, [s * 0.47, 0, 0], [0, 0, Math.PI / 2]));
      g.add(cylinder(0.11, 0.04, 0xb3261e, [s * 0.52, 0, 0], [0, 0, Math.PI / 2]));
      g.add(cylinder(0.03, 0.03, 0x8a8f99, [s * 0.42, 0, 0], [0, 0, Math.PI / 2])); // collar
    }
    g.traverse((o) => (o.castShadow = true));
    return g;
  },

  // a point in an area's frame, in the station group's frame (where the fly walks)
  inGroup(area, [x, y, z]) {
    return this.V(x, y, z).applyAxisAngle(this.V(0, 1, 0), area.rotation.y).add(area.position);
  },

  // --- bench ---------------------------------------------------------------
  rackBench() {
    this.benchArea.add(this.benchBar);
    this.benchBar.position.set(BENCH.x, BENCH.lock, 0);
    this.benchBar.rotation.set(0, Math.PI / 2, 0);
  },
  unrackBench() {
    this.fly.root.add(this.benchBar);                 // root frame: x across the bench, z towards the head
    this.benchBar.rotation.set(0, 0, 0);
    this.setBench(BENCH.lock);
  },
  // lying on its back: front and middle legs on the bar, hind legs braced against the bench
  setBench(barY, wobble = 0) {
    const { fly, V } = this;
    this.benchBar.position.set(wobble, barY, 0);
    for (const [side, sx] of SIDES) {
      fly.setFoot(side, 0, V(sx * 0.17 + wobble, barY, 0), V(sx, -0.6, 0));
      fly.setFoot(side, 1, V(sx * 0.3 + wobble, barY, 0), V(sx, -0.6, 0));
      fly.setFoot(side, 2, V(sx * 0.27, 0.22, -0.32), V(sx * 0.5, 1, 0));
    }
  },
  // u = 0: standing at the approach spot, u = 1: on its back on the bench
  lieDown(u) {
    const { fly } = this, e = ease(u);
    const yaw = this.benchArea.rotation.y - Math.PI / 2;                          // head towards the uprights
    fly.root.position.lerpVectors(this.inGroup(this.benchArea, BENCH.approach), this.inGroup(this.benchArea, [BENCH.x, 0, 0]), e);
    fly.root.rotation.y = yaw;
    fly.body.rotation.set(0, 0, Math.PI * e);
    fly.body.position.set(0, 0.85 * e + Math.sin(Math.PI * u) * 0.2, 0);
    fly.rest();
  },
  async benchRep(failed) {
    const y = (t) => this.setBench(t);
    await tween(420, (t) => y(lerp(BENCH.lock, BENCH.chest, ease(t))));            // lower to the chest
    if (!failed) return tween(520, (t) => y(lerp(BENCH.chest, BENCH.lock, ease(t)))); // press
    await tween(700, (t) => this.setBench(lerp(BENCH.chest, 0.8, Math.sin(t * Math.PI / 2)), Math.sin(t * 70) * 0.012));
    await tween(220, (t) => y(lerp(0.8, BENCH.chest, t)));                          // pinned
    await this.fly.buzz(450);
    await tween(800, (t) => this.setBench(lerp(BENCH.chest, BENCH.lock, ease(t)), Math.sin(t * 50) * 0.006));
  },

  // --- squat ---------------------------------------------------------------
  rackSquat() {
    this.squatArea.add(this.squatBar);
    this.squatBar.position.set(SQUAT.hookX, SQUAT.hookY, 0);
    this.squatBar.rotation.set(0, Math.PI / 2, 0);
  },
  unrackSquat() {
    this.fly.body.add(this.squatBar);                 // body frame: across the back, just above the thorax
    this.squatBar.position.set(0, 0.47, -0.03);
    this.squatBar.rotation.set(0, 0, 0);
  },
  // depth 0: standing tall, 1: deep squat. Upright on the hind legs, front and middle legs hold the bar.
  setSquat(depth, { lean, height, shake = 0 } = {}) {
    const { fly, V } = this;
    const l = lean ?? lerp(0.22, 0.62, depth);
    const H = height ?? lerp(0.6, 0.38, depth);
    fly.body.rotation.set(-Math.PI / 2 + l, 0, shake);
    fly.body.position.set(0, H - 0.32 * Math.sin(l), -0.02 + 0.32 * Math.cos(l));
    for (const [side, s] of SIDES) {
      const sx = -s; // no roll here: body-left stays on the left
      fly.setFoot(side, 2, V(sx * 0.13, 0, 0.05), V(sx * 0.4, 0.3, 1));            // knees forward over the toes
      fly.setFootBody(side, 0, V(sx * 0.15, 0.46, -0.03), V(sx * 0.8, 0, -1));      // hands on the bar
      fly.setFootBody(side, 1, V(sx * 0.27, 0.46, -0.03), V(sx * 0.8, 0, -1));
    }
  },
  // u = 0: standing level at the approach spot, u = 1: upright under the bar
  standUp(u) {
    const { fly } = this, e = ease(u);
    fly.root.position.lerpVectors(this.inGroup(this.squatArea, SQUAT.approach), this.inGroup(this.squatArea, [0, 0, 0]), Math.min(1, e * 1.6));
    fly.root.rotation.y = this.squatArea.rotation.y + Math.PI / 2;                  // facing out of the rack
    const l = lerp(Math.PI / 2, 0.22, e);
    const H = lerp(0.32, 0.6, e);
    fly.body.rotation.set(-Math.PI / 2 + l, 0, 0);
    fly.body.position.set(0, H - 0.32 * Math.sin(l), lerp(0, -0.02, e) + 0.32 * Math.cos(l) * e);
    fly.rest();
    if (e > 0.6) this.setSquat(0, { lean: l, height: H });
  },
  async squatRep(failed) {
    await tween(480, (t) => this.setSquat(ease(t)));                                 // down
    if (!failed) return tween(580, (t) => this.setSquat(1 - ease(t)));               // drive up
    await tween(650, (t) => this.setSquat(1 - 0.35 * Math.sin(t * Math.PI / 2), { shake: Math.sin(t * 60) * 0.03 }));
    await tween(250, (t) => this.setSquat(lerp(0.65, 1.1, t)));                      // buckles
    await this.fly.buzz(450);
    await tween(850, (t) => this.setSquat(1.1 - 1.1 * ease(t), { shake: Math.sin(t * 40) * 0.015 }));
  },

  // --- moving between stations ------------------------------------------------
  async getUp() {
    if (!this.ready) return;
    this.ready = false;
    if (this.at === 0) {
      await tween(250, (t) => this.setBench(BENCH.lock + 0.04 * Math.sin(t * Math.PI)));
      this.rackBench();
      await tween(600, (t) => this.lieDown(1 - t));
    } else {
      await tween(250, (t) => this.setSquat(0, { height: 0.6 - 0.03 * Math.sin(t * Math.PI) }));
      this.rackSquat();
      await tween(600, (t) => this.standUp(1 - t));
    }
    this.fly.reset();
  },
  async setUp() {
    if (this.ready) return;
    if (this.at === 0) {
      await tween(650, (t) => this.lieDown(t));
      this.unrackBench();
      await tween(250, (t) => this.setBench(BENCH.lock + 0.04 * Math.sin(t * Math.PI)));
    } else {
      await tween(650, (t) => this.standUp(t));
      this.unrackSquat();
      await tween(250, (t) => this.setSquat(0, { height: 0.6 + 0.03 * Math.sin(t * Math.PI) }));
    }
    this.ready = true;
  },
  async goTo(station) {
    await this.getUp();
    const spec = station === 0 ? BENCH : SQUAT;
    await this.fly.walkTo(this.inGroup(station === 0 ? this.benchArea : this.squatArea, spec.approach), 1000);
    this.at = station;
    await this.setUp();
  },

  // rest m minutes (sim time): rack the bar, stand and breathe while the clock runs down at 20x
  async rest(m) {
    const total = (m * 60 * 1000) / SIM_SPEED, start = performance.now();
    const show = (simLeft) => {
      const s = Math.max(0, Math.ceil(simLeft));
      this.timerEl.innerHTML = `<b>REST ${Math.floor(s / 60)}:${String(s % 60).padStart(2, "0")}</b>` +
        `<span class="cap">${m} min \u00b7 ${SIM_SPEED}\u00d7 speed</span>`;
    };
    show(m * 60);
    this.timer.visible = true;
    await this.getUp();
    const { fly } = this;
    fly.root.rotation.y = 0;                                                       // face the camera, catch breath
    const left = Math.max(300, total - (performance.now() - start));
    let buzzed = false;
    await tween(left, (t) => {
      const elapsed = performance.now() - start;
      show(m * 60 * (1 - elapsed / total));
      fly.body.scale.set(1 + 0.03 * Math.sin(elapsed / 180), 1 - 0.03 * Math.sin(elapsed / 180), 1);  // panting
      if (!buzzed && t > 0.5) { buzzed = true; fly.buzz(300); }                    // shake it out
    });
    fly.body.scale.set(1, 1, 1);
    this.timer.visible = false;
  },

  async play(action, caption, { fly }) {
    this.fly = fly;
    const rest = action.match(/^rest_(\d+)min$/);
    if (rest) return this.rest(Number(rest[1]));
    // the caption names the station the brain thinks it's at (also catches a reset after an episode)
    const named = caption.includes("squat") ? 1 : caption.includes("bench") ? 0 : null;
    if (action === "do_rep") {
      if (named != null && named !== this.at) await this.goTo(named);
      await this.setUp();
      const failed = caption.includes("fail");
      await (this.at === 0 ? this.benchRep(failed) : this.squatRep(failed));
    } else {
      await this.goTo(named ?? 1 - this.at);
    }
    await wait(120);
  },
};
