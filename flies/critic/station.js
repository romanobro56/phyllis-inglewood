// Critic station: Gordo Inglewood sits at a little dinner table in a napkin bib, fork in one front leg and
// knife in the other. A dish lands on the plate, Gordo cuts a bite, tastes it and reacts.
// In the world you pick the dish from the menu strip under the brain panel; Gordo only eats what it's fed.
// For the precise 0-10 score, run the tasting room: python flies/critic/tasting.py
import { tween, wait, box, cylinder } from "/web/lib/fly.js";

const DISH = {
  banana: 0xffe135, mango: 0xffa62b, grapes: 0x7b3fa0, peach: 0xffb07c, strawberry: 0xe8283c,
  cherry: 0xb1102b, blueberries: 0x3b4cc0, pineapple: 0xf2c94c, watermelon: 0xf25c6e, apple: 0xd62d20,
  orange: 0xff8c1a, lemon: 0xfff44f, corn: 0xf7d046, tomato: 0xe5322d, carrot: 0xed7d31,
  cucumber: 0x5c9e3f, "bell pepper": 0x3e9e3e, lettuce: 0x8fd16a, potato: 0xc9a66b, broccoli: 0x2f7d32,
  onion: 0xe8d9c4, "jalapeño": 0x2e8b2e, garlic: 0xf3efe2, burger: 0x8b5a2b, bacon: 0xc0504d,
  sausage: 0xa0442c, steak: 0x8b2e2e, "chicken leg": 0xd29a5c, sashimi: 0xff8a6b,
  "overripe banana": 0x9c7a1e, "fermented apple": 0x7a3b22, "compost heap": 0x4f4a2a,
  "rotten meat": 0x5e6b3a, "rotten egg": 0xbfbf8a, "spoiled milk": 0xe8e8c8, "moldy bread": 0x8aa37a,
  "cow pie": 0x5a3d1e, "dog poop": 0x6b4423, "horse apples": 0x7a5a2e, "bird droppings": 0xe6e6dc,
};
const VERDICTS = ["spit_it_out", "revolting", "bad", "meh", "decent", "good", "delicious", "chefs_kiss"];

// Gordo's chair is at x = -1.08 facing +x (the camera sees him side-on); the plate is in front of him.
const SEAT = { x: -1.08, top: 0.32 };
const PLATE = { x: -0.58, y: 0.56 };
const TABLE_TOP = 0.54;
const H = 0.75;          // thorax height while sitting upright
const TZ = 0.02;         // thorax position forward of the chair centre
const lerp = (a, b, t) => a + (b - a) * t;
const ease = (t) => (t < 0.5 ? 2 * t * t : 1 - (-2 * t + 2) ** 2 / 2);

// Poses in the fly's own (root) frame: x = Gordo's left/right, y up, z forward towards the plate.
// hand = where the leg holds the utensil, pitch = utensil tilt (0 level, + tines down, ~-2.7 tines back to the mouth)
const REST = { lean: 0.15, bob: 0, fork: { hand: [-0.13, 0.64, 0.28], pitch: 0.45 }, knife: { hand: [0.13, 0.64, 0.28], pitch: 0.45 } };
const REACH = { lean: 0.42, bob: 0, fork: { hand: [-0.04, 0.73, 0.37], pitch: 0.65 }, knife: { hand: [0.06, 0.73, 0.36], pitch: 0.75 } };
const MOUTH = { lean: 0.15, bob: 0, fork: { hand: [-0.04, 0.8, 0.3], pitch: -2.68 }, knife: { hand: [0.13, 0.66, 0.27], pitch: 0.4 } };
const HANDS_UP = { lean: 0.05, bob: 0, fork: { hand: [-0.2, 0.95, 0.18], pitch: -1.2 }, knife: { hand: [0.2, 0.95, 0.18], pitch: -1.2 } };

export default {
  title: "Gordo Inglewood",
  blurb: "Gordo Inglewood, a fruit-fly food critic. Fruit? Chef's kiss. Garlic? Spits it out.",

  build({ THREE, group, fly }) {
    this.THREE = THREE;
    this.V = (x, y, z) => new THREE.Vector3(x, y, z);
    const mat = (color, extra = {}) => new THREE.MeshStandardMaterial({ color, roughness: 0.5, ...extra });
    // table with a tablecloth that hangs over the edges
    group.add(box(1.8, 0.08, 1.2, 0xf4f1ea, [0, 0.5, 0]));
    group.add(box(1.84, 0.12, 1.24, 0xf4f1ea, [0, 0.44, 0]));
    for (const [x, z] of [[-0.8, -0.5], [0.8, -0.5], [-0.8, 0.5], [0.8, 0.5]])
      group.add(box(0.07, 0.38, 0.07, 0x5b3a1e, [x, 0.19, z]));
    group.add(cylinder(0.25, 0.025, 0xffffff, [PLATE.x, PLATE.y, 0]));                          // plate
    group.add(cylinder(0.18, 0.026, 0xeeeeee, [PLATE.x, PLATE.y + 0.002, 0]));
    // a wine glass, a candle, a folded menu
    group.add(cylinder(0.008, 0.12, 0xdfe7ef, [-0.35, 0.6, -0.36]));
    const bowl = new THREE.Mesh(new THREE.CylinderGeometry(0.06, 0.035, 0.1, 16, 1, true),
      mat(0xdfe7ef, { transparent: true, opacity: 0.45, side: THREE.DoubleSide, roughness: 0.1 }));
    bowl.position.set(-0.35, 0.71, -0.36);
    group.add(bowl, cylinder(0.045, 0.006, 0xdfe7ef, [-0.35, 0.543, -0.36]), cylinder(0.04, 0.04, 0x7a1020, [-0.35, 0.68, -0.36]));
    group.add(cylinder(0.025, 0.2, 0xfff6e0, [0.1, 0.64, -0.3]));
    this.flame = new THREE.Mesh(new THREE.SphereGeometry(0.018, 8, 6), mat(0xffb030, { emissive: 0xff8a00, emissiveIntensity: 2 }));
    this.flame.scale.set(1, 1.8, 1);
    this.flame.position.set(0.1, 0.76, -0.3);
    group.add(this.flame);
    const card = box(0.16, 0.12, 0.005, 0x2b1d14, [0.25, 0.6, 0.35]);
    card.rotation.x = -0.35;
    group.add(card);

    // Gordo's chair (with a backrest behind him)
    group.add(box(0.4, 0.04, 0.4, 0x6b4423, [SEAT.x, SEAT.top - 0.02, 0]));
    for (const [dx, dz] of [[-0.17, -0.17], [0.17, -0.17], [-0.17, 0.17], [0.17, 0.17]])
      group.add(box(0.035, SEAT.top - 0.04, 0.035, 0x5b3a1e, [SEAT.x + dx, (SEAT.top - 0.04) / 2, dz]));
    for (const dz of [-0.17, 0.17]) group.add(box(0.035, 0.6, 0.035, 0x5b3a1e, [SEAT.x - 0.19, SEAT.top + 0.28, dz]));
    group.add(box(0.035, 0.22, 0.38, 0x6b4423, [SEAT.x - 0.19, SEAT.top + 0.48, 0]));

    this.dish = new THREE.Mesh(new THREE.SphereGeometry(0.13, 20, 14), mat(0xffe135));
    this.dish.position.set(PLATE.x, PLATE.y + 0.11, 0);
    this.dish.castShadow = true;
    this.dish.visible = false;                                          // empty plate until someone feeds it
    group.add(this.dish);
    // score bar at the far end of the table: grows and turns green with the verdict
    group.add(box(0.06, 1.0, 0.06, 0x444444, [0.95, 0.95, -0.45]));
    this.meter = box(0.12, 1, 0.12, 0x3cff7a, [0.95, 0.5, -0.3]);
    this.meter.scale.y = 0.01;
    group.add(this.meter);

    // Gordo: seated, napkin bib on, cutlery in hand
    this.fly = fly;
    fly.root.position.set(SEAT.x, 0, 0);
    fly.root.rotation.y = Math.PI / 2;
    // napkin tucked in under the head, draped over the chest (the chest faces the table when sitting up)
    const bib = new THREE.Mesh(new THREE.SphereGeometry(0.142, 18, 10, 0, Math.PI * 2, Math.PI * 0.6, Math.PI * 0.4),
      mat(0xffffff, { side: THREE.DoubleSide, roughness: 0.9 }));
    bib.position.set(0, 0.32, 0);
    bib.rotation.x = -Math.PI / 4;
    fly.body.add(bib);
    this.fork = this.utensil("fork");
    this.knife = this.utensil("knife");
    this.bite = new THREE.Mesh(new THREE.SphereGeometry(0.03, 10, 8), mat(0xffe135));
    this.bite.position.set(0, 0, 0.16);
    this.bite.visible = false;
    this.fork.add(this.bite);
    fly.root.add(this.fork, this.knife);
    this.pose = structuredClone(REST);
    this.apply();
  },

  // cutlery points along its local +z from the hand
  utensil(kind) {
    const THREE = this.THREE;
    const steel = new THREE.MeshStandardMaterial({ color: 0xc3c9d1, metalness: 0.85, roughness: 0.25 });
    const grip = new THREE.MeshStandardMaterial({ color: 0x1e1a1a, roughness: 0.6 });
    const g = new THREE.Group();
    g.scale.setScalar(1.45);                                   // big enough to read from the camera
    const handle = new THREE.Mesh(new THREE.CylinderGeometry(0.011, 0.009, 0.1, 8).rotateX(Math.PI / 2).translate(0, 0, 0.0), grip);
    g.add(handle, new THREE.Mesh(new THREE.CylinderGeometry(0.006, 0.006, 0.05, 6).rotateX(Math.PI / 2).translate(0, 0, 0.07), steel));
    if (kind === "fork") {
      g.add(new THREE.Mesh(new THREE.BoxGeometry(0.036, 0.006, 0.025).translate(0, 0, 0.095), steel));
      for (const x of [-0.013, 0, 0.013])
        g.add(new THREE.Mesh(new THREE.BoxGeometry(0.005, 0.005, 0.05).translate(x, 0, 0.13), steel));
    } else {
      g.add(new THREE.Mesh(new THREE.BoxGeometry(0.004, 0.028, 0.11).translate(0, -0.006, 0.135), steel));
    }
    g.traverse((o) => (o.castShadow = true));
    return g;
  },

  // put Gordo in this.pose: body upright on the chair, legs placed, cutlery in hand
  apply() {
    const { fly, V } = this, p = this.pose, l = p.lean;
    fly.body.rotation.set(-Math.PI / 2 + l, 0, p.roll || 0);
    fly.body.position.set(0, H + p.bob - 0.32 * Math.sin(l), TZ + 0.32 * Math.cos(l));
    for (const [side, sx] of [["L", -1], ["R", 1]]) {
      fly.setFoot(side, 1, V(sx * 0.25, TABLE_TOP + 0.005, 0.22), V(sx, 0.6, 0));      // elbows on the table
      fly.setFoot(side, 2, V(sx * 0.18, 0.16, 0.16), V(sx * 0.3, 0.3, 1));             // dangling from the chair
    }
    for (const [obj, u, side, sx] of [[this.fork, p.fork, "L", -1], [this.knife, p.knife, "R", 1]]) {
      const hand = V(...u.hand);
      fly.setFoot(side, 0, hand, V(sx, -0.6, -0.4));
      obj.position.copy(hand);
      obj.rotation.set(u.pitch, sx * 0.12, 0);
    }
  },

  // tween from the current pose to `to` (any of REST / REACH / ...), with an optional per-frame extra
  async moveTo(to, ms, extra) {
    const from = structuredClone(this.pose);
    await tween(ms, (t) => {
      const e = ease(t), p = this.pose;
      p.lean = lerp(from.lean, to.lean, e);
      p.bob = lerp(from.bob, to.bob, e);
      for (const k of ["fork", "knife"]) {
        p[k].hand = from[k].hand.map((v, i) => lerp(v, to[k].hand[i], e));
        p[k].pitch = lerp(from[k].pitch, to[k].pitch, e);
      }
      extra?.(t, p);
      this.apply();
    });
  },

  serve(caption) {
    const name = (caption.split(":")[0] || "").replace(/^\S+\s/, "").trim();
    const color = DISH[name] ?? 0x999999;
    this.dish.material.color.setHex(color);
    this.bite.material.color.setHex(color);
    this.dish.scale.setScalar(1);
    this.dish.position.set(PLATE.x, PLATE.y + 0.11, 0);
    this.dish.visible = true;
  },

  // cut a piece, fork it, bring it to the mouth, chew
  async taste() {
    await this.moveTo(REACH, 300);
    const saw = structuredClone(REACH);
    await this.moveTo(saw, 380, (t, p) => (p.knife.hand[2] = REACH.knife.hand[2] + Math.sin(t * Math.PI * 6) * 0.04));
    this.bite.visible = true;
    this.dish.scale.multiplyScalar(0.94);
    await this.moveTo(MOUTH, 380);
    this.bite.visible = false;
    await this.moveTo(MOUTH, 320, (t, p) => (p.bob = Math.abs(Math.sin(t * Math.PI * 3)) * -0.02));   // chew
  },

  async setMeter(value) {
    const from = this.meter.scale.y;
    const to = Math.max(0.01, value / 10);
    const c = new this.THREE.Color().setHSL((value / 10) * 0.33, 0.9, 0.5); // red -> green
    this.meter.material.color.copy(c);
    await tween(400, (t) => {
      this.meter.scale.y = from + (to - from) * t;
      this.meter.position.y = 0.5 + this.meter.scale.y / 2;
    });
  },

  async play(action, caption, { fly }) {
    this.fly = fly;
    this.serve(caption);
    await this.taste();
    const value = (VERDICTS.indexOf(action) * 10) / 7;
    const shake = (amp, n) => (t, p) => (p.roll = amp * Math.sin(t * Math.PI * n) * (1 - t));
    switch (action) {
      case "spit_it_out": { // recoils, flings the dish off the table, throws the cutlery up
        const recoil = { ...structuredClone(HANDS_UP), lean: -0.12 };
        await Promise.all([this.moveTo(recoil, 300), fly.buzz(500),
          tween(500, (t) => this.dish.position.set(PLATE.x + t * 1.0, PLATE.y + 0.11 + Math.sin(t * Math.PI) * 0.5, t * 0.3))]);
        this.dish.visible = false;
        await this.moveTo(REST, 400);
        break;
      }
      case "revolting": // shudders, cutlery down
        await this.moveTo({ ...structuredClone(REST), lean: 0.02 }, 600, shake(0.18, 9));
        this.pose.roll = 0;
        await this.moveTo(REST, 200);
        break;
      case "bad": // pushes the plate away with the knife
        await this.moveTo(REACH, 250);
        await tween(350, (t) => (this.dish.position.x = PLATE.x + 0.25 * ease(t)));
        await this.moveTo({ ...structuredClone(REST), lean: 0.05 }, 400);
        break;
      case "meh": // a shrug: both hands up a little, wings twitch
        await Promise.all([this.moveTo({ ...structuredClone(REST), fork: { hand: [-0.16, 0.74, 0.22], pitch: 0.1 }, knife: { hand: [0.16, 0.74, 0.22], pitch: 0.1 } }, 300), fly.buzz(250)]);
        await this.moveTo(REST, 350);
        break;
      case "decent": // one approving nod
        await this.moveTo(REST, 400, (t, p) => (p.lean = REST.lean + Math.sin(t * Math.PI) * 0.18));
        break;
      case "good": // two nods
        await this.moveTo(REST, 650, (t, p) => (p.lean = REST.lean + Math.abs(Math.sin(t * Math.PI * 2)) * 0.18));
        break;
      case "delicious": // goes straight back in for another bite, wings buzzing
        await Promise.all([this.taste(), fly.buzz(700)]);
        await this.moveTo(REST, 250);
        break;
      case "chefs_kiss": // cleans the plate, kisses the fork, both arms up, bounces on the chair
        await this.taste();
        await tween(300, (t) => this.dish.scale.setScalar(Math.max(0.01, 0.8 * (1 - t))));
        this.dish.visible = false;
        await this.moveTo(MOUTH, 200);
        await Promise.all([fly.buzz(700), this.moveTo(HANDS_UP, 700, (t, p) => (p.bob = Math.abs(Math.sin(t * Math.PI * 3)) * 0.06))]);
        await this.moveTo(REST, 350);
        break;
      default:
        await this.moveTo(REST, 300);
    }
    await Promise.all([this.setMeter(value), this.moveTo(REST, 300)]);
  },
};
