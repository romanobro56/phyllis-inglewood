// Native station: the host owns the actual Fly, camera, lights and neural inspector.
import { tween, box, cylinder } from '/web/lib/fly.js';
import { parseCaption, Playback, SLOT_COUNT, PITCH } from './logic.js';

const C = { ink: 0x11232d, brass: 0xc7a66b, mint: 0x62debe, red: 0xf36e7d, ivory: 0xf4eedc };
const HOME = { x: 0, z: 1.65 };
const ease = t => t * t * (3 - 2 * t);

export default {
  title: 'Pi Fly',
  blurb: 'A connectome learning a finite prefix, one carried digit at a time.',

  build({ THREE, group, fly }) {
    Object.assign(this, { THREE, group, fly, playback: new Playback(), serial: 0, replenishments: 0, advances: 0 });
    this.deck = [];
    this.slots = [];
    const plinth = box(4.9, .18, 4.15, C.ink, [0, .09, .15]);
    plinth.material.roughness = .32;
    plinth.material.metalness = .35;
    group.add(plinth);
    group.add(box(4.65, .025, .025, C.brass, [0, .19, -1.77]));
    group.add(box(4.65, .025, .025, C.brass, [0, .19, 2.05]));
    for (const x of [-2.2, 2.2]) for (const z of [-1.55, 1.85])
      group.add(cylinder(.09, .20, C.brass, [x, .12, z]));
    const heading = this.text('π   /   THE DIGIT ATELIER', 3.3, .32, '#d8c197');
    heading.position.set(0, .43, -1.52);
    group.add(heading);
    const subtitle = this.text('TWO NEURAL CHOICES  ·  ONE DIGIT', 2.9, .18, '#91a9b5');
    subtitle.position.set(0, .22, 1.98);
    group.add(subtitle);
    // Persistent home sockets. Every physical tile has a digit identity and home.
    for (let d = 0; d < 10; d++) {
      const home = this.home(d);
      const socket = box(.60, .025, .64, 0x1d3641, [home.x, .2, home.z]);
      group.add(socket);
      this.deck[d] = this.tile(d);
      this.deck[d].position.copy(home);
      group.add(this.deck[d]);
    }
    for (let i = 0; i < SLOT_COUNT; i++) this.slots.push(this.slot(i));
    this.stats = this.text('', 4.1, .23, '#b4cbd1');
    this.stats.position.set(0, .28, -0.31);
    group.add(this.stats);
    this.feedback = this.text('Choose a group: 0–4 or 5–9', 3.5, .22, '#7edbc4');
    this.feedback.position.set(0, .29, 1.56);
    group.add(this.feedback);
    fly.root.position.set(HOME.x, .20, HOME.z);
    fly.face({ x: 0, z: -1 });
    this.setText(this.stats, '0 DIGITS    /    STREAK 0    /    MISSES 0    /    REWARD 0.00');
    this.assertIdle();
  },

  home(d) { return new this.THREE.Vector3(((d % 5) - 2) * .83, .30, d < 5 ? .34 : 1.06); },

  text(value, width, height, color = '#f4eedc') {
    const canvas = document.createElement('canvas');
    canvas.width = Math.ceil(128 * width / height); canvas.height = 128;
    const texture = new this.THREE.CanvasTexture(canvas);
    texture.colorSpace = this.THREE.SRGBColorSpace;
    const sprite = new this.THREE.Sprite(new this.THREE.SpriteMaterial({ map: texture, transparent: true, depthWrite: false }));
    sprite.scale.set(width, height, 1);
    sprite.userData.canvas = canvas;
    sprite.userData.color = color;
    this.setText(sprite, value);
    return sprite;
  },

  setText(sprite, value) {
    const canvas = sprite.userData.canvas, ctx = canvas.getContext('2d');
    ctx.clearRect(0, 0, canvas.width, canvas.height);
    ctx.font = '600 72px ui-monospace, SFMono-Regular, monospace';
    ctx.textAlign = 'center'; ctx.textBaseline = 'middle';
    ctx.fillStyle = sprite.userData.color;
    ctx.fillText(value, canvas.width / 2, 64, canvas.width - 20);
    sprite.material.map.needsUpdate = true;
  },

  tile(digit) {
    const tile = new this.THREE.Group();
    tile.userData = { digit, id: ++this.serial, home: this.home(digit) };
    const slab = box(.48, .105, .51, C.ivory);
    slab.material.roughness = .28;
    slab.material.metalness = .1;
    tile.add(slab);
    const glyph = this.text(String(digit), .32, .37, '#132d36');
    glyph.position.set(0, .20, .015);
    tile.add(glyph);
    const edge = box(.35, .01, .018, C.brass, [0, .06, .215]);
    tile.add(edge);
    return tile;
  },

  slot(i) {
    const slot = new this.THREE.Group();
    slot.position.set((i - 3) * PITCH, .20, -.98);
    const base = box(.54, .045, .59, 0x233e48);
    base.material.emissive = new this.THREE.Color(C.mint);
    base.material.emissiveIntensity = i === 6 ? .24 : .02;
    slot.add(base);
    slot.userData.base = base;
    slot.userData.active = i === 6;
    if (i === 6) {
      const mark = this.text('·', .25, .3, '#7bdec4');
      mark.position.y = .12;
      slot.add(mark); slot.userData.mark = mark;
    }
    this.group.add(slot);
    return slot;
  },

  dispose(object) {
    object.removeFromParent();
    object.traverse(o => {
      o.geometry?.dispose();
      const materials = o.material ? (Array.isArray(o.material) ? o.material : [o.material]) : [];
      for (const m of materials) { m.map?.dispose(); m.dispose(); }
    });
  },

  resetScene() {
    for (const slot of this.slots) this.dispose(slot);
    this.slots = Array.from({ length: SLOT_COUNT }, (_, i) => this.slot(i));
    this.replenishments = this.advances = 0;
    this.index = 0;
  },

  async move(fly, target, ms) {
    // Both promises are awaited; the carried object follows the real fly body.
    await Promise.all([fly.walkTo(target, ms), fly.buzz(ms)]);
  },

  async pickup(fly, tile) {
    fly.face(tile.position);
    fly.body.attach(tile); // preserves world transform before lowering into its grip
    const from = tile.position.clone();
    const grip = new this.THREE.Vector3(0, .14, .34);
    await tween(150, t => tile.position.lerpVectors(from, grip, ease(t)));
  },

  async lower(tile, parent, target) {
    parent.attach(tile);
    tile.rotation.set(0, 0, 0);
    const from = tile.position.clone();
    await tween(160, t => tile.position.lerpVectors(from, target, ease(t)));
  },

  async shift() {
    const starts = this.slots.map(s => s.position.x);
    await tween(300, t => this.slots.forEach((s, i) => { s.position.x = starts[i] - PITCH * ease(t); }));
    this.dispose(this.slots.shift());
    this.slots.push(this.slot(6)); // exactly one new destination
    this.advances++;
  },

  assertIdle() {
    if (this.deck.length !== 10 || new Set(this.deck).size !== 10 || this.deck.some((tile, d) =>
      !tile || tile.userData.digit !== d || tile.parent !== this.group || tile.position.distanceTo(this.home(d)) > .00001))
      throw new Error('Pi Fly deck invariant violated');
    if (this.slots.length !== SLOT_COUNT || this.slots.some((slot, i) => Math.abs(slot.position.x - (i - 3) * PITCH) > .00001))
      throw new Error('Pi Fly strip invariant violated');
  },

  play(action, caption, { fly }) {
    const event = parseCaption(action, caption);
    return this.playback.enqueue(event, async (e, fresh) => {
      if (fresh) this.resetScene();
      if (e.kind === 'group') {
        this.setText(this.stats, `${e.index} DIGITS    /    STREAK ${e.streak}    /    MISSES ${e.wrong}    /    REWARD ${e.reward.toFixed(2)}`);
        this.setText(this.feedback, `GROUP ${e.group === 0 ? '0–4' : '5–9'}  ·  choosing an offset`);
        await fly.buzz(180);
        return;
      }
      const previousIndex = e.index - (e.kind === 'correct' ? 1 : 0);
      if (previousIndex !== (this.index || 0)) throw new Error('Pi Fly sequence out of sync');
      const tile = this.deck[e.digit], home = this.home(e.digit), destination = this.slots.at(-1);
      this.setText(this.feedback, `CARRYING ${e.digit}  →  POSITION ${previousIndex + 1}`);
      await this.move(fly, { x: home.x, z: home.z + .32 }, 380);
      await this.pickup(fly, tile);
      await this.move(fly, { x: destination.position.x, z: destination.position.z + .34 }, 520);
      await this.lower(tile, destination, new this.THREE.Vector3(0, .12, 0));
      const success = e.kind === 'correct';
      destination.userData.mark.visible = false;
      const mat = destination.userData.base.material;
      mat.emissive.setHex(success ? C.mint : C.red);
      this.setText(this.feedback, success ? `${e.digit} ACCEPTED  ·  ${e.index} VERIFIED DIGITS` : `${e.digit} REJECTED  ·  returning the same tile`);
      await tween(250, t => { mat.emissiveIntensity = .3 + Math.sin(t * Math.PI) * 1.5; });
      if (success) {
        destination.userData.active = false;
        destination.userData.index = e.index - 1;
        destination.userData.digit = e.digit;
        if (e.index === 1) {
          const dot = this.text('.', .13, .28, '#eadab6');
          dot.position.set(.265, .16, 0); destination.add(dot);
        }
        await this.shift();
        this.deck[e.digit] = this.tile(e.digit);
        this.deck[e.digit].position.copy(home);
        this.group.add(this.deck[e.digit]);
        this.replenishments++;
        this.index = e.index;
      } else {
        // Retrieve that exact physical tile; do not construct a replacement.
        this.group.attach(tile);
        await this.pickup(fly, tile);
        await this.move(fly, { x: home.x, z: home.z + .32 }, 520);
        await this.lower(tile, this.group, home);
        destination.userData.mark.visible = true;
        await tween(100, t => { mat.emissiveIntensity = .3 * (1 - t); });
        mat.emissive.setHex(C.mint); mat.emissiveIntensity = .24;
      }
      this.setText(this.stats, `${e.index} DIGITS    /    STREAK ${e.streak}    /    MISSES ${e.wrong}    /    REWARD ${e.reward.toFixed(2)}`);
      await this.move(fly, HOME, 300);
      fly.face({ x: 0, z: -1 });
      this.assertIdle();
      if (e.status !== 'live') this.setText(this.feedback, e.status === 'complete' ? 'FINITE PREFIX COMPLETE  ·  next episode' : 'ATTEMPT LIMIT  ·  next episode');
    });
  },
};
