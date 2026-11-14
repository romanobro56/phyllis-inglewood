import { tween, wait, box, cylinder } from "/web/lib/fly.js";

export default {
  title: "Mona Lisa fly",
  blurb: "64 daubs, eight pigments, one connectome. Repeated inference, not live learning.",

  build({ THREE, group, fly }) {
    this.THREE = THREE;
    this.ready = fetch('/flies/painting/reference.json').then(r => {
      if (!r.ok) throw new Error('Painting reference unavailable');
      return r.json();
    }).then(data => { this.data = data; this.clear(); });
    group.add(box(3.8, 0.08, 3.3, 0x302b29, [0, 0.04, 0]));
    for (const x of [-0.65, 0.65]) {
      const leg = box(0.10, 2.8, 0.12, 0x80512e, [x, 1.4, -0.35]);
      leg.rotation.z = x > 0 ? 0.12 : -0.12;
      group.add(leg);
    }
    group.add(box(0.12, 2.4, 0.12, 0x58371e, [0, 1.2, -0.85]));
    group.add(box(1.75, 0.12, 0.4, 0xa4713c, [0, 0.8, -0.25]));
    group.add(box(1.7, 2.05, 0.12, 0xd2aa61, [0, 1.85, -0.32]));
    this.cells = Array.from({length:64}, (_, i) => {
      const cell = box(0.181, 0.225, 0.025, 0x808080,
        [-0.65 + (i % 8) * 0.186, 2.64 - Math.floor(i / 8) * 0.23, -0.24]);
      group.add(cell);
      return cell;
    });
    // A framed, locally stored public-domain reference, never painted by JS.
    group.add(box(0.92, 1.35, 0.08, 0xa4713c, [-1.45, 1.8, -0.5]));
    const texture = new THREE.TextureLoader().load('/flies/painting/source.jpg');
    texture.colorSpace = THREE.SRGBColorSpace;
    const reference = new THREE.Mesh(new THREE.PlaneGeometry(0.80, 1.21),
      new THREE.MeshBasicMaterial({map:texture}));
    reference.position.set(-1.45, 1.8, -0.45);
    group.add(reference);
    group.add(cylinder(0.45, 0.08, 0xb48856, [1.15, 0.8, 0.25]));
    this.swatches = Array.from({length:8}, (_, i) => {
      const angle = i * Math.PI / 4;
      const swatch = cylinder(0.08, 0.02, 0x808080,
        [1.15 + 0.30*Math.cos(angle), 0.855, 0.25 + 0.30*Math.sin(angle)]);
      group.add(swatch);
      return swatch;
    });
    this.brush = new THREE.Group();
    this.brush.add(cylinder(0.018, 0.38, 0x9d6535));
    this.tip = cylinder(0.033, 0.10, 0x222222, [0, 0.24, 0]);
    this.brush.add(this.tip);
    this.brush.rotation.x = -0.6;
    group.add(this.brush);
    this.brush.position.set(1.15, 1, 0.25);
    this.labelCanvas = document.createElement('canvas');
    this.labelCanvas.width = 768; this.labelCanvas.height = 128;
    this.labelTexture = new THREE.CanvasTexture(this.labelCanvas);
    const label = new THREE.Mesh(new THREE.PlaneGeometry(2.8, 0.47),
      new THREE.MeshBasicMaterial({map:this.labelTexture}));
    label.position.set(0, 3.15, -0.2);
    group.add(label);
    this.label('Blank canvas • inference only');
    fly.root.position.set(0, 0.1, 1.0);
    fly.face({x:0, z:-0.3});
  },

  color(rgb) { return new this.THREE.Color(`rgb(${rgb.join(',')})`); },
  clear() {
    this.cells.forEach(c => c.material.color.copy(this.color(this.data.blank)));
    this.swatches.forEach((c,i) => c.material.color.copy(this.color(this.data.palette[i])));
  },
  label(text) {
    const ctx = this.labelCanvas.getContext('2d');
    ctx.fillStyle = '#181b20'; ctx.fillRect(0,0,768,128);
    ctx.fillStyle = '#f4dfab'; ctx.font = '30px sans-serif';
    ctx.textAlign = 'center'; ctx.fillText(text,384,75);
    this.labelTexture.needsUpdate = true;
  },

  async play(action, caption, { fly }) {
    await this.ready;
    // Every supported action maps directly to that pigment; no color correction.
    const pigment = {
      paint_0:0, paint_1:1, paint_2:2, paint_3:3,
      paint_4:4, paint_5:5, paint_6:6, paint_7:7,
    }[action];
    const match = /cell (\d+)\/64; similarity ([\d.]+)%/.exec(caption);
    if (pigment === undefined || !match) throw new Error('Invalid painting decision');
    const count = Number(match[1]);
    if (count < 1 || count > 64) throw new Error('Invalid painting cell');
    if (count === 1) this.clear();
    const final = count === 64;
    const cell = this.cells[count-1];
    const color = this.color(this.data.palette[pigment]);
    this.tip.material.color.copy(color);
    fly.face({x:cell.position.x,z:-0.3});
    const flyFrom = fly.root.position.clone();
    const flyTo = new this.THREE.Vector3(cell.position.x, cell.position.y - 0.45, 0.22);
    await Promise.all([
      tween(final ? 80 : 350, t => fly.root.position.lerpVectors(flyFrom,flyTo,t)),
      fly.buzz(final ? 80 : 350),
    ]);
    const from = this.brush.position.clone();
    const to = cell.position.clone().add(new this.THREE.Vector3(0,-0.20,0.14));
    await tween(final ? 90 : 450, t => this.brush.position.lerpVectors(from,to,t));
    await tween(final ? 90 : 400, t => {
      this.brush.position.x = to.x + Math.sin(t*Math.PI)*0.06;
      if (t >= 0.5) cell.material.color.copy(color);
    });
    // Only the chosen cell receives the chosen pigment.
    cell.material.color.copy(color);
    this.label(`${final ? 'Finished' : `${count}/64 daubs`} • similarity ${match[2]}%`);
    if (final) {
      await fly.walkTo({x:0,z:1.15},180);
      fly.face({x:0,z:-0.3});
      await wait(3000); // Total final animation 3440ms < viewer's 4000ms watchdog.
      this.clear();
      this.label('Fresh canvas • inference only');
    } else {
      await tween(300, t => { this.brush.position.z = to.z + 0.25*t; });
    }
  },
};
