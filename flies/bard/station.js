// Bard station: a fly at a typewriter, writing Shakespeare one word at a time.
// Eight keys carry the eight words on its tray (ctx.view.tray, sent by the server just before the step).
// The fly walks to the key its brain picked, stamps on it, and the word lands on the page behind.
import { tween, wait, box, cylinder } from "/web/lib/fly.js";

const DESK = 0.74;                 // desk top height
const KEY_W = 0.42, KEY_D = 0.3;   // a key is roughly fly-sized
const ROW_Z = [0.42, 0.05];        // front row (slots 0-3), back row (slots 4-7)
const INK = "#2a1c12";

export default {
  title: "Bard fly",
  blurb: "A fruit fly at a typewriter, picking Shakespeare one word at a time.",
  playTimeout: 3000,

  build({ THREE, group, fly }) {
    this.THREE = THREE;
    // desk
    group.add(box(2.4, 0.08, 1.5, 0x5a3a22, [0, DESK - 0.04, 0.1]));
    for (const [x, z] of [[-1.1, -0.55], [1.1, -0.55], [-1.1, 0.75], [1.1, 0.75]])
      group.add(box(0.08, DESK - 0.08, 0.08, 0x3b2615, [x, (DESK - 0.08) / 2, z]));

    // typewriter body behind the keys, with the platen (roller) on top
    group.add(box(2.0, 0.16, 0.42, 0x1f3a2c, [0, DESK + 0.08, -0.38]));
    this.platen = cylinder(0.07, 1.7, 0x111111, [0, DESK + 0.22, -0.48], [0, 0, Math.PI / 2]);
    group.add(this.platen);
    for (const s of [-1, 1]) group.add(cylinder(0.05, 0.08, 0xb8b8b8, [s * 0.9, DESK + 0.22, -0.48], [0, 0, Math.PI / 2]));

    // the page: a canvas texture standing up out of the roller, facing the camera (+z)
    this.pageCanvas = document.createElement("canvas");
    this.pageCanvas.width = 1024;
    this.pageCanvas.height = 760;
    this.pageTex = new THREE.CanvasTexture(this.pageCanvas);
    this.pageTex.colorSpace = THREE.SRGBColorSpace;
    this.pageTex.anisotropy = 4;
    const page = new THREE.Mesh(new THREE.PlaneGeometry(1.3, 0.965),
      new THREE.MeshStandardMaterial({ map: this.pageTex, roughness: 0.9 }));
    page.position.set(0, DESK + 0.22 + 0.47, -0.5);
    page.rotation.x = -0.12;
    page.castShadow = true;
    group.add(page);
    this.words = [];
    this.drawPage("");

    // eight keys: a dark base and an ivory cap with the word on it
    this.keys = [];
    for (let i = 0; i < 8; i++) {
      const x = ((i % 4) - 1.5) * (KEY_W + 0.06);
      const z = ROW_Z[i < 4 ? 0 : 1];
      const key = new THREE.Group();
      key.position.set(x, DESK, z);
      key.add(box(KEY_W, 0.05, KEY_D, 0x1a1a1a, [0, 0.025, 0]));
      const canvas = document.createElement("canvas");
      canvas.width = 256;
      canvas.height = 160;
      const tex = new THREE.CanvasTexture(canvas);
      tex.colorSpace = THREE.SRGBColorSpace;
      const cap = new THREE.Mesh(new THREE.BoxGeometry(KEY_W - 0.04, 0.03, KEY_D - 0.04), [
        ...Array(2).fill(new THREE.MeshStandardMaterial({ color: 0xe8dfc8 })),
        new THREE.MeshStandardMaterial({ map: tex, roughness: 0.6 }),   // +y face: the word
        ...Array(3).fill(new THREE.MeshStandardMaterial({ color: 0xe8dfc8 })),
      ]);
      cap.position.y = 0.065;
      cap.castShadow = cap.receiveShadow = true;
      key.add(cap);
      group.add(key);
      this.keys.push({ key, cap, canvas, tex, word: "" });
      this.label(i, "");
    }

    // a quill in an inkpot, a candle and a small skull (alas)
    group.add(cylinder(0.08, 0.12, 0x101018, [0.95, DESK + 0.06, 0.55]));
    const quill = box(0.02, 0.5, 0.06, 0xf2efe6, [0.97, DESK + 0.32, 0.55]);
    quill.rotation.z = -0.35;
    group.add(quill);
    group.add(cylinder(0.05, 0.22, 0xf3e7c9, [-0.98, DESK + 0.11, 0.5]));
    const flame = new THREE.Mesh(new THREE.SphereGeometry(0.03, 10, 8), new THREE.MeshBasicMaterial({ color: 0xffb347 }));
    flame.position.set(-0.98, DESK + 0.26, 0.5);
    group.add(flame);
    const glow = new THREE.PointLight(0xffa040, 1.2, 3);
    glow.position.copy(flame.position);
    group.add(glow);
    const skull = new THREE.Mesh(new THREE.SphereGeometry(0.11, 18, 14), new THREE.MeshStandardMaterial({ color: 0xe9e2cf, roughness: 0.8 }));
    skull.scale.set(1, 0.9, 1.15);
    skull.position.set(-0.75, DESK + 0.1, 0.6);
    group.add(skull);
    for (const s of [-1, 1]) {
      const eye = new THREE.Mesh(new THREE.SphereGeometry(0.028, 8, 6), new THREE.MeshBasicMaterial({ color: 0x1a1a1a }));
      eye.position.set(-0.75 + s * 0.04, DESK + 0.12, 0.71);
      group.add(eye);
    }

    // the fly starts in the gap between the two rows of keys, facing the page
    fly.root.position.set(0, DESK + 0.1, 0.24);
    fly.root.rotation.y = Math.PI;
    this.home = fly.root.position.clone();
  },

  // word on key i (canvas on the cap's top face; readable from the front)
  label(i, word) {
    const k = this.keys[i];
    k.word = word;
    const c = k.canvas.getContext("2d");
    c.fillStyle = "#efe6cf";
    c.fillRect(0, 0, c.canvas.width, c.canvas.height);
    c.fillStyle = INK;
    c.textAlign = "center";
    c.textBaseline = "middle";
    let size = 64;
    do c.font = `600 ${size}px Georgia, "Times New Roman", serif`;
    while (c.measureText(word).width > 236 && (size -= 4) > 18);
    c.fillText(word, c.canvas.width / 2, c.canvas.height / 2 + 4);   // the top face's texture top is the far edge
    k.tex.needsUpdate = true;
  },

  drawPage(text, fresh = "") {
    const c = this.pageCanvas.getContext("2d");
    const W = c.canvas.width, H = c.canvas.height;
    c.fillStyle = "#f6f0e1";
    c.fillRect(0, 0, W, H);
    c.fillStyle = "rgba(120,90,50,.08)";
    for (let y = 0; y < H; y += 8) c.fillRect(0, y, W, 1);
    c.font = '600 30px Georgia, "Times New Roman", serif';
    c.fillStyle = "#7a5a3a";
    c.fillText("THE TRAGEDIE OF THE FLY", 70, 90);
    c.font = '44px "Courier New", Courier, monospace';
    // wrap; like a real typewriter the newest line sits just above the roller and older lines scroll up
    const lines = [];
    let line = "";
    for (const w of text.split(" ").filter(Boolean)) {
      const next = line ? line + " " + w : w;
      if (c.measureText(next).width > W - 140 && line) { lines.push(line); line = w; } else line = next;
    }
    if (line) lines.push(line);
    const shown = lines.slice(-7);
    const top = H - 290 - (shown.length - 1) * 56;   // newest line clears the fly's head
    shown.forEach((l, j) => {
      const last = j === shown.length - 1, y = top + j * 56;
      c.fillStyle = INK;
      if (last && fresh && l.endsWith(fresh)) {
        const head = l.slice(0, l.length - fresh.length);
        c.fillText(head, 70, y);
        c.fillStyle = "#a8432a";
        c.fillText(fresh, 70 + c.measureText(head).width, y);
      } else c.fillText(l, 70, y);
    });
    this.pageTex.needsUpdate = true;
  },

  async play(action, caption, { fly, view }) {
    const i = Number(/slot_(\d)/.exec(action)?.[1] ?? 0);
    const tray = view?.tray ?? [];
    tray.forEach((w, j) => this.label(j, w));
    const k = this.keys[i];
    const page = view?.page ?? this.words.join(" ");
    let word = tray[i] ?? (/"(.+)"/.exec(caption)?.[1] || "");
    if (word === "i" || word.startsWith("i'")) word = "I" + word.slice(1);
    if (!page || /[.?!]$/.test(page)) word = word.charAt(0).toUpperCase() + word.slice(1);   // as the page prints it

    // walk to the chosen key, then stamp on it
    const at = new this.THREE.Vector3().copy(k.key.position);
    await fly.walkTo({ x: at.x, z: at.z }, 450);
    fly.face(new this.THREE.Vector3(at.x, 0, -1));
    await fly.hop(0.18, 200);
    await tween(90, (t) => (k.cap.position.y = 0.065 - 0.03 * t));
    await tween(120, (t) => {
      k.cap.position.y = 0.035 + 0.03 * t;
      this.platen.rotation.x += 0.08;
    });

    // the word lands on the page; a new page when the server started a new episode
    const text = page ? page + " " + word : word;
    const ends = /^[,.?!]$/.test(word);
    const shown = ends ? text.replace(/ ([,.?!])$/, "$1") : text;
    this.words = shown.split(" ");
    this.drawPage(shown, ends ? "" : word);
    if (/[.?!]$/.test(word)) await fly.buzz(250);
    await wait(250);

    // back to the middle between words
    await fly.walkTo(this.home, 350);
    fly.face(new this.THREE.Vector3(this.home.x, 0, -1));
  },
};
