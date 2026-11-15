// The world's set dressing, OVERTIME: a fly-sized open-plan office at 3:47 am. Every employee in their own
// cubicle under one buzzing light, carpet tiles, an LED board thanking them for their dedication, a birthday
// cake nobody came to. main.js calls build() once, place() after the stations exist, update()/render() each frame.
import * as THREE from "three";
import { setupRenderer, roomEnvironment, loadFonts, addCss, canvasTexture, blotches, speckle, rng, wrapText,
  Post, arcOf, footprint, easeFov, lightCone } from "./kit.js";

const SANS = "'Inter Tight', 'Inter', system-ui, sans-serif";
const MONO = "'JetBrains Mono', ui-monospace, monospace";
const LENS = 30;
const TICKER = "AVERAGE LIFESPAN: 40 DAYS   ·   TODAY IS DAY 31   ·   THANK YOU FOR YOUR CONTINUED DEDICATION   ·   " +
  "THE BUILDING CLOSES AT 6 PM   ·   NOBODY HAS GONE HOME   ·   ";

export default {
  bg: 0x07090d,
  fog: { ground: [30, 120], wall: [16, 115] },
  maxDistance: 120,
  panel: { band: 0x14181f, base: 0x5a6b7e, hot: 0xfff1c9, input: 0x39d0b0, baseAlpha: 0.22, additive: true },

  async build({ scene, renderer, camera }) {
    this.scene = scene;
    setupRenderer(renderer, { exposure: 1.3 });
    roomEnvironment(renderer, scene, 0.05);
    await loadFonts(["Inter+Tight:wght@400;500;600;700", "JetBrains+Mono:wght@400;600", "Caveat:wght@600"],
      [`600 20px ${SANS}`, `400 12px ${MONO}`, "600 40px Caveat"]);
    this.css();

    // carpet tiles, laid quarter-turned like every office
    const tile = (g, w, h, rot) => {
      g.save(); g.translate(w / 2, h / 2); g.rotate(rot); g.translate(-w / 2, -h / 2);
      g.fillStyle = "#2a3140"; g.fillRect(0, 0, w, h);
      const r = rng(7 + rot * 10);
      for (let y = 0; y < h; y += 3) {        // the loop-pile rows
        g.fillStyle = `rgba(${r() < 0.5 ? "70,82,104" : "18,22,30"},${0.25 + r() * 0.3})`;
        g.fillRect(0, y, w, 1.5);
      }
      g.restore();
      speckle(g, w, h, { n: 2600, colors: ["#4a5872", "#151a24", "#5b6a86", "#3a2f2a"], size: 2.2, alpha: 0.8, seed: 3 + rot * 5 });
    };
    const carpet = canvasTexture(1024, 1024, (g, w, h) => {
      const half = w / 2;
      for (let i = 0; i < 4; i++) {
        const c = document.createElement("canvas");
        c.width = c.height = half;
        tile(c.getContext("2d"), half, half, (i === 1 || i === 2) ? Math.PI / 2 : 0);
        g.drawImage(c, (i % 2) * half, Math.floor(i / 2) * half);
      }
      g.strokeStyle = "rgba(5,7,10,.65)"; g.lineWidth = 3;
      g.strokeRect(0, 0, half, half); g.strokeRect(half, 0, half, half); g.strokeRect(0, half, half, half); g.strokeRect(half, half, half, half);
      blotches(g, w, h, { n: 30, rMin: 30, rMax: 120, alpha: 0.25, color: "10,10,14", seed: 8 });   // coffee, years of it
    }, { repeat: [60, 60] });
    const floor = new THREE.Mesh(new THREE.PlaneGeometry(400, 400), new THREE.MeshStandardMaterial({ map: carpet, roughness: 1, color: 0xb0b8c8 }));
    floor.rotation.x = -Math.PI / 2;
    floor.receiveShadow = true;
    scene.add(floor);
    scene.add(new THREE.HemisphereLight(0x34476a, 0x07080a, 0.6));

    this.post = new Post(renderer, scene, camera, {
      ao: true, aoRadius: 0.7, aoIntensity: 1.1, tiltShift: 0, bloom: { strength: 0.5, radius: 0.5, threshold: 0.9 },
      grade: { uGrain: 0.055, uVignette: 0.55, uAberration: 0.004, uLift: [0.01, 0.018, 0.03], uGain: [0.98, 1.02, 1.0], uSat: 0.9 },
    });
    this.resize(innerWidth, innerHeight);
    camera.fov = LENS;
    camera.updateProjectionMatrix();
  },

  css() {
    addCss(`
      body { --ink: #e7ecf2; --muted: #7d8a99; --claude: #ffd166; --mono: ${MONO}; }
      #status { color: #4d5866; font-family: ${MONO}; }
      .label { background: rgba(16,19,25,.86); border: 1px solid rgba(255,255,255,.08); border-radius: 6px; padding: 5px 10px 5px 22px;
               box-shadow: 0 6px 18px rgba(0,0,0,.45); color: #e7ecf2; font: 12px ${SANS}; position: relative; }
      .label::before { content: ""; position: absolute; left: 9px; top: 9px; width: 7px; height: 7px; border-radius: 50%;
                       background: #3ccf6e; box-shadow: 0 0 0 2px rgba(60,207,110,.2); }
      .label b { color: #f3f5f7; font-weight: 600; }
      .label.untrained::before { background: #f5b83d; }
      .label.untrained b { color: #f5b83d; }
      .label .cap { color: #8e9aab; font: 11px ${SANS}; margin-top: 1px; }
      .label.lesioned { border-color: rgba(255,90,78,.6); }
      .label.lesioned::before { background: #6b7684; box-shadow: none; }
      #nav button { border-color: rgba(220,230,240,.22); background: rgba(16,19,25,.55); color: #d9e0e8; }
      #nav button:hover { background: rgba(40,46,56,.8); }
      #nav-label { color: #6c7888; }
      #panel { color: #e7ecf2; border-bottom-color: rgba(255,255,255,.08); }
      #p-sub, .hint, .bar .pct, #p-legend { color: #7d8a99; }
      #p-lesion { border-color: rgba(231,236,242,.3); }
      .bar .track { background: rgba(255,255,255,.08); }
      .kicker { color: #ff5a4e; }
      #p-title { font-family: ${SANS}; letter-spacing: -.02em; }
    `);
  },

  overviewPose(R, camera) {
    const c = this.center ?? new THREE.Vector3(0, 0, R * 0.35);
    const target = c.clone().add(new THREE.Vector3(0, 2.6, -R * 0.62));
    const halfH = Math.atan(Math.tan(THREE.MathUtils.degToRad(LENS / 2)) * Math.min(camera.aspect, 2));
    const d = (R + 9) / Math.tan(halfH);
    const tilt = THREE.MathUtils.degToRad(21);
    return { pos: target.clone().add(new THREE.Vector3(-d * 0.06, d * Math.sin(tilt), d * Math.cos(tilt))), target };
  },

  place(flies) {
    const { R, center } = arcOf(flies);
    this.center = center;
    this.lights = [];
    const r = rng(17);
    const fabric = this.fabric();
    const trim = new THREE.MeshStandardMaterial({ color: 0x9aa1aa, metalness: 0.7, roughness: 0.35 });
    const pins = this.pinUps();
    flies.forEach((f, i) => {
      const b = footprint(f.group);
      if (b.isEmpty()) b.set(new THREE.Vector3(-1, 0, -1), new THREE.Vector3(1, 1, 1));
      const x0 = b.min.x - 0.45, x1 = b.max.x + 0.45, zb = b.min.z - 0.35;
      const H = 2.3, T = 0.12;
      const cub = new THREE.Group();
      const panel = (w, x, z, ry, h = H) => {
        const m = new THREE.Mesh(new THREE.BoxGeometry(w, h, T), fabric);
        m.position.set(x, h / 2, z);
        m.rotation.y = ry;
        m.castShadow = m.receiveShadow = true;
        const cap = new THREE.Mesh(new THREE.BoxGeometry(w + 0.02, 0.05, T + 0.04), trim);
        cap.position.set(x, h + 0.025, z);
        cap.rotation.y = ry;
        cub.add(m, cap);
      };
      panel(x1 - x0, (x0 + x1) / 2, zb, 0);
      const depth = Math.min(2.2, (b.max.z - zb) * 0.6);
      panel(depth, x0, zb + depth / 2, Math.PI / 2, H * 0.72);
      panel(depth, x1, zb + depth / 2, Math.PI / 2, H * 0.72);
      // two pinned-up sheets on the back panel
      const picks = [pins[i % pins.length], pins[(i * 3 + 1) % pins.length]];
      picks.forEach((p, k) => {
        const m = new THREE.Mesh(new THREE.PlaneGeometry(p.w, p.h), new THREE.MeshStandardMaterial({ map: p.tex, roughness: 0.9 }));
        const span = x1 - x0;
        m.position.set(x0 + span * (k ? 0.78 : 0.22), H * 0.66 + (k ? -0.05 : 0.08), zb + T / 2 + 0.01);
        m.rotation.z = (r() - 0.5) * 0.08;
        m.receiveShadow = true;
        cub.add(m);
      });
      cub.traverse((o) => (o.userData.lookIgnore = true));
      f.group.add(cub);

      // the light over the desk: a warm-cold fluorescent pool, with soft shadows
      const spot = new THREE.SpotLight(0xe9f3ee, 190, 24, 0.58, 0.8, 1.6);
      spot.position.set(0, 9.5, 0.3);
      const haze = lightCone(0.4, 5.0, 9.5, 0xcfe6dc, 0.028);
      haze.position.set(0, 9.5, 0.3);
      haze.userData.lookIgnore = true;
      f.group.add(haze);
      spot.target.position.set(0, 0, -0.1);
      spot.castShadow = true;
      spot.shadow.mapSize.set(1024, 1024);
      spot.shadow.bias = -0.0005;
      spot.shadow.radius = 4;
      f.group.add(spot, spot.target);
      this.lights.push({ spot, haze, base: spot.intensity, flicker: i === Math.floor(flies.length * 0.7) });
    });
    this.backWall(R, center);
    this.cake(center.clone().add(new THREE.Vector3(1.5, 0, -R * 0.42)));
  },

  cake(at) {
    // in the middle of the floor, under no light at all: a birthday cake nobody came to
    const g = new THREE.Group();
    const sponge = new THREE.Mesh(new THREE.CylinderGeometry(0.75, 0.78, 0.55, 48), new THREE.MeshStandardMaterial({ color: 0xf1e6d6, roughness: 0.8 }));
    sponge.position.y = 0.3;
    const drip = new THREE.Mesh(new THREE.TorusGeometry(0.74, 0.06, 8, 48), new THREE.MeshStandardMaterial({ color: 0xe86f9a, roughness: 0.5 }));
    drip.rotation.x = Math.PI / 2;
    drip.position.y = 0.56;
    const plate = new THREE.Mesh(new THREE.CylinderGeometry(1.05, 0.95, 0.05, 48), new THREE.MeshStandardMaterial({ color: 0xffffff, roughness: 0.3 }));
    plate.position.y = 0.03;
    const candle = new THREE.Mesh(new THREE.CylinderGeometry(0.035, 0.035, 0.42, 12), new THREE.MeshStandardMaterial({ color: 0x6ec3ff }));
    candle.position.y = 0.79;
    this.flame = new THREE.Mesh(new THREE.SphereGeometry(0.05, 12, 8), new THREE.MeshBasicMaterial({ color: 0xffd38a }));
    this.flame.scale.set(1, 2.2, 1);
    this.flame.position.y = 1.08;
    this.candleLight = new THREE.PointLight(0xffa64d, 2.2, 7, 1.8);
    this.candleLight.position.y = 1.2;
    const slice = new THREE.Mesh(new THREE.CylinderGeometry(0.76, 0.76, 0.56, 24, 1, false, 0, 0.6), new THREE.MeshStandardMaterial({ color: 0xc89a6a }));
    slice.position.y = 0.3;           // the one missing slice's cut face (nobody took it, it just isn't there)
    const card = new THREE.Mesh(new THREE.PlaneGeometry(0.9, 0.5), new THREE.MeshStandardMaterial({ roughness: 0.9, side: THREE.DoubleSide,
      map: canvasTexture(360, 200, (c, w, h) => {
        c.fillStyle = "#fbfaf6"; c.fillRect(0, 0, w, h);
        c.fillStyle = "#c2185b"; c.font = "600 54px Caveat"; c.textAlign = "center";
        c.fillText("happy birthday", w / 2, 86); c.fillStyle = "#333"; c.font = "600 40px Caveat"; c.fillText("(day 31)", w / 2, 150);
      }) }));
    card.position.set(0.25, 0.25, 1.15);
    card.rotation.set(-0.5, 0.25, 0);
    g.add(plate, sponge, drip, candle, this.flame, this.candleLight, card);
    g.traverse((o) => { if (o.isMesh && o !== this.flame) o.castShadow = o.receiveShadow = true; });
    g.position.copy(at);
    this.scene.add(g);
  },

  fabric() {
    const tex = canvasTexture(512, 512, (g, w, h) => {
      g.fillStyle = "#5f6874"; g.fillRect(0, 0, w, h);
      speckle(g, w, h, { n: 30000, colors: ["#717b88", "#4a525d", "#68717e"], size: 1.6, alpha: 0.7, seed: 4 });
      blotches(g, w, h, { n: 12, rMin: 40, rMax: 120, alpha: 0.12, color: "30,30,30", seed: 9 });
    }, { repeat: [2, 2] });
    return new THREE.MeshStandardMaterial({ map: tex, roughness: 1, color: 0xc8ccd2 });
  },

  pinUps() {
    const sheet = (w, h, draw, pxw = 512) => {
      const pxh = Math.round(pxw * h / w);
      return { w, h, tex: canvasTexture(pxw, pxh, (g) => { draw(g, pxw, pxh); }) };
    };
    const pin = (g, x, y, col = "#d33") => { g.fillStyle = col; g.beginPath(); g.arc(x, y, 9, 0, 7); g.fill(); };
    return [
      sheet(0.62, 0.8, (g, w, h) => {          // the calendar, crossed off
        g.fillStyle = "#f4f2ec"; g.fillRect(0, 0, w, h);
        g.fillStyle = "#c22"; g.fillRect(0, 0, w, 90);
        g.fillStyle = "#fff"; g.font = `700 52px ${SANS}`; g.fillText("NOVEMBER", 30, 64);
        g.font = `500 28px ${SANS}`;
        for (let d = 1; d <= 30; d++) {
          const cx = 28 + ((d - 1) % 7) * 66, cy = 150 + Math.floor((d - 1) / 7) * 100;
          g.fillStyle = "#333"; g.fillText(String(d), cx, cy);
          if (d < 14) { g.strokeStyle = "#1f3fbf"; g.lineWidth = 5; g.beginPath(); g.moveTo(cx - 6, cy - 30); g.lineTo(cx + 40, cy + 20); g.moveTo(cx + 40, cy - 30); g.lineTo(cx - 6, cy + 20); g.stroke(); }
        }
        g.strokeStyle = "#c22"; g.lineWidth = 5; g.beginPath(); g.ellipse(28 + 2 * 66 + 18, 150 + 3 * 100 - 10, 40, 34, 0, 0, 7); g.stroke();
        g.fillStyle = "#c22"; g.font = "600 40px Caveat"; g.fillText("last day?", 250, 560);
        pin(g, w / 2, 20, "#2a6");
      }),
      sheet(0.7, 0.5, (g, w, h) => {           // motivational poster
        const grd = g.createLinearGradient(0, 0, 0, h);
        grd.addColorStop(0, "#1b2a44"); grd.addColorStop(1, "#0c1220");
        g.fillStyle = grd; g.fillRect(0, 0, w, h);
        g.fillStyle = "#e9e3d0"; g.textAlign = "center";
        g.font = `700 54px ${SANS}`; g.fillText("PERSEVERANCE", w / 2, h * 0.58);
        g.font = `400 21px ${SANS}`; g.fillStyle = "#a9b4c6";
        g.fillText("you were bred for this.", w / 2, h * 0.75);
        pin(g, 20, 20); pin(g, w - 20, 20);
      }),
      sheet(0.45, 0.45, (g, w, h) => {          // post-it
        g.fillStyle = "#f7e36a"; g.fillRect(0, 0, w, h);
        g.fillStyle = "#223"; g.font = "600 62px Caveat";
        wrapText(g, "remember to eat", 40, 150, w - 70, 72);
      }),
      sheet(0.6, 0.78, (g, w, h) => {            // the review
        g.fillStyle = "#f6f6f3"; g.fillRect(0, 0, w, h);
        g.fillStyle = "#111"; g.font = `700 30px ${SANS}`; g.fillText("PERFORMANCE REVIEW", 30, 60);
        g.font = `400 20px ${SANS}`; g.fillStyle = "#555";
        const rows = ["Punctuality", "Initiative", "Team spirit", "Is alive"];
        rows.forEach((t, k) => {
          g.fillStyle = "#333"; g.fillText(t, 30, 140 + k * 70);
          g.strokeStyle = "#888"; g.lineWidth = 2; g.strokeRect(w - 90, 115 + k * 70, 34, 34);
          if (k < 3) { g.strokeStyle = "#1f3fbf"; g.lineWidth = 5; g.beginPath(); g.moveTo(w - 86, 132 + k * 70); g.lineTo(w - 74, 146 + k * 70); g.lineTo(w - 50, 108 + k * 70); g.stroke(); }
        });
        g.fillStyle = "#1f3fbf"; g.font = "600 44px Caveat"; g.fillText("pending", w - 200, 140 + 3 * 70 + 50);
        pin(g, w / 2, 20, "#36c");
      }),
      sheet(0.5, 0.36, (g, w, h) => {            // the drawing from home
        g.fillStyle = "#fffdf6"; g.fillRect(0, 0, w, h);
        g.strokeStyle = "#e0452b"; g.lineWidth = 6; g.lineCap = "round";
        g.beginPath(); g.arc(w * 0.3, h * 0.55, 40, 0, 7); g.stroke();
        g.beginPath(); g.arc(w * 0.7, h * 0.55, 30, 0, 7); g.stroke();
        g.strokeStyle = "#2b7a2b"; g.beginPath(); g.moveTo(0, h * 0.9); g.lineTo(w, h * 0.88); g.stroke();
        g.fillStyle = "#333"; g.font = "600 44px Caveat"; g.fillText("dad + me", w * 0.3, h * 0.22);
        pin(g, w / 2, 16, "#fb3");
      }),
    ];
  },

  backWall(R, center) {
    const S = this.scene;
    const zw = center.z - R - 6.5, W = 110, H = 10;
    const paint = canvasTexture(1024, 256, (g, w, h) => {
      g.fillStyle = "#59606a"; g.fillRect(0, 0, w, h);
      blotches(g, w, h, { n: 140, rMin: 20, rMax: 90, alpha: 0.12, color: "20,20,25", seed: 5 });
      speckle(g, w, h, { n: 8000, colors: ["#666d77", "#4c525b"], size: 1.4, alpha: 0.5, seed: 6 });
    }, { repeat: [6, 1] });
    const wall = new THREE.Mesh(new THREE.BoxGeometry(W, H, 0.5), new THREE.MeshStandardMaterial({ map: paint, roughness: 0.95 }));
    wall.position.set(0, H / 2, zw);
    wall.receiveShadow = true;
    S.add(wall);
    const base = new THREE.Mesh(new THREE.BoxGeometry(W, 0.35, 0.6), new THREE.MeshStandardMaterial({ color: 0x1b1e22, roughness: 0.6 }));
    base.position.set(0, 0.175, zw + 0.05);
    S.add(base);

    // the LED board: a dot-matrix shader reading a long strip of text
    const cols = 210, rows = 14;
    const strip = canvasTexture(2048 * 4, 64, (g, w, h) => {
      g.fillStyle = "#000"; g.fillRect(0, 0, w, h);
      g.fillStyle = "#fff"; g.font = `700 60px ${MONO}`; g.textBaseline = "middle";
      g.fillText(TICKER + TICKER, 0, h / 2 + 2);
    }, { srgb: false, aniso: 1 });
    strip.wrapS = THREE.RepeatWrapping;
    strip.minFilter = THREE.LinearFilter;
    strip.generateMipmaps = false;
    const textW = (() => { const c = document.createElement("canvas").getContext("2d"); c.font = `700 60px ${MONO}`; return c.measureText(TICKER).width / (2048 * 4); })();
    this.led = new THREE.ShaderMaterial({
      uniforms: { tText: { value: strip }, uOff: { value: 0 }, uGrid: { value: new THREE.Vector2(cols, rows) },
        uView: { value: 0.105 }, uLoop: { value: textW } },
      vertexShader: `varying vec2 vUv; void main() { vUv = uv; gl_Position = projectionMatrix * modelViewMatrix * vec4(position, 1.0); }`,
      fragmentShader: `uniform sampler2D tText; uniform float uOff, uView, uLoop; uniform vec2 uGrid; varying vec2 vUv;
        void main() {
          vec2 cell = floor(vUv * uGrid); vec2 f = fract(vUv * uGrid) - 0.5;
          vec2 s = (cell + 0.5) / uGrid;
          float u = mod(uOff + s.x * uView, uLoop);
          float on = step(0.45, texture2D(tText, vec2(u, s.y)).r);
          float dotm = smoothstep(0.42, 0.28, length(f));
          vec3 c = mix(vec3(0.09, 0.012, 0.01), vec3(4.0, 0.42, 0.18), on) * dotm;
          gl_FragColor = vec4(c, 1.0);
        }`,
    });
    const bw = 30, bh = (bw * rows) / cols;
    const board = new THREE.Mesh(new THREE.PlaneGeometry(bw, bh), this.led);
    board.position.set(0, 6.2, zw + 0.4);
    const frame = new THREE.Mesh(new THREE.BoxGeometry(bw + 0.6, bh + 0.6, 0.35), new THREE.MeshStandardMaterial({ color: 0x0b0c0e, roughness: 0.4, metalness: 0.4 }));
    frame.position.set(0, 6.2, zw + 0.2);
    S.add(frame, board);
    const glow = new THREE.RectAreaLight(0xff3a20, 2.2, bw, bh);
    glow.position.set(0, 6.2, zw + 0.5);
    glow.lookAt(0, 4, zw + 10);
    S.add(glow);

    // the clock, stuck at a quarter to four (the second hand still goes round)
    const face = canvasTexture(512, 512, (g, w) => {
      g.fillStyle = "#f2f0ea"; g.beginPath(); g.arc(w / 2, w / 2, w / 2 - 6, 0, 7); g.fill();
      g.fillStyle = "#111";
      for (let k = 0; k < 60; k++) {
        g.save(); g.translate(w / 2, w / 2); g.rotate((k / 60) * Math.PI * 2);
        g.fillRect(-(k % 5 ? 2 : 5), -w / 2 + 22, k % 5 ? 4 : 10, k % 5 ? 14 : 34);
        g.restore();
      }
      g.font = `600 30px ${SANS}`; g.textAlign = "center"; g.fillStyle = "#444"; g.fillText("FLYWORKS", w / 2, w * 0.36);
    });
    const clock = new THREE.Group();
    const rim = new THREE.Mesh(new THREE.CylinderGeometry(2.1, 2.1, 0.35, 64), new THREE.MeshStandardMaterial({ color: 0x15171a, roughness: 0.5 }));
    rim.rotation.x = Math.PI / 2;
    const disc = new THREE.Mesh(new THREE.CircleGeometry(1.92, 64), new THREE.MeshStandardMaterial({ map: face, roughness: 0.7 }));
    disc.position.z = 0.18;
    const hand = (len, wdt, color, z) => {
      const m = new THREE.Mesh(new THREE.BoxGeometry(wdt, len, 0.03), new THREE.MeshStandardMaterial({ color }));
      m.geometry.translate(0, len / 2 - 0.12, 0);
      m.position.z = z;
      clock.add(m);
      return m;
    };
    const hr = hand(1.0, 0.12, 0x111111, 0.2), mn = hand(1.55, 0.08, 0x111111, 0.22);
    this.sec = hand(1.65, 0.03, 0xc8201a, 0.24);
    hr.rotation.z = -((3 + 47 / 60) / 12) * Math.PI * 2;
    mn.rotation.z = -(47 / 60) * Math.PI * 2;
    clock.add(rim, disc);
    clock.position.set(-21, 6.4, zw + 0.45);
    S.add(clock);
    const clockLight = new THREE.SpotLight(0xdfe8ff, 18, 14, 0.5, 0.9, 1.5);
    clockLight.position.set(-21, 11, zw + 5);
    clockLight.target = clock;
    S.add(clockLight);

    // EXIT, over a wall with no door in it
    const exitTex = canvasTexture(512, 192, (g, w, h) => {
      g.fillStyle = "#0b5a2a"; g.fillRect(0, 0, w, h);
      g.fillStyle = "#d9ffe6"; g.font = `700 120px ${SANS}`; g.textBaseline = "middle"; g.fillText("EXIT", 40, h / 2 + 6);
      g.beginPath(); g.moveTo(400, 60); g.lineTo(470, h / 2); g.lineTo(400, h - 60); g.closePath(); g.fill();
    });
    const exit = new THREE.Mesh(new THREE.BoxGeometry(3.2, 1.2, 0.3), [0, 0, 0, 0, 1, 0].map((k) => k
      ? new THREE.MeshStandardMaterial({ map: exitTex, emissive: 0xffffff, emissiveMap: exitTex, emissiveIntensity: 2.2 })
      : new THREE.MeshStandardMaterial({ color: 0x1a1d20 })));
    exit.position.set(-32, 7.8, zw + 0.45);
    S.add(exit);
    const exitGlow = new THREE.PointLight(0x2bff7a, 6, 12, 1.8);
    exitGlow.position.set(-32, 7.0, zw + 1.6);
    S.add(exitGlow);

    // the vending machine, humming in the dark. It sells fruit. The fruit is not fresh.
    this.vending(new THREE.Vector3(R + 7, 0, zw + 2.4));
  },

  vending(at) {
    const S = this.scene;
    const front = canvasTexture(512, 900, (g, w, h) => {
      g.fillStyle = "#111"; g.fillRect(0, 0, w, h);
      const glass = g.createLinearGradient(0, 0, w, 0);
      glass.addColorStop(0, "#fff7e2"); glass.addColorStop(1, "#ffe4b0");
      g.fillStyle = glass; g.fillRect(24, 24, w * 0.7, h - 160);
      const fruit = [["#d8b23a", "#5a3a10"], ["#b9441f", "#3a1205"], ["#86a83a", "#3d4a12"], ["#6b3a6b", "#2a0f2a"]];
      const r = rng(31);
      for (let row = 0; row < 6; row++) {
        const y = 60 + row * 118;
        g.fillStyle = "#c9c2b0"; g.fillRect(24, y + 70, w * 0.7, 6);
        for (let k = 0; k < 4; k++) {
          const [c, rot] = fruit[(row + k) % 4];
          const x = 50 + k * 84;
          g.fillStyle = c; g.beginPath(); g.ellipse(x + 20, y + 44, 24, 26, 0, 0, 7); g.fill();
          g.fillStyle = rot; g.globalAlpha = 0.4 + r() * 0.5;      // brown spots
          for (let s = 0; s < 5; s++) { g.beginPath(); g.arc(x + 8 + r() * 26, y + 30 + r() * 28, 3 + r() * 6, 0, 7); g.fill(); }
          g.globalAlpha = 1;
          g.fillStyle = "#555"; g.font = `600 14px ${MONO}`; g.fillText(`${String.fromCharCode(65 + row)}${k + 1}`, x + 10, y + 96);
        }
      }
      g.fillStyle = "#222"; g.fillRect(w * 0.76, 40, w * 0.2, 300);
      g.fillStyle = "#0f0"; g.font = `600 22px ${MONO}`; g.fillText("0.00", w * 0.79, 80);
      g.fillStyle = "#333"; g.fillRect(40, h - 110, w * 0.62, 70);
      g.fillStyle = "#ddd"; g.font = `700 30px ${SANS}`; g.fillText("FRESH FRUIT", w * 0.76 - 300, h - 60);
    });
    const body = new THREE.Mesh(new THREE.BoxGeometry(2.8, 5.0, 2.0), [
      new THREE.MeshStandardMaterial({ color: 0x22262b, roughness: 0.5, metalness: 0.3 }),
      new THREE.MeshStandardMaterial({ color: 0x22262b, roughness: 0.5, metalness: 0.3 }),
      new THREE.MeshStandardMaterial({ color: 0x22262b }), new THREE.MeshStandardMaterial({ color: 0x22262b }),
      new THREE.MeshStandardMaterial({ map: front, emissive: 0xffffff, emissiveMap: front, emissiveIntensity: 0.45, roughness: 0.25 }),
      new THREE.MeshStandardMaterial({ color: 0x22262b })]);
    body.position.copy(at).add(new THREE.Vector3(0, 2.5, 0));
    body.rotation.y = -0.35;
    body.castShadow = true;
    S.add(body);
    const glow = new THREE.PointLight(0xffd59a, 10, 14, 1.6);
    glow.position.copy(at).add(new THREE.Vector3(-1.2, 2.4, 2.6));
    S.add(glow);
  },

  update(t, dt, camera, mode) {
    easeFov(camera, mode === "overview" ? LENS : 50, dt);
    if (this.led) this.led.uniforms.uOff.value = (t * 0.018) % this.led.uniforms.uLoop.value;
    if (this.flame) {
      const f = 1 + Math.sin(t * 13) * 0.08 + Math.sin(t * 29.3) * 0.06;
      this.flame.scale.set(f, 2.2 * f, f);
      this.candleLight.intensity = 2.2 * f;
    }
    if (this.sec) this.sec.rotation.z = -Math.floor(t % 60) / 60 * Math.PI * 2;
    for (const l of this.lights ?? []) {
      if (!l.flicker) continue;
      // a dying tube: mostly on, with bursts of stutter
      const burst = Math.sin(t * 0.7) > 0.55;
      const on = !burst || Math.sin(t * 53) * Math.sin(t * 31.7) > -0.1;
      l.spot.intensity = l.base * (on ? (burst ? 0.75 : 1) : 0.08);
      l.haze.visible = on;
    }
  },

  resize(w, h) { this.post?.setSize(w, h); },
  render(renderer, scene, camera) { this.post.render(performance.now() / 1000); },
};
