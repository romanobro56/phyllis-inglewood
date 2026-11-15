// The matrix view: one fly's brain as a green point cloud, replaying each decision's activity.
import * as THREE from "three";
import { OrbitControls } from "three/addons/controls/OrbitControls.js";

const ACTION_COLORS = [0xff4fd8, 0x4fc3ff, 0xffd84f, 0xff8a4f, 0xb04fff, 0x4fffc8, 0xff4f6a, 0xa8ff4f];
const STEP_MS = 90; // playback speed per brain step

export class Matrix {
  constructor({ onClose, onLesion }) {
    this.onClose = onClose;
    this.onLesion = onLesion;
    this.el = document.getElementById("matrix");
    this.fly = null;
    this.cache = {};
    this.renderer = new THREE.WebGLRenderer({ antialias: true, alpha: true });
    this.renderer.setPixelRatio(Math.min(devicePixelRatio, 2));
    document.getElementById("brain").appendChild(this.renderer.domElement);
    this.scene = new THREE.Scene();
    this.camera = new THREE.PerspectiveCamera(45, 1, 0.01, 100);
    this.camera.position.set(0, 0.3, 2.6);
    this.controls = new OrbitControls(this.camera, this.renderer.domElement);
    this.controls.autoRotate = true;
    this.controls.autoRotateSpeed = 0.6;
    this.controls.enableDamping = true;
    this.ray = new THREE.Raycaster();
    this.ray.params.Points.threshold = 0.012;
    this.tip = document.getElementById("tip");
    this.lesioned = false;

    document.getElementById("m-close").onclick = () => this.close();
    document.getElementById("m-lesion").onclick = () => {
      this.lesioned = !this.lesioned;
      this.updateLesionButton();
      this.onLesion(this.fly, this.lesioned);
    };
    addEventListener("keydown", (e) => e.key === "Escape" && this.fly && this.close());
    this.renderer.domElement.addEventListener("mousemove", (e) => this.hover(e));
    this.renderer.domElement.addEventListener("click", () => {
      if (this.hovered != null && this.meta.root_id[this.hovered] && !this.meta.synthetic)
        window.open(`https://codex.flywire.ai/app/cell_details?root_id=${this.meta.root_id[this.hovered]}`, "_blank");
    });
    addEventListener("resize", () => this.resize());
    this.rain = new Rain(document.getElementById("rain"));
    this.renderer.setAnimationLoop(() => {
      if (!this.fly) return;
      this.controls.update();
      this.renderer.render(this.scene, this.camera);
      this.rain.draw();
    });
  }

  updateLesionButton() {
    const b = document.getElementById("m-lesion");
    b.textContent = `Lesion connectome: ${this.lesioned ? "ON (brain disconnected)" : "OFF"}`;
    b.classList.toggle("on", this.lesioned);
  }

  resize() {
    this.renderer.setSize(innerWidth, innerHeight);
    this.camera.aspect = innerWidth / innerHeight;
    this.camera.updateProjectionMatrix();
    this.rain.resize();
  }

  async load(preset) {
    if (!this.cache[preset]) {
      const [buf, meta] = await Promise.all([
        fetch(`/data/${preset}.points.bin`).then((r) => r.arrayBuffer()),
        fetch(`/data/${preset}.meta.json`).then((r) => r.json()),
      ]);
      this.cache[preset] = { pos: new Float32Array(buf), meta };
    }
    return this.cache[preset];
  }

  async open(info, title) {
    this.fly = info.name;
    this.info = info;
    this.lesioned = false;
    this.updateLesionButton();
    this.el.classList.add("open");
    this.resize();
    document.getElementById("m-title").textContent = title;
    document.getElementById("m-sub").textContent =
      `${info.trained ? "trained" : "UNTRAINED"} | ${info.preset} brain` + (info.synthetic ? " | SYNTHETIC (not a fly)" : "");
    document.getElementById("m-caption").textContent = "waiting for the next decision...";
    document.getElementById("m-bars").innerHTML = "";
    const { pos, meta } = await this.load(info.preset);
    this.meta = meta;
    this.build(pos, meta, info);
  }

  build(pos, meta, info) {
    this.scene.clear();
    const n = meta.n;
    const g = new THREE.BufferGeometry();
    g.setAttribute("position", new THREE.BufferAttribute(pos, 3));
    this.base = new Float32Array(n * 3);
    for (let i = 0; i < n; i++) {
      const c = meta.classes[meta.class_index[i]];
      const dim = c === "optic" ? 0.05 : 0.09;
      this.base[i * 3] = 0; this.base[i * 3 + 1] = dim; this.base[i * 3 + 2] = dim * 0.4;
    }
    for (const i of meta.input_idx) { this.base[i * 3] = 0; this.base[i * 3 + 1] = 0.25; this.base[i * 3 + 2] = 0.3; }
    this.colors = new Float32Array(this.base);
    g.setAttribute("color", new THREE.BufferAttribute(this.colors, 3));
    this.points = new THREE.Points(g, new THREE.PointsMaterial({
      size: 0.012, vertexColors: true, transparent: true, opacity: 0.9,
      blending: THREE.AdditiveBlending, depthWrite: false }));
    this.scene.add(this.points);

    // the action groups: bigger, coloured points
    this.groupPoints = [];
    const legend = [];
    info.groups.forEach((idx, k) => {
      const gp = new Float32Array(idx.length * 3);
      idx.forEach((j, m) => gp.set(pos.subarray(j * 3, j * 3 + 3), m * 3));
      const geo = new THREE.BufferGeometry();
      geo.setAttribute("position", new THREE.BufferAttribute(gp, 3));
      const mat = new THREE.PointsMaterial({ size: 0.045, color: ACTION_COLORS[k % 8], transparent: true, opacity: 0.4 });
      const p = new THREE.Points(geo, mat);
      this.scene.add(p);
      this.groupPoints.push(p);
      legend.push(`<span style="color:#${ACTION_COLORS[k % 8].toString(16).padStart(6, "0")}">&#9679;</span> ${info.actions[k]} neurons`);
    });
    legend.push(`<span style="color:#3fd">&#9679;</span> sensory input neurons`);
    document.getElementById("legend").innerHTML = legend.join("<br>");
  }

  // a decision arrived for the fly we're inside
  async show(m) {
    document.getElementById("m-caption").textContent = `> ${m.caption}`;
    const bars = this.info.actions.map((a, k) => {
      const p = m.probs[a];
      const col = "#" + ACTION_COLORS[k % 8].toString(16).padStart(6, "0");
      return `<div class="bar ${a === m.action ? "chosen" : ""}"><span class="name" style="color:${col}">${a}</span> ${(p * 100).toFixed(0)}%
              <div class="track"><div class="fill" style="width:${p * 100}%;background:${col}"></div></div></div>`;
    });
    document.getElementById("m-bars").innerHTML = bars.join("");
    if (!m.activity || !this.points) return;
    const raw = Uint8Array.from(atob(m.activity.data), (c) => c.charCodeAt(0));
    const { steps, n } = m.activity;
    const attr = this.points.geometry.getAttribute("color");
    for (let s = 0; s < steps; s++) {
      const off = s * n;
      for (let i = 0; i < n; i++) {
        const v = raw[off + i] / 255;
        this.colors[i * 3] = this.base[i * 3] + v * 0.55;
        this.colors[i * 3 + 1] = this.base[i * 3 + 1] + v * 1.2;
        this.colors[i * 3 + 2] = this.base[i * 3 + 2] + v * 0.45;
      }
      attr.needsUpdate = true;
      this.groupPoints.forEach((gp, k) => {
        const idx = this.info.groups[k];
        let sum = 0;
        for (const j of idx) sum += raw[off + j];
        gp.material.opacity = 0.25 + 0.75 * Math.min(1, (sum / idx.length / 255) * 2.5);
        gp.material.size = this.info.actions[k] === m.action && s === steps - 1 ? 0.07 : 0.045;
      });
      await new Promise((r) => setTimeout(r, STEP_MS));
    }
  }

  hover(e) {
    if (!this.points) return;
    const p = new THREE.Vector2((e.clientX / innerWidth) * 2 - 1, -(e.clientY / innerHeight) * 2 + 1);
    this.ray.setFromCamera(p, this.camera);
    const hit = this.ray.intersectObject(this.points)[0];
    this.hovered = hit ? hit.index : null;
    if (!hit) { this.tip.style.display = "none"; return; }
    const i = hit.index, meta = this.meta;
    this.tip.style.display = "block";
    this.tip.style.left = e.clientX + 12 + "px";
    this.tip.style.top = e.clientY + 12 + "px";
    this.tip.textContent = `${meta.cell_type[i] || "unnamed"} | ${meta.classes[meta.class_index[i]]} | ${meta.synthetic ? "synthetic #" : "FlyWire "}${meta.root_id[i]}`;
  }

  close() {
    if (this.lesioned) this.onLesion(this.fly, false);
    this.el.classList.remove("open");
    this.fly = null;
    this.tip.style.display = "none";
    this.onClose();
  }
}

// falling green glyphs behind the brain
class Rain {
  constructor(canvas) { this.c = canvas; this.ctx = canvas.getContext("2d"); this.resize(); }
  resize() {
    this.c.width = innerWidth; this.c.height = innerHeight;
    this.cols = Math.floor(innerWidth / 16);
    this.y = Array.from({ length: this.cols }, () => Math.random() * innerHeight);
  }
  draw() {
    const ctx = this.ctx;
    ctx.fillStyle = "rgba(0,0,0,0.08)";
    ctx.fillRect(0, 0, this.c.width, this.c.height);
    ctx.fillStyle = "#0f0";
    ctx.font = "14px monospace";
    for (let i = 0; i < this.cols; i++) {
      ctx.fillText("ACGT01".charAt((Math.random() * 6) | 0), i * 16, this.y[i]);
      this.y[i] = this.y[i] > this.c.height && Math.random() > 0.975 ? 0 : this.y[i] + 16;
    }
  }
}
