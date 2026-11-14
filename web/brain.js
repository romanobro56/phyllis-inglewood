// Brains as point clouds: one dot per real neuron (FlyWire positions), lit by the activity the server
// recorded while the fly made its last decision. One geometry per fly, drawn by the top panel and by its pod.
import * as THREE from "three";

export const ACTION_COLORS = [0xd6336c, 0x1c7ed6, 0xe8a200, 0xe8590c, 0x7048e8, 0x0ca678, 0xc92a2a, 0x5c940d];
export const hex = (c) => "#" + c.toString(16).padStart(6, "0");
const STEP_MS = 90; // replay speed: one recorded brain step per 90 ms

// shared across every brain material
// uWorldScale: drawing-buffer height / (2 tan(fov/2)) of the world camera, so point sizes are world units
export const brainShared = { uTime: { value: 0 }, uWorldScale: { value: 800 } };

const presets = {};
export function loadPreset(preset) {
  presets[preset] ||= Promise.all([
    fetch(`/data/${preset}.points.bin`).then((r) => { if (!r.ok) throw new Error(`no data/${preset}.points.bin`); return r.arrayBuffer(); }),
    fetch(`/data/${preset}.meta.json`).then((r) => r.json()),
  ]).then(([buf, meta]) => ({ pos: new THREE.BufferAttribute(new Float32Array(buf), 3), meta }));
  return presets[preset];
}

const vertexShader = /* glsl */ `
  attribute float act;    // 0..1 firing rate
  attribute float kind;   // 0 neuron, 1 optic lobe, 2 sensory input, 3+k action group k
  uniform vec3 uBase, uHot, uInput, uGroup[8];
  uniform float uBaseAlpha, uSize, uScale, uIdle, uSeed, uTime, uFogNear, uFogFar;
  varying vec3 vColor;
  varying float vAlpha;
  void main() {
    vec4 mv = modelViewMatrix * vec4(position, 1.0);
    float a = clamp(act, 0.0, 1.0);
    if (uIdle > 0.0) {   // dormant pods: a few random neurons twitch now and then
      float h = fract(sin(dot(position, vec3(12.9898, 78.233, 37.719)) * 43758.5453 + floor(uTime * 5.0 + uSeed) * 1.618));
      a = max(a, uIdle * smoothstep(0.975, 1.0, h));
    }
    vec3 base = uBase; float sz = 1.0; float ba = uBaseAlpha;
    if (kind > 2.5) { base = uGroup[int(kind - 2.5)]; sz = 2.4; ba = min(1.0, uBaseAlpha * 3.0 + 0.25); }
    else if (kind > 1.5) { base = uInput; sz = 1.5; ba = min(1.0, uBaseAlpha * 2.0 + 0.1); }
    else if (kind > 0.5) { ba *= 0.6; }
    vColor = mix(base, uHot, kind > 2.5 ? a * 0.35 : a);
    float fog = 1.0 - smoothstep(uFogNear, uFogFar, -mv.z);
    vAlpha = mix(ba, 1.0, a) * fog;
    gl_PointSize = uSize * sz * (1.0 + 1.4 * a) * uScale / -mv.z;
    gl_Position = projectionMatrix * mv;
  }`;
const fragmentShader = /* glsl */ `
  varying vec3 vColor;
  varying float vAlpha;
  void main() {
    float d = length(gl_PointCoord - 0.5);
    if (d > 0.5) discard;
    gl_FragColor = vec4(vColor, vAlpha * (1.0 - smoothstep(0.3, 0.5, d)));
  }`;

// base: idle neuron colour, hot: firing colour. additive=true for glowing brains in dark gel.
export function brainMaterial({ base, hot, input = 0x0b7285, baseAlpha = 0.2, size = 0.012, additive = false,
                                idle = 0, seed = 0, fog = false, scale = brainShared.uWorldScale }) {
  const m = new THREE.ShaderMaterial({
    uniforms: {
      uBase: { value: new THREE.Color(base) }, uHot: { value: new THREE.Color(hot) },
      uInput: { value: new THREE.Color(input) }, uGroup: { value: ACTION_COLORS.map((c) => new THREE.Color(c)) },
      uBaseAlpha: { value: baseAlpha }, uSize: { value: size }, uScale: scale,
      uIdle: { value: idle }, uSeed: { value: seed }, uTime: brainShared.uTime,
      uFogNear: fog ? fog.near : { value: 1e6 }, uFogFar: fog ? fog.far : { value: 2e6 },
    },
    vertexShader, fragmentShader, transparent: true, depthWrite: false,
    blending: additive ? THREE.AdditiveBlending : THREE.NormalBlending,
  });
  return m;
}

// point size that looks right whatever the brain size (5k or 134k neurons)
export const pointSize = (n) => Math.min(0.022, Math.max(0.005, 0.005 * Math.sqrt(134000 / n)));

// Live activity of one fly's brain.
export class BrainActivity {
  constructor(info, { pos, meta }) {
    this.info = info;
    this.meta = meta;
    const n = (this.n = meta.n);
    const kind = new Float32Array(n);
    const optic = meta.classes.indexOf("optic");
    for (let i = 0; i < n; i++) if (meta.class_index[i] === optic) kind[i] = 1;
    for (const i of meta.input_idx) kind[i] = 2;
    info.groups.forEach((idx, k) => { for (const j of idx) kind[j] = 3 + (k % 8); });
    this.act = new Float32Array(n);
    this.actAttr = new THREE.BufferAttribute(this.act, 1).setUsage(THREE.DynamicDrawUsage);
    this.geometry = new THREE.BufferGeometry();
    this.geometry.setAttribute("position", pos);
    this.geometry.setAttribute("kind", new THREE.BufferAttribute(kind, 1));
    this.geometry.setAttribute("act", this.actAttr);
    this.geometry.computeBoundingSphere();
    this.queue = null;
    this.level = 0;       // how lit the brain is overall (0..1), for decay and pod glow
    this.pulse = 0;       // bumps on every decision, decays: drives twitches and lights
  }

  // a decision arrived: queue its recorded activity for replay
  push(activity) {
    this.pulse = 1;
    if (!activity) return;
    const bin = atob(activity.data);
    const raw = new Uint8Array(bin.length);
    for (let i = 0; i < bin.length; i++) raw[i] = bin.charCodeAt(i);
    if (activity.n !== this.n) return;
    this.queue = { raw, steps: activity.steps, s: 0, next: performance.now() };
  }

  update(now, dt) {
    this.pulse *= Math.exp(-dt * 3);
    const q = this.queue;
    if (q && now >= q.next) {
      const off = q.s * this.n, act = this.act, raw = q.raw;
      for (let i = 0; i < this.n; i++) act[i] = raw[off + i] / 255;
      this.level = 1;
      this.actAttr.needsUpdate = true;
      q.s++;
      q.next += STEP_MS;
      if (q.s >= q.steps) this.queue = null;
    } else if (!q && this.level > 0.02) {         // fade out between decisions
      const k = Math.exp(-dt * 1.4);
      this.level *= k;
      const act = this.act;
      for (let i = 0; i < this.n; i++) act[i] *= k;
      this.actAttr.needsUpdate = true;
    }
  }
}
