// Shared pieces for the world's set dressing (web/looks/overtime.js):
// renderer setup, a post-processing stack (ambient occlusion, tilt-shift, bloom, a film grade), and canvas textures.
import * as THREE from "three";
import { EffectComposer } from "three/addons/postprocessing/EffectComposer.js";
import { RenderPass } from "three/addons/postprocessing/RenderPass.js";
import { GTAOPass } from "three/addons/postprocessing/GTAOPass.js";
import { UnrealBloomPass } from "three/addons/postprocessing/UnrealBloomPass.js";
import { ShaderPass } from "three/addons/postprocessing/ShaderPass.js";
import { OutputPass } from "three/addons/postprocessing/OutputPass.js";
import { HorizontalTiltShiftShader } from "three/addons/shaders/HorizontalTiltShiftShader.js";
import { VerticalTiltShiftShader } from "three/addons/shaders/VerticalTiltShiftShader.js";
import { RoomEnvironment } from "three/addons/environments/RoomEnvironment.js";
import { RectAreaLightUniformsLib } from "three/addons/lights/RectAreaLightUniformsLib.js";

export function setupRenderer(renderer, { exposure = 1, toneMapping = THREE.ACESFilmicToneMapping } = {}) {
  renderer.toneMapping = toneMapping;
  renderer.toneMappingExposure = exposure;
  renderer.outputColorSpace = THREE.SRGBColorSpace;
  renderer.shadowMap.type = THREE.PCFSoftShadowMap;
  RectAreaLightUniformsLib.init();
}

export function roomEnvironment(renderer, scene, intensity = 0.4) {
  const pmrem = new THREE.PMREMGenerator(renderer);
  scene.environment = pmrem.fromScene(new RoomEnvironment(), 0.04).texture;
  scene.environmentIntensity = intensity;
}

// Fonts from Google Fonts, awaited so canvas textures are drawn with them.
export async function loadFonts(families, faces) {
  const link = document.createElement("link");
  link.rel = "stylesheet";
  link.href = `https://fonts.googleapis.com/css2?${families.map((f) => "family=" + f).join("&")}&display=block`;
  document.head.appendChild(link);
  await new Promise((r) => { link.onload = r; link.onerror = r; setTimeout(r, 4000); });
  await Promise.all(faces.map((f) => document.fonts.load(f).catch(() => null)));
}

export function addCss(css) {
  const s = document.createElement("style");
  s.textContent = css;
  document.head.appendChild(s);
}

export function canvasTexture(w, h, draw, { repeat, srgb = true, aniso = 8 } = {}) {
  const c = document.createElement("canvas");
  c.width = w;
  c.height = h;
  const g = c.getContext("2d");
  draw(g, w, h);
  const t = new THREE.CanvasTexture(c);
  if (srgb) t.colorSpace = THREE.SRGBColorSpace;
  t.anisotropy = aniso;
  if (repeat) { t.wrapS = t.wrapT = THREE.RepeatWrapping; t.repeat.set(...repeat); }
  return t;
}

export function rng(seed = 1) {
  return () => {
    seed = (seed + 0x6d2b79f5) | 0;
    let t = Math.imul(seed ^ (seed >>> 15), 1 | seed);
    t = (t + Math.imul(t ^ (t >>> 7), 61 | t)) ^ t;
    return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
  };
}

// Soft blotchy noise (for roughness maps, smudges, stains): many blurred random dots.
export function blotches(g, w, h, { n = 400, rMin = 4, rMax = 40, alpha = 0.08, color = "255,255,255", seed = 3 } = {}) {
  const r = rng(seed);
  for (let i = 0; i < n; i++) {
    const x = r() * w, y = r() * h, rad = rMin + r() * (rMax - rMin);
    const grd = g.createRadialGradient(x, y, 0, x, y, rad);
    grd.addColorStop(0, `rgba(${color},${alpha * r()})`);
    grd.addColorStop(1, `rgba(${color},0)`);
    g.fillStyle = grd;
    g.fillRect(x - rad, y - rad, rad * 2, rad * 2);
  }
}

export function speckle(g, w, h, { n = 20000, colors = ["#000"], size = 1.5, alpha = 0.5, seed = 5 } = {}) {
  const r = rng(seed);
  for (let i = 0; i < n; i++) {
    g.globalAlpha = alpha * r();
    g.fillStyle = colors[(r() * colors.length) | 0];
    const s = size * (0.4 + r());
    g.fillRect(r() * w, r() * h, s, s);
  }
  g.globalAlpha = 1;
}

// Multi-line text, wrapped to maxW. Returns the y after the last line.
export function wrapText(g, text, x, y, maxW, lineH) {
  for (const para of text.split("\n")) {
    let line = "";
    for (const word of para.split(" ")) {
      const test = line ? line + " " + word : word;
      if (g.measureText(test).width > maxW && line) { g.fillText(line, x, y); y += lineH; line = word; }
      else line = test;
    }
    g.fillText(line, x, y);
    y += lineH;
  }
  return y;
}

// The final grade: chromatic aberration at the edges, lift/gain, vignette, film grain.
const GradeShader = {
  uniforms: {
    tDiffuse: { value: null }, uTime: { value: 0 }, uRes: { value: new THREE.Vector2(1, 1) },
    uGrain: { value: 0.04 }, uVignette: { value: 0.35 }, uAberration: { value: 0.006 },
    uLift: { value: new THREE.Vector3(0, 0, 0) }, uGain: { value: new THREE.Vector3(1, 1, 1) }, uSat: { value: 1 },
  },
  vertexShader: `varying vec2 vUv; void main() { vUv = uv; gl_Position = projectionMatrix * modelViewMatrix * vec4(position, 1.0); }`,
  fragmentShader: `
    uniform sampler2D tDiffuse; uniform float uTime, uGrain, uVignette, uAberration, uSat; uniform vec2 uRes;
    uniform vec3 uLift, uGain; varying vec2 vUv;
    float hash(vec2 p) { return fract(sin(dot(p, vec2(12.9898, 78.233))) * 43758.5453); }
    void main() {
      vec2 d = vUv - 0.5;
      float ab = uAberration * dot(d, d) * 4.0;
      vec3 c = vec3(texture2D(tDiffuse, vUv - d * ab).r, texture2D(tDiffuse, vUv).g, texture2D(tDiffuse, vUv + d * ab).b);
      float l = dot(c, vec3(0.299, 0.587, 0.114));
      c = mix(vec3(l), c, uSat);
      c = c * uGain + uLift * (1.0 - c);
      float v = smoothstep(0.95, 0.2, length(d * vec2(1.0, 1.15)));
      c *= mix(1.0 - uVignette, 1.0, v);
      c += (hash(floor(vUv * uRes) + fract(uTime * 7.13) * 91.0) - 0.5) * uGrain;
      gl_FragColor = vec4(c, 1.0);
    }`,
};

export class Post {
  constructor(renderer, scene, camera, { ao = true, aoRadius = 0.6, aoIntensity = 1, tiltShift = 0, bloom = null, grade = {} } = {}) {
    this.renderer = renderer;
    this.composer = new EffectComposer(renderer);
    this.composer.addPass(new RenderPass(scene, camera));
    if (ao) {
      this.ao = new GTAOPass(scene, camera, 1, 1);
      this.ao.updateGtaoMaterial({ radius: aoRadius, distanceExponent: 1.5, thickness: 1, scale: 1, samples: 16 });
      this.ao.blendIntensity = aoIntensity;
      this.composer.addPass(this.ao);
    }
    if (bloom) {
      this.bloom = new UnrealBloomPass(new THREE.Vector2(1, 1), bloom.strength ?? 0.5, bloom.radius ?? 0.6, bloom.threshold ?? 0.85);
      this.composer.addPass(this.bloom);
    }
    this.composer.addPass(new OutputPass());
    this.tilt = tiltShift;
    if (tiltShift) {
      this.th = new ShaderPass(HorizontalTiltShiftShader);
      this.tv = new ShaderPass(VerticalTiltShiftShader);
      this.composer.addPass(this.th);
      this.composer.addPass(this.tv);
    }
    this.grade = new ShaderPass(GradeShader);
    for (const [k, v] of Object.entries(grade)) {
      const u = this.grade.uniforms[k];
      if (u?.value?.set && Array.isArray(v)) u.value.set(...v);
      else if (u) u.value = v;
    }
    this.composer.addPass(this.grade);
    this.focus = 0.5;
  }

  setSize(w, h) {
    this.composer.setPixelRatio(this.renderer.getPixelRatio());
    this.composer.setSize(w, h);
    const dpr = this.renderer.getPixelRatio();
    this.grade.uniforms.uRes.value.set(w * dpr * 0.5, h * dpr * 0.5);
    if (this.th) { this.th.uniforms.h.value = 1 / (w * dpr); this.tv.uniforms.v.value = 1 / (h * dpr); }
  }

  // tilt: blur strength (0 = off). focus: screen height (0 bottom .. 1 top) that stays sharp.
  render(t, { tilt = this.tilt, focus } = {}) {
    if (this.th) {
      const W = innerWidth, H = innerHeight, dpr = this.renderer.getPixelRatio();
      this.th.enabled = this.tv.enabled = tilt > 0;
      this.th.uniforms.h.value = tilt / (W * dpr);
      this.tv.uniforms.v.value = tilt / (H * dpr);
      if (focus == null) {   // the middle of the space under the brain panel
        const band = Math.max(0, Math.min(H, document.getElementById("panel").getBoundingClientRect().bottom));
        focus = (H - band) / 2 / H;
      }
      this.focus += (focus - this.focus) * 0.08;
      this.th.uniforms.r.value = this.tv.uniforms.r.value = this.focus;
    }
    this.grade.uniforms.uTime.value = t;
    this.composer.render();
  }
}

// A light-box cone of light (for spot and fluorescent haze): additive, fades along its length.
export function lightCone(r0, r1, h, color, opacity = 0.12) {
  const geo = new THREE.CylinderGeometry(r0, r1, h, 48, 1, true).translate(0, -h / 2, 0);
  const mat = new THREE.ShaderMaterial({
    transparent: true, depthWrite: false, blending: THREE.AdditiveBlending, side: THREE.DoubleSide,
    uniforms: { uColor: { value: new THREE.Color(color) }, uOpacity: { value: opacity }, uH: { value: h } },
    vertexShader: `varying float vY; varying vec3 vN; varying vec3 vV;
      void main() { vY = position.y; vec4 mv = modelViewMatrix * vec4(position, 1.0);
        vN = normalize(normalMatrix * normal); vV = normalize(-mv.xyz); gl_Position = projectionMatrix * mv; }`,
    fragmentShader: `uniform vec3 uColor; uniform float uOpacity, uH; varying float vY; varying vec3 vN; varying vec3 vV;
      void main() { float along = 1.0 - clamp(-vY / uH, 0.0, 1.0);
        float rim = pow(abs(dot(normalize(vN), normalize(vV))), 1.5);
        gl_FragColor = vec4(uColor * uOpacity * along * along * rim, 1.0); }`,
  });
  return new THREE.Mesh(geo, mat);
}

// A station's footprint in its own frame (stations are built facing +z, at y = 0).
export function footprint(group) {
  const pos = group.position.clone(), rot = group.rotation.y;
  group.position.set(0, 0, 0);
  group.rotation.y = 0;
  group.updateMatrixWorld(true);
  const box = new THREE.Box3();
  group.traverse((o) => {
    if (!o.isMesh || o.userData.lookIgnore || !o.geometry) return;
    if (o.geometry.boundingBox == null) o.geometry.computeBoundingBox();
    const b = o.geometry.boundingBox.clone().applyMatrix4(o.matrixWorld);
    if (b.max.y < 6 && Math.abs(b.min.x) < 8 && Math.abs(b.max.z) < 8) box.union(b);   // ignore far-flung or sky-high bits
  });
  group.position.copy(pos);
  group.rotation.y = rot;
  group.updateMatrixWorld(true);
  return box;
}

// Centre and radius of the stations' half circle (it opens towards +z).
export function arcOf(flies) {
  let R = 5, cz = -Infinity;
  for (const f of flies) { R = Math.max(R, Math.abs(f.group.position.x)); cz = Math.max(cz, f.group.position.z); }
  return { R, center: new THREE.Vector3(0, 0, cz) };
}

// Ease the camera's field of view towards fov (looks use a long lens for the overview, the normal one up close).
export function easeFov(camera, fov, dt, rate = 3) {
  if (Math.abs(camera.fov - fov) < 0.01) return;
  camera.fov += (fov - camera.fov) * Math.min(1, dt * rate);
  camera.updateProjectionMatrix();
}
