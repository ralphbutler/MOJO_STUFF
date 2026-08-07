import * as THREE from "three";
import { OrbitControls } from "three/addons/controls/OrbitControls.js";

// ---------------------------------------------------------------- constants
const COL = {
  hider: 0x4f8cff, hiderDark: 0x2f5fd0,
  seeker: 0xff5a5a, seekerDark: 0xc83030,
  box: 0xffd24a, boxLocked: 0xff9d2e,
  ramp: 0x8d99ab, wall: 0x39414f, floor: 0x161b22,
};
const CELL = 1;

// ---------------------------------------------------------------- scene
const app = document.getElementById("app");
const renderer = new THREE.WebGLRenderer({ antialias: true });
renderer.setPixelRatio(Math.min(devicePixelRatio, 2));
renderer.shadowMap.enabled = true;
app.appendChild(renderer.domElement);

const scene = new THREE.Scene();
scene.background = new THREE.Color(0x0e1116);
scene.fog = new THREE.Fog(0x0e1116, 28, 60);

const camera = new THREE.PerspectiveCamera(50, 1, 0.1, 200);
const controls = new OrbitControls(camera, renderer.domElement);
controls.enableDamping = true;
controls.maxPolarAngle = Math.PI * 0.49;

scene.add(new THREE.HemisphereLight(0xbcd0ff, 0x202830, 0.9));
const sun = new THREE.DirectionalLight(0xffffff, 1.4);
sun.position.set(8, 16, 6);
sun.castShadow = true;
sun.shadow.mapSize.set(1024, 1024);
sun.shadow.camera.left = sun.shadow.camera.bottom = -20;
sun.shadow.camera.right = sun.shadow.camera.top = 20;
scene.add(sun);

// dynamic content lives under this group; rebuilt per trace
let world = new THREE.Group();
scene.add(world);

// ---------------------------------------------------------------- builders
function gridToWorld(x, y, n) {
  // center the board on the origin; y(grid) -> z(world)
  return [x - n / 2 + 0.5, y - n / 2 + 0.5];
}

function makeCharacter(team) {
  const g = new THREE.Group();
  const main = team === "hider" ? COL.hider : COL.seeker;
  const dark = team === "hider" ? COL.hiderDark : COL.seekerDark;
  const bodyMat = new THREE.MeshStandardMaterial({ color: main, roughness: 0.5, metalness: 0.1 });

  // rounded body (capsule) like the paper's spherical agents, a touch taller
  const body = new THREE.Mesh(new THREE.CapsuleGeometry(0.26, 0.22, 6, 14), bodyMat);
  body.position.y = 0.42; body.castShadow = true; g.add(body);

  // head
  const head = new THREE.Mesh(new THREE.SphereGeometry(0.2, 18, 14), bodyMat);
  head.position.y = 0.82; head.castShadow = true; g.add(head);

  // visor band (shows facing) + two eyes
  const visor = new THREE.Mesh(
    new THREE.BoxGeometry(0.3, 0.1, 0.06),
    new THREE.MeshStandardMaterial({ color: dark, roughness: 0.3 }));
  visor.position.set(0, 0.84, 0.17); g.add(visor);
  const eyeMat = new THREE.MeshStandardMaterial({ color: 0xffffff, emissive: 0x222222 });
  for (const dx of [-0.07, 0.07]) {
    const eye = new THREE.Mesh(new THREE.SphereGeometry(0.03, 8, 8), eyeMat);
    eye.position.set(dx, 0.84, 0.21); g.add(eye);
  }
  // little feet
  for (const dx of [-0.13, 0.13]) {
    const foot = new THREE.Mesh(new THREE.SphereGeometry(0.1, 10, 8), bodyMat);
    foot.position.set(dx, 0.08, 0); foot.castShadow = true; g.add(foot);
  }

  // "seen!" alarm ring (hidden unless spotted)
  const ring = new THREE.Mesh(
    new THREE.TorusGeometry(0.42, 0.04, 8, 24),
    new THREE.MeshBasicMaterial({ color: 0xff3030 }));
  ring.rotation.x = Math.PI / 2; ring.position.y = 1.12; ring.visible = false;
  g.add(ring); g.userData.ring = ring;
  g.userData.facing = g; // whole group rotates to face
  return g;
}

function makeBox() {
  const m = new THREE.Mesh(
    new THREE.BoxGeometry(0.86, 0.86, 0.86),
    new THREE.MeshStandardMaterial({ color: COL.box, roughness: 0.6 }));
  m.position.y = 0.43; m.castShadow = true; m.receiveShadow = true;
  return m;
}

function makeRamp() {
  // right-triangle prism wedge
  const shape = new THREE.Shape();
  shape.moveTo(-0.45, 0); shape.lineTo(0.45, 0); shape.lineTo(-0.45, 0.6); shape.lineTo(-0.45, 0);
  const geo = new THREE.ExtrudeGeometry(shape, { depth: 0.8, bevelEnabled: false });
  geo.translate(0, 0, -0.4);
  const m = new THREE.Mesh(geo, new THREE.MeshStandardMaterial({ color: COL.ramp, roughness: 0.8 }));
  m.castShadow = true; m.receiveShadow = true;
  return m;
}

function lockTint(mesh, locked, base, lockedColor) {
  mesh.material.color.setHex(locked ? lockedColor : base);
  mesh.material.emissive = new THREE.Color(locked ? 0x331a00 : 0x000000);
}

// ---------------------------------------------------------------- trace state
let trace = null, n = 12;
const nodes = { agents: {}, boxes: {}, ramps: {} };
let sightLines = [];
let frameIdx = 0, playing = true, speed = 1;
const FRAME_DUR = 0.42; // seconds per frame at 1x
let acc = 0;

function clearWorld() {
  scene.remove(world);
  world.traverse(o => { if (o.geometry) o.geometry.dispose(); });
  world = new THREE.Group(); scene.add(world);
  nodes.agents = {}; nodes.boxes = {}; nodes.ramps = {};
  sightLines.forEach(l => l.geometry.dispose());
  sightLines = [];
}

function buildWorld(t) {
  clearWorld();
  trace = t; n = t.meta.grid;

  // floor + grid
  const floor = new THREE.Mesh(
    new THREE.PlaneGeometry(n, n),
    new THREE.MeshStandardMaterial({ color: COL.floor, roughness: 1 }));
  floor.rotation.x = -Math.PI / 2; floor.receiveShadow = true;
  world.add(floor);
  const grid = new THREE.GridHelper(n, n, 0x2a3240, 0x222a36);
  grid.position.y = 0.01; world.add(grid);

  // walls
  const wallMat = new THREE.MeshStandardMaterial({ color: COL.wall, roughness: 0.9 });
  for (const [x, y] of t.walls) {
    const w = new THREE.Mesh(new THREE.BoxGeometry(CELL, 0.7, CELL), wallMat);
    const [wx, wz] = gridToWorld(x, y, n);
    w.position.set(wx, 0.35, wz); w.castShadow = true; w.receiveShadow = true;
    world.add(w);
  }

  const f0 = t.frames[0];
  for (const a of f0.agents) {
    const c = makeCharacter(a.team); world.add(c); nodes.agents[a.id] = c;
  }
  for (const b of f0.boxes) { const m = makeBox(); world.add(m); nodes.boxes[b.id] = m; }
  for (const r of f0.ramps) { const m = makeRamp(); world.add(m); nodes.ramps[r.id] = m; }

  // frame nav UI
  frameIdx = 0; acc = 0;
  scrub.max = String(t.frames.length - 1); scrub.value = "0";
  meta.innerHTML =
    `grid ${n}×${n} · ${t.frames.length} frames · prep ${t.meta.prep_turns} turns<br>` +
    `result: <b>${t.result.winner}</b> won · hidden ${(t.result.hider_reward * 100) | 0}% of seek phase`;
  applyFrame(0, 0);
}

// interpolate node transforms between frame i and i+1 by alpha
function applyFrame(i, alpha) {
  const A = trace.frames[i], B = trace.frames[Math.min(i + 1, trace.frames.length - 1)];
  const lerp = (a, b) => a + (b - a) * alpha;

  const posOf = (frame, list, id) => {
    const e = frame[list].find(x => x.id === id);
    return e ? e.pos : null;
  };

  for (const a of A.agents) {
    const node = nodes.agents[a.id]; if (!node) continue;
    const pb = posOf(B, "agents", a.id) || a.pos;
    const [x0, z0] = gridToWorld(a.pos[0], a.pos[1], n);
    const [x1, z1] = gridToWorld(pb[0], pb[1], n);
    node.position.set(lerp(x0, x1), 0, lerp(z0, z1));
    const f = a.face && (a.face[0] || a.face[1]) ? a.face : [0, 1];
    node.rotation.y = Math.atan2(f[0], f[1]);
    node.userData.ring.visible = !!a.seen;
  }
  // who is carrying what this frame, so held objects render ON the carrier
  const carriedBy = {};
  for (const a of A.agents) if (a.carry) carriedBy[a.carry] = a.id;

  for (const list of ["boxes", "ramps"]) {
    for (const o of A[list]) {
      const node = nodes[list][o.id]; if (!node) continue;
      if (list === "boxes") lockTint(node, o.locked, COL.box, COL.boxLocked);
      else node.material.color.setHex(o.locked ? 0xb0bdd0 : COL.ramp);
      const baseY = list === "boxes" ? 0.43 : 0.0;
      const carrier = carriedBy[o.id];
      if (carrier && nodes.agents[carrier]) {
        // hoist the held object above the carrier's head — unmistakably "carried",
        // never overlapping the body or trailing into the cell behind.
        const ap = nodes.agents[carrier].position;
        node.position.set(ap.x, baseY + 0.95, ap.z);
        continue;
      }
      const pb = posOf(B, list, o.id) || o.pos;
      const [x0, z0] = gridToWorld(o.pos[0], o.pos[1], n);
      const [x1, z1] = gridToWorld(pb[0], pb[1], n);
      node.position.set(lerp(x0, x1), baseY, lerp(z0, z1));
    }
  }
  // sightlines (no interpolation; snap)
  sightLines.forEach(l => world.remove(l));
  sightLines = [];
  for (const [sid, hid] of A.sight) {
    const s = nodes.agents[sid], h = nodes.agents[hid];
    if (!s || !h) continue;
    const geo = new THREE.BufferGeometry().setFromPoints([
      new THREE.Vector3(s.position.x, 0.8, s.position.z),
      new THREE.Vector3(h.position.x, 0.8, h.position.z)]);
    const line = new THREE.Line(geo, new THREE.LineBasicMaterial({ color: 0xff3030 }));
    world.add(line); sightLines.push(line);
  }

  const phase = A.phase === "prep" ? "🟡 prep" : "🔴 seek";
  tlabel.textContent = `${i} / ${trace.frames.length - 1}  ${phase}`;
}

// ---------------------------------------------------------------- controls
const scrub = document.getElementById("scrub");
const tlabel = document.getElementById("tlabel");
const meta = document.getElementById("meta");
const playBtn = document.getElementById("play");
const speedBtn = document.getElementById("speed");

playBtn.onclick = () => { playing = !playing; playBtn.textContent = playing ? "⏸ pause" : "▶ play"; };
speedBtn.onclick = () => {
  speed = ({ 1: 2, 2: 4, 4: 0.5 })[speed] || 1;
  speedBtn.textContent = speed + "×";
};
scrub.oninput = () => { playing = false; playBtn.textContent = "▶ play";
  frameIdx = +scrub.value; acc = 0; applyFrame(frameIdx, 0); };

// ---------------------------------------------------------------- loading
function loadTrace(obj) {
  try {
    buildWorld(obj);
    const r = (n * 0.9);
    camera.position.set(r, r * 0.85, r);
    controls.target.set(0, 0, 0);
    playing = true; playBtn.textContent = "⏸ pause";
  } catch (e) {
    meta.textContent = "Failed to load trace: " + e.message;
    console.error(e);
  }
}

document.getElementById("file").addEventListener("change", ev => {
  const f = ev.target.files[0]; if (!f) return;
  f.text().then(t => loadTrace(JSON.parse(t)));
});

const drop = document.getElementById("drop");
addEventListener("dragover", e => { e.preventDefault(); drop.style.display = "flex"; });
addEventListener("dragleave", e => { if (e.target === drop) drop.style.display = "none"; });
addEventListener("drop", e => {
  e.preventDefault(); drop.style.display = "none";
  const f = e.dataTransfer.files[0]; if (!f) return;
  f.text().then(t => loadTrace(JSON.parse(t)));
});

// VENDORED from HIDE_SEEK_LLM/viewer (PLAN.md decision 8). One change only:
// a ?trace= query parameter, so compare.html can point two iframes at a real
// trace and a dreamed one. Everything else is byte-identical upstream.
// on first open, prefer the most recent run (last_trace.json), then fall back to
// the bundled sample. Drag any traces/episode_*.json on to view a specific episode.
fetch(new URLSearchParams(location.search).get("trace") || "./last_trace.json")
  .then(r => r.ok ? r.json() : Promise.reject())
  .catch(() => fetch("./sample_trace.json").then(r => r.json()))
  .then(loadTrace)
  .catch(() => { meta.textContent = "Drop a trace JSON to begin."; });

// ---------------------------------------------------------------- loop
const clock = new THREE.Clock();
function tick() {
  requestAnimationFrame(tick);
  const dt = clock.getDelta();
  if (trace && playing) {
    acc += dt * speed;
    while (acc >= FRAME_DUR) {
      acc -= FRAME_DUR;
      frameIdx = (frameIdx + 1) % trace.frames.length;
      scrub.value = String(frameIdx);
    }
    applyFrame(frameIdx, Math.min(acc / FRAME_DUR, 1));
  }
  controls.update();
  renderer.render(scene, camera);
}
function resize() {
  camera.aspect = innerWidth / innerHeight; camera.updateProjectionMatrix();
  renderer.setSize(innerWidth, innerHeight);
}
addEventListener("resize", resize); resize(); tick();
