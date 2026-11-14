#!/usr/bin/env python3
"""Build web/assets/fly/fly.glb (+ meta.json) from the flybody fruit-fly model.

Geometry and body tree: TuragaLab/flybody (Apache-2.0), flybody/fruitfly/assets/fruitfly.xml.
Motion:
  walk  - REAL: Spotlight free-walking recording bundled with FlyGym v2 (Apache-2.0),
          flygym_demo/spotlight_data/assets/spotlight_behavior_clip.npz (330 Hz, 3D leg keypoints),
          retargeted onto the flybody legs by inverse kinematics, cut into a seamless loop.
  groom - REAL: NeuroMechFly v1 (Apache-2.0) DeepFly3D 3D pose of a tethered fly grooming
          (data/joint_tracking/grooming/fly1/df3d/pose_result__*.pkl, 100 Hz), front legs
          retargeted by IK; mid/hind legs held in the standing pose.
  idle  - PROCEDURAL (small head/antenna/abdomen motion).
  flap  - PROCEDURAL: one wing-beat cycle of flybody's built-in approximate wing pattern
          (flybody/tasks/pattern_generators.py, used when no measured pattern is supplied).

Usage (paths are the checkouts made in /tmp/flyasset):
  python build_fly.py --flybody /tmp/flyasset/flybody --flygym /tmp/flyasset/flygym \
      --nmf /tmp/flyasset/nmf --out /path/to/web/assets/fly [--preview]
Needs: mujoco numpy scipy trimesh fast-simplification pygltflib (pillow for --preview).
"""
import argparse, glob, json, os, pickle, struct
import numpy as np
import mujoco as mj
from scipy.optimize import least_squares
from scipy.signal import savgol_filter, medfilt
from scipy.interpolate import CubicSpline

FPS = 60            # key rate of every clip
SLOW = 4.0          # real clips play at 1/SLOW of real time at playback rate 1 (see meta.json)
TRI_BUDGET = 140_000
MATERIAL_SUFFIXES = ["bristle-brown", "membrane", "ocelli", "black", "brown", "lower", "red"]

LEGS = [("T1", "left"), ("T2", "left"), ("T3", "left"), ("T1", "right"), ("T2", "right"), ("T3", "right")]
LEG_TAGS = ["lf", "lm", "lh", "rf", "rm", "rh"]
LEG_BODIES = ["coxa", "femur", "tibia", "tarsus", "claw"]          # keypoint bodies ThC, CTr, FTi, TiTa, (claw tip)
LEG_JOINTS = ["coxa_abduct", "coxa_twist", "coxa", "femur_twist", "femur", "tibia", "tarsus"]
REST_SPRINGREF_JOINTS = ("wing_", "rostrum", "haustellum")      # folded wings, retracted proboscis


# ----------------------------------------------------------------------------- model
class Fly:
    def __init__(self, xml):
        self.m = mj.MjModel.from_xml_path(xml)
        self.d = mj.MjData(self.m)
        m = self.m
        self.bid = lambda n: mj.mj_name2id(m, mj.mjtObj.mjOBJ_BODY, n)
        self.jid = lambda n: mj.mj_name2id(m, mj.mjtObj.mjOBJ_JOINT, n)
        self.qa = lambda n: m.jnt_qposadr[self.jid(n)]
        q = m.qpos0.copy()
        q[:7] = [0, 0, 0, 1, 0, 0, 0]
        for j in range(m.njnt):
            if m.joint(j).name.startswith(REST_SPRINGREF_JOINTS):
                q[m.jnt_qposadr[j]] = m.qpos_spring[m.jnt_qposadr[j]]
        self.qrest = q

    def fk(self, q):
        self.d.qpos[:] = q
        mj.mj_kinematics(self.m, self.d)
        mj.mj_comPos(self.m, self.d)
        return self.d


# ----------------------------------------------------------------------------- IK retarget
class LegIK:
    """Least-squares IK of the 7 leg DOFs/leg onto 3D keypoints (thorax frame)."""

    def __init__(self, fly):
        self.f = fly; m = fly.m
        self.kb = [[fly.bid(f"{b}_{T}_{s}") for b in LEG_BODIES] for T, s in LEGS]
        jn = [[f"{j}_{T}_{s}" for j in LEG_JOINTS] for T, s in LEGS]
        self.qadr = np.array([[m.jnt_qposadr[fly.jid(n)] for n in js] for js in jn])
        self.vadr = np.array([[m.jnt_dofadr[fly.jid(n)] for n in js] for js in jn])
        self.lo = np.array([[m.jnt_range[fly.jid(n)][0] for n in js] for js in jn])
        self.hi = np.array([[m.jnt_range[fly.jid(n)][1] for n in js] for js in jn])
        # claw tip = claw-mesh vertex farthest from the claw body origin (body frame)
        self.tip = []
        for li in range(6):
            b = self.kb[li][4]
            pts = [geom_verts_body(m, g) for g in range(m.body_geomadr[b], m.body_geomadr[b] + m.body_geomnum[b])
                   if m.geom_type[g] == mj.mjtGeom.mjGEOM_MESH]
            v = np.concatenate(pts)
            self.tip.append(v[np.argmax(np.linalg.norm(v, axis=1))])

    def keypoints(self, q):
        d = self.f.fk(q)
        P = np.zeros((6, 5, 3))
        for li in range(6):
            for k in range(4):
                P[li, k] = d.xpos[self.kb[li][k]]
            b = self.kb[li][4]
            P[li, 4] = d.xpos[b] + d.xmat[b].reshape(3, 3) @ self.tip[li]
        Rt = d.xmat[1].reshape(3, 3)
        return (P - d.xpos[1]) @ Rt

    def solve(self, targets, legs=range(6), w=(0, 0.5, 1, 1, 2), reg=0.05):
        m = self.f.m; legs = list(legs); W = np.array(w)
        idx = self.qadr[legs].ravel(); vidx = self.vadr[legs].ravel()
        lo = self.lo[legs].ravel(); hi = self.hi[legs].ravel()
        qbase = self.f.qrest.copy()
        out = np.tile(qbase, (len(targets), 1))
        x = np.clip(qbase[idx], lo + 1e-6, hi - 1e-6)
        jacp = np.zeros((3, m.nv))
        for t, tgt in enumerate(targets):
            xprev = x.copy()

            def res(x):
                q = qbase.copy(); q[idx] = x; P = self.keypoints(q)
                r = [W[k] * (P[li, k] - tgt[li, k]) for li in legs for k in range(1, 5)]
                return np.concatenate([np.ravel(r), reg * (x - xprev)])

            def jac(x):
                q = qbase.copy(); q[idx] = x; d = self.f.fk(q)
                Rt = d.xmat[1].reshape(3, 3); rows = []
                for li in legs:
                    for k in range(1, 5):
                        b = self.kb[li][k]
                        pt = d.xpos[b] if k < 4 else d.xpos[b] + d.xmat[b].reshape(3, 3) @ self.tip[li]
                        mj.mj_jac(m, d, jacp, None, pt, b)
                        rows.append(W[k] * (Rt.T @ jacp[:, vidx]))
                return np.vstack(rows + [reg * np.eye(len(x))])

            x = least_squares(res, x, jac=jac, bounds=(lo, hi), method="trf", max_nfev=60, xtol=1e-7).x
            out[t, idx] = x
        return out


def frame_from_points(thc, claws):
    """Rows (fwd, left, up) from ThC points (6,3) and support-claw points (n,3)."""
    f = thc[[0, 3]].mean(0) - thc[[2, 5]].mean(0); f /= np.linalg.norm(f)
    up = thc.mean(0) - claws.mean(0); up -= f * (up @ f); up /= np.linalg.norm(up)
    return np.stack([f, np.cross(up, f), up])


def retarget_targets(rest_kp, K, R, ground_fix=False):
    """Map source keypoints K (T,6,5,3) to flybody thorax-frame targets.
    Each leg keeps its source shape relative to its ThC point, scaled to the flybody leg length."""
    Lfb = np.linalg.norm(np.diff(rest_kp, axis=1), axis=-1).sum(1)
    Ls = np.median(np.linalg.norm(np.diff(K, axis=2), axis=-1).sum(2), 0)
    s = Lfb / Ls
    T = rest_kp[None, :, 0:1] + s[None, :, None, None] * np.einsum("ij,tlkj->tlki", R, K - K[:, :, 0:1])
    if ground_fix:  # put each leg's stance phase on the flybody standing ground plane
        ground = rest_kp[:, 4, 2].mean()
        cz = T[:, :, 4, 2]
        dz = np.array([ground - np.median(cz[:, li][cz[:, li] <= np.percentile(cz[:, li], 40)]) for li in range(6)])
        T[:, :, 2:, 2] += dz[None, :, None]
        # pin stance claws onto the ground: heights below h1 -> 0, above h2 unchanged, smooth in between
        L = np.ptp(rest_kp[..., 0])          # rough body scale (model units)
        h1, h2 = 0.03 * L, 0.15 * L
        c = T[:, :, 4, 2] - ground
        u = np.clip((c - h1) / (h2 - h1), 0, 1)
        T[:, :, 4, 2] = ground + np.where(c > h1, c, 0.0) * (u * u * (3 - 2 * u))
    return T, s


def best_loop(Q, cols, Lmin, Lmax):
    """Pick [a, a+L) minimising pose+velocity mismatch between its ends."""
    X = Q[:, cols]; V = np.gradient(X, axis=0)
    best = None
    for L in range(Lmin, Lmax + 1):
        for a in range(0, len(X) - L - 1):
            c = np.linalg.norm(X[a] - X[a + L]) + 3 * np.linalg.norm(V[a] - V[a + L])
            if best is None or c < best[0]:
                best = (c, a, L)
    return best


def make_loop(Q, a, L, src_fps, cols):
    """Seamless loop: remove end mismatch with a linear ramp, then resample periodically at FPS (slowed)."""
    seg = Q[a:a + L + 1].copy()
    drift = seg[-1] - seg[0]
    ramp = np.linspace(0, 1, L + 1)[:, None]
    seg[:, cols] -= ramp * drift[None, cols]
    seg[-1] = seg[0]
    t_src = np.arange(L + 1) / src_fps
    dur_real = L / src_fps
    n = int(round(dur_real * SLOW * FPS))
    t = np.arange(n + 1) / n * dur_real
    out = np.tile(seg[0], (n + 1, 1))
    out[:, cols] = CubicSpline(t_src, seg[:, cols], bc_type="periodic")(t)
    out[-1] = out[0]
    return out, dur_real


# ----------------------------------------------------------------------------- clips
def clip_walk(fly, ik, flygym_dir):
    d = np.load(os.path.join(flygym_dir, "src/flygym_demo/spotlight_data/assets/spotlight_behavior_clip.npz"),
                allow_pickle=True)
    fps = float(d["data_fps"])
    K = d["fwdkin_egoxyz"].reshape(-1, 6, 5, 3).astype(float)     # ego frame: x fwd, y left, z up (ground z=0)
    assert list(d["legs"]) == LEG_TAGS
    a0, a1 = 380, 660                                                # regular straight-walking section
    K = K[a0:a1]
    K = medfilt(K, kernel_size=(5, 1, 1, 1))                         # kill single-frame tracking spikes
    K = savgol_filter(K, 9, 3, axis=0)
    rest_kp = ik.keypoints(fly.qrest)
    T, s = retarget_targets(rest_kp, K, np.eye(3), ground_fix=True)
    Q = ik.solve(T, w=(0, 0.5, 1, 1, 4))
    cols = ik.qadr.ravel()
    Q[:, cols] = savgol_filter(Q[:, cols], 7, 3, axis=0)
    c, a, L = best_loop(Q, cols, int(0.45 * fps), int(0.8 * fps))
    loop, dur_real = make_loop(Q, a, L, fps, cols)
    info = dict(source_frames=[int(d["framerange_in_raw_recording"][0]) + a0 + a,
                               int(d["framerange_in_raw_recording"][0]) + a0 + a + L],
                experiment_trial=str(d["experiment_trial"]), source_fps=fps,
                real_duration_s=dur_real, leg_scale_source_to_model=s.round(3).tolist(), loop_mismatch=float(c))
    return loop, info


def clip_groom(fly, ik, nmf_dir):
    f = glob.glob(os.path.join(nmf_dir, "data/joint_tracking/grooming/fly1/df3d/pose_result__*.pkl"))[0]
    P = pickle.load(open(f, "rb"))["points3d"]                       # (900, 38, 3), 100 Hz
    fps = 100.0
    # DeepFly3D layout: 0-14 one side (front, mid, hind x ThC,CTr,FTi,TiTa,Claw), 19-33 other side.
    K = np.stack([P[:, i:i + 5] for i in [0, 5, 10, 19, 24, 29]], 1).astype(float)
    sup = [1, 2, 4, 5]
    Fs = frame_from_points(K[:, :, 0].mean(0), K[:, sup, 4].mean(0))
    # side check: with a right-handed frame, the first block must be the fly's left
    side = (K[:, :3, 0].mean((0, 1)) - K[:, 3:, 0].mean((0, 1))) @ Fs[1]
    if side < 0:
        K = K[:, [3, 4, 5, 0, 1, 2]]
        Fs = frame_from_points(K[:, :, 0].mean(0), K[:, sup, 4].mean(0))
    rest_kp = ik.keypoints(fly.qrest)
    Ff = frame_from_points(rest_kp[:, 0], rest_kp[sup, 4])
    a0, a1 = 150, 380                                                # antennal/head grooming bout
    K = savgol_filter(K[a0:a1], 7, 3, axis=0)
    T, s = retarget_targets(rest_kp, K, Ff.T @ Fs)
    Q = ik.solve(T, legs=[0, 3])
    cols = ik.qadr[[0, 3]].ravel()
    Q[:, cols] = savgol_filter(Q[:, cols], 5, 3, axis=0)
    c, a, L = best_loop(Q, cols, int(1.2 * fps), int(2.1 * fps))
    loop, dur_real = make_loop(Q, a, L, fps, cols)
    info = dict(source_file=os.path.relpath(f, nmf_dir), source_frames=[a0 + a, a0 + a + L], source_fps=fps,
                real_duration_s=dur_real, leg_scale_source_to_model=s.round(3).tolist(), loop_mismatch=float(c))
    return loop, info


def clip_idle(fly, dur=4.0):
    n = int(dur * FPS); t = np.arange(n + 1) / FPS
    Q = np.tile(fly.qrest, (n + 1, 1))
    w = lambda k: 2 * np.pi * k / dur          # integer cycles per loop => seamless
    def add(j, v): Q[:, fly.qa(j)] += v
    add("head", 0.05 * np.sin(w(1) * t) + 0.015 * np.sin(w(3) * t + 1))
    add("head_abduct", 0.06 * np.sin(w(1) * t + 0.7))
    add("head_twist", 0.04 * np.sin(w(2) * t + 2.0))
    for side, ph in (("left", 0.0), ("right", 1.3)):
        add(f"antenna_{side}", 0.08 * np.sin(w(3) * t + ph) + 0.04 * np.sin(w(7) * t + 2 * ph))
        add(f"antenna_abduct_{side}", 0.06 * np.sin(w(2) * t + ph))
    for i, j in enumerate(["abdomen", "abdomen_2", "abdomen_3", "abdomen_4", "abdomen_5", "abdomen_6", "abdomen_7"]):
        add(j, 0.012 * np.sin(w(2) * t - 0.3 * i))                   # breathing ripple
    add("abdomen_abduct", 0.02 * np.sin(w(1) * t))
    for side, ph in (("left", 0.0), ("right", 0.5)):
        add(f"haltere_{side}", 0.05 * np.sin(w(4) * t + ph))
    # occasional front-tarsus tap is skipped: legs stay planted
    return Q


def clip_flap(fly, n=30):
    x = np.linspace(0, 2 * np.pi, n + 1)
    yaw = 1.1 * np.sin(x - np.pi / 2) + 0.3          # flybody WingBeatPatternGenerator default pattern
    roll = 0.25 * np.sin(1.5 * x) - 0.1
    pitch = 1.35 * np.sin(x) + 0.8
    Q = np.tile(fly.qrest, (n + 1, 1))
    for side in ("left", "right"):
        Q[:, fly.qa(f"wing_yaw_{side}")] = yaw
        Q[:, fly.qa(f"wing_roll_{side}")] = roll
        Q[:, fly.qa(f"wing_pitch_{side}")] = pitch
    Q[-1] = Q[0]
    return Q


# ----------------------------------------------------------------------------- geometry
def geom_verts_body(m, g):
    mid = m.geom_dataid[g]
    v = m.mesh_vert[m.mesh_vertadr[mid]:m.mesh_vertadr[mid] + m.mesh_vertnum[mid]].astype(float)
    R = np.zeros(9); mj.mju_quat2Mat(R, m.geom_quat[g])
    return v @ R.reshape(3, 3).T + m.geom_pos[g]


def geom_faces(m, g):
    mid = m.geom_dataid[g]
    return m.mesh_face[m.mesh_faceadr[mid]:m.mesh_faceadr[mid] + m.mesh_facenum[mid]].copy()


def material_of(meshname):
    for s in MATERIAL_SUFFIXES:
        if meshname.endswith("_" + s):
            return s
    return "body"


# glTF axes: X = fly left (mj +y), Y = up (mj +z), Z = forward (mj +x). Proper rotation (det +1).
PERM = np.array([[0, 1, 0], [0, 0, 1], [1, 0, 0]], float)


def to_gl_vec(v):
    return v @ PERM.T


def to_gl_quat(q):  # mujoco (w,x,y,z) -> glTF (x,y,z,w) in permuted axes
    q = np.atleast_2d(q)
    out = np.stack([q[:, 2], q[:, 3], q[:, 1], q[:, 0]], 1)
    return out


def quat_conj(q): return q * np.array([1, -1, -1, -1])


def quat_mul(a, b):
    w1, x1, y1, z1 = a.T; w2, x2, y2, z2 = b.T
    return np.stack([w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2, w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2,
                     w1 * y2 - x1 * z2 + y1 * w2 + z1 * x2, w1 * z2 + x1 * y2 - y1 * x2 + z1 * w2], 1)


def local_trs(fly, Q):
    """Per-frame local (pos, quat) of every body (index 1..nbody-1) in mujoco axes."""
    m = fly.m; nb = m.nbody
    pos = np.zeros((len(Q), nb, 3)); quat = np.zeros((len(Q), nb, 4))
    for t, q in enumerate(Q):
        d = fly.fk(q)
        xp, xq = d.xpos.copy(), d.xquat.copy()
        par = m.body_parentid
        for b in range(1, nb):
            p = par[b]
            if p == 0:
                pos[t, b], quat[t, b] = xp[b], xq[b]
            else:
                Rp = d.xmat[p].reshape(3, 3)
                pos[t, b] = Rp.T @ (xp[b] - xp[p])
                quat[t, b] = quat_mul(quat_conj(xq[p][None]), xq[b][None])[0]
    quat /= np.linalg.norm(quat, axis=-1, keepdims=True).clip(1e-12)
    # keep quaternion sign continuous for clean slerp
    for t in range(1, len(Q)):
        flip = (quat[t] * quat[t - 1]).sum(-1) < 0
        quat[t, flip] *= -1
    return pos, quat


# ----------------------------------------------------------------------------- glTF writer
class Bin:
    def __init__(self): self.buf = bytearray(); self.views = []; self.accs = []

    def add(self, arr, comp, typ, target=None, minmax=False):
        import pygltflib as G
        arr = np.ascontiguousarray(arr)
        while len(self.buf) % 4: self.buf.append(0)
        off = len(self.buf); self.buf += arr.tobytes()
        self.views.append(G.BufferView(buffer=0, byteOffset=off, byteLength=arr.nbytes, target=target))
        cnt = arr.shape[0]
        acc = G.Accessor(bufferView=len(self.views) - 1, componentType=comp, count=cnt, type=typ)
        if minmax:
            flat = arr.reshape(cnt, -1)
            acc.min = flat.min(0).astype(float).tolist(); acc.max = flat.max(0).astype(float).tolist()
        self.accs.append(acc)
        return len(self.accs) - 1


def build(args):
    import pygltflib as G
    import trimesh, fast_simplification

    fly = Fly(os.path.join(args.flybody, "flybody/fruitfly/assets/fruitfly.xml"))
    m = fly.m
    ik = LegIK(fly)

    # ---- rest pose, scale, placement
    d = fly.fk(fly.qrest)
    body_names = [m.body(b).name for b in range(m.nbody)]
    mesh_geoms = [g for g in range(m.ngeom) if m.geom_type[g] == mj.mjtGeom.mjGEOM_MESH]
    trunk = ("thorax", "head", "rostrum", "haustellum", "labrum", "antenna", "abdomen")
    def world_verts(g):
        b = m.geom_bodyid[g]
        return geom_verts_body(m, g) @ d.xmat[b].reshape(3, 3).T + d.xpos[b]
    tv = np.concatenate([world_verts(g) for g in mesh_geoms if body_names[m.geom_bodyid[g]].startswith(trunk)])
    L = tv[:, 0].max() - tv[:, 0].min()
    claw_v = np.concatenate([world_verts(g) for g in mesh_geoms if body_names[m.geom_bodyid[g]].startswith("claw")])
    center = np.array([(tv[:, 0].max() + tv[:, 0].min()) / 2, (tv[:, 1].max() + tv[:, 1].min()) / 2, claw_v[:, 2].min()])
    S = 1.0 / L
    print(f"body length {L:.4f} model units (cm); scale {S:.3f}")

    # ---- meshes (with decimation)
    raw = []
    for g in mesh_geoms:
        mn = m.mesh(m.geom_dataid[g]).name
        mat = material_of(mn)
        name = mn if mn.endswith("_" + mat) else f"{mn}_body"
        raw.append((g, name, mat, geom_verts_body(m, g), geom_faces(m, g)))
    total = sum(len(f) for *_, f in raw)
    special = {"head_ocelli": 0.12, "head_red": 0.30}
    small = sum(len(f) for _, n, _, _, f in raw if len(f) < 800 and n not in special)
    spec_tris = sum(len(f) * special[n] for _, n, _, _, f in raw if n in special)
    big = sum(len(f) for _, n, _, _, f in raw if len(f) >= 800 and n not in special)
    keep = min(1.0, (TRI_BUDGET - small - spec_tris) / big)
    print(f"triangles before {total}; keep fraction for meshes >=800 tris: {keep:.3f}")
    meshes = []
    for g, name, mat, v, f in raw:
        tm = trimesh.Trimesh(v, f, process=True)
        frac = special.get(name, keep if len(f) >= 800 else 1.0)
        if frac < 0.999:
            vv, ff = fast_simplification.simplify(tm.vertices.astype(np.float32), tm.faces.astype(np.int64), 1 - frac)
            tm = trimesh.Trimesh(vv, ff, process=True)
        tm.update_faces(tm.nondegenerate_faces())
        tm.remove_unreferenced_vertices()
        meshes.append((m.geom_bodyid[g], name, mat, tm))
    ntri = sum(len(t.faces) for *_, t in meshes)
    print(f"triangles after {ntri}")

    # ---- clips
    clips = {}; clip_info = {}
    Qw, iw = clip_walk(fly, ik, args.flygym); clips["walk"] = Qw; clip_info["walk"] = iw
    Qg, ig = clip_groom(fly, ik, args.nmf); clips["groom"] = Qg; clip_info["groom"] = ig
    clips["idle"] = clip_idle(fly); clip_info["idle"] = {}
    clips["flap"] = clip_flap(fly); clip_info["flap"] = {}
    rest_pos, rest_quat = local_trs(fly, fly.qrest[None])

    def gl_pos(b, p):  # local position in glTF space, root offset applied for thorax
        return to_gl_vec((p - center) * S) if m.body_parentid[b] == 0 else to_gl_vec(p * S)

    # ---- walk speed from stance claws (in body lengths per clip-second at playback rate 1)
    kps = np.array([ik.keypoints(q) for q in Qw])                   # thorax frame, model units
    tip = kps[:, :, 4]; dt = 1.0 / FPS
    vx = np.gradient(tip[:, :, 0], axis=0) / dt
    ground = ik.keypoints(fly.qrest)[:, 4, 2].mean()
    stance = (tip[:, :, 2] < ground + 0.02 * L) & (vx < 0)
    speed_units = float(np.median(-vx[stance]))
    walk_speed_bl = speed_units * S
    print(f"walk speed {walk_speed_bl:.3f} BL/s at playback 1 ({walk_speed_bl * SLOW:.2f} BL/s real time)")
    stance_err = float(np.percentile(np.abs(tip[:, :, 2][stance] - ground) * S, 90))

    # ---- assemble glTF
    B = Bin(); gl = G.GLTF2(asset=G.Asset(generator="flyhack build_fly.py", version="2.0"))
    mat_rgba = {}
    for i in range(m.nmat):
        mat_rgba[m.mat(i).name] = (m.mat_rgba[i].tolist(), float(m.mat_shininess[i]))
    mat_index = {}
    for mat in sorted({x[2] for x in meshes}):
        rgba, shin = mat_rgba.get(mat, ([0.5, 0.5, 0.5, 1], 0.5))
        gm = G.Material(name=mat, doubleSided=mat in ("membrane", "brown", "bristle-brown", "black"),
                        pbrMetallicRoughness=G.PbrMetallicRoughness(baseColorFactor=rgba, metallicFactor=0.0,
                                                                    roughnessFactor=round(1 - 0.7 * shin, 3)))
        if rgba[3] < 1: gm.alphaMode = "BLEND"
        mat_index[mat] = len(gl.materials); gl.materials.append(gm)

    node_of_body = {}
    for b in range(1, m.nbody):
        n = G.Node(name=body_names[b], translation=gl_pos(b, rest_pos[0, b]).tolist(),
                   rotation=np.clip(to_gl_quat(rest_quat[0, b])[0].astype(np.float32), -1, 1).astype(float).tolist(), children=[])
        node_of_body[b] = len(gl.nodes); gl.nodes.append(n)
    for b in range(1, m.nbody):
        p = m.body_parentid[b]
        if p: gl.nodes[node_of_body[p]].children.append(node_of_body[b])
    for b, name, mat, tm in meshes:
        V = to_gl_vec(np.asarray(tm.vertices) * S).astype(np.float32)
        N = to_gl_vec(np.asarray(tm.vertex_normals))
        nl = np.linalg.norm(N, axis=1, keepdims=True)
        N = np.where(nl > 1e-6, N / np.maximum(nl, 1e-12), [0.0, 1.0, 0.0]).astype(np.float32)
        F = np.asarray(tm.faces)
        F = F.astype(np.uint16 if len(V) < 65535 else np.uint32)
        ia = B.add(F.reshape(-1), G.UNSIGNED_SHORT if F.dtype == np.uint16 else G.UNSIGNED_INT, G.SCALAR, G.ELEMENT_ARRAY_BUFFER)
        pa = B.add(V, G.FLOAT, G.VEC3, G.ARRAY_BUFFER, minmax=True)
        na = B.add(N, G.FLOAT, G.VEC3, G.ARRAY_BUFFER)
        gl.meshes.append(G.Mesh(name=name, primitives=[G.Primitive(attributes=G.Attributes(POSITION=pa, NORMAL=na),
                                                                    indices=ia, material=mat_index[mat])]))
        gl.nodes.append(G.Node(name=name, mesh=len(gl.meshes) - 1))
        gl.nodes[node_of_body[b]].children.append(len(gl.nodes) - 1)
    for n in gl.nodes:
        if n.children == []: n.children = None

    anim_meta = {}
    for cname, Q in clips.items():
        pos, quat = local_trs(fly, Q)
        nT = len(Q); times = (np.arange(nT) / FPS).astype(np.float32)
        ta = B.add(times, G.FLOAT, G.SCALAR, minmax=True)
        samplers, channels, animated = [], [], []
        for b in range(1, m.nbody):
            dq = 1 - np.abs((quat[:, b] * rest_quat[0, b]).sum(-1))
            dp = np.abs(pos[:, b] - rest_pos[0, b]).max()
            if dq.max() > 1e-7:
                oa = B.add(to_gl_quat(quat[:, b]).astype(np.float32), G.FLOAT, G.VEC4)
                samplers.append(G.AnimationSampler(input=ta, output=oa, interpolation="LINEAR"))
                channels.append(G.AnimationChannel(sampler=len(samplers) - 1,
                                                   target=G.AnimationChannelTarget(node=node_of_body[b], path="rotation")))
                animated.append(body_names[b])
            if dp > 1e-9:
                oa = B.add(np.array([gl_pos(b, p) for p in pos[:, b]], np.float32), G.FLOAT, G.VEC3)
                samplers.append(G.AnimationSampler(input=ta, output=oa, interpolation="LINEAR"))
                channels.append(G.AnimationChannel(sampler=len(samplers) - 1,
                                                   target=G.AnimationChannelTarget(node=node_of_body[b], path="translation")))
        anim_meta[cname] = dict(duration_s=float(times[-1]), keys=nT, fps=FPS, loop=True, animated_nodes=len(set(animated)))
        if cname == "walk": anim_meta[cname]["speed_body_lengths_per_s"] = round(walk_speed_bl, 4)
        gl.animations.append(G.Animation(name=cname, samplers=samplers, channels=channels,
                                         extras=dict(anim_meta[cname])))

    root = node_of_body[1]
    wing_nodes = ["wing_left", "wing_right"]
    leg_tips = [f"claw_{T}_{s}" for T, s in LEGS]
    gl.scenes = [G.Scene(name="fly", nodes=[root], extras=dict(walk_speed_body_lengths_per_s=round(walk_speed_bl, 4),
                                                               wing_nodes=wing_nodes, leg_tip_nodes=leg_tips))]
    gl.scene = 0
    gl.buffers = [G.Buffer(byteLength=len(B.buf))]
    gl.bufferViews = B.views; gl.accessors = B.accs
    gl.set_binary_blob(bytes(B.buf))
    os.makedirs(args.out, exist_ok=True)
    out_glb = os.path.join(args.out, "fly.glb")
    gl.save_binary(out_glb)
    size = os.path.getsize(out_glb)
    print(f"wrote {out_glb} ({size / 1e6:.2f} MB)")

    meta = dict(
        source="flybody fruit fly model (TuragaLab/flybody, Google DeepMind & HHMI Janelia), "
               "fruitfly.xml + per-part OBJ meshes; body tree and node names are flybody body names",
        license="Model geometry: Apache-2.0 (flybody). Motion data: Apache-2.0 (FlyGym v2 Spotlight sample clip; "
                "NeuroMechFly v1 DeepFly3D grooming data). See README.md.",
        units=dict(length="1.0 = fly body length, head front to abdomen tip", up="+Y", forward="+Z",
                   left="+X", feet_at_y=0.0, flybody_cm_per_unit=round(L, 5)),
        triangle_count=int(ntri), triangle_count_original=int(total), file_size_bytes=size,
        playback_note=f"Real-data clips (walk, groom) run {SLOW:g}x slower than real time at playback rate 1 "
                      f"(set timeScale={SLOW:g} for real time). idle is in natural time. flap is one wing-beat "
                      f"cycle; real flies beat ~200 Hz, so play it as fast as you like.",
        real_time_factor=dict(walk=1 / SLOW, groom=1 / SLOW, idle=1.0, flap=None),
        walk_speed_body_lengths_per_s=round(walk_speed_bl, 4),
        walk_speed_body_lengths_per_s_real_time=round(walk_speed_bl * SLOW, 4),
        walk_foot_plant_error_90pct_body_lengths=round(stance_err, 4),
        clips=dict(
            walk=dict(anim_meta["walk"], data="REAL recorded: freely walking fly (Spotlight rig, EPFL Neuroengineering lab), "
                      "3D leg keypoints from FlyGym v2 spotlight_behavior_clip.npz, retargeted to flybody legs by IK; "
                      "seamless loop cut from the recording (small end mismatch blended out); body root held in place",
                      real=True, **clip_info["walk"]),
            groom=dict(anim_meta["groom"], data="REAL recorded: tethered fly antennal/head grooming (optogenetic aDN "
                       "activation), NeuroMechFly v1 DeepFly3D 3D pose, front legs retargeted to flybody by IK; "
                       "mid/hind legs held in the standing pose; loop cut from the recording",
                       real=True, **clip_info["groom"]),
            idle=dict(anim_meta["idle"], data="PROCEDURAL: small head, antenna, haltere and abdomen-breathing motion; legs planted",
                      real=False),
            flap=dict(anim_meta["flap"], data="PROCEDURAL: one cycle of flybody's built-in approximate wing-beat pattern "
                      "(WingBeatPatternGenerator default; the measured pattern is in the GPL figshare dataset, not used); wings only",
                      real=False)),
        wing_nodes=wing_nodes, leg_tip_nodes=leg_tips,
        rest_pose="flybody default standing pose; wings folded over the abdomen and proboscis retracted (joint springref)",
    )
    with open(os.path.join(args.out, "meta.json"), "w") as fh:
        json.dump(meta, fh, indent=2)
    if args.preview:
        preview(fly, clips, os.path.join(args.preview_dir or "/tmp/flyasset/work", "preview"))
    return meta


def preview(fly, clips, prefix):
    import PIL.Image as Image
    m = fly.m; r = mj.Renderer(m, 240, 320)
    for cname, Q in clips.items():
        rows = []
        for el, az in ((-5, 90), (-60, 150)):
            ims = []
            for t in np.linspace(0, len(Q) - 1, 6).astype(int):
                d = fly.fk(Q[t]); cam = mj.MjvCamera(); cam.azimuth = az; cam.elevation = el
                cam.distance = 0.6; cam.lookat[:] = d.xpos[1]; r.update_scene(d, cam); ims.append(r.render())
            rows.append(np.concatenate(ims, 1))
        Image.fromarray(np.concatenate(rows, 0)).save(f"{prefix}_{cname}.png")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--flybody", required=True); ap.add_argument("--flygym", required=True)
    ap.add_argument("--nmf", required=True); ap.add_argument("--out", required=True)
    ap.add_argument("--preview", action="store_true"); ap.add_argument("--preview-dir")
    build(ap.parse_args())
