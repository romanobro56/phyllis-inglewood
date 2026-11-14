# fly.glb: animated fruit fly (flybody)

`fly.glb` is the *Drosophila* body model from **flybody** (Google DeepMind & HHMI Janelia), exported as a rigid node
hierarchy with four embedded animation clips. Clip details, the walk speed and the node lists are in `meta.json`.

- **Nodes** use the flybody body names (`thorax`, `head`, `wing_left`, `coxa_T1_left`, `femur_T1_left`, ...,
  `claw_T3_right`) and follow the flybody body tree. Each body node has one child mesh node per material group. The
  mesh names match the flybody OBJ names (`head_red`, `thorax_black`, `wing_left_membrane`, `tarsal_claw_T1_left_brown`,
  `abdomen_1_body`, ...). Each mesh's glTF material is named after its group (`body`, `red`, `ocelli`, `black`,
  `bristle-brown`, `lower`, `brown`, `membrane`), so the app can swap materials by name.
- **Axes:** +Y is up, the fly faces +Z, and +X is the fly's left. The body (head front to abdomen tip) is 1.0 unit long
  and centred on the origin in X and Z. In the rest pose the feet stand on y = 0.
- **Size:** about 140k triangles (decimated from 272k), 2.8 MB, uncompressed. The file passes the Khronos glTF
  validator with no errors or warnings.
- **Rest pose:** the flybody standing pose, with the wings folded over the abdomen and the proboscis retracted.

| clip | data | notes |
|------|------|-------|
| `walk`  | **Real recording.** A freely walking fly in EPFL's Spotlight rig, from the 330 Hz sample in FlyGym v2 (`spotlight_behavior_clip.npz`). The 3D leg keypoints were retargeted onto the flybody legs with IK. | 0.545 s of the real recording, cut into a seamless loop. The body root stays in place, and the stance feet are pinned to y = 0. Speed is in `meta.json`. |
| `groom` | **Real recording.** A tethered fly doing antennal/head grooming, from the NeuroMechFly v1 DeepFly3D 3D pose (100 Hz). The front legs were retargeted with IK. | 1.56 s real loop. The mid and hind legs stay in the standing pose. |
| `idle`  | **Procedural.** Small head, antenna, haltere and abdomen "breathing" motion. | 4 s loop, legs planted. |
| `flap`  | **Procedural.** One cycle of flybody's built-in approximate wing-beat pattern. | Wings only. Play it fast. |

**Timing:** at playback rate 1, `walk` and `groom` run **4x slower than real time** (`timeScale = 4` gives real time).
`walk` moves at `walk_speed_body_lengths_per_s` (≈1.58 BL/s at rate 1, ≈6.3 BL/s real time). If you move the fly at
that speed times your timeScale, the feet don't slide. Keys are 60 fps with LINEAR interpolation, and the last key of
each loop equals the first.

## How it was made

`build_fly.py` does the following:

1. Loads flybody's `fruitfly.xml` with the MuJoCo Python bindings.
2. Reads each mesh geom in its body frame, decimates the larger meshes with `fast-simplification`, and converts from
   MuJoCo (z-up, +x forward) to glTF (y-up, +z forward) at a scale of 1/body length.
3. Retargets the recorded keypoints (thorax-coxa, coxa-trochanter, femur-tibia, tibia-tarsus, claw) to the flybody leg
   joints by least-squares IK in MuJoCo. Each leg keeps its recorded shape relative to its coxa, scaled to the flybody
   leg length.
4. Finds the best loop window in the recording, blends out the small mismatch at the loop ends, resamples to 60 fps,
   and writes per-node rotation tracks with `pygltflib`.

To re-run it:

```bash
mkdir -p /tmp/flyasset && cd /tmp/flyasset
git clone --depth 1 https://github.com/TuragaLab/flybody
git clone --depth 1 https://github.com/NeLy-EPFL/flygym          # Spotlight walking clip (v2.1.1)
git clone --depth 1 --filter=blob:none --no-checkout https://github.com/NeLy-EPFL/NeuroMechFly nmf
(cd nmf && git checkout HEAD -- LICENSE data/joint_tracking/grooming)
python3 -m venv venv && ./venv/bin/pip install mujoco numpy scipy trimesh fast-simplification pygltflib pillow
./venv/bin/python /path/to/flyhack/web/assets/fly/build_fly.py \
    --flybody /tmp/flyasset/flybody --flygym /tmp/flyasset/flygym --nmf /tmp/flyasset/nmf \
    --out /path/to/flyhack/web/assets/fly            # add --preview for MuJoCo contact sheets
```

## Attribution and licenses

- **flybody** model and meshes: Apache-2.0. © Google DeepMind / HHMI Janelia. See Vaxenburg et al., *Whole-body
  physics simulation of fruit fly locomotion*, Nature (2025), and https://github.com/TuragaLab/flybody.
- **FlyGym v2** Spotlight sample clip (`flygym_demo/spotlight_data`): Apache-2.0, NeLy lab (EPFL). See Wang-Chen et al.
  2024 (FlyGym/NeuroMechFly v2) and Wang-Chen et al. 2026 (Spotlight/PoseForge). https://github.com/NeLy-EPFL/flygym
- **NeuroMechFly v1** DeepFly3D grooming pose data: Apache-2.0, NeLy lab (EPFL). See Lobato-Rios et al., Nature Methods
  (2022), and https://github.com/NeLy-EPFL/NeuroMechFly.
- Not used: the flybody figshare datasets (measured wing-beat pattern, walking imitation, grooming poses). They are
  GPL-3.0+ and could not be downloaded here, so `flap` uses flybody's built-in approximate pattern, which is
  Apache-2.0 code.
