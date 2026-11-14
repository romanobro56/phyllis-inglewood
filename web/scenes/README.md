# Scenes

Each fly lives in a photoreal scene: one film still turned into a Gaussian splat by World Labs Marble,
drawn with Spark in `web/world.js`. Flies that aren't in any scene get a dark stage of their own.

```
web/scenes/index.json            the list of scenes
web/scenes/<scene>/still.jpg     the source frame (source.txt says where it came from)
web/scenes/<scene>/clean.jpg     the same frame with the people removed
web/scenes/<scene>/scene.spz     the splat (Marble)            collider.glb  its collision mesh
web/scenes/<scene>/aerial.jpg    the building from above, for the first step of the switch
web/scenes/<scene>/scene.json    camera, light, and where each fly stands
web/assets/map/map.jpg + map.json  the map the switch flies over, and each scene's pin on it
```

## Make a scene

```bash
python scripts/scene.py clean <scene>     # still.jpg -> clean.jpg (needs OPENAI_API_KEY, or do it in ChatGPT)
python scripts/scene.py splat <scene>     # clean.jpg -> scene.spz + collider.glb (needs WLT_API_KEY in .env)
python scripts/scene.py aerial <scene>    # aerial.jpg
python scripts/scene.py map               # web/assets/map/map.jpg
```

Until `scene.spz` exists the still is shown as a flat backdrop, so a scene can be laid out first.

## Place the flies

Open `http://localhost:8000/web/?edit&fly=<fly>`. The camera is free (drag to orbit, right-drag to pan).

| key | |
| --- | --- |
| shift + click | the fly stands where you clicked |
| `[` `]` | turn it |
| `-` `=` | smaller / bigger |
| `c` | this view becomes the fly's own frame |
| `v` | this view becomes the scene's frame (the default for its flies) |
| `k` | show Marble's collision mesh (clicks land on it) |
| `p` | copy the scene's `scene.json`; paste it into `web/scenes/<scene>/scene.json` |

Marble puts the photo's camera at the origin looking down -z, so `"camera": {"position": [0,0,0], "target": [0,0,-1]}`
is the original shot. Stay close to it: the splat only holds up within a few degrees of that view.
Station props (the boxes in `station.js`) are hidden here; the scene's own furniture stands in for them.
Set `"props": true` in a scene to show them.

## scene.json

```json
{
  "camera": { "position": [0, 0, 0], "target": [0, -0.3, -3], "fov": 50 },
  "splat": { "file": "scene.spz", "quaternion": [1, 0, 0, 0], "position": [0, 0, 0], "scale": 1 },
  "light": { "direction": [-0.5, 0.8, 0.4], "intensity": 1.2, "ambient": 0.35 },
  "flies": { "inbox": { "position": [0, -1.2, -3.2], "yaw": 2.6, "scale": 1.2, "camera": { "...": "optional" } } }
}
```
