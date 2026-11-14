"""Turn a film still into a Gaussian-splat scene for the 3D world (web/scenes/<scene>/).

    python scripts/scene.py clean <scene>            # still.jpg -> clean.jpg (people removed, OpenAI image edit)
    python scripts/scene.py splat <scene> [--draft]  # clean.jpg -> scene.spz + collider.glb + world.json (World Labs Marble)
    python scripts/scene.py all <scene>              # clean, then splat
    python scripts/scene.py map                      # web/assets/map/map.jpg: top-down satellite map of the fly city
    python scripts/scene.py aerial <scene>           # web/scenes/<scene>/aerial.jpg: satellite view of that building
    python scripts/scene.py credits                  # World Labs credit balance (free call, checks the key)

Keys come from the environment or the repo-root .env (never committed):
    OPENAI_API_KEY=...      # clean / map / aerial. Without it the prompt is printed for pasting into ChatGPT.
    WLT_API_KEY=...         # splat / credits (platform.worldlabs.ai/api-keys; WORLDLABS_API_KEY also accepted)

Costs: OpenAI bills per image (Sunburst at high quality ~ a few tens of cents). World Labs bills credits,
1,250 credits = $1: marble-1.1 image world = 1,580 (~$1.26), --draft = 230 (~$0.18), --plus = 1,580-3,080.
Docs: developers.openai.com/api/docs/guides/image-generation, docs.worldlabs.ai/api
"""
import argparse
import base64
import json
import os
import sys
import time
import urllib.error
import urllib.request
import uuid
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SCENES = ROOT / "web" / "scenes"
MAP = ROOT / "web" / "assets" / "map" / "map.jpg"

OPENAI = "https://api.openai.com/v1/images"
IMAGE_MODEL = "gpt-image-2.6-sunburst"   # newest GPT Image (Nov 2026); Sunburst = the precise-editing variant
WORLDLABS = "https://api.worldlabs.ai/marble/v1"
MARBLE = {"default": "marble-1.1", "draft": "marble-1.0-draft", "plus": "marble-1.1-plus"}
SPZ_ORDER = ["full_res", "500k", "150k", "100k"]   # best first

CLEAN_PROMPT = (
    "Remove every person from this film still. Keep the exact same camera position, framing, crop, lens and "
    "perspective, and keep the lighting, shadows, film grain, color grade and exposure identical. Do not move, "
    "add, restyle or remove any object, furniture, instrument, prop, wall, window or light. Fill in the areas "
    "the people covered with what would really be behind them (floor, walls, furniture), continuing the "
    "surrounding textures and lines naturally. The result must look like the same frame of the same empty "
    "room, photographed a moment before anyone walked in. No text, no watermark."
)
SATELLITE_STYLE = (
    "Photorealistic top-down satellite imagery, straight-down orthographic nadir view, crisp high-resolution "
    "aerial photograph in the style of the Grand Theft Auto V pause-menu satellite map: rich saturated colors, "
    "soft daylight shadows, sharp detail on every roof, tree, car and road marking. No text, no labels, no "
    "street names, no icons, no map pins, no legend, no UI, no borders, no watermark."
)
MAP_PROMPT = (
    "A huge sprawling city built for flies, seen from directly above. " + SATELLITE_STYLE + " The city mixes "
    "dense downtown blocks, suburbs with houses and backyards, a river with bridges, parks, a harbor, highways "
    "with interchanges, strip malls, a stadium and an airport, but scaled and themed for flies: fruit markets, "
    "sugar cubes, honey ponds, banana-peel rooftops, compost hills and window-screen fences tucked into a "
    "realistic urban layout. Thousands of buildings, every district detailed edge to edge."
)
AERIAL = {
    "living-room": "a single-story suburban American house with a pitched shingle roof, a front porch, a "
                   "driveway with an old sedan, a fenced backyard and big shade trees, on a quiet residential street",
    "band-room": "a grand old music conservatory building in Manhattan, New York: a stone and brick "
                 "pre-war block with a flat tar roof, rooftop water tower and HVAC units, a light well, "
                 "surrounded by busy avenues, yellow cabs and neighboring high-rises",
    "diamond-gym": "a small strip-mall gym building on a corner of Irvington Avenue in Maplewood, New Jersey: "
                   "a two-story brick commercial block with a flat roof and rooftop HVAC units, a striped "
                   "parking lot with cars, a row of neighboring storefronts, sidewalks, street trees and "
                   "suburban houses around it",
}
SQUARE = "2880x2880"   # largest square: pixel budget 8,294,400 (3840x2160), edges multiples of 16


# ---------------------------------------------------------------- helpers

def key(*names):
    env = {}
    if (ROOT / ".env").exists():
        for line in (ROOT / ".env").read_text().splitlines():
            k, sep, v = line.partition("=")
            if sep and not k.strip().startswith("#"):
                env[k.strip()] = v.strip().strip("'\"")
    for n in names:
        if os.environ.get(n) or env.get(n):
            return os.environ.get(n) or env[n]
    return None


def http(method, url, headers=None, data=None, timeout=600):
    if isinstance(data, dict):
        data, headers = json.dumps(data).encode(), {**(headers or {}), "Content-Type": "application/json"}
    req = urllib.request.Request(url, data=data, method=method, headers=headers or {})
    try:
        with urllib.request.urlopen(req, timeout=timeout) as r:
            return r.read()
    except urllib.error.HTTPError as e:
        raise SystemExit(f"{method} {url.split('?')[0]} -> HTTP {e.code}\n{e.read().decode(errors='replace')[:2000]}")


def download(url, dest):
    dest.parent.mkdir(parents=True, exist_ok=True)
    dest.write_bytes(http("GET", url))
    print(f"  saved {dest.relative_to(ROOT)} ({dest.stat().st_size / 1e6:.1f} MB)")


def scene_dir(scene):
    d = SCENES / scene
    if not d.is_dir():
        raise SystemExit(f"no such scene folder: {d.relative_to(ROOT)}  (have: {', '.join(p.name for p in SCENES.iterdir() if p.is_dir())})")
    return d


def jpeg_size(path):
    """(width, height) of a JPEG, read from its SOF header (stdlib only)."""
    b, i = path.read_bytes(), 2
    while i < len(b):
        marker, length = b[i + 1], int.from_bytes(b[i + 2:i + 4], "big")
        if 0xC0 <= marker <= 0xCF and marker not in (0xC4, 0xC8, 0xCC):
            return int.from_bytes(b[i + 7:i + 9], "big"), int.from_bytes(b[i + 5:i + 7], "big")
        i += 2 + length
    raise SystemExit(f"can't read JPEG size of {path}")


def biggest_size(w, h, max_edge=3840, max_px=8_294_400):
    """Largest WxH (multiples of 16, ratio <= 3:1) with the same aspect as the input, for the edit endpoint."""
    r = min(max(w / h, 1 / 3), 3)
    W = int(min(max_edge, max_edge * r, (max_px * r) ** 0.5)) // 16 * 16
    return f"{W}x{int(W / r) // 16 * 16}"


def multipart(fields, files):
    bnd = uuid.uuid4().hex
    out = b""
    for k, v in fields.items():
        out += f'--{bnd}\r\nContent-Disposition: form-data; name="{k}"\r\n\r\n{v}\r\n'.encode()
    for k, path in files:
        out += (f'--{bnd}\r\nContent-Disposition: form-data; name="{k}"; filename="{path.name}"\r\n'
                f"Content-Type: image/jpeg\r\n\r\n").encode() + path.read_bytes() + b"\r\n"
    return out + f"--{bnd}--\r\n".encode(), f"multipart/form-data; boundary={bnd}"


def no_openai_key(prompt, dest, attach=None):
    print("No OPENAI_API_KEY in the environment or .env, so nothing was generated.\n"
          "Do it by hand in ChatGPT instead:")
    if attach:
        print(f"  1. Attach {attach.relative_to(ROOT)}")
    print(f"  {'2' if attach else '1'}. Paste this prompt:\n\n{prompt}\n\n"
          f"  {'3' if attach else '2'}. Download the result and save it as {dest.relative_to(ROOT)} (JPG)")


def openai_image(prompt, dest, size, image=None, quality="high"):
    k = key("OPENAI_API_KEY")
    if not k:
        return no_openai_key(prompt, dest, image)
    fields = {"model": IMAGE_MODEL, "prompt": prompt, "size": size, "quality": quality,
              "output_format": "jpeg", "n": 1}
    print(f"  {IMAGE_MODEL} {'edit' if image else 'generate'} {size} quality={quality} (takes ~1-3 min)...")
    auth = {"Authorization": f"Bearer {k}"}
    if image:
        body, ctype = multipart(fields, [("image[]", image)])
        r = http("POST", f"{OPENAI}/edits", {**auth, "Content-Type": ctype}, body)
    else:
        r = http("POST", f"{OPENAI}/generations", auth, fields)
    dest.parent.mkdir(parents=True, exist_ok=True)
    dest.write_bytes(base64.b64decode(json.loads(r)["data"][0]["b64_json"]))
    print(f"  saved {dest.relative_to(ROOT)}")


# ---------------------------------------------------------------- commands

def clean(scene, quality="high"):
    d = scene_dir(scene)
    src = d / "still.jpg"
    if not src.exists():
        raise SystemExit(f"missing {src.relative_to(ROOT)}")
    openai_image(CLEAN_PROMPT, d / "clean.jpg", biggest_size(*jpeg_size(src)), image=src, quality=quality)


def splat(scene, model="default", prompt=None, res=None):
    d = scene_dir(scene)
    src = d / "clean.jpg" if (d / "clean.jpg").exists() else d / "still.jpg"
    k = key("WLT_API_KEY", "WORLDLABS_API_KEY")
    if not k:
        raise SystemExit("No WLT_API_KEY in the environment or .env. Make one at https://platform.worldlabs.ai/api-keys\n"
                         "and add a line WLT_API_KEY=... to the repo-root .env")
    auth = {"WLT-Api-Key": k}
    print(f"[{scene}] uploading {src.relative_to(ROOT)}")
    prep = json.loads(http("POST", f"{WORLDLABS}/media-assets:prepare_upload", auth,
                           {"file_name": f"{scene}.jpg", "kind": "image", "extension": "jpg"}))
    up = prep["upload_info"]
    http(up.get("upload_method", "PUT"), up["upload_url"],
         {"Content-Type": "image/jpeg", **(up.get("required_headers") or {})}, src.read_bytes())

    world_prompt = {"type": "image", "is_pano": "auto",
                    "image_prompt": {"source": "media_asset", "media_asset_id": prep["media_asset"]["media_asset_id"]}}
    if prompt:
        world_prompt["text_prompt"] = prompt
    op = json.loads(http("POST", f"{WORLDLABS}/worlds:generate", auth,
                         {"display_name": f"flyhack {scene}", "model": MARBLE[model],
                          "world_prompt": world_prompt, "permission": {"public": False}}))
    print(f"[{scene}] generating with {MARBLE[model]} (operation {op['operation_id']}, usually ~5 min)")
    t0 = time.time()
    while not op.get("done"):
        time.sleep(10)
        op = json.loads(http("GET", f"{WORLDLABS}/operations/{op['operation_id']}", auth))
        print(f"  {time.time() - t0:4.0f}s  {json.dumps(op.get('metadata') or {})[:120]}", flush=True)
    if op.get("error"):
        raise SystemExit(f"generation failed: {op['error']}")

    world = op.get("response") or {}
    if not (world.get("assets") or {}).get("splats"):
        world = json.loads(http("GET", f"{WORLDLABS}/worlds/{world.get('world_id') or op['metadata']['world_id']}", auth))
    (d / "world.json").write_text(json.dumps({"world": world, "operation": {**op, "response": None}}, indent=2))
    print(f"  saved {(d / 'world.json').relative_to(ROOT)}  cost: {json.dumps(op.get('cost'))}")

    assets = world.get("assets") or {}
    spz = (assets.get("splats") or {}).get("spz_urls") or {}
    pick = res if res in spz else next((r for r in SPZ_ORDER if r in spz), next(iter(spz), None))
    if not pick:
        raise SystemExit("world has no spz_urls; see world.json")
    print(f"[{scene}] splat resolution {pick} (available: {', '.join(spz)})")
    download(spz[pick], d / "scene.spz")
    collider = (assets.get("mesh") or {}).get("collider_mesh_url")
    if collider:
        download(collider, d / "collider.glb")
    else:
        print("  no collider mesh offered")
    print(f"Done. Marble viewer: {world.get('world_marble_url', '-')}")


def credits():
    k = key("WLT_API_KEY", "WORLDLABS_API_KEY")
    if not k:
        raise SystemExit("No WLT_API_KEY in the environment or .env")
    print(json.dumps(json.loads(http("GET", f"{WORLDLABS}/credits", {"WLT-Api-Key": k})), indent=2))


def main():
    ap = argparse.ArgumentParser(description="Film still -> Gaussian-splat scene (OpenAI image edit + World Labs Marble).",
                                 formatter_class=argparse.RawDescriptionHelpFormatter, epilog=__doc__)
    sub = ap.add_subparsers(dest="cmd", required=True)
    for name, help_ in [("clean", "still.jpg -> clean.jpg with people removed"),
                        ("splat", "clean.jpg (or still.jpg) -> scene.spz, collider.glb, world.json"),
                        ("all", "clean, then splat"),
                        ("aerial", "satellite photo of the building -> aerial.jpg")]:
        p = sub.add_parser(name, help=help_)
        p.add_argument("scene", help="folder name in web/scenes/, e.g. band-room")
        if name in ("clean", "all", "aerial"):
            p.add_argument("--quality", default="high", choices=["low", "medium", "high", "xhigh", "max"])
        if name in ("splat", "all"):
            g = p.add_mutually_exclusive_group()
            g.add_argument("--draft", action="store_true", help="marble-1.0-draft: 230 credits, rough")
            g.add_argument("--plus", action="store_true", help="marble-1.1-plus: bigger spaces, up to 3,080 credits")
            p.add_argument("--prompt", help="optional text prompt to guide the world")
            p.add_argument("--res", choices=SPZ_ORDER, help="splat resolution to save (default: best available)")
    p = sub.add_parser("map", help="top-down satellite map of the fly city -> web/assets/map/map.jpg")
    p.add_argument("--quality", default="high", choices=["low", "medium", "high", "xhigh", "max"])
    sub.add_parser("credits", help="World Labs credit balance (free)")
    a = ap.parse_args()

    if a.cmd in ("clean", "all"):
        clean(a.scene, a.quality)
    if a.cmd in ("splat", "all"):
        if a.cmd == "all" and not (scene_dir(a.scene) / "clean.jpg").exists():
            raise SystemExit("no clean.jpg yet (make it in ChatGPT with the prompt above), then run: splat " + a.scene)
        splat(a.scene, "draft" if a.draft else "plus" if a.plus else "default", a.prompt, a.res)
    if a.cmd == "map":
        openai_image(MAP_PROMPT, MAP, SQUARE, quality=a.quality)
    if a.cmd == "aerial":
        if a.scene not in AERIAL:
            raise SystemExit(f"no aerial prompt for {a.scene!r}; add one to AERIAL in scripts/scene.py")
        prompt = f"A satellite photo of {AERIAL[a.scene]}, centered in the frame. " + SATELLITE_STYLE
        openai_image(prompt, scene_dir(a.scene) / "aerial.jpg", SQUARE, quality=a.quality)
    if a.cmd == "credits":
        credits()


if __name__ == "__main__":
    main()
