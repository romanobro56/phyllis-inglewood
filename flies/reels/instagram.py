"""Give Scrolly its own Instagram: a Chrome window on instagram.com/reels/, shown live on the 3D phone.

    python flies/reels/instagram.py              # opens Chrome; log in to the fly's account the first time
    python flies/reels/instagram.py --headless   # later, once logged in: no window

Leave it running next to `python -m server.serve`. The station connects to it, streams this Chrome tab onto
the phone, and forwards the brain's choices as real gestures on Instagram:
  scroll_<N>s  -> after N seconds, scroll to the next reel
  comments     -> open the comments, scroll them down, close them
  like         -> WATCH-ONLY: the heart is drawn on our phone, nothing is sent to Instagram
The brain still decides from its own simulated feed (task.py), not from what is actually on Instagram.

If this isn't running, is logged out, or the account is blocked, the phone falls back to the videos in
raw/<kind>/ (see media.py) and then to the cartoon reels.

The login lives in a separate Chrome profile in raw/chrome-profile (git-ignored, never uploaded to the GPU).
You type the password yourself in the Chrome window; this script never sees it.
Instagram's terms don't allow automated accounts, so expect the account to get flagged eventually.

Needs Google Chrome (or Chromium) installed; uses only Python packages the repo already has.
"""
import argparse
import asyncio
import base64
import json
import shutil
import signal
import subprocess
import sys
import time
import urllib.request
from http import HTTPStatus
from pathlib import Path

from websockets.asyncio.client import connect
from websockets.asyncio.server import serve
from websockets.exceptions import ConnectionClosed

HERE = Path(__file__).resolve().parent
RAW = HERE / "raw"
PROFILE = RAW / "chrome-profile"
MANIFEST = RAW / "reels.json"            # the station reads this; we add {"instagram": {"port": ...}} while running
START_URL = "https://www.instagram.com/reels/"
VIEW = (360, 640)                        # the phone screen's layout size in CSS px (9:16, like the station canvas)
CHROMES = ["/Applications/Google Chrome.app/Contents/MacOS/Google Chrome",
           "/Applications/Chromium.app/Contents/MacOS/Chromium",
           "google-chrome", "google-chrome-stable", "chromium", "chromium-browser",
           r"C:\Program Files\Google\Chrome\Application\chrome.exe"]

# Runs in the Instagram page every second: where are we, and where are the buttons of the reel in view.
STATUS_JS = r"""(() => {
  const W = innerWidth, H = innerHeight;
  const vis = (r) => r.width > 0 && r.height > 0 && r.bottom > 0 && r.top < H && r.right > 0 && r.left < W;
  const find = (labels) => {
    let best = null, bd = 1e9;
    for (const l of labels) for (const s of document.querySelectorAll(`svg[aria-label="${l}"]`)) {
      const r = s.getBoundingClientRect();
      if (!vis(r)) continue;
      const d = Math.abs((r.top + r.bottom) / 2 - H * 0.62);
      if (d < bd) { bd = d; best = r; }
    }
    return best && [(best.left + best.width / 2) / W, (best.top + best.height / 2) / H];
  };
  const path = location.pathname, text = ((document.body && document.body.innerText) || "").slice(0, 4000);
  return {
    path,
    login: /^\/accounts\/(login|emailsignup)/.test(path) || !!document.querySelector('input[name="password"]'),
    blocked: /^\/(challenge|accounts\/suspended|accounts\/disabled)/.test(path) || /(suspended|disabled) your account/i.test(text),
    video: !!document.querySelector("video"),
    ui: { comment: find(["Comment"]), like: find(["Like", "Unlike"]), close: find(["Close"]) },
  };
})()"""

# Centre of the innermost scrollable box that isn't the reels feed itself (the comments list), or null.
COMMENTS_BOX_JS = r"""(() => {
  const H = innerHeight;
  const boxes = [...document.querySelectorAll("div, ul, section")].filter((e) => {
    const s = getComputedStyle(e);
    if (!/(auto|scroll)/.test(s.overflowY) || e.scrollHeight <= e.clientHeight + 10) return false;
    const r = e.getBoundingClientRect();
    return r.height > 80 && r.height < H * 0.95 && r.bottom > 0 && r.top < H;
  });
  boxes.sort((a, b) => b.getBoundingClientRect().top - a.getBoundingClientRect().top);
  const r = boxes[0] && boxes[0].getBoundingClientRect();
  return r ? [r.left + r.width / 2, r.top + r.height / 2] : null;
})()"""


def find_chrome(given):
    for c in ([given] if given else []) + CHROMES:
        if c and (Path(c).exists() or shutil.which(c)):
            return shutil.which(c) or c
    raise SystemExit("Google Chrome not found; pass --chrome /path/to/chrome")


def set_flag(port):
    """Tell the station (via raw/reels.json) that the bridge is up, or remove the note (port=None)."""
    RAW.mkdir(parents=True, exist_ok=True)
    data = json.loads(MANIFEST.read_text()) if MANIFEST.exists() else {"reels": []}
    if port:
        data["instagram"] = {"port": port}
    else:
        data.pop("instagram", None)
    MANIFEST.write_text(json.dumps(data, indent=2, ensure_ascii=False) + "\n")


class CDP:
    """Just enough of the Chrome DevTools Protocol, over the websockets package."""

    def __init__(self, ws, on_event):
        self.ws, self.on_event, self.n, self.waiting = ws, on_event, 0, {}

    async def call(self, method, **params):
        self.n += 1
        fut = asyncio.get_running_loop().create_future()
        self.waiting[self.n] = fut
        await self.ws.send(json.dumps({"id": self.n, "method": method, "params": params}))
        return await asyncio.wait_for(fut, 15)

    async def pump(self):
        async for raw in self.ws:
            m = json.loads(raw)
            if "id" in m:
                fut = self.waiting.pop(m["id"], None)
                if fut and not fut.done():
                    fut.set_exception(RuntimeError(m["error"].get("message"))) if "error" in m else fut.set_result(m.get("result", {}))
            else:
                self.on_event(m.get("method"), m.get("params", {}))

    async def js(self, expr):
        r = await self.call("Runtime.evaluate", expression=expr, returnByValue=True)
        return r.get("result", {}).get("value")


class Bridge:
    def __init__(self, a):
        self.a = a
        self.clients = set()
        self.busy = set()
        self.status = {"type": "status", "state": "starting", "detail": "starting Chrome"}
        self.cdp = None
        self.frames = 0

    # --- Chrome ----------------------------------------------------------------------------------------
    def launch(self):
        PROFILE.mkdir(parents=True, exist_ok=True)
        args = [find_chrome(self.a.chrome), f"--user-data-dir={PROFILE}", f"--remote-debugging-port={self.a.cdp_port}",
                "--no-first-run", "--no-default-browser-check", "--window-size=420,780", f"--app={START_URL}"]
        if self.a.headless:
            args.append("--headless=new")
        self.chrome = subprocess.Popen(args, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        for _ in range(100):
            try:
                pages = json.loads(urllib.request.urlopen(f"http://127.0.0.1:{self.a.cdp_port}/json/list", timeout=1).read())
                pages = [p for p in pages if p.get("type") == "page" and "webSocketDebuggerUrl" in p]
                if pages:
                    return next((p for p in pages if "instagram" in p.get("url", "")), pages[0])["webSocketDebuggerUrl"]
            except OSError:
                pass
            if self.chrome.poll() is not None:
                raise SystemExit("Chrome exited right away (is another copy using raw/chrome-profile?)")
            time.sleep(0.1)
        raise SystemExit("Chrome didn't open its debugging port")

    def on_event(self, method, p):
        if method == "Page.screencastFrame":
            asyncio.ensure_future(self.cdp.call("Page.screencastFrameAck", sessionId=p["sessionId"]))
            self.frames += 1
            self.broadcast(base64.b64decode(p["data"]))

    async def setup_page(self):
        w, h = VIEW
        await self.cdp.call("Page.enable")
        await self.cdp.call("Emulation.setDeviceMetricsOverride", width=w, height=h, deviceScaleFactor=2, mobile=False)
        await self.cdp.call("Emulation.setFocusEmulationEnabled", enabled=True)   # keep playing when not focused
        await self.cdp.call("Page.startScreencast", format="jpeg", quality=70, maxWidth=w * 3 // 2, maxHeight=h * 3 // 2)

    # --- gestures (always trusted input events, at human pace: one per decision of the fly) -------------
    async def click(self, x, y):
        for t in ("mousePressed", "mouseReleased"):
            await self.cdp.call("Input.dispatchMouseEvent", type=t, x=x, y=y, button="left", clickCount=1)

    async def wheel(self, x, y, dy):
        await self.cdp.call("Input.dispatchMouseEvent", type="mouseMoved", x=x, y=y)
        await self.cdp.call("Input.dispatchMouseEvent", type="mouseWheel", x=x, y=y, deltaX=0, deltaY=dy)

    async def key(self, key, code):
        for t in ("keyDown", "keyUp"):
            await self.cdp.call("Input.dispatchKeyEvent", type=t, key=key, code=code, windowsVirtualKeyCode=code_vk(code))

    async def do(self, m):
        w, h = VIEW
        ui = self.status.get("ui") or {}
        what = m.get("do")
        if self.status.get("state") != "ok":
            return
        if what == "next":                                   # swipe up to the next reel
            await self.wheel(w / 2, h / 2, h)
        elif what == "open_comments" and ui.get("comment"):
            u, v = ui["comment"]
            await self.click(u * w, v * h)
        elif what == "scroll_comments":
            at = await self.cdp.js(COMMENTS_BOX_JS) or [w / 2, h * 0.75]
            await self.wheel(at[0], at[1], max(0, min(600, int(m.get("px", 250)))))
        elif what == "close_comments":
            if ui.get("close"):
                u, v = ui["close"]
                await self.click(u * w, v * h)
            else:
                await self.key("Escape", "Escape")
        # "like" is deliberately not handled: watch-only

    # --- talking to the station ---------------------------------------------------------------------------
    def broadcast(self, data):
        for ws in list(self.clients):
            if ws in self.busy:                              # slow client: drop this frame
                continue
            self.busy.add(ws)

            async def send(ws=ws):
                try:
                    await ws.send(data)
                except ConnectionClosed:
                    pass
                finally:
                    self.busy.discard(ws)
            asyncio.ensure_future(send())

    async def serve_client(self, ws):
        self.clients.add(ws)
        await ws.send(json.dumps(self.status))
        try:
            async for raw in ws:
                if isinstance(raw, str):
                    try:
                        await self.do(json.loads(raw))
                    except Exception as e:                   # a missed gesture must never kill the bridge
                        print("gesture failed:", e)
        except ConnectionClosed:
            pass
        finally:
            self.clients.discard(ws)

    async def watch(self):
        """Every second: login / blocked / ok, and where the buttons are."""
        last = None
        while True:
            try:
                s = await self.cdp.js(STATUS_JS) or {}
                state = "blocked" if s.get("blocked") else "login" if s.get("login") else "ok" if s.get("video") else "loading"
                detail = {"blocked": "account blocked or challenged: using the backup videos",
                          "login": "log in to the fly's account in the Chrome window",
                          "loading": "loading Instagram..."}.get(state, "live")
                self.status = {"type": "status", "state": state, "detail": detail, "ui": s.get("ui") or {}, "path": s.get("path")}
                if state == "ok" and not (s.get("path") or "").startswith("/reels"):
                    self.status["detail"] = "live (not on the reels tab)"
            except Exception as e:
                self.status = {"type": "status", "state": "loading", "detail": f"page busy ({e})", "ui": {}}
            if (self.status["state"], self.status["detail"]) != last:
                last = (self.status["state"], self.status["detail"])
                print(f"[instagram] {self.status['state']}: {self.status['detail']}  ({self.frames} frames so far)", flush=True)
            for ws in list(self.clients):
                try:
                    await ws.send(json.dumps(self.status))
                except ConnectionClosed:
                    pass
            await asyncio.sleep(1)


def code_vk(code):
    return {"Escape": 27, "ArrowDown": 40}.get(code, 0)


def local_only(connection, request):
    """Only pages served from this machine may drive the account (any website could try ws://127.0.0.1)."""
    origin = request.headers.get("Origin", "")
    host = origin.split("://", 1)[-1].split(":")[0]
    if origin and host not in ("localhost", "127.0.0.1", "[::1]"):
        return connection.respond(HTTPStatus.FORBIDDEN, "local pages only\n")
    return None


async def main_async(a):
    b = Bridge(a)
    url = b.launch()
    stop = asyncio.Event()
    loop = asyncio.get_running_loop()
    for sig in (signal.SIGINT, signal.SIGTERM):
        try:
            loop.add_signal_handler(sig, stop.set)
        except NotImplementedError:                          # Windows
            pass
    try:
        async with connect(url, max_size=None) as ws:
            b.cdp = CDP(ws, b.on_event)
            pump = asyncio.ensure_future(b.cdp.pump())
            await b.setup_page()
            async with serve(b.serve_client, "127.0.0.1", a.port, max_size=None, process_request=local_only):
                set_flag(a.port)
                print(f"Instagram bridge on ws://127.0.0.1:{a.port}. Leave this running; Ctrl+C to stop.", flush=True)
                watch = asyncio.ensure_future(b.watch())
                ended = asyncio.ensure_future(asyncio.to_thread(b.chrome.wait))
                await asyncio.wait([pump, ended, asyncio.ensure_future(stop.wait())], return_when=asyncio.FIRST_COMPLETED)
                watch.cancel()
    finally:
        set_flag(None)
        if b.chrome.poll() is None:
            b.chrome.terminate()
        print("bridge stopped; the phone is back on the backup videos")


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--headless", action="store_true", help="no window (log in once without this first)")
    ap.add_argument("--port", type=int, default=8765, help="where the station connects")
    ap.add_argument("--cdp-port", type=int, default=9333, help="Chrome's debugging port (localhost only)")
    ap.add_argument("--chrome", help="path to Chrome/Chromium")
    try:
        asyncio.run(main_async(ap.parse_args()))
    except KeyboardInterrupt:
        pass
    return 0


if __name__ == "__main__":
    sys.exit(main())
