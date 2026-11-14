"""Fly-local asset server. Explicit allowlist; cannot serve vaults or checkpoints."""
import http.server
from pathlib import Path
import threading
from urllib.parse import urlsplit

ROOT=Path(__file__).resolve().parents[2]
ALLOW={'/flies/jobs/station_view.html','/flies/jobs/station.js',
       '/flies/jobs/vendor/three.module.js','/web/lib/fly.js'}


class StationServer:
    def __enter__(self):
        class Handler(http.server.BaseHTTPRequestHandler):
            def do_GET(self):
                name=urlsplit(self.path).path
                if name not in ALLOW:
                    self.send_error(404);return
                file=ROOT/name.lstrip('/')
                if not file.is_file():
                    self.send_error(404);return
                data=file.read_bytes()
                self.send_response(200)
                self.send_header('Content-Type','text/html; charset=utf-8' if file.suffix=='.html' else 'text/javascript; charset=utf-8')
                self.send_header('Content-Length',str(len(data)))
                self.end_headers();self.wfile.write(data)
            def log_message(self,*args):pass
        self.server=http.server.ThreadingHTTPServer(('127.0.0.1',0),Handler)
        self.thread=threading.Thread(target=self.server.serve_forever,daemon=True)
        self.thread.start()
        self.origin=f'http://127.0.0.1:{self.server.server_port}'
        return self

    def __exit__(self,*args):
        self.server.shutdown();self.server.server_close();self.thread.join(timeout=2)
