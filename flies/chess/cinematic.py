"""Feature-local live arena; shared fly and Matrix assets are served read-only."""
import argparse
import json
import mimetypes
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import urlsplit

import chess

from flies.chess.live_match import LiveMatch, ProtocolError
from flies.chess.storage import ROOT, write_json

REPO = ROOT.parents[1]


def checkpoint_path(value):
    if not value: return None
    path = (ROOT / value).resolve()
    if ROOT not in path.parents or not path.is_file() or path.suffix != '.pt' or path.name == 'resume.pt':
        raise ProtocolError('Choose a trusted chess checkpoint inside flies/chess/')
    return str(path)


def catalog():
    paths = sorted(ROOT.glob('artifacts/**/checkpoint_*_[01].pt'))
    paths += sorted(ROOT.glob('artifacts/**/brain_[01].pt'))
    return [str(p.relative_to(ROOT)) for p in paths]


def handler_for(controller, defaults):
    class Handler(BaseHTTPRequestHandler):
        def log_message(self, *_): pass

        def reply(self, value, status=200):
            data = json.dumps(value, allow_nan=False).encode()
            self.send_response(status); self.send_header('Content-Type', 'application/json')
            self.send_header('Content-Length', str(len(data))); self.send_header('Cache-Control', 'no-store')
            self.end_headers(); self.wfile.write(data)

        def do_POST(self):
            try:
                # Local same-origin mutations only; never expose a pickle upload API.
                origin = self.headers.get('Origin')
                if origin and urlsplit(origin).netloc != self.headers.get('Host'):
                    raise ProtocolError('Cross-origin command rejected')
                n = int(self.headers.get('Content-Length', 0))
                if not 0 < n <= 262144: raise ProtocolError('Invalid command size')
                body = json.loads(self.rfile.read(n))
                path = urlsplit(self.path).path
                if path == '/api/new':
                    args = dict(defaults); args.update({k: body[k] for k in
                        ('size', 'white', 'black', 'seed', 'max_plies', 'fen', 'greedy') if k in body})
                    args['white'] = checkpoint_path(args.get('white'))
                    args['black'] = checkpoint_path(args.get('black'))
                    result = controller.new(**args)
                elif path == '/api/claim': result = controller.claim(body.get('match_id'))
                elif path == '/api/state': result = controller.state(body)
                elif path == '/api/telemetry':
                    controller.state(body); result = controller.telemetry(body.get('side'))
                elif path == '/api/browser-report':
                    # Test harness outputs are explicitly not gameplay records.
                    write_json(ROOT / 'artifacts/cinematic-browser.json', body); result = {'saved': True}
                elif path in ('/api/next', '/api/ack', '/api/pause', '/api/cancel'):
                    result = getattr(controller, path.rsplit('/', 1)[1])(body)
                else: return self.reply({'error': 'Unknown command'}, 404)
                self.reply(result)
            except (ValueError, KeyError, TypeError) as exc:
                self.reply({'error': str(exc)}, 409)
            except Exception as exc:
                self.reply({'error': str(exc)}, 500)

        def do_GET(self):
            path = urlsplit(self.path).path
            if path == '/api/state': return self.reply(controller.state())
            if path == '/api/catalog': return self.reply(dict(checkpoints=catalog(), defaults=defaults))
            # Serve only explicit source roots and Matrix metadata, never biological npz or secrets.
            if path.startswith('/web/'):
                base = REPO / 'web'; candidate = base / path[len('/web/'):]
            elif path.startswith('/data/'):
                base = REPO / 'data'; candidate = base / path[len('/data/'):]
                if candidate.name not in {f'{s}.{suffix}' for s in ('tiny','small','central','full')
                                          for suffix in ('points.bin','meta.json')}:
                    return self.reply({'error': 'Not served'}, 404)
            else:
                base = ROOT; candidate = ROOT / ('arena.html' if path == '/' else path.lstrip('/'))
            candidate = candidate.resolve()
            if base.resolve() not in candidate.parents or not candidate.is_file() or candidate.suffix in ('.pt','.npz'):
                return self.reply({'error': 'Not found'}, 404)
            data = candidate.read_bytes()
            content = mimetypes.guess_type(str(candidate))[0] or 'application/octet-stream'
            self.send_response(200); self.send_header('Content-Type', content)
            self.send_header('Content-Length', str(len(data))); self.send_header('Cache-Control','no-cache')
            self.end_headers(); self.wfile.write(data)
    return Handler


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--port', type=int, default=8770)
    ap.add_argument('--size', choices=('tiny','small','central','full'), default='tiny')
    ap.add_argument('--white-checkpoint', dest='white')
    ap.add_argument('--black-checkpoint', dest='black')
    ap.add_argument('--seed', type=int, default=42)
    ap.add_argument('--max-plies', type=int, default=160)
    ap.add_argument('--fen', default=chess.STARTING_FEN,
                    help='Optional legal initial position; never a move-selection oracle')
    ap.add_argument('--greedy', action='store_true')
    a = vars(ap.parse_args()); port = a.pop('port')
    for side in ('white','black'):
        if a[side]: a[side] = str(Path(a[side]).resolve().relative_to(ROOT))
    controller = LiveMatch()
    server = ThreadingHTTPServer(('127.0.0.1', port), handler_for(controller, a))
    print(f'Chess Fly arena: http://127.0.0.1:{port}/ (CPU, inference only; no training)', flush=True)
    try: server.serve_forever(poll_interval=.2)
    except KeyboardInterrupt: pass
    finally: server.server_close(); controller.close()


if __name__ == '__main__': main()
