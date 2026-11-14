"""Feature-local live two-brain viewer and match launcher. No shared server edits."""
import argparse
from functools import partial
from http.server import SimpleHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
import threading
import traceback
from urllib.parse import quote

from flies.chess.selfplay import run_matches
from flies.chess.storage import ROOT, prepare_run, write_json


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--white-checkpoint');p.add_argument('--black-checkpoint')
    p.add_argument('--size',default='tiny');p.add_argument('--device',default='cpu')
    p.add_argument('--seed',type=int,default=20261114);p.add_argument('--games',type=int,default=1)
    p.add_argument('--max-plies',type=int,default=512);p.add_argument('--delay',type=float,default=.25)
    p.add_argument('--output',default=str(ROOT/'artifacts/demo'))
    p.add_argument('--port',type=int,default=8766);p.add_argument('--no-activity',action='store_true')
    args=p.parse_args()
    output=prepare_run(args.output)
    server=ThreadingHTTPServer(('127.0.0.1',args.port),partial(SimpleHTTPRequestHandler,directory=str(ROOT)))
    stop=threading.Event()
    def match():
        try:
            run_matches(output,args.white_checkpoint,args.black_checkpoint,games=args.games,seed=args.seed,
                        size=args.size,device=args.device,max_plies=args.max_plies,delay=args.delay,
                        activity=not args.no_activity,live=True,should_stop=stop.is_set)
        except Exception as error:
            write_json(output/'live.json',dict(error=str(error),status='error'))
            traceback.print_exc()
    worker=threading.Thread(target=match,name='two-FlyBrain-match',daemon=True)
    source=quote(str((output/'live.json').relative_to(ROOT)))
    print(f'Viewer: http://127.0.0.1:{server.server_port}/viewer.html?source={source}&follow=1',flush=True)
    print('The server remains available for replay after the match. Ctrl-C saves a stopped match at a full-move boundary.',flush=True)
    worker.start()
    try:
        server.serve_forever(poll_interval=.2)
    except KeyboardInterrupt:
        stop.set();worker.join(timeout=15)
    finally:
        server.server_close()


if __name__=='__main__':main()
