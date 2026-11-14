"""Optional integration: exercise the unchanged live server caption protocol.

No browser simulation: validates messages/FENs over the real server websocket.
Run server.serve separately, then this module. --seconds 60 runs for one minute.
"""
import argparse
import asyncio
import json
import time

import chess
import websockets
from flies.chess.storage import ROOT, write_json


async def verify(seconds, url):
    count, moves, episodes = 0, 0, 0
    t0 = time.monotonic()
    last_fen = None
    async with websockets.connect(url) as ws:
        await ws.send(json.dumps(dict(type="hello")))
        init = json.loads(await ws.recv())
        assert any(f["name"] == "chess" for f in init["flies"])
        while time.monotonic() - t0 < seconds:
            await ws.send(json.dumps(dict(type="ready", fly="chess")))
            message = json.loads(await asyncio.wait_for(ws.recv(), 10))
            assert message["type"] == "decision" and message["fly"] == "chess"
            assert message["action"] in [f"slot_{i}" for i in range(8)]
            payload = json.loads(message["caption"].split("|CHESS|")[1])
            assert payload["mode"] == "fly-vs-random-demo"
            for event in payload["events"]:
                board = chess.Board(event["before"])
                if last_fen is not None:
                    assert last_fen == event["before"]
                move = chess.Move.from_uci(event["uci"])
                assert move in board.legal_moves
                board.push(move)
                assert board.fen() == event["fen"]
                last_fen = event["fen"]
                moves += 1
            assert chess.Board(payload["fen"]).is_valid()
            count += 1
            if message["episode_done"]:
                episodes += 1
                last_fen = None
            await asyncio.sleep(.05)
    result = dict(seconds=time.monotonic() - t0, decisions=count, plies=moves, episodes=episodes,
                  verified="real live server protocol, legality and FEN captions; no browser render assertion")
    write_json(ROOT / "artifacts/live-protocol.json", result)
    print(result)


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--seconds", type=float, default=60)
    ap.add_argument("--url", default="ws://127.0.0.1:8765")
    args = ap.parse_args()
    asyncio.run(verify(args.seconds, args.url))


if __name__ == "__main__":
    main()
