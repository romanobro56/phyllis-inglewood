// FEN is authoritative. This world adapter is the explicitly labeled random-opponent demo.
import { tween, wait, box, cylinder } from "/web/lib/fly.js";

const START = "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1";
const xy = (sq) => ({ x: ((sq.charCodeAt(0) - 97) - 3.5) * 0.48,
  z: (3.5 - (Number(sq[1]) - 1)) * 0.48 });

export default {
  title: "Chess Fly · vs random demo",
  blurb: "Four or five descending-neuron decisions per move. Black is random here; two-brain matches use the chess runner.",

  build({ THREE, group, fly }) {
    // built at about 1.7x the other stations; shrink it so it sits in the shared world at the same scale
    const board = new THREE.Group(); board.scale.setScalar(0.6); group.add(board); group = board;
    this.THREE = THREE; this.group = group; this.pieces = new Map(); this.fen = null;
    group.add(box(4.05, 0.12, 4.05, 0x604936, [0, 0.08, 0]));
    this.squares = [];
    for (let rank = 0; rank < 8; rank++) for (let file = 0; file < 8; file++) {
      const square = box(0.48, 0.04, 0.48, (rank + file) % 2 ? 0xe4d3ad : 0x625046,
        [(file - 3.5) * 0.48, 0.16, (3.5 - rank) * 0.48]);
      group.add(square); this.squares.push(square);
    }
    this.signCanvas = document.createElement("canvas"); this.signCanvas.width = 1024; this.signCanvas.height = 160;
    this.signTexture = new THREE.CanvasTexture(this.signCanvas);
    const sign = new THREE.Sprite(new THREE.SpriteMaterial({ map: this.signTexture }));
    sign.scale.set(4.2, 0.65, 1); sign.position.set(0, 0.9, -2.2); group.add(sign);
    this.label("FlyBrain White · random legal Black", "Board from actual FEN captions");
    this.renderFen(START);
    fly.root.position.set(-1.45, 0.07, 1.0);
    fly.face({ x: 0, z: 0 });
  },

  label(first, second) {
    const ctx = this.signCanvas.getContext("2d");
    ctx.fillStyle = "#16202b"; ctx.fillRect(0, 0, 1024, 160);
    ctx.fillStyle = "#f5edda"; ctx.font = "bold 37px sans-serif"; ctx.fillText(first, 20, 57);
    ctx.font = "28px sans-serif"; ctx.fillText(second, 20, 115);
    this.signTexture.needsUpdate = true;
  },

  piece(symbol) {
    const THREE = this.THREE, white = symbol === symbol.toUpperCase();
    const root = new THREE.Group();
    const heights = { p: .23, n: .36, b: .40, r: .32, q: .47, k: .53 };
    const h = heights[symbol.toLowerCase()];
    const color = white ? 0xf1e4c6 : 0x283546;
    root.add(cylinder(.16, .06, color, [0, .02, 0]));
    root.add(cylinder(.09, h, color, [0, h / 2, 0]));
    const canvas = document.createElement("canvas"); canvas.width = canvas.height = 128;
    const ctx = canvas.getContext("2d");
    ctx.fillStyle = white ? "#f4ead5" : "#263447"; ctx.beginPath(); ctx.arc(64,64,60,0,Math.PI*2); ctx.fill();
    ctx.fillStyle = white ? "#263447" : "#f4ead5"; ctx.font = "bold 88px serif";
    ctx.textAlign = "center"; ctx.textBaseline = "middle"; ctx.fillText(symbol.toUpperCase(),64,68);
    const texture = new THREE.CanvasTexture(canvas);
    const sprite = new THREE.Sprite(new THREE.SpriteMaterial({ map: texture }));
    sprite.scale.set(.32,.32,1); sprite.position.y = h + .12; root.add(sprite);
    return root;
  },

  renderFen(fen) {
    // Dispose resources when snapshots replace the board; long live games remain bounded.
    for (const piece of this.pieces.values()) {
      this.group.remove(piece);
      piece.traverse(o => { if (o.geometry) o.geometry.dispose(); if (o.material) {
        if (o.material.map) o.material.map.dispose(); o.material.dispose();
      } });
    }
    this.pieces.clear();
    const rows = fen.split(" ")[0].split("/");
    rows.forEach((row, r) => {
      let f = 0;
      for (const symbol of row) {
        if (/\d/.test(symbol)) { f += Number(symbol); continue; }
        const sq = String.fromCharCode(97 + f++) + (8 - r);
        const piece = this.piece(symbol), p = xy(sq);
        piece.position.set(p.x, .22, p.z); this.group.add(piece); this.pieces.set(sq, piece);
      }
    });
    this.fen = fen;
  },

  async play(action, caption, { fly, info }) {
    let slot;
    switch (action) {
      case "slot_0": slot = 0; break;
      case "slot_1": slot = 1; break;
      case "slot_2": slot = 2; break;
      case "slot_3": slot = 3; break;
      case "slot_4": slot = 4; break;
      case "slot_5": slot = 5; break;
      case "slot_6": slot = 6; break;
      case "slot_7": slot = 7; break;
      default: throw new Error(`Unknown chess action: ${action}`);
    }
    const marker = caption.indexOf("|CHESS|");          // older servers sent the board inside the caption
    const data = info?.fen ? info : marker >= 0 ? JSON.parse(caption.slice(marker + 7)) : null;
    if (!data) throw new Error("Chess decision is missing authoritative board state");
    for (const square of this.squares) square.material.emissive.setHex(0);
    for (const event of data.events) {
      for (const square of this.squares) square.material.emissive.setHex(0);
      if (this.fen !== event.before) this.renderFen(event.before);
      const src = event.uci.slice(0, 2), dst = event.uci.slice(2, 4), piece = this.pieces.get(src);
      const from = xy(src), to = xy(dst);
      for (const sq of [src, dst]) {
        const i = (Number(sq[1]) - 1) * 8 + sq.charCodeAt(0) - 97;
        this.squares[i].material.emissive.setHex(0x245522);
      }
      this.label(`${event.side}: ${event.san} (${event.uci}) · legal`, event.controller);
      if (!piece) throw new Error(`Recorded source ${src} is empty`);
      await tween(480, t => piece.position.set(from.x + (to.x-from.x)*t, .22 + Math.sin(t*Math.PI)*.3, from.z+(to.z-from.z)*t));
      // Complete FEN also handles castling rook, en passant capture, promotion.
      this.renderFen(event.fen);
    }
    if (this.fen !== data.fen) this.renderFen(data.fen);
    const o = data.outcome || {};
    this.label(`Fly vs random · ${data.fen.split(" ")[1] === "w" ? "White" : "Black"} to move`,
      o.truncated ? "Compute cap · unfinished (*)" : o.termination ? `${o.result} · ${o.termination}` : `${data.stage} · slot ${slot}`);
    await fly.buzz(180); await wait(120);
  },
};
