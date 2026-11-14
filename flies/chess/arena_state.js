// Presentation rules: no move selection, no engine, no legal action ranking.
export function placement(fen) {
  const rows = fen.split(' ')[0].split('/');
  if (rows.length !== 8) throw Error('Invalid FEN rows');
  const map = {};
  rows.forEach((row,r) => {
    let f=0;
    for (const p of row) {
      if (/[1-8]/.test(p)) f+=Number(p);
      else { if (!/[prnbqkPRNBQK]/.test(p) || f>7) throw Error('Invalid piece'); map['abcdefgh'[f++]+(8-r)] = p; }
    }
    if(f!==8) throw Error('Invalid FEN width');
  });
  return map;
}
export function boardFen(map) {
  return Array.from({length:8},(_,r)=>{
    let row='',empty=0;
    for(let f=0;f<8;f++) {
      const p=map['abcdefgh'[f]+(8-r)];
      if(p){if(empty)row+=empty;empty=0;row+=p;}else empty++;
    }
    return row+(empty||'');
  }).join('/');
}
export const sideOf = p => p===p.toUpperCase()?'white':'black';
export function movePlan(event) {
  const before=placement(event.before),after=placement(event.fen);
  const source=event.uci.slice(0,2),destination=event.uci.slice(2,4),piece=before[source];
  const side=event.side || (event.before.split(' ')[1]==='w'?'white':'black');
  if(!piece || sideOf(piece)!==side || event.before.split(' ')[1] === event.fen.split(' ')[1]) throw Error('Wrong moving side');
  const map={...before}, capture=before[destination]?destination:null;
  let captured=capture,rook=null;
  if(piece.toLowerCase()==='p' && source[0]!==destination[0] && !capture) captured=destination[0]+source[1];
  if(captured && (!map[captured]||sideOf(map[captured])===side)) throw Error('Invalid capture');
  if(captured)delete map[captured];
  delete map[source];
  map[destination]=event.uci[4]?(side==='white'?event.uci[4].toUpperCase():event.uci[4]):piece;
  if(piece.toLowerCase()==='k' && Math.abs(source.charCodeAt(0)-destination.charCodeAt(0))===2) {
    const kingSide=destination[0]==='g';rook=[(kingSide?'h':'a')+source[1],(kingSide?'f':'d')+source[1]];
    if(map[rook[0]]!==(side==='white'?'R':'r')) throw Error('Missing castling rook');
    map[rook[1]]=map[rook[0]];delete map[rook[0]];
  }
  if(boardFen(map)!==boardFen(after)) throw Error('Move does not reconstruct authoritative FEN');
  const plan={source,destination,capture:captured,rook,promotion:event.uci[4]||null,side};
  if(event.visual) for(const k of ['source','destination','capture','rook','promotion'])
    if(JSON.stringify(plan[k])!==JSON.stringify(event.visual[k]))throw Error('Visual metadata mismatch: '+k);
  return plan;
}
export function squarePosition(square,y=0.31) {return [square.charCodeAt(0)-97-3.5,y,3.5-(Number(square[1])-1)];}

export class EventGate {
  constructor(){this.reset(null,0);}
  reset(id,epoch){this.id=id;this.epoch=epoch;this.sequence=0;this.running=null;this.generation=(this.generation||0)+1;}
  begin(e){
    if(e.match_id!==this.id || e.sequence!==this.sequence+1 || this.running)throw Error('Stale, duplicate or unordered animation');
    this.running=e.sequence;return this.generation;
  }
  complete(e,generation){
    if(generation!==this.generation || e.match_id!==this.id || this.running!==e.sequence)throw Error('Stale animation callback');
    this.sequence=e.sequence;this.running=null;
  }
}
