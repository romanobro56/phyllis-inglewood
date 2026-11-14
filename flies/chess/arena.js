import {Arena} from './arena_scene.js';
import {Director} from './director.js';
import {EventGate,placement,boardFen,movePlan} from './arena_state.js';
import {Inspector} from './inspector.js';

const $=id=>document.getElementById(id),START='rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1';
const reduced=matchMedia('(prefers-reduced-motion: reduce)').matches;
let state=null,credentials=null,mode='live',busy=false,polling=false,disconnected=false,generation=0,lastEvent=0,replay=null,replayIndex=0,userPaused=false,inspectionPaused=false,resumeAfterInspect=false,contextLost=false,newPending=false,recovery=null,superseded=false;
const gate=new EventGate();
const errors=[];window.addEventListener('error',e=>errors.push(e.message));window.addEventListener('unhandledrejection',e=>errors.push(String(e.reason)));
const originalConsoleError=console.error.bind(console);console.error=(...args)=>{errors.push(args.map(String).join(' '));originalConsoleError(...args);};
function error(e){$('error').hidden=false;$('error').textContent=String(e.message||e);console.error(e);}
const stale=e=>/Stale connection epoch|Stale match or invalid token/.test(e.message);
function loseControl(){superseded=true;director.paused=true;$('connection').textContent='CONTROL MOVED · PAUSED';$('pause').disabled=false;$('pause').textContent='Take control';$('detail').textContent='Another tab or server session owns this game. Take control to reconnect deliberately.';}
async function request(path,body={}){
  const response=await fetch(path,{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify({...credentials,...body}),signal:AbortSignal.timeout(15000)});
  const result=await response.json();if(!response.ok)throw Error(result.error||`HTTP ${response.status}`);return result;
}
const arena=new Arena($('arena'),{onFly:side=>inspect(side),onContext:async restored=>{
  if(!restored){contextLost=true;director.paused=true;await pause(true);error('WebGL context lost. Play is paused; no animation acknowledged.');}
  else{contextLost=false;await recover();$('error').hidden=true;}
}});
const director=new Director(arena);director.speed=Number($('speed').value)*(reduced?1.6:1);
const inspector=new Inspector(request,()=>{inspectionPaused=false;if(resumeAfterInspect&&!userPaused)pause(false);},live=>{inspectionPaused=!live;userPaused=!live;return pause(!live);});
let previous=performance.now(),frames=0,frameTime=0,lowTime=0,renderId;
function frame(now){
  const dt=Math.min((now-previous)/1000,.05);previous=now;
  if(!document.hidden&&!contextLost){director.update(dt);if(!inspector.side)arena.update(dt,director);frames++;frameTime+=dt;
    if(frameTime>2){const fps=frames/frameTime;$('performance').textContent=`${Math.round(fps)} FPS · ${arena.quality} · drag to orbit`;
      lowTime=fps<32?lowTime+frameTime:0;if(lowTime>5&&arena.quality!=='low'){arena.setQuality('low');$('quality').value='low';lowTime=0;}frames=frameTime=0;}}
  renderId=requestAnimationFrame(frame);
}
function setState(s){
  if(state?.match_id===s.match_id&&(s.epoch<state.epoch||s.event_id<lastEvent))return;
  state=s;if(s.phase==='empty')return;
  credentials={match_id:s.match_id,token:s.token,epoch:s.epoch};
  director.paused=s.paused||inspectionPaused||contextLost;
  arena.active=s.pending?.side||s.side;
  $('pause').disabled=['loading','finished','cancelled'].includes(s.phase);
  $('pause').textContent=s.paused?'Resume':'Pause';$('restart').disabled=false;
  for(const side of ['white','black']){const p=s.players[side];$(`${side}-model`).textContent=p?`${p.label} · ${p.preset} · ${p.neurons.toLocaleString()} neurons`:s.phase==='loading'?'Loading independent FlyBrain…':'Model unavailable';
    document.querySelector(`.player-${side}`).classList.toggle('active',s.phase==='thinking'?s.side===side:s.pending?.side===side);}
  $('turn').textContent=`${s.paused?'PAUSED · ':''}${s.phase==='finished'?'MATCH ENDED':(s.pending?.side||s.side).toUpperCase()+' · '+s.phase.toUpperCase()} · MOVE ${s.fen.split(' ')[5]} · PLY ${s.pending?.sequence||s.sequence}`;
  $('caption').textContent=s.phase==='loading'?'Two brains are waking.':s.phase==='thinking'?`${s.side==='white'?'White':'Black'} is thinking.`:s.pending?`${s.pending.side==='white'?'White':'Black'} plays ${s.pending.san}`:s.phase==='finished'?outcomeText(s.outcome):'The next connection.';
  $('detail').textContent=s.error|| (s.pending?'Carrying a real neural decision. The next brain waits.':s.phase==='thinking'?'Eight descending-neuron groups · legal coordinate selection':s.paused?'Resume to continue. Click either fly to inspect its latest real inference.':'Click either fly to explore its simulated neuron rates.');
  if(s.error)error(s.error);
  history(s.history,s.outcome);$('record-link').href=s.recording;$('pgn-link').href=s.pgn;$('record-link').hidden=$('pgn-link').hidden=!s.players.white;
  lastEvent=s.event_id; // server snapshots are monotonic per current match
}
function outcomeText(o){return o.truncated?'Stopped at the ply cap · no chess result':o.termination?`${o.result} · ${o.termination.replaceAll('_',' ').toLowerCase()}`:'Game in progress';}
function history(moves,outcome){
  $('moves').replaceChildren();
  for(let i=0;i<moves.length;i+=2){for(const value of [String(i/2+1)+'.',moves[i]?.san||'',moves[i+1]?.san||'']){const el=document.createElement('span');el.textContent=value;if(value.endsWith('.'))el.className='number';$('moves').append(el);}}
  $('result').textContent=outcomeText(outcome);
}
async function pause(paused,manual=false){
  const localGeneration=generation,epoch=credentials?.epoch;
  if(manual)userPaused=paused;
  director.paused=paused||inspectionPaused||contextLost;
  if(mode==='replay'){$('pause').textContent=paused?'Resume':'Pause';return;}
  if(credentials){try{const s=await request('/api/pause',{paused});if(localGeneration===generation)setState(s);}catch(e){if(localGeneration===generation&&epoch===credentials?.epoch){if(stale(e))loseControl();else error(e);}}}
}
async function inspect(side){
  try{if(mode==='replay')throw Error('Replay has no per-neuron rate recording. Start a live match to inspect a real brain.');
    if(!state?.infos?.[side])throw Error('Start a live match and wait for model initialization.');
    resumeAfterInspect=!state.paused;inspector.live=false;inspectionPaused=true;director.paused=true;await pause(true);await inspector.open(side,state);
  }catch(e){error(e);}
}
function cancelVisual(){generation++;director.cancel();gate.reset(null,0);busy=false;}
async function newMatch(){
  if(newPending)return;newPending=true;
  try{if(recovery)await recovery;if(inspector.matrix?.fly)inspector.matrix.close();cancelVisual();mode='live';$('mode').textContent='LIVE INFERENCE';$('error').hidden=true;userPaused=false;inspectionPaused=false;disconnected=false;superseded=false;lastEvent=0;
    const s=await request('/api/new',{size:$('preset').value,white:$('white-checkpoint').value,black:$('black-checkpoint').value,seed:Number($('seed').value),max_plies:Number($('max-plies').value)});
    gate.reset(s.match_id,s.epoch);arena.reset(s.initial_fen);setState(s);$('settings').close();
    $('connection').textContent='LOCAL · CPU';
  }catch(e){error(e);}finally{newPending=false;}
}
async function animatePending(s){
  if(busy||s.paused||inspectionPaused)return;
  const e=s.pending,g=gate.begin(e),localGeneration=generation;busy=true;arena.active=e.side;
  try{
    const fen=await director.play(e);
    if(localGeneration!==generation||mode!=='live')return;
    const rendered=arena.assertBoard(fen);
    const acked=await request('/api/ack',{sequence:e.sequence,rendered_fen:fen,rendered_placement:rendered});
    if(localGeneration!==generation)return;
    gate.complete(e,g);setState(acked);
  }catch(e){if(localGeneration===generation){if(stale(e))loseControl();else{disconnected=true;await pause(true);error(e);}}}
  finally{if(localGeneration===generation)busy=false;}
}
function recover(){
  if(recovery)return recovery;
  recovery=recoverCore().finally(()=>{recovery=null;});return recovery;
}
async function recoverCore(){
  if(mode!=='live')return;
  cancelVisual();const current=await fetch('/api/state').then(r=>r.json());
  if(current.phase==='empty'){state=null;credentials=null;disconnected=false;director.paused=true;$('connection').textContent='SERVER RESTARTED';$('caption').textContent='Start a fresh live match.';$('detail').textContent='Saved recordings remain available. The previous in-memory session ended with the server.';$('pause').disabled=true;for(const side of ['white','black'])$(`${side}-model`).textContent='No live model loaded';return;}
  const s=await request('/api/claim',{match_id:current.match_id});
  gate.reset(s.match_id,s.epoch);gate.sequence=s.sequence;
  arena.reset(s.pending?s.pending.before:s.fen);lastEvent=0;setState(s);userPaused=true;disconnected=false;
  superseded=false;
  $('connection').textContent='RECOVERED · RESUME WHEN READY';
  if(!s.error)$('error').hidden=true;
}
async function poll(){
  if(polling||newPending||recovery||superseded||mode!=='live'||contextLost||document.hidden)return;
  polling=true;
  const localGeneration=generation;
  try{
    if(!credentials)return;
    if(disconnected){await recover();return;}
    const s=await request('/api/state');
    if(localGeneration!==generation||mode!=='live')return;
    if(s.match_id!==state.match_id||s.epoch!==credentials.epoch)throw Error('Stale server state');
    if(s.event_id<lastEvent)return; // An older overlapping poll must never roll back an ACK.
    setState(s);
    if(inspector.side)inspector.refresh(s)?.catch(error);
    if(s.pending&&!busy&&!s.paused&&!inspectionPaused)animatePending(s).catch(error);
    else if(s.phase==='ready'&&!s.paused&&!busy&&!inspectionPaused){const next=await request('/api/next');if(localGeneration===generation){setState(next);director.state='THINKING';}}
  }catch(e){if(localGeneration!==generation)return;director.paused=true;
    if(stale(e))loseControl();
    else{disconnected=true;$('connection').textContent='CONNECTION LOST · PAUSED';error(e);}}
  finally{polling=false;}
}
async function loadReplay(record){
  if(!Array.isArray(record.moves)||!record.initial_fen||!record.outcome||typeof record.outcome.truncated!=='boolean'||!['*','1-0','0-1','1/2-1/2'].includes(record.outcome.result))throw Error('Not a complete Chess Fly game recording');
  let fen=record.initial_fen;for(const e of record.moves){if(e.before!==fen)throw Error('Replay FEN discontinuity');movePlan(e);fen=e.fen;}
  if(fen!==record.final_fen)throw Error('Replay final FEN mismatch');
  if(credentials&&mode==='live')await request('/api/pause',{paused:true});
  cancelVisual();mode='replay';superseded=false;replay=record;replayIndex=0;director.paused=true;userPaused=true;
  arena.reset(record.initial_fen);$('mode').textContent='RECORDED REPLAY';$('turn').textContent='REPLAY · PLY 0';$('caption').textContent='A recorded game.';$('detail').textContent='Playback only · no live neural inference';$('pause').disabled=false;$('pause').textContent='Play';$('replay-ply').max=record.moves.length;
  for(const side of ['white','black']){const p=record.players?.[side];$(`${side}-model`).textContent=p?`RECORDED ${p.label} · ${p.preset}`:'Recorded player · no live activity';}
  history([],record.outcome);$('replay-dialog').close();
}
async function replayStep(){
  if(mode!=='replay'||director.paused||busy||!replay||replayIndex>=replay.moves.length)return;
  const local=generation,e=replay.moves[replayIndex];busy=true;arena.active=e.side;
  try{await director.play(e);if(local!==generation)return;replayIndex++;$('replay-ply').value=replayIndex;$('turn').textContent='REPLAY · PLY '+replayIndex;$('caption').textContent=e.san;history(replay.moves.slice(0,replayIndex),replay.outcome);
    if(replayIndex===replay.moves.length){director.paused=true;$('pause').textContent='Finished';$('caption').textContent=outcomeText(replay.outcome);}}
  catch(e){if(local===generation)error(e);}finally{if(local===generation)busy=false;}
}
$('new-match').onclick=newMatch;$('restart').onclick=newMatch;$('settings-start').onclick=newMatch;$('settings-button').onclick=()=>$('settings').showModal();
$('pause').onclick=async()=>{if(superseded){await recover();if(credentials)await pause(false,true);}else await pause(!director.paused,true);};$('inspect-white').onclick=()=>inspect('white');$('inspect-black').onclick=()=>inspect('black');
$('speed').onchange=()=>director.speed=Number($('speed').value)*(reduced?1.6:1);$('camera').onchange=()=>arena.cameraMode=$('camera').value;$('quality').onchange=()=>arena.setQuality($('quality').value);
$('history-button').onclick=()=>$('history-panel').classList.toggle('open');$('history-close').onclick=()=>$('history-panel').classList.remove('open');$('replay-button').onclick=()=>$('replay-dialog').showModal();
$('replay-file').onchange=async e=>{try{await loadReplay(JSON.parse(await e.target.files[0].text()));}catch(err){error(err);}};
$('replay-current').onclick=async()=>{try{if(!state?.recording)throw Error('Play a live match first');await loadReplay(await fetch(state.recording).then(r=>r.json()));}catch(e){error(e);}};
$('replay-seek').onclick=()=>{if(!replay)return;const i=Number($('replay-ply').value);if(!Number.isInteger(i)||i<0||i>replay.moves.length)return;cancelVisual();mode='replay';director.paused=true;replayIndex=i;arena.reset(i?replay.moves[i-1].fen:replay.initial_fen);$('turn').textContent='REPLAY · PLY '+i;$('pause').textContent='Play';history(replay.moves.slice(0,i),replay.outcome);$('replay-dialog').close();};
document.addEventListener('visibilitychange',()=>{if(document.hidden)pause(true,true);});
window.addEventListener('beforeunload',()=>{cancelAnimationFrame(renderId);director.cancel();arena.dispose();});
async function initialize(){
  arena.reset(START);renderId=requestAnimationFrame(frame);
  await inspector.initialize();const cat=await fetch('/api/catalog').then(r=>r.json());
  for(const side of ['white','black'])for(const path of cat.checkpoints){const option=document.createElement('option');option.value=path;option.textContent=path.replace('artifacts/','');$(`${side}-checkpoint`).append(option);}
  for(const key of ['size','seed','max_plies','white','black']){const el=$({size:'preset',seed:'seed',max_plies:'max-plies',white:'white-checkpoint',black:'black-checkpoint'}[key]);if(cat.defaults[key]!=null)el.value=cat.defaults[key];}
  const existing=await fetch('/api/state').then(r=>r.json());if(existing.phase!=='empty')await recover();
  $('loading').remove();setInterval(()=>{poll();replayStep();},250);
}
initialize().catch(e=>{$('loading').remove();error('Unable to initialize WebGL/shared viewer: '+e.message);});
// Narrow read-only diagnostics for the local browser test page; never selects a move.
window.chessArena={get state(){return state;},arena,director,gate,inspector,errors,get mode(){return mode;},newMatch,inspect,loadReplay,pause,recover};
