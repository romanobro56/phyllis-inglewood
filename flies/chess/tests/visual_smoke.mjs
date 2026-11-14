// Execute station and viewer logic without a browser or GPU. This is not a WebGL render test.
import fs from 'node:fs/promises';
import vm from 'node:vm';
import assert from 'node:assert/strict';

class Vector { set(x,y,z){Object.assign(this,{x,y,z});return this;} }
class Object3D {
  constructor(){this.position=new Vector();this.rotation=new Vector();this.scale=new Vector();this.children=[];this.userData={};}
  add(...nodes){this.children.push(...nodes);} remove(node){this.children=this.children.filter(x=>x!==node);}
  traverse(fn){fn(this);for(const c of this.children)c.traverse(fn);}
}
class Material {constructor(values={}){Object.assign(this,values);this.emissive={setHex(){}};}dispose(){}}
class Sprite extends Object3D {constructor(material){super();this.material=material;}}
const context2d=new Proxy({}, {get:(_,key)=>['fillStyle','font','textAlign','textBaseline'].includes(key)?undefined:()=>{},set:()=>true});
class Element {constructor(){this.children=[];this.value=0;}append(x){this.children.push(x);}replaceChildren(){this.children=[];}setAttribute(k,v){this[k]=v;}getContext(){return context2d;}}
const elements=new Map();
const document={createElement:()=>new Element(),getElementById:id=>{if(!elements.has(id))elements.set(id,new Element());return elements.get(id);}};
const THREE={Group:Object3D,Sprite,SpriteMaterial:Material,CanvasTexture:class{dispose(){}}};
const object=()=>{const o=new Object3D();o.geometry={dispose(){}};o.material=new Material();return o;};
const sandbox=vm.createContext({document,console,Map,JSON,Math,Error});
const helpers=new vm.SyntheticModule(['tween','wait','box','cylinder'],function(){
 this.setExport('tween',async(ms,fn)=>{fn(0);fn(.5);fn(1);});this.setExport('wait',async()=>{});
 this.setExport('box',object);this.setExport('cylinder',object);
},{context:sandbox});
const source=await fs.readFile(new URL('../station.js',import.meta.url),'utf8');
const module=new vm.SourceTextModule(source,{context:sandbox});
await module.link(spec=>{assert.equal(spec,'/web/lib/fly.js');return helpers;});await module.evaluate();
const station=module.namespace.default,group=new Object3D(),fly={root:new Object3D(),face(){},async buzz(){}};
station.build({THREE,group,fly});assert.equal(station.squares.length,64);assert.equal(station.pieces.size,32);
const recording=JSON.parse(await fs.readFile(new URL('../artifacts/matches/game_0000.json',import.meta.url),'utf8'));
for(let slot=0;slot<8;slot++)await station.play('slot_'+slot,'test |CHESS|'+JSON.stringify({events:[],fen:recording.initial_fen,stage:'source_file',outcome:{}}),{fly});
for(const move of recording.moves){
 await station.play('slot_'+move.stages.at(-1).action,'real move |CHESS|'+JSON.stringify({events:[move],fen:move.fen,stage:'source_file',outcome:move.outcome}),{fly});
 assert.equal(station.fen,move.fen);
 let total=0;for(const c of move.fen.split(' ')[0])if(/[a-z]/i.test(c))total++;
 assert.equal(station.pieces.size,total);
}
await assert.rejects(()=>station.play('invalid','',{fly}));
await assert.rejects(()=>station.play('slot_0','no state',{fly}));
// The recording viewer must reconstruct initial, intermediate, and final board cells exactly.
const html=await fs.readFile(new URL('../viewer.html',import.meta.url),'utf8');
const timers=[];
const viewerContext=vm.createContext({document,console,JSON,Error,setInterval(fn){timers.push(fn)},URLSearchParams,
 location:{search:''},recording});
vm.runInContext(html.match(/<script>([\s\S]*?)<\/script>/)[1],viewerContext);
vm.runInContext('load(recording,true)',viewerContext);
assert.equal(elements.get('board').children.length,64);
assert.equal(elements.get('fen').textContent,recording.final_fen);
assert.equal(elements.get('counter').textContent,recording.moves.length+' / '+recording.moves.length);
vm.runInContext('index=0;render()',viewerContext);
assert.equal(elements.get('fen').textContent,recording.initial_fen);
vm.runInContext('el("play").onclick()',viewerContext);await timers[0]();
assert.equal(elements.get('fen').textContent,recording.moves[0].fen);
for(let i=0;i<recording.moves.length+1;i++)await timers[0]();
assert.equal(elements.get('play').textContent,'Play');
assert.equal(elements.get('move-list').children.length,recording.moves.length);
viewerContext.invalid=JSON.parse(JSON.stringify(recording));viewerContext.invalid.final_fen='invalid';
assert.throws(()=>vm.runInContext('load(invalid)',viewerContext),/Final FEN/);
assert.equal(elements.get('fen').textContent,recording.final_fen);
viewerContext.invalid=JSON.parse(JSON.stringify(recording));viewerContext.invalid.moves[0].stages[0].probabilities.fill(0);
assert.throws(()=>vm.runInContext('load(invalid)',viewerContext),/probabilities/);
console.log('PASS: 64 station squares, 8 slot branches, '+recording.moves.length+' actual recorded moves, FEN/piece counts, viewer replay/autoplay/move list, invalid-event rejection. No WebGL render claimed.');
