import * as THREE from 'three';
import {OrbitControls} from 'three/addons/controls/OrbitControls.js';
import {RoomEnvironment} from 'three/addons/environments/RoomEnvironment.js';
import {EffectComposer} from 'three/addons/postprocessing/EffectComposer.js';
import {RenderPass} from 'three/addons/postprocessing/RenderPass.js';
import {SSAOPass} from 'three/addons/postprocessing/SSAOPass.js';
import {OutputPass} from 'three/addons/postprocessing/OutputPass.js';
import {mergeGeometries} from 'three/addons/utils/BufferGeometryUtils.js';
import {Fly} from '/web/lib/fly.js';
import {placement,boardFen,squarePosition,sideOf} from './arena_state.js';

const HEIGHT={p:.75,r:.94,n:1.04,b:1.13,q:1.27,k:1.43};
const v=a=>new THREE.Vector3(...a);
function pieceGeometry(type){
  const profiles={
    p:[[0,0],[.27,0],[.29,.045],[.28,.10],[.21,.15],[.16,.22],[.10,.39],[.15,.43],[.15,.49],[0,.49]],
    r:[[0,0],[.29,0],[.30,.06],[.28,.13],[.20,.18],[.17,.59],[.25,.65],[.25,.80],[0,.80]],
    n:[[0,0],[.29,0],[.30,.055],[.27,.13],[.19,.20],[.17,.32],[0,.32]],
    b:[[0,0],[.29,0],[.30,.06],[.26,.14],[.19,.19],[.12,.60],[.19,.65],[.19,.71],[.10,.75],[0,.75]],
    q:[[0,0],[.31,0],[.32,.06],[.28,.14],[.20,.20],[.13,.66],[.21,.72],[.18,.83],[.24,1.02],[0,1.02]],
    k:[[0,0],[.32,0],[.33,.06],[.29,.14],[.21,.21],[.14,.71],[.22,.77],[.22,.85],[.13,.90],[.10,1.03],[0,1.03]]
  };
  const parts=[new THREE.LatheGeometry(profiles[type].map(([x,y])=>new THREE.Vector2(x,y)),48)];
  function add(g,x,y,z,ry=0){g.rotateY(ry);g.translate(x,y,z);parts.push(g);}
  if(type==='p')add(new THREE.SphereGeometry(.18,24,16),0,.57,0);
  if(type==='r')for(let i=0;i<6;i++){const a=i*Math.PI/3;add(new THREE.BoxGeometry(.13,.16,.14),Math.sin(a)*.19,.86,Math.cos(a)*.19,a);}
  if(type==='b'){
    // Two lobes leave a real diagonal slit in the bishop mitre.
    const g=new THREE.SphereGeometry(.205,28,20,0,Math.PI*.93);g.scale(.85,1.5,1);g.rotateZ(-.18);add(g,-.012,.88,0);
    const h=new THREE.SphereGeometry(.205,28,20,Math.PI*1.05,Math.PI*.93);h.scale(.85,1.5,1);h.rotateZ(-.18);add(h,.012,.88,0);
    add(new THREE.SphereGeometry(.065,16,10),-.055,1.105,0);
  }
  if(type==='q'){
    for(let i=0;i<7;i++){const a=i*Math.PI*2/7;add(new THREE.SphereGeometry(.052,14,10),Math.sin(a)*.235,1.07,Math.cos(a)*.235);}
    add(new THREE.SphereGeometry(.085,20,12),0,1.185,0);
  }
  if(type==='k'){add(new THREE.BoxGeometry(.12,.40,.12),0,1.23,0);add(new THREE.BoxGeometry(.34,.10,.12),0,1.28,0);}
  if(type==='n'){
    const s=new THREE.Shape();s.moveTo(-.18,.27);s.bezierCurveTo(-.26,.44,-.22,.79,-.10,.90);s.lineTo(-.12,1.04);s.lineTo(.015,.96);s.lineTo(.10,.98);s.lineTo(.22,.85);s.lineTo(.30,.77);s.lineTo(.27,.66);s.lineTo(.13,.65);s.lineTo(.04,.73);s.bezierCurveTo(-.02,.65,.10,.48,.18,.30);s.closePath();
    const g=new THREE.ExtrudeGeometry(s,{depth:.22,bevelEnabled:true,bevelSegments:3,steps:1,bevelSize:.035,bevelThickness:.03,curveSegments:20});
    g.translate(0,0,-.11);g.rotateY(Math.PI/2);parts.push(g);
  }
  const normalized=parts.map(g=>g.index?g.toNonIndexed():g);
  // Preserve the smooth sphere/lathe normals and the intentionally sharp crown
  // and cross normals. Recomputing after deindexing would facet every surface.
  const result=mergeGeometries(normalized,false);
  new Set([...parts,...normalized]).forEach(g=>g.dispose());return result;
}
function marbleTexture(){
  const c=document.createElement('canvas');c.width=c.height=256;const ctx=c.getContext('2d');
  ctx.fillStyle='#ded9ca';ctx.fillRect(0,0,256,256);
  // Deterministic subtle stone veins, solely a visual texture.
  for(let j=0;j<22;j++){ctx.beginPath();for(let x=0;x<=256;x+=4){const y=(j*23+x*.47+Math.sin(x*.028+j)*14)%300; x?ctx.lineTo(x,y):ctx.moveTo(x,y);}ctx.strokeStyle=`rgba(92,91,77,${j%3===0?.08:.025})`;ctx.lineWidth=j%3===0?1.4:3;ctx.stroke();}
  const t=new THREE.CanvasTexture(c);t.colorSpace=THREE.SRGBColorSpace;t.wrapS=t.wrapT=THREE.RepeatWrapping;return t;
}
export class Arena {
  constructor(host,{onFly=()=>{},onContext=()=>{}}={}){
    this.host=host;this.pieces=new Map();this.flies={};this.held={};this.pulses=[];this.cameraMode='cinematic';this.time=0;this.quality='medium';this.active='white';
    this.renderer=new THREE.WebGLRenderer({antialias:true,alpha:false,powerPreference:'high-performance'});
    this.renderer.setPixelRatio(Math.min(devicePixelRatio,1.75));this.renderer.shadowMap.enabled=true;this.renderer.shadowMap.type=THREE.PCFSoftShadowMap;
    this.renderer.toneMapping=THREE.ACESFilmicToneMapping;this.renderer.toneMappingExposure=1.05;
    host.append(this.renderer.domElement);this.scene=new THREE.Scene();this.scene.background=new THREE.Color(0x101416);this.scene.fog=new THREE.FogExp2(0x101416,.022);
    this.camera=new THREE.PerspectiveCamera(38,1,.1,100);this.camera.position.set(11.2,12.3,16.8);
    this.controls=new OrbitControls(this.camera,this.renderer.domElement);this.controls.target.set(0,.3,0);this.controls.enableDamping=true;this.controls.minDistance=8;this.controls.maxDistance=30;this.controls.maxPolarAngle=Math.PI*.47;
    this.controls.addEventListener('start',()=>{this.cameraMode='free';document.getElementById('camera').value='free';});
    const room=new RoomEnvironment(),pmrem=new THREE.PMREMGenerator(this.renderer);this.environment=pmrem.fromScene(room,.04);this.scene.environment=this.environment.texture;this.scene.environmentIntensity=.28;room.dispose();pmrem.dispose();
    this.scene.add(new THREE.HemisphereLight(0xe8eddf,0x28251f,1.25));
    const key=new THREE.DirectionalLight(0xffe7bd,4.2);key.position.set(-4,11,6);key.castShadow=true;key.shadow.mapSize.set(2048,2048);Object.assign(key.shadow.camera,{left:-8,right:8,top:8,bottom:-8,near:.5,far:30});key.shadow.bias=-.00025;key.shadow.normalBias=.025;key.shadow.radius=3;this.scene.add(key);this.key=key;
    const rim=new THREE.DirectionalLight(0xbacbcb,2.8);rim.position.set(6,7,-8);this.scene.add(rim);
    const gold=new THREE.PointLight(0xffd99a,35,18,2);gold.position.set(-7,4,-3);this.scene.add(gold);
    this.materials={white:new THREE.MeshPhysicalMaterial({color:0xeee6d3,roughness:.26,metalness:.06,clearcoat:.5,clearcoatRoughness:.22}),black:new THREE.MeshPhysicalMaterial({color:0x283033,roughness:.24,metalness:.48,clearcoat:.55,clearcoatRoughness:.22}),gold:new THREE.MeshStandardMaterial({color:0xaa925c,roughness:.3,metalness:.8})};
    this.geometries=Object.fromEntries(Object.keys(HEIGHT).map(t=>[t,pieceGeometry(t)]));
    this.pieceGroup=new THREE.Group();this.scene.add(this.pieceGroup);
    this.board();this.models();
    this.composer=new EffectComposer(this.renderer);this.composer.addPass(new RenderPass(this.scene,this.camera));
    this.ao=new SSAOPass(this.scene,this.camera,innerWidth,innerHeight,16);this.ao.kernelRadius=.12;this.ao.minDistance=.002;this.ao.maxDistance=.06;this.ao.enabled=false;this.composer.addPass(this.ao);this.composer.addPass(new OutputPass());
    this.ray=new THREE.Raycaster();this.down=null;
    this.renderer.domElement.addEventListener('pointerdown',e=>this.down=[e.clientX,e.clientY]);
    this.renderer.domElement.addEventListener('pointerup',e=>{
      if(!this.down||Math.hypot(e.clientX-this.down[0],e.clientY-this.down[1])>6)return;
      this.ray.setFromCamera(new THREE.Vector2(e.clientX/innerWidth*2-1,1-e.clientY/innerHeight*2),this.camera);
      const hit=this.ray.intersectObjects(Object.values(this.flies).map(f=>f.root),true)[0];
      if(hit){let o=hit.object;while(o&&!o.userData.side)o=o.parent;if(o)onFly(o.userData.side);}
    });
    this.renderer.domElement.addEventListener('webglcontextlost',e=>{e.preventDefault();onContext(false);});
    this.renderer.domElement.addEventListener('webglcontextrestored',()=>onContext(true));
    this.resize=()=>{const w=innerWidth,h=innerHeight;this.renderer.setSize(w,h);this.composer.setSize(w,h);this.camera.aspect=w/h;this.camera.updateProjectionMatrix();};addEventListener('resize',this.resize);this.resize();
  }
  board(){
    const baseMat=new THREE.MeshPhysicalMaterial({color:0x202820,roughness:.29,metalness:.28,clearcoat:.7});
    const addBox=(w,h,d,mat,y)=>{const m=new THREE.Mesh(new THREE.BoxGeometry(w,h,d),mat);m.position.y=y;m.castShadow=m.receiveShadow=true;this.scene.add(m);return m;};
    addBox(9.3,.34,9.3,baseMat,.04);addBox(9.12,.025,9.12,this.materials.gold,.22);addBox(9.04,.09,9.04,baseMat,.24);
    const light=new THREE.MeshPhysicalMaterial({color:0xd9d5c6,map:marbleTexture(),roughness:.38,metalness:.08,clearcoat:.4});
    const dark=new THREE.MeshPhysicalMaterial({color:0x364139,roughness:.34,metalness:.16,clearcoat:.4});
    const g=new THREE.BoxGeometry(.997,.035,.997),dummy=new THREE.Object3D();
    for(let parity=0;parity<2;parity++){
      const tiles=new THREE.InstancedMesh(g,parity?light:dark,32);let i=0;
      for(let rank=0;rank<8;rank++)for(let file=0;file<8;file++)if((rank+file)%2===parity){dummy.position.set(file-3.5,.291,3.5-rank);dummy.updateMatrix();tiles.setMatrixAt(i++,dummy.matrix);}
      tiles.receiveShadow=true;this.scene.add(tiles);
    }
    const floor=new THREE.Mesh(new THREE.PlaneGeometry(200,200),new THREE.MeshStandardMaterial({color:0x13181a,roughness:.8,metalness:.12}));floor.rotation.x=-Math.PI/2;floor.position.y=-.17;floor.receiveShadow=true;this.scene.add(floor);
    // Engraved coordinates share one texture and one plane.
    const canvas=document.createElement('canvas');canvas.width=canvas.height=1024;const c=canvas.getContext('2d');c.clearRect(0,0,1024,1024);c.font='17px Georgia';c.textAlign='center';c.fillStyle='#b7a67c';
    for(let i=0;i<8;i++){const x=(i+.5)/9*1024+1024/18;c.fillText('abcdefgh'[i],x,996);c.fillText(String(8-i),27,x+5);}
    const texture=new THREE.CanvasTexture(canvas);texture.colorSpace=THREE.SRGBColorSpace;const labels=new THREE.Mesh(new THREE.PlaneGeometry(9,9),new THREE.MeshBasicMaterial({map:texture,transparent:true,depthWrite:false}));labels.rotation.x=-Math.PI/2;labels.position.y=.308;this.scene.add(labels);
    this.markers=[];
    for(const color of [0xd4bd83,0xeee5ce,0xd0a365]){const ring=new THREE.Mesh(new THREE.RingGeometry(.36,.40,64),new THREE.MeshBasicMaterial({color,transparent:true,opacity:.6,side:THREE.DoubleSide,depthWrite:false}));ring.rotation.x=-Math.PI/2;ring.visible=false;this.scene.add(ring);this.markers.push(ring);}
  }
  models(){
    for(const [i,side] of ['white','black'].entries()){
      const fly=new Fly(i?0x776442:0x9a7a48);fly.root.scale.setScalar(2.3);fly.root.userData.side=side;fly.root.position.copy(v(this.home(side)));this.scene.add(fly.root);this.flies[side]=fly;
      this.rest(side);
      const halo=new THREE.Mesh(new THREE.RingGeometry(.58,.60,64),new THREE.MeshBasicMaterial({color:i?0x88989a:0xcbb783,transparent:true,opacity:.45,side:THREE.DoubleSide}));halo.rotation.x=-Math.PI/2;halo.position.set(fly.root.position.x,-.155,fly.root.position.z);this.scene.add(halo);
      fly.legs=[];fly.body.traverse(o=>{if(o.geometry?.type==='CylinderGeometry')fly.legs.push({mesh:o,z:o.rotation.z});});
    }
  }
  home(side){return side==='white'?[-.9,.24,5.45]:[.9,.24,-5.45];}
  rest(side){const f=this.flies[side];f.face(v([0,0,0]));this.grip(side,0);}
  flyPosition(side){return this.flies[side].root.position.toArray();}
  flyYaw(side){return this.flies[side].root.rotation.y;}
  setFly(side,p,yaw){const f=this.flies[side];f.root.position.set(...p);f.root.rotation.y=yaw;}
  pieceHeight(square){return HEIGHT[this.pieces.get(square).userData.symbol.toLowerCase()];}
  makePiece(symbol){const side=sideOf(symbol),m=new THREE.Mesh(this.geometries[symbol.toLowerCase()],this.materials[side]);m.castShadow=m.receiveShadow=true;m.userData.symbol=symbol;m.rotation.y=side==='white'?Math.PI:0;return m;}
  reset(fen){
    this.detachAll();for(const piece of this.pieces.values())piece.removeFromParent();this.pieces.clear();
    // Captures are removed from the scene and share cached geometry/materials.
    this.captureMesh?.removeFromParent();this.captureMesh=null;
    for(const [sq,p] of Object.entries(placement(fen))){const m=this.makePiece(p);m.position.set(...squarePosition(sq));this.pieceGroup.add(m);this.pieces.set(sq,m);}
    this.fen=fen;this.pulses=[];this.markers.forEach(m=>{m.visible=false;m.scale.setScalar(1);});
    for(const side of ['white','black']){this.setFly(side,this.home(side),0);this.rest(side);}
  }
  attach(side,square){const piece=this.pieces.get(square);if(!piece||sideOf(piece.userData.symbol)!==side)throw Error('Fly can only carry its own piece');this.flies[side].root.attach(piece);this.held[side]=piece;}
  release(side,from,to){const p=this.held[side];if(p!==this.pieces.get(from))throw Error('Attachment mismatch');this.pieceGroup.attach(p);p.position.set(...squarePosition(to));p.rotation.set(0,side==='white'?Math.PI:0,0);p.scale.set(1,1,1);this.pieces.delete(from);this.pieces.set(to,p);delete this.held[side];}
  detachAll(){for(const side of Object.keys(this.held)){this.pieceGroup.attach(this.held[side]);delete this.held[side];}}
  grip(side,t){for(const {mesh,z} of this.flies[side].legs||[])mesh.rotation.z=z*(1-.75*t);}
  capture(square,t){
    const p=this.pieces.get(square);if(!p)throw Error('Missing captured piece');
    p.position.y=.31-t*.9;p.scale.setScalar(1-t*.4);
    if(t===1){this.pieces.delete(square);p.removeFromParent();}
  }
  promote(square,type,side){const old=this.pieces.get(square);old.removeFromParent();const p=this.makePiece(side==='white'?type.toUpperCase():type);p.position.set(...squarePosition(square));this.pieceGroup.add(p);this.pieces.set(square,p);}
  assertBoard(fen){
    const map={};for(const [sq,p] of this.pieces){if(p.parent!==this.pieceGroup)throw Error('A piece is still attached');const expected=squarePosition(sq);if(p.position.toArray().some((x,i)=>Math.abs(x-expected[i])>.001))throw Error('Piece placement drift');map[sq]=p.userData.symbol;}
    if(boardFen(map)!==fen.split(' ')[0])throw Error('Rendered board disagrees with authoritative FEN');return boardFen(map);
  }
  highlight(from,to){[from,to].forEach((sq,i)=>{this.markers[i].position.set(...squarePosition(sq,.316));this.markers[i].visible=true;});}
  pulse(square){this.markers[1].position.set(...squarePosition(square,.316));this.pulses=[{time:this.time}];}
  check(event){const king=Object.entries(placement(event.fen)).find(([,p])=>p===(event.fen.split(' ')[1]==='w'?'K':'k'));const marker=this.markers[2];marker.visible=!!event.visual?.check;if(king)marker.position.set(...squarePosition(king[0],.318));}
  setQuality(q){this.quality=q;this.renderer.setPixelRatio(Math.min(devicePixelRatio,q==='low'?1:q==='high'?2:1.5));this.ao.enabled=q==='high';this.key.shadow.mapSize.set(q==='low'?1024:2048,q==='low'?1024:2048);this.key.shadow.map?.dispose();this.key.shadow.map=null;this.resize();}
  update(dt,director){
    this.time+=dt;const t=this.time;
    for(const [i,side] of ['white','black'].entries()){
      const f=this.flies[side],moving=director.busy&&this.active===side;
      f.body.position.y=Math.sin(t*(i?2.1:2.5)+i*2)*.013;
      f.body.rotation.x=moving?-.035:Math.sin(t*.7+i)*.015;
      f.body.rotation.y=moving?0:Math.sin(t*.48+i*2.4)*.035;
      for(const {pivot,s} of f.wings)pivot.rotation.z=s*(.18+.48*Math.abs(Math.sin(t*(i?67:61)+i)));
    }
    if(this.pulses.length){const age=t-this.pulses[0].time;this.markers[1].scale.setScalar(1+.16*Math.sin(Math.min(1,age/.8)*Math.PI));this.markers[1].material.opacity=.45+.3*Math.max(0,1-age);if(age>1)this.pulses=[];}
    if(this.cameraMode!=='free'){
      let pos=[11.2,12.3,16.8],target=[0,.3,0];
      if(this.cameraMode==='board')pos=[.01,18,.15];
      if(this.cameraMode==='showcase')pos=[Math.sin(t*.06)*17,11.7,Math.cos(t*.06)*17];
      if(this.cameraMode==='follow'){const p=this.flyPosition(this.active);pos=[p[0]*.4+8,9.5,p[2]*.3+11];target=[p[0]*.35,.7,p[2]*.35];}
      const blend=1-Math.exp(-dt*2);this.camera.position.lerp(v(pos),blend);this.controls.target.lerp(v(target),blend);
    }
    this.controls.update();this.composer.render();
  }
  dispose(){removeEventListener('resize',this.resize);this.controls.dispose();this.composer.dispose();this.ao.dispose();this.environment.dispose();const resources=new Set();this.scene.traverse(o=>{if(o.geometry)resources.add(o.geometry);for(const m of Array.isArray(o.material)?o.material:[o.material])if(m){resources.add(m);for(const val of Object.values(m))if(val?.isTexture)resources.add(val);}});resources.forEach(r=>r.dispose());this.renderer.dispose();this.renderer.domElement.remove();}
}
