import {box,tween,wait} from '/web/lib/fly.js';
export default {
 title:'Elton Inglewood',
 blurb:'Displays selected neural actions. Checkpoint provenance is provided by the local runner.',
 build({THREE,group,fly}) {
  group.add(box(3.8,.12,2.6,0x203c58,[0,.06,0]));this.cards=[];
  for(let i=0;i<8;i++){
   const card=box(.65,.08,.55,0xb8cbdc,[(i%4-1.5)*.85,.17,Math.floor(i/4)*.85-.5]);
   // label on the card's top face: what this slot does right now (ctx.view.actions, from task.view())
   const canvas=document.createElement('canvas');canvas.width=256;canvas.height=216;
   const tex=new THREE.CanvasTexture(canvas);tex.colorSpace=THREE.SRGBColorSpace;
   const side=card.material;
   card.material=[side,side,new THREE.MeshStandardMaterial({map:tex,roughness:.7}),side,side,side];
   card.userData={canvas,tex,top:card.material[2]};
   group.add(card);this.cards.push(card);this.label(card,'');
  }
  fly.root.position.set(0,.13,1.05);
 },
 label(card,text) {
  const {canvas,tex}=card.userData,c=canvas.getContext('2d');
  c.fillStyle=text?'#dfe8f1':'#9fb3c6';c.fillRect(0,0,256,216);
  c.fillStyle='#14283c';c.textAlign='center';c.textBaseline='middle';c.font='600 30px Helvetica, Arial, sans-serif';
  const words=text.split(/\s+/).filter(Boolean),lines=[];let line='';
  for(const w of words){const n=line?line+' '+w:w;if(c.measureText(n).width>228&&line){lines.push(line);line=w;}else line=n;}
  if(line)lines.push(line);
  lines.slice(0,5).forEach((l,j,a)=>c.fillText(l,128,108+(j-(a.length-1)/2)*36));
  tex.needsUpdate=true;
 },
 async play(action,caption,{fly,event,view}) {
  const match=/^workday_slot_([0-7])$/.exec(action);
  if(!match)return;
  (view?.actions||[]).forEach((t,i)=>this.cards[i]&&this.label(this.cards[i],t));
  const card=this.cards[Number(match[1])];
  await fly.walkTo({x:card.position.x,z:card.position.z},350);
  const top=card.userData.top,color=top.color.getHex();
  const outcome=event?.event||caption;
  top.color.setHex(/incorrect|error|rejected|invalid|unnecessary/.test(outcome)?0xe76f51:
    /correct|uploaded|created|login_success|simulated_submission/.test(outcome)?0x64c792:0xf5bf50);
  await tween(180,t=>card.position.y=.17+Math.sin(Math.PI*t)*.15);
  if(/simulated_submission/.test(outcome))await fly.buzz(350);
  top.color.setHex(color);await wait(60);
 }
};
