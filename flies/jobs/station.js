import {box,tween,wait} from '/web/lib/fly.js';
export default {
 title:'Elton — simulated candidate',
 blurb:'Displays selected neural actions. Checkpoint provenance is provided by the local runner.',
 build({THREE,group,fly}) {
  group.add(box(3.8,.12,2.6,0x203c58,[0,.06,0]));this.cards=[];
  for(let i=0;i<8;i++){
   const card=box(.65,.08,.55,0xb8cbdc,[(i%4-1.5)*.85,.17,Math.floor(i/4)*.85-.5]);
   group.add(card);this.cards.push(card);
  }
  fly.root.position.set(0,.13,1.05);
 },
 async play(action,caption,{fly,event}) {
  const match=/^workday_slot_([0-7])$/.exec(action);
  if(!match)return;
  const card=this.cards[Number(match[1])];
  await fly.walkTo({x:card.position.x,z:card.position.z},350);
  const color=card.material.color.getHex();
  const outcome=event?.event||caption;
  card.material.color.setHex(/incorrect|error|rejected|invalid|unnecessary/.test(outcome)?0xe76f51:
    /correct|uploaded|created|login_success|simulated_submission/.test(outcome)?0x64c792:0xf5bf50);
  await tween(180,t=>card.position.y=.17+Math.sin(Math.PI*t)*.15);
  if(/simulated_submission/.test(outcome))await fly.buzz(350);
  card.material.color.setHex(color);await wait(60);
 }
};
