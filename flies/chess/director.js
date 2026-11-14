import {movePlan, squarePosition} from './arena_state.js';
const ease=t=>t*t*(3-2*t);
export const STATES=['IDLE','THINKING','APPROACHING','PICKUP','CARRYING','PLACING','RELEASING','RETURNING','TURN_COMPLETE'];
export class Director {
  constructor(arena){this.arena=arena;this.state='IDLE';this.speed=1;this.paused=false;this.generation=0;this.step=null;this.busy=false;}
  update(dt){
    const s=this.step;if(!s||this.paused)return;
    s.elapsed=Math.min(s.duration,s.elapsed+Math.min(dt,.05)*this.speed);
    s.fn(ease(s.elapsed/s.duration));
    if(s.elapsed>=s.duration){this.step=null;s.resolve();}
  }
  wait(duration,fn=()=>{}){
    if(this.step)throw Error('Overlapping animation step');
    return new Promise((resolve,reject)=>{this.step={duration,elapsed:0,fn,resolve,reject};fn(0);});
  }
  cancel(){this.generation++;if(this.step){const s=this.step;this.step=null;s.reject(Error('Animation cancelled'));}this.busy=false;this.state='IDLE';this.arena.detachAll?.();}
  async flight(side,to,duration,arc=.5){
    const from=this.arena.flyPosition(side);
    const previous=this.arena.flyYaw(side);
    const yaw=Math.atan2(to[0]-from[0],to[2]-from[2]);
    const delta=Math.atan2(Math.sin(yaw-previous),Math.cos(yaw-previous));
    await this.wait(duration,t=>{
      const p=from.map((x,i)=>x+(to[i]-x)*t+(i===1?Math.sin(Math.PI*t)*arc:0));
      this.arena.setFly(side,p,previous+delta*Math.min(1,t*3));
    });
  }
  async transport(side,from,to){
    const height=this.arena.pieceHeight(from),src=squarePosition(from),dst=squarePosition(to);
    this.state='APPROACHING';
    await this.flight(side,[src[0],src[1]+height+.95,src[2]],1.15,.65);
    this.state='PICKUP';await this.flight(side,[src[0],src[1]+height-.05,src[2]],.6,0);
    this.arena.attach(side,from);await this.wait(.24,t=>this.arena.grip(side,t));
    await this.flight(side,[src[0],src[1]+height+.9,src[2]],.5,0);
    this.state='CARRYING';await this.flight(side,[dst[0],dst[1]+height+.9,dst[2]],1.3,.5);
    this.state='PLACING';await this.flight(side,[dst[0],dst[1]+height-.05,dst[2]],.7,0);
    this.arena.release(side,from,to);this.state='RELEASING';
    await this.wait(.26,t=>this.arena.grip(side,1-t));this.arena.pulse(to);
    await this.flight(side,[dst[0],dst[1]+height+.8,dst[2]],.4,0);
  }
  async play(event){
    if(this.busy)throw Error('Move animations cannot overlap');
    const plan=movePlan(event),generation=this.generation;this.busy=true;
    try{
      this.arena.assertBoard(event.before);this.arena.highlight(plan.source,plan.destination);
      if(plan.capture)await this.wait(.45,t=>this.arena.capture(plan.capture,t));
      await this.transport(plan.side,plan.source,plan.destination);
      if(plan.rook)await this.transport(plan.side,...plan.rook);
      if(plan.promotion)this.arena.promote(plan.destination,plan.promotion,plan.side);
      this.state='RETURNING';await this.flight(plan.side,this.arena.home(plan.side),1.15,.6);
      this.arena.rest(plan.side);this.arena.assertBoard(event.fen);this.arena.fen=event.fen;
      if(generation!==this.generation)throw Error('Stale move completion');
      this.state='TURN_COMPLETE';this.arena.check?.(event);return event.fen;
    }finally{if(generation===this.generation)this.busy=false;}
  }
}
