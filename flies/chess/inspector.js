// Adapter around the ORIGINAL shared Matrix, never a replacement neural viewer.
export class Inspector {
  constructor(api,onClose,onLive){this.api=api;this.onClose=onClose;this.onLive=onLive;this.side=null;this.serial=Promise.resolve();this.generation=0;this.last=null;this.live=false;this.packets={};}
  async initialize(){
    const response=await fetch('/web/index.html');if(!response.ok)throw Error('Shared brain viewer host unavailable');
    const host=new DOMParser().parseFromString(await response.text(),'text/html');
    const style=document.createElement('style');style.textContent=host.querySelector('style').textContent;document.head.prepend(style);
    document.body.append(document.importNode(host.getElementById('matrix'),true));
    const context=document.createElement('aside');context.id='neural-context';document.body.append(context);this.context=context;
    const {Matrix}=await import('/web/matrix.js');
    this.matrix=new Matrix({onClose:()=>{this.side=null;this.generation++;context.classList.remove('open');document.body.classList.remove('inspecting');this.onClose();},onLesion:()=>{}});
    this.matrix.renderer.setPixelRatio(Math.min(devicePixelRatio,1.5));
    const lesion=document.getElementById('m-lesion');lesion.disabled=true;lesion.textContent='Live inspection · weights fixed';
    document.getElementById('rain').setAttribute('aria-label','Decorative glyph background, not measured neural activity');
  }
  disposePoints(){for(const p of [this.matrix.points,...(this.matrix.groupPoints||[])])if(p){p.geometry.dispose();p.material.dispose();}this.matrix.points=null;this.matrix.groupPoints=[];}
  open(side,state){
    if(!state?.infos?.[side])return Promise.reject(Error('Wait for the independent models to finish loading'));
    const generation=++this.generation;this.side=side;this.last=null;
    // Matrix.show is asynchronous. Wait for it before changing its source to
    // prevent old White packets being applied to Black's point buffers.
    this.serial=this.serial.catch(()=>{}).then(async()=>{
      if(generation!==this.generation)return;
      this.disposePoints();await this.matrix.open(state.infos[side],`${side.toUpperCase()} · CHESS FLY`);
      document.body.classList.add('inspecting');this.context.classList.add('open');
      const b=document.getElementById('m-lesion');b.disabled=true;b.textContent='Live inspection · weights fixed';
      await this.display(side,state,generation);
    });return this.serial;
  }
  refresh(state){
    if(!this.side||this.queued)return;
    const side=this.side,generation=this.generation;this.queued=true;
    this.serial=this.serial.catch(()=>{}).then(()=>this.display(side,state,generation)).finally(()=>this.queued=false);
    return this.serial;
  }
  async display(side,state,generation){
    if(generation!==this.generation)return;
    const model=state.players[side];
    const cached=this.packets[side],id=state.activity_ids?.[side];
    let packet;
    try{packet=!id?null:cached?.match_id===state.match_id&&cached?.decision_id===id?cached:await this.api('/api/telemetry',{side});}
    catch(e){if(generation!==this.generation)return;throw e;}
    if(packet)this.packets[side]=packet; // At most one measured packet per side.
    if(generation!==this.generation)return;
    this.context.replaceChildren();
    const title=document.createElement('strong');title.textContent=`${side.toUpperCase()} · SIMULATED NEURON RATES`;this.context.append(title);
    const text=document.createElement('p');
    const computing=state.phase==='thinking'&&state.side===side&&!state.paused;
    text.textContent=`${model.label} · ${model.preset} · ${model.neurons.toLocaleString()} neurons\n${model.checkpoint?.split('/').slice(-2).join('/')||'Fresh initialization'}\nGraph ${model.graph_hash}\n${computing?'COMPUTING · next update pending':'Stored most recent measured activity'}\n${packet?'Verified inference · '+packet.timestamp:'No inference recorded for this brain yet.'}\n${packet?`${packet.stage} → ${packet.action}\nOutput groups: ${packet.output_groups.map(x=>x.toFixed(4)).join(', ')}\nObservation: ${packet.observation_encoding} · 64 floats\nDecision ${packet.sequence}`:''}`;
    this.context.append(text);
    const note=document.createElement('p');note.className='neural-note';note.textContent='Measured rate iterations replay at 90 ms/step. Point colors encode sqrt(rate/2), clipped to uint8. Base colors and glyph rain are decorative. Inspection initially pauses play. Live updates continue the physical animation off-screen; weights stay fixed.';this.context.append(note);
    for(const s of ['white','black']){const b=document.createElement('button');b.textContent=`Inspect ${s}`;b.onclick=()=>this.open(s,state);this.context.append(b);}
    const live=document.createElement('button');live.textContent=this.live?'Pause game updates':'Observe live updates';live.onclick=async()=>{this.live=!this.live;await this.onLive(this.live);};this.context.append(live);
    if(!packet){document.getElementById('m-caption').textContent='No measured inference yet. Close and play a move.';return;}
    if(packet.match_id!==state.match_id||packet.side!==side)throw Error('Neural activity provenance mismatch');
    if(packet.decision_id===this.last)return;
    this.last=packet.decision_id;
    await this.matrix.show(packet);
  }
}
