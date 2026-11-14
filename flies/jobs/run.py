"""Sequential persistent LOCAL Workday-inspired execution, with visible browser.

Unlike server.serve, this command fails closed for absent/incompatible checkpoints.
It runs inference only. Training uses flies.jobs.learning, in a separate process.
"""
import argparse
from contextlib import ExitStack
import gzip
import hashlib
import json
from pathlib import Path
import time
import numpy as np
import torch
from core.brain import FlyBrain,pick_device
from flies.jobs.task import JobsFly
from flies.jobs.policy import load_model, probabilities, digest, checkpoint_path
from flies.jobs.sim_state import SimulationState, SecretVault, atomic_json, private_vault_path
from flies.jobs.contract import SCHEMA
from flies.jobs.jobs import load_jobs, job_id


def main():
    parser=argparse.ArgumentParser(__doc__)
    parser.add_argument('--checkpoint',type=Path)
    parser.add_argument('--device',default='auto')
    parser.add_argument('--policy',choices=['trained','brain','random','untrained','lesioned'],default='trained')
    parser.add_argument('--state-dir',type=Path,default=Path('flies/jobs/local_state'))
    parser.add_argument('--jobs-file',type=Path,default=Path('flies/jobs/elton_fly_job_links.txt'))
    parser.add_argument('--episodes',type=int,default=1,help='Decision chunks; pending forms remain first in queue')
    parser.add_argument('--level',type=int,default=9)
    parser.add_argument('--browser',action='store_true')
    parser.add_argument('--headed',action='store_true')
    parser.add_argument('--station',action='store_true',help='Embed station rendering of the SAME execution events')
    parser.add_argument('--delay',type=float,default=.3)
    parser.add_argument('--status',action='store_true')
    parser.add_argument('--verbose',action='store_true')
    parser.add_argument('--fresh-forms',action='store_true',help='Demo mode: every chunk is a NEW form (a timeout ends that job); '
                        'forms and action draws come from --seed and the chunk number, so different policies see identical forms')
    parser.add_argument('--seed',type=int,default=41)
    args=parser.parse_args()
    if args.episodes<1 or not 1<=args.level<=10 or args.delay<0:
        parser.error('positive episodes, level 1..10 and nonnegative delay required')
    if args.policy in ('trained','brain','lesioned') and args.checkpoint is None and not args.status:
        parser.error('an explicit --checkpoint is required; no untrained fallback')
    if args.station and not args.browser:
        parser.error('--station requires --browser')
    args.state_dir.mkdir(parents=True,exist_ok=True)
    world_path=args.state_dir/'simulation_state.json'
    world=SimulationState.load(world_path)
    if args.status:
        print(json.dumps({'simulated_accounts':len(world.accounts),'simulated_applications':len(world.applications),
                          'queue':{'terminal':world.drafts.get('queue',{}).get('terminal',{}),
                                   'pending_url':(world.drafts.get('queue',{}).get('pending') or {}).get('job',{}).get('url')},'mode':'simulation_only'},indent=2));return
    vault_file=private_vault_path(args.state_dir)
    if world.accounts and not vault_file.exists():
        raise ValueError('Credential vault missing for persisted simulation accounts; restore it before continuing')
    vault=SecretVault.open(vault_file)
    world.bind_vault(vault)
    jobs=load_jobs(args.jobs_file)
    torch.set_num_threads(2)
    device=pick_device(args.device)
    if device.type=='cuda' and not torch.cuda.is_available():parser.error('CUDA is unavailable')
    torch.manual_seed(2026)
    brain=None;checkpoint='uniform random'
    if args.policy=='untrained':
        brain=FlyBrain('tiny',64,8,'jobs').to(device);checkpoint='explicit fresh initialization seed 2026'
    elif args.policy!='random':
        brain=load_model(args.checkpoint,JobsFly,device);checkpoint=str(checkpoint_path(args.checkpoint))
    parameter_hash=digest(brain) if brain is not None else None
    rng=np.random.default_rng(args.seed)
    scoreboard=[]
    with ExitStack() as stack:
        log=stack.enter_context(gzip.open(args.state_dir/'actions.jsonl.gz','at',encoding='utf-8'))
        mirror=None
        if args.browser:
            from playwright.sync_api import sync_playwright
            from flies.jobs.browser import BrowserMirror
            driver=stack.enter_context(sync_playwright())
            browser=driver.chromium.launch(headless=not args.headed)
            stack.callback(browser.close)
            # Station runs get a taller page so the form, all 8 action choices and the 3D fly fit on screen.
            mirror=BrowserMirror(browser.new_page(viewport={'width':1440,'height':1000}) if args.station else browser.new_page())
            if args.station:
                from flies.jobs.station_server import StationServer
                station=stack.enter_context(StationServer())
                mirror.station_origin=station.origin
        queue=world.drafts.setdefault('queue',{'schema':SCHEMA,'terminal':{},'pending':None})
        if queue['schema']!=SCHEMA:
            raise ValueError('incompatible pending simulation queue')
        print(f'LOCAL SIMULATION ONLY · policy={args.policy} · checkpoint={checkpoint}',flush=True)
        for chunk in range(args.episodes):
            if args.fresh_forms:
                # Never resume a stuck form; paired seeds keep forms identical across policies.
                queue['pending']=None
                rng=np.random.default_rng([args.seed,chunk])
            pending=queue.get('pending')
            job=pending['job'] if pending else next((j for j in jobs if job_id(j) not in queue['terminal']),None)
            if job is None:
                print('Simulation queue exhausted.');break
            env=JobsFly(level=args.level,split='test',jobs=[job],state=world,vault=vault,trace=True)
            if pending:
                env.restore(pending['environment'],resume_chunk=True)
                rng.bit_generator.state=pending['rng']
            else:
                env.reset(rng)
            if mirror:
                mirror.open(env)
            for _ in range(env.max_steps):
                obs=mirror.observation() if mirror else env.obs()
                p=probabilities(brain,[obs],[env.mask()],args.policy,device)[0]
                action=int(rng.choice(8,p=p/p.sum()))
                try:
                    if mirror:mirror.execute(env,action)
                    _,_,done,info=env.step(action,rng)
                    event={**env.traces[-1],'policy':args.policy,'checkpoint':checkpoint,'parameter_hash':parameter_hash,'probabilities':p.tolist()}
                    if mirror:
                        mirror.sync(env,event,args.policy,checkpoint)
                        mirror.check(env)
                except Exception as error:
                    # Never print Playwright's raw arguments, which can contain secrets.
                    queue['pending']={'job':job,'environment':env.snapshot(),'rng':rng.bit_generator.state}
                    world.save(world_path)
                    raise RuntimeError('Browser execution failed; pending state saved. Error type: '+type(error).__name__) from None
                log.write(json.dumps(event)+'\n');log.flush()
                queue['pending']={'job':job,'environment':env.snapshot(),'rng':rng.bit_generator.state}
                if done and (env.status!='timed_out' or args.fresh_forms):
                    queue['terminal'][job_id(job)]={'url':job['url'],'status':env.status,'metrics':env.metrics(),'policy':args.policy,'checkpoint':checkpoint}
                    queue['pending']=None
                world.save(world_path)
                if args.verbose:print(json.dumps(event),flush=True)
                if args.delay:time.sleep(args.delay)
                if done:
                    print(json.dumps({'url':job['url'],'status':env.status,**env.metrics()}),flush=True)
                    scoreboard.append(env.metrics())
                    break
        if args.fresh_forms and scoreboard:
            print(f"SCOREBOARD policy={args.policy} seed={args.seed}: completed {sum(m['completed'] for m in scoreboard)}/{len(scoreboard)} "
                  f"applications, mean field accuracy {np.mean([m['field_accuracy'] for m in scoreboard]):.0%}, "
                  f"mean actions {np.mean([m['actions'] for m in scoreboard]):.0f}",flush=True)
        if mirror and args.headed:
            print('Run finished. Browser remains open until you close it or press Ctrl+C.',flush=True)
            try:
                while not mirror.page.is_closed():mirror.page.wait_for_timeout(1000)
            except KeyboardInterrupt:
                pass


if __name__=='__main__':
    main()
