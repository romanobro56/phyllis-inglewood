"""Visible local Workday-inspired DOM adapter. No real employer navigation.

DOM operations execute exactly one selected Action. Browser observations are
read back from accessible controls, the profile picker and public action palette.
No expected answer table or private simulation state is embedded in HTML.
"""
import copy
import json
from html import escape
from flies.jobs.contract import encode

STYLE = '''body{margin:0;background:#edf1f5;color:#1e293b;font:16px system-ui}header{background:#123b64;color:white;padding:18px 30px}
main{display:grid;grid-template-columns:minmax(400px,2fr) minmax(280px,1fr);gap:20px;padding:24px;max-width:1450px;margin:auto}
article,aside{background:white;border-radius:10px;padding:24px;box-shadow:0 2px 8px #152a4515}h1{font-size:25px}label{display:block;font-weight:600;margin:14px 0 5px}
input,textarea,select{font:inherit;box-sizing:border-box;width:100%;padding:10px;border:1px solid #93a6b8;border-radius:5px}textarea{height:85px}
button{cursor:pointer;border:1px solid #2668a0;border-radius:5px;background:#edf6ff;color:#103e65;padding:10px;margin:4px;font:inherit}
button:hover{background:#cce7ff}.error{background:#fff0ec;color:#8b2516;padding:8px}.badge{background:#fff3be;color:#5b4100;padding:5px;border-radius:4px}
small{color:#536779}.grid #fields{display:grid;grid-template-columns:1fr 1fr;gap:15px}.cards .field{border:1px solid #bac8d5;padding:12px;margin:8px}
#palette button{display:block;width:100%;text-align:left}pre{white-space:pre-wrap;overflow-wrap:anywhere;max-height:260px;overflow:auto;font-size:12px}
[aria-current=true]{outline:3px solid #e4ae31}iframe{width:100%;height:340px;border:0}#station{position:sticky;bottom:0;background:#111c29;border-radius:8px}.summary{font-size:13px;border-bottom:1px solid #eee;padding:5px}
'''


def html(view):
    controls = []
    for e in view['elements']:
        ident, label = escape(e['id'], quote=True), escape(e['label'])
        if e['kind'] == 'button':
            continue
        req = ' required' if e['required'] else ''
        value = '' if e['kind'] in ('password','file') else escape(e['value'], quote=True)
        attrs = f'id="{ident}" data-kind="{e["kind"]}" aria-label="{label}" data-constraint="{escape(e["constraint"],quote=True)}"{req}'
        if e['kind']=='radio':
            options=''.join(f'<label><input type="radio" name="{ident}" value="{escape(o,quote=True)}"'+(' checked' if o==e['value'] else '')+req+'>'+escape(o)+'</label>' for o in e['options'])
            control=f'<fieldset {attrs}>{options}</fieldset>'
        elif e['kind'] in ('select', 'combobox'):
            options = '<option value="">Select an option</option>' + ''.join(f'<option value="{escape(o,quote=True)}"'+(' selected' if o == e['value'] else '')+'>'+escape(o)+'</option>' for o in e['options'])
            control = f'<select {attrs}>{options}</select>'
            if e['kind'] == 'combobox':
                control = f'<input aria-label="Search {label}" data-search-for="{ident}" placeholder="Search options">' + control
        elif e['kind'] == 'textarea':
            control = f'<textarea {attrs}>{value}</textarea>'
        else:
            kind = e['kind'] if e['kind'] in ('email','password','date','file','tel') else 'text'
            control = f'<input type="{kind}" {attrs}' + (f' value="{value}"' if kind != 'file' else '') + '>'
        if e['kind']=='file':control+=f'<small id="file-status-{ident}">{escape(e["value"])}</small>'
        controls.append(f'<div class="field"><label for="{ident}">{label}'+(' *' if e['required'] else '')+f'</label>{control}<small>{escape(e["constraint"])}</small></div>')
    buttons = ''.join(f'<button type="button" id="{escape(e["id"],quote=True)}">{escape(e["label"])}</button>' for e in view['elements'] if e['kind']=='button')
    summary = ''.join(f'<div class="summary">{escape(f["label"])}: {escape(f["value"] or ("Unavailable" if f.get("unavailable") else "Empty"))}</div>' for f in view['summary'])
    return '<!doctype html><meta charset="utf-8"><title>Elton — local applicant simulation</title><style>'+STYLE+'</style>'+\
        '<header><strong>ELTON INGLEWOOD · CANDIDATE HOME</strong> <span class="badge">LOCAL SIMULATION — NO REAL SUBMISSIONS</span></header>'+\
        f'<main><article class="{view["layout"]}"><small id="tenant">{escape(view["tenant"])}</small><h1>{escape(view["stage"].replace("_"," ").title())}</h1>'+\
        '<nav>'+escape(' → '.join(view['steps']))+'</nav><div id="messages"></div><form><div id="fields">'+''.join(controls)+'</div><div id="navigation">'+buttons+\
        '</div></form>'+summary+'<pre id="event"></pre></article><aside><h2>Elton’s action choices</h2><p>These choices are selected by the neural policy.</p><div id="palette"></div><h3>Profile / resume facts</h3><pre id="profile"></pre><div id="station"></div></aside></main>'+\
        '<script>document.querySelector("form").addEventListener("submit",e=>e.preventDefault());document.querySelectorAll("#navigation button").forEach(b=>b.onclick=()=>document.body.dataset.clicked=b.id);</script>'


class BrowserMirror:
    def __init__(self, page):
        self.page = page
        self.signature = None
        self.sequence = 0
        # All employer/external requests blocked, including navigations caused by HTML.
        self.station_origin = None
        self.page.route('**/*', lambda route: route.continue_() if self.station_origin and route.request.url.startswith(self.station_origin+'/') else route.abort())

    def open(self, env):
        self.signature = None
        self.sync(env)

    def sync(self, env, event=None, policy='unknown', checkpoint='none'):
        view = env.public_view()
        signature = (view['stage'], tuple((e['id'],e['kind']) for e in view['elements']))
        if signature != self.signature:
            self.page.set_content(html(view))
            self.signature = signature
            if self.station_origin:
                self.page.evaluate('''origin => {
                  window.stationReady=false;window.stationAck=0;
                  const frame=document.createElement('iframe');frame.id='execution-station';
                  frame.src=origin+'/flies/jobs/station_view.html';document.querySelector('#station').append(frame);
                  window.addEventListener('message',e=>{
                    if(e.source!==frame.contentWindow)return;
                    if(e.data.type==='station_ready')window.stationReady=true;
                    if(e.data.type==='station_failed')window.stationReady='failed';
                    if(e.data.type==='station_ack')window.stationAck=e.data.sequence;
                  });
                }''',self.station_origin)
                self.page.wait_for_function('window.stationReady',timeout=15000)
                if self.page.evaluate('window.stationReady')=='failed':
                    raise RuntimeError('3D station failed to initialize')
            # Restore ONLY earlier selected values on a revisited page, including
            # opaque saved credentials. No answer lookup occurs here.
            for e in view['elements']:
                if (e['kind'] == 'password' or e.get('protected')) and e['value']:
                    try:self.page.locator('#'+e['id']).fill(env._password(e['id']))
                    except Exception:raise RuntimeError('Protected credential restoration failed') from None
        for e in view['elements']:
            # Mirror the simulator's outcome of "information unavailable": the field is shown empty.
            if e.get('unavailable') and not e['value'] and e['kind'] not in ('button', 'file'):
                control = self.page.locator('#'+e['id'])
                if control.count():
                    if control.get_attribute('data-kind')=='radio':control.evaluate('(n)=>n.querySelectorAll("input").forEach(i=>i.checked=false)')
                    elif control.evaluate('(e)=>e.tagName')=='SELECT':control.select_option(value='')
                    else:control.fill('')
        public = copy.deepcopy(view)
        self.page.evaluate('''v => {
          window.publicMeta=v;
          const palette=document.querySelector('#palette');palette.replaceChildren();
          v.elements.filter(e=>e.kind==='file').forEach(e=>{const n=document.getElementById('file-status-'+e.id);if(n)n.textContent=e.value;});
          v.actions.forEach((a,i)=>{if(a){const b=document.createElement('button');b.textContent=a.label;
            b.dataset.slot=i;b.dataset.action=JSON.stringify(a);palette.append(b);}});
          const count=document.querySelector('#tenant');
          count.textContent=v.tenant+((v.stage==='experience'||v.stage==='education')?' · entries: '+v.section_entry_count[v.stage]:'');
          const m=document.querySelector('#messages');m.replaceChildren();
          v.messages.forEach(text=>{const d=document.createElement('p');d.className='error';d.textContent=text;m.append(d);});
          document.querySelector('#profile').textContent=JSON.stringify({facts:v.profile,provenance:v.profile_provenance},null,2);
          document.querySelectorAll('[aria-current]').forEach(n=>n.removeAttribute('aria-current'));
          if(v.focus) document.getElementById(v.focus)?.setAttribute('aria-current','true');
        }''', public)
        if event:
            self.sequence += 1
            payload = {'sequence': self.sequence, 'policy': policy, 'checkpoint': str(checkpoint), **event}
            self.page.locator('#event').evaluate('(e,v)=>e.textContent=JSON.stringify(v,null,2)',payload)
            self.page.evaluate('(v)=>window.lastExecutionEvent=v',payload)
            if self.station_origin:
                self.page.evaluate('(event)=>document.querySelector("#execution-station").contentWindow.postMessage({type:"execution",event},"*")',payload)
                self.page.wait_for_function('(sequence)=>window.stationAck>=sequence',arg=self.sequence,timeout=10000)

    def observation_view(self):
        return self.page.evaluate('''() => {
          const v=structuredClone(window.publicMeta);
          v.elements=v.elements.map(e=>{
            const n=document.getElementById(e.id);
            if(e.kind==='button') return {...e,label:n.textContent};
            const p=n.type==='password'||e.protected, radio=e.kind==='radio';
            return {...e,label:n.getAttribute('aria-label'),kind:n.dataset.kind,required:radio?n.querySelector('input').required:n.required,
              constraint:n.dataset.constraint,value:radio?(n.querySelector('input:checked')?.value||''):p?(n.value?'[protected]':''):n.type==='file'?(n.files[0]?.name||document.getElementById('file-status-'+e.id)?.textContent||''):n.value,
              options:radio?[...n.querySelectorAll('input')].map(o=>o.value):n.tagName==='SELECT'?[...n.options].filter(o=>o.value&&!o.hidden).map(o=>o.value):[]};
          });
          const choices=Array(8).fill(null);
          document.querySelectorAll('#palette button').forEach(b=>{
            const a=JSON.parse(b.dataset.action);a.label=b.textContent;choices[Number(b.dataset.slot)]=a;
          });v.actions=choices;
          v.messages=[...document.querySelectorAll('#messages p')].map(n=>n.textContent);
          return v;
        }''')

    def observation(self, env=None):
        return encode(self.observation_view())

    def execute(self, env, action_index):
        action = env.available_actions()[action_index]
        if action.kind == 'focus':
            self.page.locator('#'+action.target).focus()
        elif action.kind in ('enter', 'choose', 'clear'):
            control = self.page.locator('#'+action.target)
            if action.kind == 'choose':
                if control.get_attribute('data-kind')=='radio':
                    control.get_by_role('radio',name=action.value,exact=True).check()
                else:control.select_option(value=action.value)
            else:
                value = '' if action.kind == 'clear' else env.vault.resolve(env.tenant, 'password_weak' if action.ref=='password_C' else action.ref) if action.secret else action.value
                # HTML date controls reject arbitrary text; leave empty, matching
                # the simulator's explicit date constraint handling.
                if control.get_attribute('type') == 'date':
                    import re
                    value = value if re.fullmatch(r'\d{4}-\d{2}-\d{2}', value) else ''
                if control.get_attribute('data-kind')=='radio':
                    control.evaluate('(n)=>n.querySelectorAll("input").forEach(i=>i.checked=false)')
                elif action.secret:
                    try:control.fill(value)
                    except Exception:raise RuntimeError('Protected credential entry failed') from None
                else:control.fill(value)
        elif action.kind == 'review' and action.target:
            # Whether the field is really unavailable is the simulator's decision (it clears only fields
            # without verified information). Clearing here unconditionally made the DOM diverge whenever
            # review was requested on a field that has an answer; sync() mirrors the actual outcome.
            pass
        elif action.kind == 'upload':
            self.page.locator('#'+action.target).set_input_files(env.documents[action.ref]['path'])
        elif action.kind == 'search':
            self.page.locator('[data-search-for="'+action.target+'"]').fill(action.value)
        elif action.kind == 'click':
            if env.stage=='auth':self.check(env)
            self.page.locator('#'+action.target).click()

    def check(self, env):
        view = self.observation_view()
        actual = {e['id']: e for e in view['elements']}
        for expected in env.elements():
            if expected['kind'] in ('button','file'):
                continue
            if expected['kind']=='password' or expected.get('protected'):
                actual_secret=self.page.locator('#'+expected['id']).input_value()
                if actual_secret!=env._password(expected['id']):
                    raise RuntimeError('Protected DOM credential differs from the selected handle')
            got = actual[expected['id']]['value']
            wanted = '[protected]' if (expected['kind']=='password' or expected.get('protected')) and expected['value'] else expected['value']
            if got != wanted:
                raise RuntimeError('DOM value diverged from selected action: '+expected['id'])
        return view


class SimulatedBrowserEnvironment:
    """ApplicationEnvironment facade: same public slots/encoder as the simulator.

A future WorkdayBrowserEnvironment should implement this contract using its own
DOM discovery, credential resolver and protected non-submitting executor. It must
not import the simulation's oracle or scenario generator.
"""
    def __init__(self, engine, page):
        self.engine=engine
        self.mirror=BrowserMirror(page)
        self.rng=None

    def reset(self,rng):
        self.rng=rng
        self.engine.reset(rng)
        self.mirror.open(self.engine)
        return self.obs()

    def public_view(self):
        return self.mirror.observation_view()

    def obs(self):
        return encode(self.public_view())

    def available_actions(self):
        from flies.jobs.contract import Action
        result=[]
        for a in self.public_view()['actions']:
            if a:
                a=dict(a)
                if a['secret']:a['value']=''
                result.append(Action(**a))
        return result

    def mask(self):
        return [a is not None for a in self.public_view()['actions']]

    def step(self,action,rng):
        self.mirror.execute(self.engine,action)
        _,reward,done,info=self.engine.step(action,rng)
        self.mirror.sync(self.engine,info)
        self.mirror.check(self.engine)
        return self.obs(),reward,done,info

    def execute(self,action):
        index=self.available_actions().index(action)
        return self.step(index,self.rng)[3]
