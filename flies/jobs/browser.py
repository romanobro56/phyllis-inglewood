"""Non-submitting DOM adapter. It never navigates to job URLs or fills defaults."""
from html import escape


def html(env):
    view = env.public_view()
    controls = []
    for field in env.fields:
        identity = escape(field['id'], quote=True)
        label = escape(field['label'])
        required = ' required' if field['required'] else ''
        # Saved values reflect earlier brain actions; initial forms are blank.
        value = escape(env.values.get(field['id'], ''), quote=True)
        if field['kind'] == 'select':
            options = '<option value="">Choose an option</option>' + ''.join(
                f'<option value="{escape(o, quote=True)}"' + (' selected' if env.values.get(field['id']) == o else '') + f'>{escape(o)}</option>'
                for o in field['options'])
            control = f'<select id="{identity}" name="{identity}"{required}>{options}</select>'
        elif field['kind'] == 'textarea':
            control = f'<textarea id="{identity}" name="{identity}"{required}>{value}</textarea>'
        else:
            control = f'<input id="{identity}" name="{identity}" value="{value}"{required}>'
        hidden = ' hidden' if field['page'] != env.page else ''
        controls.append(f'<section data-page="{field["page"]}"{hidden}><label for="{identity}">{label}</label>{control}</section>')
    return '''<!doctype html><meta charset="utf-8"><title>Non-submitting application test</title>
    <style>body{font:18px sans-serif;margin:2rem}section{margin:1rem}label{display:block}input,textarea,select{width:35rem;max-width:90%}[hidden]{display:none}</style>
    <h1>Local application test — no employer submissions</h1><form>''' + ''.join(controls) + '''
    <button type="button" id="advance">Continue / test submission</button></form>
    <pre id="feedback"></pre><script>
    document.querySelector('form').addEventListener('submit',e=>e.preventDefault());
    document.querySelector('#advance').onclick=()=>{document.body.dataset.attempted='true'};
    </script>'''


class BrowserMirror:
    def __init__(self, page):
        self.page = page
        self.page.route('http://**/*', lambda route: route.abort())
        self.page.route('https://**/*', lambda route: route.abort())

    def open(self, env):
        self.page.set_content(html(env))
        layout = env.layout['layout']
        if layout == 'grid':
            self.page.add_style_tag(content='form{display:grid;grid-template-columns:repeat(2,minmax(0,1fr))}input,textarea,select{width:95%}')
        elif layout == 'cards':
            self.page.add_style_tag(content='section{border:1px solid #999;border-radius:12px;padding:1rem;background:#f4f6fa}')

    def observation(self, env):
        # Read labels, options, values, required flags and input kinds from DOM.
        # Resume snippets and focus/mode are public controller state, not answers.
        view = env.public_view()
        fields = self.page.evaluate("""() => [...document.querySelectorAll('section')].map(s => {
          const c = s.querySelector('input,textarea,select');
          return {id:c.id, label:s.querySelector('label').textContent,
            kind:c.tagName.toLowerCase()==='select'?'select':c.tagName.toLowerCase()==='textarea'?'textarea':'text',
            required:c.required, value:c.value,
            options:c.tagName==='SELECT'?[...c.options].filter(o=>o.value).map(o=>o.value):[]};
        })""")
        current = next(f for f in fields if f['id'] == view['current']['id'])
        view['current'] = current
        view['values'] = {f['id']: f['value'] for f in fields}
        if current['kind'] == 'select':
            view['candidates'] = [{'label': v, 'value': v} for v in current['options']]
        return env.encode_view(view)

    def execute(self, env, action):
        if action < 5 and env.mask()[action]:
            field = env.current
            mode = 'select' if field['kind'] == 'select' else 'type'
            if env.input_mode != mode:
                return
            value = env.candidates()[action]['value']
            control = self.page.locator('#' + field['id'])
            if mode == 'select':
                control.select_option(value=value)
            else:
                control.fill('')
                control.press_sequentially(value)
        elif action == 6:
            self.page.locator('#advance').click()

    def check(self, env):
        actual = self.page.evaluate("() => Object.fromEntries(new FormData(document.querySelector('form')))")
        if actual != {f['id']: env.values.get(f['id'], '') for f in env.fields}:
            raise RuntimeError('DOM values diverged from selected brain actions')
        # Only visibility changes after environment transitions, never answer values.
        self.page.evaluate("page => document.querySelectorAll('[data-page]').forEach(e=>e.hidden=Number(e.dataset.page)!==page)", env.page)
        if env.last_caption == 'validation rejected':
            self.page.locator('#feedback').evaluate('(node,text)=>node.textContent=text', env.layout['error_message'])
        else:
            self.page.locator('#feedback').evaluate('(node)=>node.textContent=""')
        return actual
