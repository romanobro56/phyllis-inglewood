"""Bundle the folder into one self-contained file, standalone.html, that opens without a server."""
import json, os
D = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..')
rd = lambda p: open(os.path.join(D, p), encoding='utf-8').read()
html = rd('index.html')
engine = '\n'.join(rd('engine/' + f) for f in ('flysim.js', 'songear.js', 'bandear.js', 'worker.js'))
inline = json.dumps({'drawing': json.loads(rd('data/drawing.json')), 'network': rd('data/network.json'), 'engine': engine}).replace('</', '<\\/')
html = html.replace('<link rel="stylesheet" href="ui/style.css">', '<style>\n' + rd('ui/style.css') + '</style>')
app = rd('ui/app.js'); assert '</script' not in app
html = html.replace('<script src="ui/app.js"></script>', '<script>window.FLYBAND_INLINE = ' + inline + ';</script>\n<script>\n' + app + '</script>')
open(os.path.join(D, 'standalone.html'), 'w', encoding='utf-8').write(html)
print('standalone.html', round(len(html) / 1e6, 2), 'MB')
