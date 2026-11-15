# The Fly Band, by Ear

Upload a song. It is split into drums, bass, vocals and keys, and each part is played to one simulated
fruit fly. For each fly you can listen to the part as played, the movement of its antennae, the song its
output neurons would sing, and a recreation of the part rebuilt from the moments its brain found alike.

What is modeled, what is invented, and what was checked is in [WRITEUP.md](WRITEUP.md).

## Run it

**Quickest:** open `standalone.html` in a browser. It is the whole thing in one file and needs no server.

**From the separate files:** browsers will not load the data files or start the background threads from a
page opened straight from disk, so serve the folder:

```
cd fly-band-by-ear
python3 -m http.server 8000
```

then open <http://localhost:8000>. Everything runs in the browser; nothing is uploaded anywhere.

After changing any file, rebuild the single-file version with `python3 tools/build_standalone.py`.

## What is where

```
index.html              the page's markup only
ui/
  style.css             look and layout
  app.js                controls, sound playback, drawing. No modelling
engine/
  flysim.js             the neuron model: steps the network 0.1 ms at a time
  bandear.js            the ear, one brain's run, the wing, the split into parts, the recreation
  songear.js            beat finding and choice of the 30-second excerpt
  worker.js             runs one fly per background thread and talks to the page
data/
  network.json          the 8,161-neuron network: connections, antennal classes by side, output neurons
  drawing.json          where to draw each neuron, and the brain outline
standalone.html         everything bundled into one file
tools/
  build_standalone.py   makes standalone.html
  data-prep/            scripts that built data/ (see below)
  tests/                test songs and the measurements quoted in the writeup
```

The page and the engine only meet in `engine/worker.js`: the page posts a song or a part to a worker, and
the worker posts back sound and spikes. To change the model, edit `engine/`; to change the page, edit `ui/`.

## Where to change things

| To change | Look at |
| --- | --- |
| How loud the parts are played, the antenna's tuning, firing-rate curves | `EAR` in `engine/bandear.js` |
| How output neurons become song | `WING` and `sing()` in `engine/bandear.js` |
| The split into four parts | `TUNE` and `split4()` in `engine/bandear.js` |
| The recreation rule | `brainStates()`, `choose()`, `assemble()` in `engine/bandear.js` |
| Excerpt length | `EXCERPT` in `ui/app.js` |
| Colours, layout, text | `ui/style.css`, `index.html`, `ui/app.js` |

## tools/

`tools/data-prep/` holds the Python scripts that built `data/` from the FlyWire annotations and the
Shiu et al. connectivity tables. They are included to show how the network was made. They carry paths from
the workspace they were written in and need those downloads to rerun, so expect to edit paths first.

`tools/tests/` holds the Node scripts and synthetic test songs behind the numbers in the writeup. The Node
scripts load the engine from this folder but read test audio from `/tmp`, which the Python scripts there
write.

## Data and credits

The wiring comes from the FlyWire connectome and its annotations, and the neuron model and connectivity
tables from Shiu et al. Links are in the Sources section of the writeup. Check their licences before
redistributing `data/network.json`.
