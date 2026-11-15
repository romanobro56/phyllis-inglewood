// node --experimental-vm-modules flies/pi/tests/run_js.mjs /tmp/pi-three.module.mjs
// Dependency is the exact Three 0.170.0 module already used by web/index.html.
import fs from 'node:fs/promises';
import vm from 'node:vm';
import path from 'node:path';
import { fileURLToPath } from 'node:url';

export async function run(fs, vm, root, threePath) {
  const canvas = () => ({ width: 0, height: 0,
    getContext: () => ({ clearRect() {}, fillText() {}, font: '', textAlign: '', textBaseline: '', fillStyle: '' }) });
  const context = vm.createContext({ console, document: { createElement: canvas }, setTimeout, clearTimeout,
    performance: { now: () => 0 } });
  const three = new vm.SourceTextModule(await fs.readFile(threePath, 'utf8'), { context });
  await three.link(() => {}); await three.evaluate();
  const fly = new vm.SourceTextModule(await fs.readFile(`${root}/web/lib/fly.js`, 'utf8'), { context });
  await fly.link(() => three); await fly.evaluate();
  const tests = new vm.SourceTextModule(await fs.readFile(`${root}/flies/pi/tests/station_tests.js`, 'utf8'), { context });
  await tests.link(() => {}); await tests.evaluate();
  const result = await tests.namespace.runTests({ THREE: three.namespace,
    Fly: fly.namespace.Fly, box: fly.namespace.box, cylinder: fly.namespace.cylinder,
    stationSource: await fs.readFile(`${root}/flies/pi/station.js`, 'utf8'),
    logicSource: await fs.readFile(`${root}/flies/pi/logic.js`, 'utf8'),
    fixtures: JSON.parse(await fs.readFile(`${root}/flies/pi/tests/fixtures.json`, 'utf8')) });
  return result;
}

if (process.argv[1] === fileURLToPath(import.meta.url)) {
  const root = path.resolve(path.dirname(fileURLToPath(import.meta.url)), '../../..');
  const result = await run(fs, vm, root, process.argv[2]);
  console.log(`PASS: ${result.assertions} station assertions (real Three.js/Fly, virtual clock, canvas stub)`);
}
