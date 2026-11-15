// Runs with real Three.js objects and the real shared Fly/model/helpers.
// Only the clock and canvas drawing are substituted; this is NOT a WebGL test.
// Export allows the identical suite to run in a browser or Node vm module runner.
export async function runTests({ THREE, Fly, box, cylinder, stationSource, logicSource, fixtures }) {
  const results = [];
  const check = (value, name) => { if (!value) throw new Error(name); results.push(name); };
  const equal = (a, b, name) => check(JSON.stringify(a) === JSON.stringify(b), name);
  const logic = new Function(logicSource.replace(/export /g, '') + '\nreturn {ACTIONS, parseCaption, Playback, SLOT_COUNT, PITCH};')();
  const { parseCaption, Playback, PITCH, SLOT_COUNT, ACTIONS } = logic;
  let elapsed = 0, samples = [];
  const tween = async (ms, fn) => { const end = elapsed + ms; await Promise.resolve(); fn(0); fn(.5); await Promise.resolve(); fn(1); elapsed = Math.max(elapsed, end); samples.push(ms); };
  const factory = () => new Function('tween', 'box', 'cylinder', 'parseCaption', 'Playback', 'SLOT_COUNT', 'PITCH',
    stationSource.replace(/^import .*;\n/gm, '').replace('export default', 'return'))(
      tween, box, cylinder, parseCaption, Playback, SLOT_COUNT, PITCH);
  const group = new THREE.Group(), fly = new Fly(); group.add(fly.root);
  // Actual shared Fly method geometry/attachment is retained. Clock-controlled
  // walk/buzz avoid wall-clock delay while exercising station await order.
  fly.walkTo = async (target, ms) => {
    const from = fly.root.position.clone(); fly.face(target);
    const to = new THREE.Vector3(target.x, from.y, target.z);
    await tween(ms, t => fly.root.position.lerpVectors(from, to, t));
  };
  fly.buzz = async ms => tween(ms, () => {});
  const station = factory(); station.build({ THREE, group, fly });
  check(station.deck.length === 10 && new Set(station.deck).size === 10, 'ten unique physical deck tiles');
  equal(station.deck.map(t => t.userData.digit), [0,1,2,3,4,5,6,7,8,9], 'deck digit identities');
  for (const action of ACTIONS) {
    const a = +action.at(-1);
    const caption = `π1 1:2 | ${a} ✗ | P0:_ | S0 W1 R-0.20 | live`;
    if (a < 5) check(parseCaption(action, caption).digit === a, `offset ${a} supported`);
    else {
      let rejected = false; try { parseCaption(action, caption); } catch { rejected = true; }
      check(rejected, `masked output ${a} rejected`);
    }
  }
  let rejected = false; try { parseCaption('choose_0', 'not an event'); } catch { rejected = true; }
  check(rejected, 'malformed event fails explicitly');
  let lastEpisode = 0, previousIndex = 0;
  for (const f of fixtures) {
    const event = parseCaption(f.action, f.caption);
    const oldDeck = station.deck.slice(), oldSlots = station.slots.slice();
    const generation = station.serial, replenishments = station.replenishments;
    elapsed = 0; samples = [];
    const promise = station.play(f.action, f.caption, { fly });
    let resolved = false; promise.then(() => { resolved = true; });
    check(!resolved, `transition ${event.episode}:${event.transition} asynchronous`);
    await promise;
    check(resolved, 'play resolves only after complete animation');
    station.assertIdle();
    if (event.kind !== 'group') {
      // Virtual clock measures the complete awaited critical path.
      check(elapsed < 4000, `full ${event.kind} path below host timeout budget (${elapsed}ms virtual)`);
      check(tileCount(group, THREE) === 10 + Math.min(6, event.index), 'no unexpected scene tile duplicates');
      check(fly.body.children.every(o => o.userData.digit === undefined), 'no carried tile after resolution');
      check(fly.root.position.distanceTo(new THREE.Vector3(0,.2,1.65)) < 1e-6, 'fly home before resolution');
      if (event.kind === 'incorrect') {
        check(station.deck[event.digit] === oldDeck[event.digit], 'rejection returns identical physical tile');
        check(station.serial === generation, 'rejection constructs zero tiles');
        check(station.replenishments === replenishments, 'rejection replenishes zero tiles');
        check(station.slots.every((s, i) => s === oldSlots[i]), 'rejection does not shift strip');
      } else {
        check(station.serial === generation + 1, 'acceptance creates exactly one tile');
        check(station.replenishments === replenishments + 1, 'acceptance replenishes exactly once');
        check(station.deck[event.digit] !== oldDeck[event.digit], 'accepted physical tile leaves deck');
        check(station.slots.slice(0,6).every((s, i) => s === oldSlots[i+1]), 'acceptance shifts existing slots exactly once left');
        check(!oldSlots.includes(station.slots.at(-1)), 'exactly one new destination slot');
        check(station.slots.at(-2).children.includes(oldDeck[event.digit]), 'accepted tile occupies previous destination');
        check(event.index === previousIndex + 1, 'accepted ordering follows authoritative events');
        if (event.index > 6) check(oldSlots[0].parent === null, 'offscreen slot disposed');
      }
      previousIndex = event.index;
    } else {
      check(station.serial === generation, 'group selection never moves or replenishes tile');
      if (event.episode !== lastEpisode) previousIndex = 0;
    }
    lastEpisode = event.episode;
    const serial = station.serial;
    await station.play(f.action, f.caption, { fly });
    check(station.serial === serial, 'duplicate callback ignored without side effects');
  }
  const q = new Playback();
  let release, running = 0, maximum = 0, order = [];
  const gate = new Promise(r => { release = r; });
  const animate = async e => { maximum = Math.max(maximum, ++running); order.push(e.transition); if (e.transition === 1) await gate; running--; };
  const p1 = q.enqueue({episode: 1, transition: 1}, animate);
  const p2 = q.enqueue({episode: 1, transition: 2}, animate);
  let secondDone = false; p2.then(() => { secondDone = true; });
  await Promise.resolve(); await Promise.resolve();
  check(!secondDone, 'overlapping call waits for first complete animation');
  release(); await Promise.all([p1,p2]);
  equal(order, [1,2], 'queued playback preserves action ordering');
  check(maximum === 1 && q.phase === 'idle', 'no overlapping animation state');
  await q.enqueue({episode: 0, transition: 20}, animate);
  equal(order, [1,2], 'stale episode callback ignored');
  let gapFailed = false;
  try { await q.enqueue({episode: 1, transition: 4}, animate); } catch { gapFailed = true; }
  check(gapFailed, 'missing transition fails closed');
  return { assertions: results.length, results };
}

function tileCount(group) {
  let count = 0; group.traverse(o => { if (o.userData.digit !== undefined && o.userData.id !== undefined) count++; });
  return count;
}
