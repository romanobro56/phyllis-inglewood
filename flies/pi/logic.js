// Feature-local protocol and serialization. No pi reference or grading in the renderer.
export const ACTIONS = Array.from({ length: 8 }, (_, i) => `choose_${i}`);
export const SLOT_COUNT = 7;
export const PITCH = 0.60;

export function parseCaption(action, caption) {
  if (!ACTIONS.includes(action)) throw new Error(`Unknown Pi Fly action: ${action}`);
  const m = /^π1 (\d+):(\d+) \| (group [01]|[0-9] [✓✗]) \| P(\d+):([0-9]{1,6}|_) \| S(\d+) W(\d+) R(-?\d+\.\d{2}) \| (live|complete|limit)$/.exec(caption);
  if (!m) throw new Error('Invalid Pi Fly v1 caption');
  const e = { episode: +m[1], transition: +m[2], index: +m[4], suffix: m[5] === '_' ? '' : m[5],
    streak: +m[6], wrong: +m[7], reward: +m[8], status: m[9] };
  e.kind = m[3].startsWith('group') ? 'group' : m[3].endsWith('✓') ? 'correct' : 'incorrect';
  e.group = e.kind === 'group' ? +m[3].at(-1) : null;
  e.digit = e.kind === 'group' ? null : +m[3][0];
  const a = ACTIONS.indexOf(action);
  if (e.episode < 1 || e.transition < 1 || (e.kind === 'group' ? a !== e.group : a !== e.digit % 5)
      || e.suffix.length !== Math.min(6, e.index)) throw new Error('Inconsistent Pi Fly event');
  return e;
}

export class Playback {
  constructor() { this.tail = Promise.resolve(); this.episode = 0; this.transition = 0; this.phase = 'idle'; }
  enqueue(event, animate) {
    // Every caller awaits its entire queued animation, including duplicate callbacks.
    const work = this.tail.then(async () => {
      if (event.episode < this.episode || (event.episode === this.episode && event.transition <= this.transition)) return;
      const fresh = event.episode > this.episode;
      if ((!fresh && event.transition !== this.transition + 1) || (fresh && event.transition !== 1))
        throw new Error('Pi Fly transition gap; reconnect to reset the station');
      this.phase = 'playing';
      try {
        await animate(event, fresh);
        this.episode = event.episode;
        this.transition = event.transition;
      } finally { this.phase = 'idle'; }
    });
    // A failed animation cannot be silently retried in a partially changed scene.
    this.tail = work;
    return work;
  }
}
