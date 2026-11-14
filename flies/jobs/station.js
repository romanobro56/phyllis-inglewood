import { box, tween, wait } from "/web/lib/fly.js";

export default {
  title: "Jobs fly — simulation",
  blurb: "Elton learns to type, select dropdowns, and complete accurate applications.",

  build({ THREE, group, fly }) {
    group.add(box(3.4, 0.12, 2.5, 0x28364b, [0, 0.06, 0]));
    this.cards = [];
    for (let i = 0; i < 5; i++) {
      const x = (i % 3 - 1) * 0.95;
      const z = Math.floor(i / 3) * 0.9 - 0.55;
      const card = box(0.7, 0.08, 0.65, 0xd5e5ed, [x, 0.17, z]);
      group.add(card);
      for (let line = 0; line < 3; line++) {
        group.add(box(0.46, 0.015, 0.035, 0x528aa1, [x, 0.22, z - 0.16 + line * 0.14]));
      }
      this.cards.push(card);
    }
    this.review = box(0.55, 0.12, 0.4, 0xf0b44d, [1.25, 0.2, 1]);
    group.add(this.review);
    fly.root.position.set(0, 0.13, 1.1);
  },

  async play(action, caption, { fly }) {
    if (/^slot_[0-4]$/.test(action)) {
      const card = this.cards[Number(action.slice(5))];
      await fly.walkTo({ x: card.position.x, z: card.position.z }, 650);
      const original = card.material.color.getHex();
      card.material.color.setHex(/incorrect|invalid|rejected/.test(caption) ? 0xdf6868 : /correct/.test(caption) ? 0x68c9a1 : 0xf0b44d);
      await tween(400, t => { card.position.y = 0.17 + Math.sin(t * Math.PI) * 0.2; });
      card.material.color.setHex(original);
    } else if (action === "continue_or_submit") {
      await fly.walkTo({ x: -1.35, z: 1 }, 650);
      await fly.buzz(400);
    } else if (action === "toggle_input_mode") {
      await fly.walkTo({ x: 1.25, z: 1 }, 650);
      await fly.hop(0.3, 400);
    } else if (action === "next_field") {
      await fly.walkTo({ x: 0, z: 0.8 }, 450);
      await fly.hop(0.12, 250);
    }
    await wait(150);
  },
};
