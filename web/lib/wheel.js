// The GTA V character wheel: hold Tab, point at a fly (mouse or arrow keys), let go to switch to it.
// Slots sit on a ring, a portrait and the fly's name in each. Clicking a slot also switches.
export class Wheel {
  constructor({ onPick, onOpen }) {
    this.el = document.getElementById("wheel");
    this.ring = this.el.querySelector(".ring");
    this.onPick = onPick;
    this.onOpen = onOpen;
    this.items = [];
    this.sel = 0;
    this.open = false;
    addEventListener("keydown", (e) => {
      if (e.key === "Tab") { e.preventDefault(); if (!e.repeat) this.show(); }
      else if (this.open && (e.key === "ArrowRight" || e.key === "ArrowDown")) { e.preventDefault(); this.select(this.sel + 1); }
      else if (this.open && (e.key === "ArrowLeft" || e.key === "ArrowUp")) { e.preventDefault(); this.select(this.sel - 1); }
    });
    addEventListener("keyup", (e) => { if (e.key === "Tab") { e.preventDefault(); this.hide(true); } });
    addEventListener("blur", () => this.hide(false));
    addEventListener("pointermove", (e) => {
      if (!this.open || this.items.length < 2) return;
      const dx = e.clientX - innerWidth / 2, dy = e.clientY - innerHeight / 2;
      if (Math.hypot(dx, dy) < 60) return;                                     // dead zone in the middle
      const a = (Math.atan2(dx, -dy) + Math.PI * 2) % (Math.PI * 2);           // 0 at the top, clockwise
      this.select(Math.round(a / ((Math.PI * 2) / this.items.length)));
    });
    this.ring.addEventListener("click", (e) => {
      const s = e.target.closest(".slot");
      if (s) { this.select(+s.dataset.i); this.hide(true); }
    });
  }

  // items: [{name, title, portrait, untrained}], current: index of the fly on screen
  set(items, current) {
    this.items = items;
    this.ring.innerHTML = items.map((it, i) =>
      `<div class="slot" data-i="${i}"><div class="face"><img alt=""></div><div class="name">${esc(it.title)}</div></div>`).join("");
    this.current = current;
    this.layout();
  }

  layout() {
    const n = this.items.length, R = Math.min(innerWidth, innerHeight) * (n > 6 ? 0.34 : 0.27);
    [...this.ring.children].forEach((s, i) => {
      const a = (i / n) * Math.PI * 2;
      s.style.transform = `translate(${Math.sin(a) * R}px, ${-Math.cos(a) * R}px)`;
    });
  }

  portrait(i, url) {
    const img = this.ring.children[i]?.querySelector("img");
    if (img && url && img.getAttribute("src") !== url) img.src = url;
  }

  select(i) {
    const n = this.items.length;
    if (!n) return;
    this.sel = ((i % n) + n) % n;
    [...this.ring.children].forEach((s, j) => s.classList.toggle("sel", j === this.sel));
  }

  show() {
    if (this.open || !this.items.length) return;
    this.open = true;
    this.layout();
    this.select(this.current ?? 0);
    this.el.classList.add("open");
    this.onOpen?.(true);
  }

  hide(pick) {
    if (!this.open) return;
    this.open = false;
    this.el.classList.remove("open");
    this.onOpen?.(false);
    if (pick && this.sel !== this.current) this.onPick(this.sel);
  }
}

const esc = (s) => String(s).replace(/[&<>"]/g, (c) => ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;" })[c]);
