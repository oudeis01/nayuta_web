// Discourse nebula — sibling background geometry to the structural Helix.
// Faithful port of graphics_consumer/src/screens/nebula.{h,cpp}. Where the helix
// is BERT's deterministic 12×12 compute topology, the nebula is the EMERGENT
// semantic graph: a drifting constellation of recently whispered lemmas linked to
// their nearest embedding neighbors. Force-laid-out, slowly reorganizing,
// monochrome wireframe with small word labels — a 2D plane with subtle parallax,
// the visual opposite of the rigid helix ("structure vs. meaning").
//
// Fed resolved words from /bert/whisper, which IS present in the Phase-1
// captures, so the nebula runs on real data.

const MAX_NODES = 44;
const MAX_EDGES = 80;
const TOP_NBRS = 3; // neighbors used per whisper
// The marker/offset/lineWidth/sway literals below were tuned at this font size
// (the --dev small panel, s≈1). They are raw px, so at a fullscreen panel the font
// scales up while the markers stay tiny and the constellation loses visual mass —
// the log then visually dominates (mon D feedback 2026-06-05). Scale them by
// s = fontPx / REF_FONT so the nebula's weight tracks the panel size.
const REF_FONT = 11;
const REST_MIN = 0.35; // model-unit rest length floor
const REST_SCALE = 0.95; // + dist * this
const FLASH_DECAY = 1.6; // birth-flash fade rate

function frand(): number {
  return Math.random();
}
function clampf(x: number, lo: number, hi: number): number {
  return x < lo ? lo : x > hi ? hi : x;
}

interface Node {
  word: string;
  x: number;
  y: number;
  vx: number;
  vy: number;
  z: number; // depth in [0.3,1.0]; nearer = larger/brighter
  flash: number; // 1 at birth/refresh, decays to 0
  age: number; // seconds since last refresh (eviction key)
  bridge: boolean;
  id: number;
}
interface Edge {
  a: number; // node id
  b: number; // node id
  dist: number;
  flash: number;
  age: number;
}

export class Nebula {
  // Live-tunable knobs (match the C++ defaults).
  spring = 2.0; // edge spring stiffness toward rest length
  repulse = 1.7; // pairwise node repulsion (spreads the cloud)
  centerPull = 0.22; // gentle pull toward origin (keeps it bounded)
  damp = 4.0; // velocity damping rate
  swayPx = 18.0; // parallax sway amplitude (screen px)

  private nodes: Node[] = [];
  private edges: Edge[] = [];
  private nextId = 1;
  private t = 0;
  private cxS = 0;
  private cyS = 0;
  private scaleS = 0;

  private findNode(word: string): number {
    for (let i = 0; i < this.nodes.length; i++) if (this.nodes[i].word === word) return i;
    return -1;
  }
  private nodeIndex(id: number): number {
    for (let i = 0; i < this.nodes.length; i++) if (this.nodes[i].id === id) return i;
    return -1;
  }
  private addNode(word: string, bridge: boolean, nx: number, ny: number): number {
    const n: Node = {
      word,
      x: nx + (frand() - 0.5) * 0.4,
      y: ny + (frand() - 0.5) * 0.4,
      vx: 0,
      vy: 0,
      z: 0.35 + 0.6 * frand(),
      flash: 1.0,
      age: 0.0,
      bridge,
      id: this.nextId++,
    };
    this.nodes.push(n);
    return n.id;
  }
  private touchEdge(idA: number, idB: number, dist: number): void {
    for (const e of this.edges)
      if ((e.a === idA && e.b === idB) || (e.a === idB && e.b === idA)) {
        e.dist = dist;
        e.flash = 1.0;
        e.age = 0.0;
        return;
      }
    if (this.edges.length >= MAX_EDGES) {
      // drop the oldest edge
      let mi = 0;
      for (let i = 1; i < this.edges.length; i++) if (this.edges[i].age > this.edges[mi].age) mi = i;
      this.edges.splice(mi, 1);
    }
    this.edges.push({ a: idA, b: idB, dist, flash: 1.0, age: 0.0 });
  }
  private evictIfNeeded(): void {
    while (this.nodes.length > MAX_NODES) {
      // remove the stalest node + its edges
      let mi = 0;
      for (let i = 1; i < this.nodes.length; i++) if (this.nodes[i].age > this.nodes[mi].age) mi = i;
      const dead = this.nodes[mi].id;
      this.nodes.splice(mi, 1);
      this.edges = this.edges.filter((e) => e.a !== dead && e.b !== dead);
    }
  }

  // src word + its nearest neighbors (word, semantic distance), bridge flag.
  addWhisper(src: string, nbrs: [string, number][], isBridge: boolean): void {
    if (!src) return;

    let si = this.findNode(src);
    let srcId: number;
    if (si < 0) {
      // place near the current centroid so it joins the cloud, not the edge
      let cx = 0;
      let cy = 0;
      if (this.nodes.length) {
        for (const n of this.nodes) {
          cx += n.x;
          cy += n.y;
        }
        cx /= this.nodes.length;
        cy /= this.nodes.length;
      }
      srcId = this.addNode(src, isBridge, cx, cy);
      si = this.nodeIndex(srcId);
    } else {
      srcId = this.nodes[si].id;
      this.nodes[si].flash = 1.0;
      this.nodes[si].age = 0.0;
      this.nodes[si].bridge = this.nodes[si].bridge || isBridge;
    }
    const sx = this.nodes[si].x;
    const sy = this.nodes[si].y;

    let used = 0;
    for (const [word, dist] of nbrs) {
      if (used >= TOP_NBRS) break;
      if (!word || word === src) continue;
      const rest = REST_MIN + dist * REST_SCALE;
      const ni = this.findNode(word);
      let nbrId: number;
      if (ni < 0) {
        const ang = frand() * 6.2831853;
        nbrId = this.addNode(word, false, sx + Math.cos(ang) * rest, sy + Math.sin(ang) * rest);
      } else {
        nbrId = this.nodes[ni].id;
        this.nodes[ni].flash = 1.0;
        this.nodes[ni].age = 0.0;
      }
      this.touchEdge(srcId, nbrId, dist);
      used++;
    }

    this.evictIfNeeded();
  }

  update(dt: number): void {
    this.t += dt;
    const N = this.nodes.length;
    if (N === 0) return;

    for (const n of this.nodes) {
      n.age += dt;
      n.flash = clampf(n.flash - dt * FLASH_DECAY, 0.0, 1.0);
    }
    for (const e of this.edges) {
      e.age += dt;
      e.flash = clampf(e.flash - dt * FLASH_DECAY, 0.0, 1.0);
    }

    const fx = new Float32Array(N);
    const fy = new Float32Array(N);

    // Pairwise repulsion (soft, capped).
    for (let i = 0; i < N; i++)
      for (let j = i + 1; j < N; j++) {
        const dx = this.nodes[i].x - this.nodes[j].x;
        const dy = this.nodes[i].y - this.nodes[j].y;
        const d2 = dx * dx + dy * dy + 0.05;
        const inv = this.repulse / d2;
        const fxi = dx * inv;
        const fyi = dy * inv;
        fx[i] += fxi;
        fy[i] += fyi;
        fx[j] -= fxi;
        fy[j] -= fyi;
      }

    // Edge springs toward rest length (rest grows with semantic distance).
    for (const e of this.edges) {
      const ia = this.nodeIndex(e.a);
      const ib = this.nodeIndex(e.b);
      if (ia < 0 || ib < 0) continue;
      const dx = this.nodes[ib].x - this.nodes[ia].x;
      const dy = this.nodes[ib].y - this.nodes[ia].y;
      const d = Math.sqrt(dx * dx + dy * dy) + 1e-4;
      const rest = REST_MIN + e.dist * REST_SCALE;
      const f = (this.spring * (d - rest)) / d;
      fx[ia] += dx * f;
      fy[ia] += dy * f;
      fx[ib] -= dx * f;
      fy[ib] -= dy * f;
    }

    // Centering pull + integrate with damping.
    const dmp = Math.max(0.0, 1.0 - this.damp * dt);
    for (let i = 0; i < N; i++) {
      fx[i] -= this.nodes[i].x * this.centerPull;
      fy[i] -= this.nodes[i].y * this.centerPull;
      this.nodes[i].vx = (this.nodes[i].vx + fx[i] * dt) * dmp;
      this.nodes[i].vy = (this.nodes[i].vy + fy[i] * dt) * dmp;
      this.nodes[i].x += this.nodes[i].vx * dt;
      this.nodes[i].y += this.nodes[i].vy * dt;
    }
  }

  // Draw into [ox, oy, w, h] as an isolated canvas (recentered on centroid, fit
  // by max radial extent), so the constellation floats centered with no drift.
  draw(
    ctx: CanvasRenderingContext2D,
    ox: number,
    oy: number,
    w: number,
    h: number,
    fontPx: number,
    baseAlpha = 1.0,
  ): void {
    if (this.nodes.length === 0) return;

    // Panel scale: the raw-px literals were tuned at REF_FONT; track the panel size.
    const s = fontPx / REF_FONT;

    let cx = 0;
    let cy = 0;
    for (const n of this.nodes) {
      cx += n.x;
      cy += n.y;
    }
    cx /= this.nodes.length;
    cy /= this.nodes.length;
    let halfr = 1e-3;
    for (const n of this.nodes) {
      halfr = Math.max(halfr, Math.abs(n.x - cx));
      halfr = Math.max(halfr, Math.abs(n.y - cy));
    }
    const fill = 0.88;
    const target = (fill * Math.min(w * 0.5, h * 0.5)) / halfr;
    this.cxS = this.scaleS <= 0 ? cx : this.cxS + (cx - this.cxS) * 0.06;
    this.cyS = this.scaleS <= 0 ? cy : this.cyS + (cy - this.cyS) * 0.06;
    this.scaleS = this.scaleS <= 0 ? target : this.scaleS + (target - this.scaleS) * 0.06;

    const cx0 = ox + w * 0.5;
    const cy0 = oy + h * 0.5;
    // Parallax sway: nearer nodes (larger z) shift more — subtle 2.5D depth.
    const swx = Math.sin(this.t * 0.13) * this.swayPx * s;
    const swy = Math.cos(this.t * 0.19) * this.swayPx * s;
    const mapX = (n: Node) => cx0 + (n.x - this.cxS) * this.scaleS + (n.z - 0.6) * swx;
    const mapY = (n: Node) => cy0 + (n.y - this.cyS) * this.scaleS + (n.z - 0.6) * swy;

    // Edges — faint, brighter on birth flash.
    ctx.lineWidth = Math.max(1, s);
    for (const e of this.edges) {
      const ia = this.nodeIndex(e.a);
      const ib = this.nodeIndex(e.b);
      if (ia < 0 || ib < 0) continue;
      const a = (0.16 + 0.55 * e.flash) * baseAlpha;
      ctx.strokeStyle = `rgba(200,200,200,${clampf(a, 0, 1)})`;
      ctx.beginPath();
      ctx.moveTo(mapX(this.nodes[ia]), mapY(this.nodes[ia]));
      ctx.lineTo(mapX(this.nodes[ib]), mapY(this.nodes[ib]));
      ctx.stroke();
    }

    // Nodes — markers + small word labels.
    ctx.font = `${fontPx}px JetBrains Mono, ui-monospace, monospace`;
    ctx.textBaseline = "top";
    ctx.textAlign = "left";
    for (const n of this.nodes) {
      const px = mapX(n);
      const py = mapY(n);
      const depth = clampf((n.z - 0.3) / 0.7, 0.0, 1.0); // 0 far .. 1 near
      const na = clampf((0.35 + 0.45 * depth + 0.2 * n.flash) * baseAlpha, 0.0, 1.0);
      const rad = (1.6 + 1.4 * depth + 1.5 * n.flash) * s;
      ctx.fillStyle = `rgba(235,235,235,${na})`;
      ctx.strokeStyle = `rgba(235,235,235,${na})`;

      if (n.bridge) {
        // diamond marker for bridge words
        ctx.beginPath();
        ctx.moveTo(px, py - rad - s);
        ctx.lineTo(px + rad + s, py);
        ctx.lineTo(px, py + rad + s);
        ctx.lineTo(px - rad - s, py);
        ctx.closePath();
        ctx.stroke();
      } else {
        ctx.beginPath();
        ctx.arc(px, py, rad, 0, Math.PI * 2);
        ctx.fill();
      }

      // word label, dimmer than its node, offset up-right
      const la = clampf((0.22 + 0.4 * depth + 0.3 * n.flash) * baseAlpha, 0.0, 1.0);
      ctx.fillStyle = `rgba(210,210,210,${la})`;
      ctx.fillText(n.word, px + rad + 3.0 * s, py - fontPx * 0.5);
    }
  }
}
