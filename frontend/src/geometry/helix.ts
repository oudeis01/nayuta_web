import type { OscEvent } from "../stream/types";

// Structural helix — shared background/peripheral geometry. Faithful port of
// graphics_consumer/src/screens/helix.{h,cpp} (spec: docs/20260526-helix-
// prototype-spec.md). Draws BERT's 12-layer × 12-head nested-loop topology as a
// monochrome orthographic wireframe (12 rings + 12 strands) that slowly tumbles.
//
// Data in the Phase-1 captures: /bert/clock is present (drives tumble + phase
// sync); the axis events (token_att / token_layer / sequence_start) that grow and
// reset the rings are absent. The built-in heartbeat kick (every kick_interval
// seconds) guarantees visible motion and ring refresh regardless, exactly as the
// C++ comments note, so no fallback is needed. The "entropy" twist drift reads
// the helix's own internal conc_ diffuseness — it never depended on /bert/att_w.

export type HelixView = "side" | "cross";

const NLAYERS = 12;
const NHEADS = 12;
const LAYER_GAP = 1.0;
const R0 = 3.2;
const RMIN = 0.35;

function frand(): number {
  return Math.random();
}
function clampf(x: number, lo: number, hi: number): number {
  return x < lo ? lo : x > hi ? hi : x;
}

// ── Self-drawing route (draw-on animation, user decision 2026-05-31) ──────────
// The helix data is full from the start (authentic; we never fake growth). The
// draw-on is a *rendering* flourish: a single pen traces the ENTIRE wireframe in
// one continuous stroke, periodically, so the cylinder visibly wires itself up.
//
// "One stroke covering every edge" = an Eulerian circuit. The wireframe (144
// vertices = 12 rings × 12 heads; ring edges cyclic in h, strand edges between
// adjacent layers) has 24 odd-degree vertices (all of the two end rings, degree
// 3), so a pure Eulerian path is impossible. We make every degree even by
// DOUBLING 12 ring edges (adjacent pairs on each end ring); those 12 of 276 edges
// get traced twice, which is visually invisible. Hierholzer then yields a circuit
// through all edges. Different neighbour-ordering / start policies give visually
// distinct "drawings" (FAMILIES) that we cycle on each heartbeat trigger.
interface HEdge {
  to: number;
  id: number;
}
function buildHelixGraph(nLayers: number, nHeads: number): { adj: HEdge[][]; edgeCount: number } {
  const V = nLayers * nHeads;
  const adj: HEdge[][] = Array.from({ length: V }, () => []);
  let id = 0;
  const add = (a: number, b: number): void => {
    adj[a].push({ to: b, id });
    adj[b].push({ to: a, id });
    id++;
  };
  // ring edges: head h ↔ h+1 (mod nHeads) within each layer
  for (let L = 0; L < nLayers; L++)
    for (let h = 0; h < nHeads; h++) add(L * nHeads + h, L * nHeads + ((h + 1) % nHeads));
  // strand edges: (L,h) ↔ (L+1,h)
  for (let L = 0; L < nLayers - 1; L++)
    for (let h = 0; h < nHeads; h++) add(L * nHeads + h, (L + 1) * nHeads + h);
  // double 12 ring edges on the two end rings → all degrees even
  for (const L of [0, nLayers - 1])
    for (let h = 0; h < nHeads; h += 2) add(L * nHeads + h, L * nHeads + ((h + 1) % nHeads));
  return { adj, edgeCount: id };
}
function isRingEdge(from: number, to: number, nHeads: number): boolean {
  return Math.floor(from / nHeads) === Math.floor(to / nHeads);
}
// Hierholzer with a per-policy neighbour ordering. Returns a vertex sequence of
// length edgeCount+1 visiting every (multi)edge exactly once.
function eulerCircuit(
  adj: HEdge[][],
  edgeCount: number,
  start: number,
  ringFirst: boolean,
  ascendTo: boolean,
  nHeads: number,
): number[] {
  const order: HEdge[][] = adj.map((lst, v) =>
    lst.slice().sort((a, b) => {
      const ra = isRingEdge(v, a.to, nHeads) ? 0 : 1;
      const rb = isRingEdge(v, b.to, nHeads) ? 0 : 1;
      const ka = ringFirst ? ra : 1 - ra;
      const kb = ringFirst ? rb : 1 - rb;
      if (ka !== kb) return ka - kb;
      return ascendTo ? a.to - b.to : b.to - a.to;
    }),
  );
  const used = new Uint8Array(edgeCount);
  const ptr = new Int32Array(adj.length);
  const stack: number[] = [start];
  const out: number[] = [];
  while (stack.length) {
    const v = stack[stack.length - 1];
    let advanced = false;
    while (ptr[v] < order[v].length) {
      const e = order[v][ptr[v]++];
      if (used[e.id]) continue;
      used[e.id] = 1;
      stack.push(e.to);
      advanced = true;
      break;
    }
    if (!advanced) out.push(stack.pop()!);
  }
  return out.reverse();
}
// 6 visually distinct families (policy × start), plus reverses for cheap variety.
function buildHelixRoutes(nLayers: number, nHeads: number): number[][] {
  const { adj, edgeCount } = buildHelixGraph(nLayers, nHeads);
  const mid = Math.floor(nLayers / 2) * nHeads + Math.floor(nHeads / 2);
  const last = nLayers * nHeads - 1;
  const specs: Array<[number, boolean, boolean]> = [
    [0, true, true], // ring-major, ascending, from bottom
    [0, false, true], // strand-major, ascending, from bottom
    [mid, true, false], // ring-major, descending, from middle
    [last, false, true], // strand-major, ascending, from top
  ];
  const base = specs.map(([s, rf, asc]) => eulerCircuit(adj, edgeCount, s, rf, asc, nHeads));
  return [base[0], base[1], base[2], base[3], base[0].slice().reverse(), base[1].slice().reverse()];
}

export class Helix {
  // Live-tunable knobs (match the C++ defaults).
  twist = 0.18; // baseline helical phase advance per unit z
  twistEnt = 0.16; // entropy-driven drift amplitude around baseline
  spinBase = 0.12; // baseline long-axis angular velocity (rad/s)
  kick = 3.2; // angular impulse on data update (rad/s)
  damp = 2.5; // exponential return-to-baseline rate
  tumbleAx = 0.043; // X tumble rate (rad/s)
  tumbleAy = 0.027; // Y tumble rate (rad/s)
  kickInterval = 6.0; // heartbeat kick cadence (s); <=0 disables

  private conc = new Float32Array(NLAYERS * NHEADS);
  private activeLayer = NLAYERS - 1;
  private spinVel = 0.12;
  private spinAng = 0;
  private twistCur = 0.18;
  private clock = 0; // seconds, from /bert/clock elapsed (else dt accumulation)
  private sinceKick = 0;
  private kickRingCursor = 0;
  private scaleSmooth = 0;

  // Draw-on self-construction (lazy-built routes; cycles family on each heartbeat).
  drawDur = 2.4; // seconds for one full-wireframe stroke
  morphDur = 0.9; // seconds to settle from the platonic cylinder to the real shape
  private routes: number[][] | null = null;
  private familyIdx = 0;
  private drawT = 0; // elapsed within the current stroke
  private drawing = true; // self-draw once on load, then on every heartbeat
  // morph 0 = platonic cylinder (uniform radius, regular rings); 1 = real conc shape.
  // The stroke draws the ideal cylinder; once complete, the form warps into the
  // data-driven (dented) shape. Starts at the platonic ideal on load.
  private morph = 0;
  private morphT = 0;

  constructor() {
    this.twistCur = this.twist;
    this.regenAll();
  }

  private startDraw(): void {
    if (!this.routes) this.routes = buildHelixRoutes(NLAYERS, NHEADS);
    this.familyIdx = (this.familyIdx + 1) % this.routes.length;
    this.drawT = 0;
    this.drawing = true;
    this.morph = 0; // redraw the platonic ideal, then warp to real again
    this.morphT = 0;
  }

  private regenAll(): void {
    for (let L = 0; L < NLAYERS; L++) this.regenRing(L);
  }
  private regenRing(L: number): void {
    // Mostly diffuse attention (large radius), a few concentrated heads (small).
    for (let h = 0; h < NHEADS; h++) {
      let c = 0.35 + 0.25 * frand();
      if (frand() < 0.15) c = 0.05 + 0.15 * frand();
      this.conc[L * NHEADS + h] = c;
    }
  }
  private jitter(energy: number): void {
    const amp = 0.01 * clampf(energy, 0.2, 4.0);
    for (let i = 0; i < this.conc.length; i++)
      this.conc[i] = clampf(this.conc[i] + (frand() - 0.5) * amp, 0.02, 0.95);
  }
  private fireKick(strength: number): void {
    this.spinVel += strength;
    this.sinceKick = 0;
  }
  private advanceSpin(dt: number): void {
    this.spinVel += (this.spinBase - this.spinVel) * Math.min(1, this.damp * dt);
    this.spinAng += this.spinVel * dt;
  }

  // energy: jitter amplitude scale (mean |r| proxy on the op-stream panels, else 1).
  update(events: OscEvent[], dt: number, energy = 1.0): void {
    this.clock += dt;
    for (const ev of events) {
      switch (ev.path) {
        case "/bert/clock":
          this.clock = ev.args[1]; // snap to authoritative broadcast clock
          break;
        case "/bert/token_att": // axis1: a head's attention settled
          if (this.activeLayer >= 0 && this.activeLayer < NLAYERS) this.regenRing(this.activeLayer);
          this.fireKick(this.kick);
          break;
        case "/bert/token_layer": // axis2: advance to next layer ring
          this.activeLayer++;
          if (this.activeLayer >= NLAYERS) {
            this.activeLayer = 0;
            this.regenAll();
          }
          this.fireKick(this.kick * 1.5);
          break;
        case "/bert/sequence_start": // new sequence: regrow from the bottom
          this.activeLayer = 0;
          this.regenAll();
          this.fireKick(this.kick * 1.5);
          break;
        default:
          break;
      }
    }

    // Heartbeat kick: keeps the helix alive when real axis events are minutes
    // apart, and refreshes the next ring so "new circles drawn" semantics hold.
    this.sinceKick += dt;
    if (this.kickInterval > 0 && this.sinceKick >= this.kickInterval) {
      this.regenRing(this.kickRingCursor % NLAYERS);
      this.kickRingCursor++;
      this.fireKick(this.kick);
      this.startDraw(); // heartbeat retriggers the self-construction stroke
    }

    // Advance the draw-on stroke; when it completes, warp from the platonic
    // cylinder to the real data-driven shape, then hold.
    if (this.drawing) {
      this.drawT += dt;
      if (this.drawT >= this.drawDur) {
        this.drawing = false;
        this.morphT = 0;
      }
    } else if (this.morph < 1) {
      this.morphT += dt;
      this.morph = clampf(this.morphT / this.morphDur, 0, 1);
    }

    this.jitter(energy);
    this.advanceSpin(dt);

    // Entropy-driven twist drift from internal diffuseness (low conc ≈ entropy).
    let active = this.activeLayer < NLAYERS ? this.activeLayer : NLAYERS - 1;
    if (active < 0) active = 0;
    let sum = 0;
    let cnt = 0;
    for (let L = 0; L <= active; L++)
      for (let h = 0; h < NHEADS; h++) {
        sum += 1 - this.conc[L * NHEADS + h];
        cnt++;
      }
    const diffuse = cnt ? sum / cnt : 0.5;
    const target = this.twist + this.twistEnt * (diffuse - 0.5) * 2.0;
    this.twistCur += (target - this.twistCur) * Math.min(1, 0.6 * dt);
  }

  private vertex(L: number, h: number, conc: number): [number, number, number] {
    const z = (L - (NLAYERS - 1) / 2.0) * LAYER_GAP;
    const th = h * ((2.0 * Math.PI) / NHEADS) + this.twistCur * z;
    // morph 0 → concEff = 1 (uniform max radius = regular cylinder); morph 1 →
    // the real per-head conc (the dented data shape). The twist is kept in both
    // so the platonic form is the clean (slightly twisted) cylinder.
    const concEff = 1.0 + (conc - 1.0) * this.morph;
    const rad = R0 * (RMIN + (1.0 - RMIN) * concEff);
    return [rad * Math.cos(th), rad * Math.sin(th), z];
  }

  // Orthographic projection to pre-scale, pre-translate plane coords (px, py).
  private projectXY(
    vx: number,
    vy: number,
    vz: number,
    ax: number,
    ay: number,
    view: HelixView,
  ): [number, number] {
    let x = vx;
    let y = vy;
    let z = vz;

    // 1) long-axis spin (Z)
    const ca = Math.cos(this.spinAng);
    const sa = Math.sin(this.spinAng);
    const x1 = x * ca - y * sa;
    const y1 = x * sa + y * ca;
    x = x1;
    y = y1;

    // 2) camera pitch (X) — selects the viewpoint
    const pitch = view === "cross" ? Math.PI / 2.0 : 0.0;
    const cp = Math.cos(pitch);
    const sp = Math.sin(pitch);
    const y2 = y * cp - z * sp;
    const z2 = y * sp + z * cp;
    y = y2;
    z = z2;

    // 3) tumble X
    const cx = Math.cos(ax);
    const sx = Math.sin(ax);
    const y3 = y * cx - z * sx;
    const z3 = y * sx + z * cx;
    y = y3;
    z = z3;

    // 4) tumble Y
    const cy = Math.cos(ay);
    const sy = Math.sin(ay);
    x = x * cy + z * sy;

    // 5) orthographic — drop z. y up → screen y down.
    return [x, -y];
  }

  // Draw into [ox, oy, w, h] as an isolated canvas (bounding sphere normalized to
  // fit, centered). The helix floats centered regardless of the region size.
  draw(
    ctx: CanvasRenderingContext2D,
    ox: number,
    oy: number,
    w: number,
    h: number,
    view: HelixView,
    baseAlpha = 1.0,
  ): void {
    const ax = this.clock * this.tumbleAx;
    const ay = this.clock * this.tumbleAy;
    let active = this.activeLayer < NLAYERS ? this.activeLayer : NLAYERS - 1;
    if (active < 0) return;

    const rx: number[] = new Array((active + 1) * NHEADS);
    const ry: number[] = new Array((active + 1) * NHEADS);
    let halfw = 1e-3;
    let halfh = 1e-3;
    for (let L = 0; L <= active; L++) {
      for (let hh = 0; hh < NHEADS; hh++) {
        const [vx, vy, vz] = this.vertex(L, hh, this.conc[L * NHEADS + hh]);
        const [px, py] = this.projectXY(vx, vy, vz, ax, ay, view);
        rx[L * NHEADS + hh] = px;
        ry[L * NHEADS + hh] = py;
        halfw = Math.max(halfw, Math.abs(px));
        halfh = Math.max(halfh, Math.abs(py));
      }
    }

    const fill = 0.8;
    const target = fill * Math.min((w * 0.5) / halfw, (h * 0.5) / halfh);
    this.scaleSmooth =
      this.scaleSmooth <= 0 ? target : this.scaleSmooth + (target - this.scaleSmooth) * 0.08;
    const scale = this.scaleSmooth;

    const cx0 = ox + w * 0.5;
    const cy0 = oy + h * 0.5;
    const mapX = (px: number) => cx0 + px * scale;
    const mapY = (py: number) => cy0 + py * scale;

    // Draw-on: while a stroke is in progress (and the helix is full, which it
    // always is in these captures), trace the single-stroke route from black so
    // the cylinder visibly wires itself up. Otherwise hold the full wireframe.
    if (this.drawing && active === NLAYERS - 1) {
      if (!this.routes) this.routes = buildHelixRoutes(NLAYERS, NHEADS);
      const seq = this.routes[this.familyIdx];
      const segTotal = seq.length - 1;
      const p = clampf(this.drawT / this.drawDur, 0, 1);
      const eased = 1 - (1 - p) * (1 - p); // ease-out
      const drawn = eased * segTotal;
      const fullSeg = Math.floor(drawn);
      const frac = drawn - fullSeg;

      // The traced body so far, dim. Plus a partial leading segment.
      ctx.lineWidth = 1;
      ctx.strokeStyle = `rgba(220,220,220,${0.5 * baseAlpha})`;
      ctx.beginPath();
      ctx.moveTo(mapX(rx[seq[0]]), mapY(ry[seq[0]]));
      for (let i = 1; i <= fullSeg; i++) ctx.lineTo(mapX(rx[seq[i]]), mapY(ry[seq[i]]));
      let headX: number;
      let headY: number;
      if (fullSeg < segTotal) {
        const ax = mapX(rx[seq[fullSeg]]);
        const ay = mapY(ry[seq[fullSeg]]);
        const bx = mapX(rx[seq[fullSeg + 1]]);
        const by = mapY(ry[seq[fullSeg + 1]]);
        headX = ax + (bx - ax) * frac;
        headY = ay + (by - ay) * frac;
        ctx.lineTo(headX, headY);
      } else {
        headX = mapX(rx[seq[segTotal]]);
        headY = mapY(ry[seq[segTotal]]);
      }
      ctx.stroke();

      // Bright comet head over the last few segments.
      const tailStart = Math.max(0, fullSeg - 6);
      ctx.strokeStyle = `rgba(245,245,245,${0.95 * baseAlpha})`;
      ctx.beginPath();
      ctx.moveTo(mapX(rx[seq[tailStart]]), mapY(ry[seq[tailStart]]));
      for (let i = tailStart + 1; i <= fullSeg; i++) ctx.lineTo(mapX(rx[seq[i]]), mapY(ry[seq[i]]));
      ctx.lineTo(headX, headY);
      ctx.stroke();
      return;
    }

    // Strands (head h, L=0..active) — dim helical lines.
    ctx.lineWidth = 1;
    ctx.strokeStyle = `rgba(200,200,200,${0.3 * baseAlpha})`;
    for (let hh = 0; hh < NHEADS; hh++) {
      ctx.beginPath();
      for (let L = 0; L <= active; L++) {
        const px = mapX(rx[L * NHEADS + hh]);
        const py = mapY(ry[L * NHEADS + hh]);
        if (L === 0) ctx.moveTo(px, py);
        else ctx.lineTo(px, py);
      }
      ctx.stroke();
    }

    // Rings (layer L, 12 heads closed) — brighter, recent layers brightest.
    for (let L = 0; L <= active; L++) {
      const age = active - L;
      const a = clampf(0.65 - age * 0.05, 0.12, 0.75) * baseAlpha;
      ctx.strokeStyle = `rgba(235,235,235,${a})`;
      ctx.beginPath();
      for (let hh = 0; hh < NHEADS; hh++) {
        const px = mapX(rx[L * NHEADS + hh]);
        const py = mapY(ry[L * NHEADS + hh]);
        if (hh === 0) ctx.moveTo(px, py);
        else ctx.lineTo(px, py);
      }
      ctx.closePath();
      ctx.stroke();
    }
  }
}
