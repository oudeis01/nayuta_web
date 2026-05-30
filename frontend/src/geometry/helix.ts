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

  constructor() {
    this.twistCur = this.twist;
    this.regenAll();
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
    const rad = R0 * (RMIN + (1.0 - RMIN) * conc);
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
