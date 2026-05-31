import type { OpRec } from "../stream/types";

// Monitor B — i-index sweep ribbon. Replaces the structural Helix on the op
// stream panel (user decision 2026-05-31). Where the Helix is BERT's static
// architecture schematic (12×12 nested-loop topology), this is the COMPUTATION
// TRAJECTORY: every captured OpRec becomes one point on an open spiral whose
// angle is the matmul output index (k_pos, the i-index that bert.c sweeps 0..767
// in real time), whose radius is dented by the op result r, and which grows
// upward as the stream advances — so the curve is always "becoming", never full.
//
// This is an HONEST ANALOGUE of the residual-stream "manifold" helices in
// transformer-circuits' counting-task paper, not a faithful copy: that helix is
// a PCA of hidden-state vectors binned by a scalar, and our capture exports only
// per-op scalars (24-byte OpRec), not the 768-dim vectors a PCA helix needs. So
// this is labelled a SWEEP TRAJECTORY, not a residual manifold. 100% real data;
// no synthesis, no clock. Monochrome wireframe, fits its own region like Helix/
// Nebula. The 3D point spine is kept separate from the stroke so a ribbon-band
// or double-helix rendering can later be layered on the same curve.

const IDX_SPAN = 768; // k_pos (i-index) spans 0..767 on every matmul-output op
const POS_NA = 0xffff;
const WINDOW = 1400; // trailing points kept (≈1.8 sweeps ≈ 1.8 turns visible)
const TURNS_PER_SWEEP = 1; // one full revolution per 0..767 sweep
const PITCH = 3.5; // z rise per full turn (open coil; larger = more open)
const DZ = PITCH / IDX_SPAN; // z increment per point so one sweep = one pitch
const R0 = 3.2; // outer radius (matches Helix R0 for sibling scale)
const RMIN = 0.45; // inner radius factor; radius rides in [R0·RMIN, R0]
// Op records arrive in 128-record lumps (~117ms apart at 1x), so the head would
// jump 128 points then freeze for ~7 frames — reads as an ~8fps stutter. We pace
// the *reveal*: the visible head eases toward the latest point each frame, so it
// advances smoothly between lumps. This smooths presentation timing only; every
// plotted point is still a real op (cf. scaleSmooth), no synthesis.
const REVEAL_RATE = 10; // 1/s; higher = snappier reveal, lower = smoother lag
const ALPHA_BANDS = 12; // gradient quantized into N strokes (was 1 stroke/segment)

// Fixed symmetric-log radius dent, same shape as screen_rain's symMap so the two
// i-index views read consistently. symMap(v) ∈ [-1, 1] over |v| ≤ ~35.
const SYMLOG_LIN = 0.4;
const SYMLOG_MAX = Math.log1p(35 / SYMLOG_LIN);
function symMap(v: number): number {
  return (Math.sign(v) * Math.log1p(Math.abs(v) / SYMLOG_LIN)) / SYMLOG_MAX;
}

interface Pt {
  th: number; // angle around the coil
  rho: number; // radius (dented by r)
  z: number; // height along the coil axis
}

export class Ribbon {
  spinVel = 0.16; // slow turn so the coil reads as 3D (rad/s)
  tiltX = 0.34; // fixed camera tilt (rad) for depth

  private pts: Pt[] = [];
  private seq = 0; // monotonic point counter → height
  private shown = 0; // eased reveal cursor (global point units) ≤ seq
  private spinAng = 0;
  private scaleSmooth = 0;

  update(ops: OpRec[], dt: number): void {
    this.spinAng += this.spinVel * dt;

    for (const rec of ops) {
      const kValid = (rec.flags & 2) !== 0 && rec.kPos !== POS_NA;
      if (!kValid) continue; // i-index is the angle; skip the rare op without one
      const r = rec.r;
      if (!Number.isFinite(r)) continue;
      const th = (rec.kPos / (IDX_SPAN - 1)) * 2 * Math.PI * TURNS_PER_SWEEP;
      const rho = R0 * (RMIN + (1 - RMIN) * ((symMap(r) + 1) * 0.5));
      this.pts.push({ th, rho, z: this.seq * DZ });
      this.seq++;
    }
    if (this.pts.length > WINDOW) this.pts.splice(0, this.pts.length - WINDOW);

    // Ease the reveal cursor toward the newest point (smooths the 128-lump jumps).
    this.shown += (this.seq - this.shown) * Math.min(1, REVEAL_RATE * dt);
    if (this.shown > this.seq) this.shown = this.seq;
  }

  // Project a coil point (axis = vertical Y) to plane coords, pre-scale/center.
  private project(p: Pt, zCenter: number): [number, number] {
    const X = p.rho * Math.cos(p.th);
    const Yc = p.z - zCenter;
    const Z = p.rho * Math.sin(p.th);
    // spin around the vertical axis
    const cs = Math.cos(this.spinAng);
    const sn = Math.sin(this.spinAng);
    const x1 = X * cs + Z * sn;
    const z1 = -X * sn + Z * cs;
    // tilt around X so we view the coil slightly from the side/above
    const ct = Math.cos(this.tiltX);
    const st = Math.sin(this.tiltX);
    const y2 = Yc * ct - z1 * st;
    // orthographic: drop the remaining depth; y up → screen y down
    return [x1, -y2];
  }

  draw(
    ctx: CanvasRenderingContext2D,
    ox: number,
    oy: number,
    w: number,
    h: number,
    baseAlpha = 1.0,
  ): void {
    const n = this.pts.length;
    if (n < 2) return;

    // Reveal cursor → a float index into the window: how far the head has eased.
    // The newest (seq - shown) points stay hidden until the cursor reaches them.
    const hiddenTail = Math.min(Math.max(this.seq - this.shown, 0), n - 1);
    const lastVis = n - 1 - hiddenTail; // float head index within the window
    if (lastVis < 1) return;
    const iMax = Math.floor(lastVis);
    const frac = lastVis - iMax;

    // Vertical camera driven by the EASED reveal cursor, not the raw point set.
    // z grows with seq (128-op lumps, ~117ms apart), so centering on the raw mean
    // jerked the whole coil down one lump-height per arrival. Tying the center to
    // `shown` (the same eased signal the head uses) makes the downward scroll
    // smooth and locks it to the reveal. Head is pinned near the top; the body
    // flows down through it. spanZ is the visible vertical extent so the coil
    // fills the region symmetrically once the window is full.
    const headZ = this.shown * DZ;
    const spanZ = Math.min(this.shown, WINDOW) * DZ;
    const zCenter = headZ - spanZ * 0.5;

    // Project only the revealed points (+1 for the interpolated lead segment).
    // Fitting over visible points keeps the as-yet-hidden lump from nudging scale.
    const last = Math.min(iMax + 1, n - 1);
    const px = new Array<number>(last + 1);
    const py = new Array<number>(last + 1);
    let halfw = 1e-3;
    let halfh = 1e-3;
    for (let k = 0; k <= last; k++) {
      const [x, y] = this.project(this.pts[k], zCenter);
      px[k] = x;
      py[k] = y;
      if (k <= iMax) {
        halfw = Math.max(halfw, Math.abs(x));
        halfh = Math.max(halfh, Math.abs(y));
      }
    }

    const fill = 0.8;
    const target = fill * Math.min((w * 0.5) / halfw, (h * 0.5) / halfh);
    this.scaleSmooth =
      this.scaleSmooth <= 0 ? target : this.scaleSmooth + (target - this.scaleSmooth) * 0.08;
    const scale = this.scaleSmooth;
    const cx0 = ox + w * 0.5;
    const cy0 = oy + h * 0.5;
    const mapX = (x: number): number => cx0 + x * scale;
    const mapY = (y: number): number => cy0 + y * scale;

    // Single open spiral, oldest → head. Alpha ramps tail (dim) → head (bright),
    // quantized into ALPHA_BANDS contiguous strokes (cheap vs one stroke/segment).
    ctx.lineWidth = 1;
    let i = 1;
    while (i <= iMax) {
      const band = Math.min(ALPHA_BANDS - 1, Math.floor((i / lastVis) * ALPHA_BANDS));
      const a = (0.08 + 0.62 * ((band + 0.5) / ALPHA_BANDS)) * baseAlpha;
      ctx.strokeStyle = `rgba(225,225,225,${a})`;
      ctx.beginPath();
      ctx.moveTo(mapX(px[i - 1]), mapY(py[i - 1]));
      let j = i;
      while (
        j <= iMax &&
        Math.min(ALPHA_BANDS - 1, Math.floor((j / lastVis) * ALPHA_BANDS)) === band
      ) {
        ctx.lineTo(mapX(px[j]), mapY(py[j]));
        j++;
      }
      ctx.stroke();
      i = j;
    }

    // Interpolated head position (sub-point) for buttery advance between lumps.
    let hx = mapX(px[iMax]);
    let hy = mapY(py[iMax]);
    if (iMax + 1 < n && frac > 0) {
      const nx = mapX(px[iMax + 1]);
      const ny = mapY(py[iMax + 1]);
      // bright leading segment to the interpolated front
      hx += (nx - hx) * frac;
      hy += (ny - hy) * frac;
      ctx.strokeStyle = `rgba(245,245,245,${0.85 * baseAlpha})`;
      ctx.beginPath();
      ctx.moveTo(mapX(px[iMax]), mapY(py[iMax]));
      ctx.lineTo(hx, hy);
      ctx.stroke();
    }

    // Bright leading head — the live computation front.
    ctx.fillStyle = `rgba(245,245,245,${0.95 * baseAlpha})`;
    ctx.beginPath();
    ctx.arc(hx, hy, 1.8, 0, Math.PI * 2);
    ctx.fill();
  }
}
