import type { OscEvent, OpRec } from "../stream/types";

// Screen 0 — Computation Rain. Port of graphics_consumer/src/screens/screen_rain.cpp
// (spec: docs/computation_rain_spec.md). An asymmetric 768×64 heatmap (D-S1, user
// 2026-06-05): x = k_pos (the matmul i-index) now maps 1:1 to the full 768-wide grid
// (no binning), while y = symMap(result) keeps 64 vertical bands (r is continuous,
// so 768 rows would collapse to 1px horizontal hairlines). Each frame the live ops
// paint bright cells, then everything decays, leaving comet-like trails of the
// running computation. (The original C++ used x = operand_a; see the rework note.)
//
// AUTHENTICITY DEVIATION (one, deliberate — user decision 2026-05-30):
//   The canonical C++ always paints a *synthetic* Gaussian cloud (plotWeightSpace,
//   18 pts/frame) and only plots real data for MulAttnQK (q_pos, k_pos). But the
//   project's firm constraint forbids synthetic/random-walk data. So instead of
//   the Gaussian, we plot the REAL captured op operands: every incoming OpRec's
//   (a, r) is mapped into the phase space. The canonical real MulAttnQK (q,k) path
//   is kept intact but stays dormant here (see note below).
//
// X = i-INDEX SWEEP, Y = symMap(RESULT) (2026-05-31 rework, user-approved):
//   The first attempt used the canonical x = operand_a. Two problems made the
//   screen look frozen. (1) op_type diversity is confined to the first ~35% of the
//   stream; the remaining 65% (≈78 min at install pace) is 100% MulQkvProj. (2) For
//   nearly every op, operand a barely varies (QKV a is p99 ±0.10), so on the a-axis
//   each op_type collapses to a fixed dot or column and just repaints the same cells
//   for tens of minutes — a static diagonal up front, a dead central streak in the
//   tail. The diversity was in the data; the a-axis was averaging it away.
//
//   The real fast signal is k_pos, which bert.c reuses as the matmul output index
//   (the i-index) on every non-attention op, sweeping 0..767 in real time as the op
//   crawls its rows (verified: 768 distinct values; q_pos, by contrast, is only
//   {0,1,2} — three ~27-min token zones). So x = k_pos across the full width and
//   y = symMap(r). symMap is a FIXED symmetric-log over the real range (r∈[-33,23]):
//   sign(r)·log1p(|r|/LIN) normalized, so each op_type settles at its own height —
//   LnMeanAcc r≈-7 low, embedding/LnNorm r≈0 center, the 512 LnRsqrt r≈+20 flashing
//   near the top — while every op reads as a horizontal comet sweeping left→right.
//   The front LayerNorm bands and the QKV tail both stay alive. Still 100% real:
//   the only op without a k_pos (LnRsqrt, 512 recs) falls back to symMap(a). No
//   synthesis, no clock.
//
// WHY THE ATTENTION RAIN IS QUIET IN THESE CAPTURES (verified against bert.c):
//   At install pace (1092 ops/sec) a 2-hour capture is exactly 7.86M ops — only
//   enough to finish the embedding + LayerNorm + QKV projection of the first ~3
//   tokens of layer 0 (QKV proj alone is 3·768·768 = 1.77M ops per token). The
//   attention stage (op_type 10, MulAttnQK) is never reached, so op_types 10–24
//   are absent from the capture. This is not a gap; it is the work's core theme
//   (Monitor C's "43 days per layer") made literal. The phase therefore stays in
//   QKV and the screen shows the weight-space rain of real projection operands.
//
//   Likewise the axis flashes (token/layer completion white-outs) are wired to
//   real OSC axis events, NOT a wall-clock timer — firing them on a timer would
//   fabricate token completions that never happen in this window. token_att and
//   layer are absent here, so the flash stays dormant; future captures that do
//   reach those milestones will light it up.

const NX = 768; // x cells: k_pos (i-index) 1:1, full BERT hidden width
const NY = 64; // y cells: symMap(r) bands (continuous r, kept coarse on purpose)
const DECAY = 0.91;
const CUTOFF = 0.02;
const REF_H = 600;

const POS_NA = 0xffff;
const OP_MUL_ATTN_QK = 10;

// Fixed symmetric-log mapping (replaces the QKV-dominated adaptive RMS scale).
// symMap(v) = sign(v)·log1p(|v|/LIN) normalized to [-1, 1] over |v| ≤ ~35. The
// log compresses the huge tail (LnMeanAcc ≈ -7, LnRsqrt ≈ +20, extremes to ±33)
// while LIN keeps a near-linear region around the origin so the dense small-value
// ops (embedding, QKV) still spread legibly instead of collapsing to one pixel.
const SYMLOG_LIN = 0.4;
const SYMLOG_MAX = Math.log1p(35 / SYMLOG_LIN);
const FILL = 0.46; // grid fill factor: ±1 maps to ±FILL·NY from center (edge margin)
function symMap(v: number): number {
  return (Math.sign(v) * Math.log1p(Math.abs(v) / SYMLOG_LIN)) / SYMLOG_MAX;
}

const Phase = { Embedding: 0, QKV: 1, Attention: 2, FFN: 3 } as const;
type Phase = (typeof Phase)[keyof typeof Phase];
const PHASE_NAMES = ["embedding", "qkv", "attention", "ffn"];

// Flash: a structural white-out on a real axis event (§7). Kept verbatim so it
// lights up on captures that actually reach a token/layer boundary.
class Flash {
  elapsed = 0;
  fadeIn = 0;
  hold = 0;
  fadeOut = 0;
  peak = 0;
  cleared = false;

  active(): boolean {
    return this.peak > 0 && this.elapsed < this.fadeIn + this.hold + this.fadeOut;
  }
  alpha(): number {
    if (!this.active()) return 0;
    if (this.elapsed < this.fadeIn) return (this.peak * this.elapsed) / this.fadeIn;
    let t = this.elapsed - this.fadeIn;
    if (t < this.hold) return this.peak;
    t -= this.hold;
    return this.peak * (1 - t / this.fadeOut);
  }
}

export class ScreenRain {
  // cells[x*NY+y], x = k_pos (i-index) in [0,NX), y = symMap(r) in [0,NY). Render
  // flips y so +r reads upward (matches the C++ index convention).
  private cells = new Float32Array(NX * NY);
  private phase: Phase = Phase.QKV;
  private flash = new Flash();

  private fireAxis1(): void {
    // §7.1: 80ms in, 500ms hold, 420ms out, peak 1.0 (a token finished attention).
    this.flash = new Flash();
    this.flash.fadeIn = 0.08;
    this.flash.hold = 0.5;
    this.flash.fadeOut = 0.42;
    this.flash.peak = 1.0;
  }
  private fireAxis2(): void {
    // §7.2: 700ms in, 800ms hold, 800ms out, peak 0.85 (a layer completed).
    this.flash = new Flash();
    this.flash.fadeIn = 0.7;
    this.flash.hold = 0.8;
    this.flash.fadeOut = 0.8;
    this.flash.peak = 0.85;
  }

  update(events: OscEvent[], ops: OpRec[], dt: number): void {
    // 1. Real axis events drive the structural flashes (dormant in Phase-1 caps).
    for (const ev of events) {
      if (ev.path === "/bert/token_att") this.fireAxis1();
      else if (ev.path === "/bert/layer") this.fireAxis2();
    }

    // 2. Per-frame decay (§4) — runs before this frame's hits so trails fade even
    //    in gaps while fresh hits stay at full brightness.
    const cells = this.cells;
    for (let i = 0; i < cells.length; i++) {
      const v = cells[i] * DECAY;
      cells[i] = v < CUTOFF ? 0 : v;
    }

    // 3. Walk this frame's real ops: plot the canonical MulAttnQK (q,k) directly,
    //    update phase from op_type, then plot the rest. y is always symMap(r); x is
    //    k_pos (the i-index sweep) for the QKV tail, else symMap(a).
    for (const rec of ops) {
      const qValid = (rec.flags & 1) !== 0 && rec.qPos !== POS_NA;
      const kValid = (rec.flags & 2) !== 0 && rec.kPos !== POS_NA;
      if (rec.opType === OP_MUL_ATTN_QK && qValid && kValid) {
        if (rec.qPos < NX && rec.kPos < NY) cells[rec.qPos * NY + rec.kPos] = 1.0;
        this.phase = Phase.Attention;
        continue; // attention recs paint at (q,k), not in (a,r) space
      }
      const ot = rec.opType;
      if (ot <= 1) this.phase = Phase.Embedding;
      else if (ot <= 9) this.phase = Phase.QKV;
      else if (ot <= 18) this.phase = Phase.Attention;
      else this.phase = Phase.FFN;

      const r = rec.r;
      if (!Number.isFinite(r)) continue;
      const y = clampI((symMap(r) * FILL + 0.5) * NY, 0, NY - 1);

      // x = k_pos, the matmul output-dimension (i-index) that bert.c reuses on every
      // non-attention op, sweeping 0..767 in real time as the op crawls its rows. With
      // the asymmetric grid (NX=768 = BERT hidden size) it now maps 1:1, no binning. So each
      // op_type reads as a horizontal comet at its own r-height and the whole stream
      // stays alive — both the front LayerNorm bands and the QKV tail. The lone op
      // with no k_pos (LnRsqrt, 512 recs) falls back to its operand a across the width.
      let x: number;
      if (kValid) {
        x = clampI(rec.kPos, 0, NX - 1);
      } else {
        const a = rec.a;
        if (!Number.isFinite(a)) continue;
        x = clampI((symMap(a) * FILL + 0.5) * NX, 0, NX - 1);
      }
      cells[x * NY + y] = 1.0;
    }

    // 4. Advance the flash; wipe the grid at the hold→fade-out boundary (§7).
    if (this.flash.active()) {
      const prev = this.flash.elapsed;
      this.flash.elapsed += dt;
      const clearAt = this.flash.fadeIn + this.flash.hold;
      if (!this.flash.cleared && prev < clearAt && this.flash.elapsed >= clearAt) {
        this.flash.cleared = true;
        cells.fill(0);
      }
    }
  }

  // Offscreen 768×64 buffer, upscaled with nearest-neighbor to mirror GL_NEAREST.
  private off: HTMLCanvasElement | null = null;
  private offCtx: CanvasRenderingContext2D | null = null;
  private img: ImageData | null = null;

  private ensureOff(): void {
    if (this.off) return;
    this.off = document.createElement("canvas");
    this.off.width = NX;
    this.off.height = NY;
    this.offCtx = this.off.getContext("2d")!;
    this.img = this.offCtx.createImageData(NX, NY);
  }

  render(ctx: CanvasRenderingContext2D, x: number, y: number, w: number, h: number): void {
    ctx.fillStyle = "#000";
    ctx.fillRect(x, y, w, h);

    this.ensureOff();
    const img = this.img!;
    const data = img.data;
    const cells = this.cells;
    // cell (gx=k_pos, gy=r) → pixel (col=gx, row=NY-1-gy) so +r reads upward.
    for (let gx = 0; gx < NX; gx++) {
      for (let gy = 0; gy < NY; gy++) {
        const v = cells[gx * NY + gy];
        const c = v <= 0 ? 0 : Math.round(Math.min(1, v) * 255);
        const px = (NY - 1 - gy) * NX + gx;
        const o = px * 4;
        data[o] = c;
        data[o + 1] = c;
        data[o + 2] = c;
        data[o + 3] = 255;
      }
    }
    this.offCtx!.putImageData(img, 0, 0);

    const prevSmooth = ctx.imageSmoothingEnabled;
    ctx.imageSmoothingEnabled = false;
    ctx.drawImage(this.off!, x, y, w, h);
    ctx.imageSmoothingEnabled = prevSmooth;

    // Structural flash: uniform white blend over the whole panel (frag u_flash).
    const fa = this.flash.alpha();
    if (fa > 0) {
      ctx.save();
      ctx.globalAlpha = fa;
      ctx.fillStyle = "#fff";
      ctx.fillRect(x, y, w, h);
      ctx.restore();
    }

    // Faint corner label, matching the web aggregate convention (dim grey).
    const s = h / REF_H;
    ctx.font = `${13 * s}px JetBrains Mono, ui-monospace, monospace`;
    ctx.textBaseline = "top";
    ctx.textAlign = "left";
    ctx.fillStyle = "rgba(31,31,31,1)";
    ctx.fillText(`0  computation rain  ·  ${PHASE_NAMES[this.phase]}`, x + 10 * s, y + 6 * s);
  }
}

function clampI(v: number, lo: number, hi: number): number {
  const i = Math.floor(v);
  return i < lo ? lo : i > hi ? hi : i;
}
