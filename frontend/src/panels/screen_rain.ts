import type { OscEvent, OpRec } from "../stream/types";

// Screen 0 — Computation Rain. Port of graphics_consumer/src/screens/screen_rain.cpp
// (spec: docs/computation_rain_spec.md). A 64×64 phase-space heatmap: x = a-axis
// (operand_a), y = r-axis (result). Each frame the live ops paint bright cells,
// then everything decays, leaving comet-like trails of the running computation.
//
// AUTHENTICITY DEVIATION (one, deliberate — user decision 2026-05-30):
//   The canonical C++ always paints a *synthetic* Gaussian cloud (plotWeightSpace,
//   18 pts/frame) and only plots real data for MulAttnQK (q_pos, k_pos). But the
//   project's firm constraint forbids synthetic/random-walk data. So instead of
//   the Gaussian, we plot the REAL captured op operands: every incoming OpRec's
//   (a, r) goes through the same running-RMS adaptive scaling the C++ uses, so the
//   rain *is* the actual QKV-projection numbers raining down. The canonical real
//   MulAttnQK (q,k) path is kept intact but stays dormant here (see note below).
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

const N = 64;
const DECAY = 0.91;
const CUTOFF = 0.02;
const REF_H = 600;

const POS_NA = 0xffff;
const OP_MUL_ATTN_QK = 10;

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
  // cells_[x*N+y], x = a-axis, y = r-axis (matches the C++ index convention).
  private cells = new Float32Array(N * N);
  private rmsA = 0.15; // running RMS of operand_a (seed from C++)
  private rmsR = 0.012; // running RMS of result (seed from C++)
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

    // 2. Walk this frame's real ops: plot the canonical MulAttnQK (q,k) directly,
    //    update phase from op_type, and gather (a, r) for the weight-space rain.
    let sumA2 = 0;
    let sumR2 = 0;
    let nReal = 0;
    const aBuf: number[] = [];
    const rBuf: number[] = [];
    for (const rec of ops) {
      const qValid = (rec.flags & 1) !== 0 && rec.qPos !== POS_NA;
      const kValid = (rec.flags & 2) !== 0 && rec.kPos !== POS_NA;
      if (rec.opType === OP_MUL_ATTN_QK && qValid && kValid) {
        if (rec.qPos < N && rec.kPos < N) this.cells[rec.qPos * N + rec.kPos] = 1.0;
        this.phase = Phase.Attention;
        continue; // attention recs paint at (q,k), not in (a,r) space
      }
      const ot = rec.opType;
      if (ot <= 1) this.phase = Phase.Embedding;
      else if (ot <= 9) this.phase = Phase.QKV;
      else if (ot <= 18) this.phase = Phase.Attention;
      else this.phase = Phase.FFN;

      // Real operand/result → the weight-space cloud (replaces synthetic Gaussian).
      const a = rec.a;
      const r = rec.r;
      if (Number.isFinite(a) && Number.isFinite(r)) {
        aBuf.push(a);
        rBuf.push(r);
        sumA2 += a * a;
        sumR2 += r * r;
        nReal++;
      }
    }

    // 3. Per-frame decay (§4) — runs every frame so trails fade even in gaps.
    const cells = this.cells;
    for (let i = 0; i < cells.length; i++) {
      const v = cells[i] * DECAY;
      cells[i] = v < CUTOFF ? 0 : v;
    }

    // 4. Plot the real (a, r) samples through the canonical adaptive scaling.
    if (nReal > 0) {
      // §5.2.3 running RMS (0.9 old / 0.1 new), §5.2.4 ±2σ ≈ 85% of grid.
      this.rmsA = this.rmsA * 0.9 + Math.sqrt(sumA2 / nReal) * 0.1;
      this.rmsR = this.rmsR * 0.9 + Math.sqrt(sumR2 / nReal) * 0.1;
      const scaleA = (N * 0.3) / Math.max(0.005, this.rmsA);
      const scaleR = (N * 0.3) / Math.max(0.005, this.rmsR);
      for (let i = 0; i < nReal; i++) {
        const x = clampI(aBuf[i] * scaleA + N * 0.5, 0, N - 1);
        const y = clampI(rBuf[i] * scaleR + N * 0.5, 0, N - 1);
        cells[x * N + y] = 1.0;
      }
    }

    // 5. Advance the flash; wipe the grid at the hold→fade-out boundary (§7).
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

  // Offscreen 64×64 buffer, upscaled with nearest-neighbor to mirror GL_NEAREST.
  private off: HTMLCanvasElement | null = null;
  private offCtx: CanvasRenderingContext2D | null = null;
  private img: ImageData | null = null;

  private ensureOff(): void {
    if (this.off) return;
    this.off = document.createElement("canvas");
    this.off.width = N;
    this.off.height = N;
    this.offCtx = this.off.getContext("2d")!;
    this.img = this.offCtx.createImageData(N, N);
  }

  render(ctx: CanvasRenderingContext2D, x: number, y: number, w: number, h: number): void {
    ctx.fillStyle = "#000";
    ctx.fillRect(x, y, w, h);

    this.ensureOff();
    const img = this.img!;
    const data = img.data;
    const cells = this.cells;
    // cell (gx=a, gy=r) → pixel (col=gx, row=N-1-gy) so +r reads upward.
    for (let gx = 0; gx < N; gx++) {
      for (let gy = 0; gy < N; gy++) {
        const v = cells[gx * N + gy];
        const c = v <= 0 ? 0 : Math.round(Math.min(1, v) * 255);
        const px = (N - 1 - gy) * N + gx;
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
