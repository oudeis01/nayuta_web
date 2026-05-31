import type { OscEvent, OpRec } from "../stream/types";

// Monitor E — Head Mandala. Port of graphics_consumer/src/screens/mon_mandala.cpp,
// pre-attention mode reworked into a RADIAL (polar) structure map.
//
// WHY RADIAL (design decision 2026-05-31): the canonical pre-attention fallback
// plotted the op (a, r) into a cartesian phase space — but Screen 0 (computation
// rain) already owns that exact (a-x, r-y) view in the web build, so the two read
// as the same picture. The data poverty of the Phase-1 captures (only op_types
// 0–8, no attention) forces every OpRec-only panel onto the same signal, so they
// converge. To restore the panel's distinct "mandala" identity we project the SAME
// real OpRec stream onto a polar grid that no other panel uses:
//
//   angle  = op_type    — the 25 arithmetic stages of one forward pass laid around
//                         the circle in execution order (EMB → LayerNorm → QKV …
//                         → FFN residual). A complete pass would light a full ring;
//                         these captures only reach the first ~9 stages, so the
//                         mandala is an honest partial arc.
//   radius = q_pos      — token position. Inner = position 0, rim = the furthest
//                         position the computation has crawled to (adaptive).
//   value  = |r| accum  — the magnitude of the result, accumulated then decayed.
//
// This stays fully real-data (no synthetic fill, consistent with Screen 0's
// authenticity rule) while diverging visually from the cartesian rain. The att_w
// per-head heatmap (the panel's post-attention identity) is left for captures that
// reach the attention stage. Verified fields (tate_nnn, first 400k recs): op_type
// 0–7 present, q_pos 100% valid range 0–74, layer/head 0% valid.

const N_SECT = 25; // OpType::Count — one angular wedge per arithmetic stage
const QSTORE = 128; // q_pos bins (clamp); install reaches ~75, headroom to grow
const ACCUM = 0.08; // per-rec blend into a cell (1-ACCUM keeps history)
const DECAY = 0.985; // per-frame fade so the mandala breathes
const CUTOFF = 1e-3;
const GAMMA = 0.6; // lift dim cells, matching the canonical frag pow(v,0.6)
const REF_H = 600;
const TWO_PI = Math.PI * 2;

const POS_NA = 0xffff;

export class MonMandala {
  // cells[sect*QSTORE + q]: accumulated |r| for (op_type sect, token position q).
  private cells = new Float32Array(N_SECT * QSTORE);
  private rEmaAbs = 0.01; // slow magnitude EMA (seed matches the C++ default)
  private maxQ = 1; // furthest token position seen, for adaptive radius

  update(_events: OscEvent[], ops: OpRec[], _dt: number): void {
    const cells = this.cells;

    for (const rec of ops) {
      const ot = rec.opType;
      if (ot < 0 || ot >= N_SECT) continue;
      if ((rec.flags & 1) === 0 || rec.qPos === POS_NA) continue;
      if (!Number.isFinite(rec.r)) continue;

      this.rEmaAbs = this.rEmaAbs * 0.9999 + Math.abs(rec.r) * 0.0001;
      const scale = 0.3 / Math.max(this.rEmaAbs, 1e-6);
      const contribution = Math.min(1, Math.abs(rec.r) * scale);

      const q = rec.qPos < QSTORE ? rec.qPos : QSTORE - 1;
      if (q > this.maxQ) this.maxQ = q;
      const idx = ot * QSTORE + q;
      cells[idx] = cells[idx] * (1 - ACCUM) + contribution * ACCUM;
    }

    for (let i = 0; i < cells.length; i++) {
      const v = cells[i] * DECAY;
      cells[i] = v < CUTOFF ? 0 : v;
    }
  }

  render(ctx: CanvasRenderingContext2D, x: number, y: number, w: number, h: number): void {
    ctx.fillStyle = "#000";
    ctx.fillRect(x, y, w, h);

    const cx = x + w / 2;
    const cy = y + h / 2;
    const rMax = Math.min(w, h) * 0.46;
    const rInner = rMax * 0.06; // small hole so position 0 isn't a singularity
    const span = rMax - rInner;
    const maxQ = Math.max(1, this.maxQ);
    const sw = TWO_PI / N_SECT; // angular width per op_type wedge
    const gap = sw * 0.08; // thin seam between wedges
    const ringT = Math.max(1, span / maxQ); // radial thickness of one q ring
    const cells = this.cells;

    for (let s = 0; s < N_SECT; s++) {
      // Execution order around the circle, starting at the top, clockwise.
      const a0 = -Math.PI / 2 + s * sw + gap * 0.5;
      const a1 = -Math.PI / 2 + (s + 1) * sw - gap * 0.5;
      const base = s * QSTORE;
      const qLim = Math.min(QSTORE - 1, maxQ);
      for (let q = 0; q <= qLim; q++) {
        const v = cells[base + q];
        if (v <= 0) continue;
        const rNorm = q / maxQ; // 0..1
        const rc = rInner + rNorm * span;
        const r0 = Math.max(rInner, rc - ringT * 0.5);
        // Clamp the outer edge to the disc: the rim ring (q=maxQ) sits at rc=rMax,
        // and when maxQ is small ringT = span/maxQ balloons, so an unclamped
        // r1 = rc + ringT/2 would shoot far past rMax and bleed into neighbouring
        // panels on the shared canvas.
        const r1 = Math.min(rMax, rc + ringT * 0.5);
        const c = Math.round(Math.pow(Math.min(1, v), GAMMA) * 255);
        ctx.fillStyle = `rgb(${c},${c},${c})`;
        ctx.beginPath();
        ctx.arc(cx, cy, r1, a0, a1, false);
        ctx.arc(cx, cy, r0, a1, a0, true);
        ctx.closePath();
        ctx.fill();
      }
    }

    // Faint corner label, matching the aggregate convention.
    const s = h / REF_H;
    ctx.font = `${13 * s}px JetBrains Mono, ui-monospace, monospace`;
    ctx.textBaseline = "top";
    ctx.textAlign = "left";
    ctx.fillStyle = "rgba(31,31,31,1)";
    ctx.fillText("E  head mandala  ·  stage×pos", x + 10 * s, y + 6 * s);
  }
}
