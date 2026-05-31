import type { OscEvent, OpRec } from "../stream/types";

// Monitor F — Entropy / |R| Magnitude. Port of
// graphics_consumer/src/screens/mon_entropy.cpp.
//
// Dual-mode like Monitor E: before attention weights arrive it shows |R|, the
// running magnitude of the computation result, with a slow sparkline; once
// /bert/att_w arrives it switches to per-token attention entropy H = -Σ w·log w.
// The Phase-1 captures never reach the attention stage (see screen_rain.ts), so
// att_w is absent — the same regime where the install shows the |R| magnitude
// view. We port that pre-attention mode faithfully (it runs on the real captured
// OpRec results) and leave the entropy branch for captures that reach attention.
//
// Magnitude math (verbatim from the C++ pre-attention branch):
//   R_ema  = R_ema*0.999 + |r|*0.001     (per-rec EMA)
//   R_peak = max(R_peak, |r|)
//   norm   = R_ema / R_peak               (bar + sparkline height)
//   sparkline: push one normalized point every R_SPARK_INTERVAL seconds.

const R_SPARK_INTERVAL = 10.0; // capture-seconds between sparkline samples
const SPARK_MAX = 256;
const LAYER_NA = 0xff;
const POS_NA = 0xffff;
const REF_H = 600;

export class MonEntropy {
  private rEma = 0.0;
  private rPeak = 0.001; // seed matches the C++ default
  private sparkTimer = 0.0;
  private history: number[] = [];

  // Context read off the most recent op (layer / q position / op type).
  private zmqLayer = 0;
  private zmqQPos = 0;

  update(_events: OscEvent[], ops: OpRec[], dt: number): void {
    for (const rec of ops) {
      const absR = Math.abs(rec.r);
      if (!Number.isFinite(absR)) continue;
      this.rEma = this.rEma * 0.999 + absR * 0.001;
      if (absR > this.rPeak) this.rPeak = absR;
      if (rec.layer !== LAYER_NA) this.zmqLayer = rec.layer;
      if ((rec.flags & 1) !== 0 && rec.qPos !== POS_NA) this.zmqQPos = rec.qPos;
    }

    // Sparkline: one point every R_SPARK_INTERVAL of capture time, while ops flow.
    if (ops.length > 0) {
      this.sparkTimer += dt;
      if (this.sparkTimer >= R_SPARK_INTERVAL) {
        this.sparkTimer -= R_SPARK_INTERVAL;
        const norm = this.rPeak > 0 ? Math.min(1, this.rEma / this.rPeak) : 0;
        this.history.push(norm);
        if (this.history.length > SPARK_MAX) this.history.shift();
      }
    }
  }

  render(ctx: CanvasRenderingContext2D, x: number, y: number, w: number, h: number): void {
    ctx.fillStyle = "#000";
    ctx.fillRect(x, y, w, h);

    const s = h / REF_H;
    const mono = "JetBrains Mono, ui-monospace, monospace";
    const pad = 10 * s;

    // Top labels: "F  magnitude" left, "|R|" right (matches the C++ headers).
    ctx.textBaseline = "top";
    ctx.font = `${12 * s}px ${mono}`;
    ctx.fillStyle = "rgba(51,51,51,1)";
    ctx.textAlign = "left";
    ctx.fillText("F  magnitude", x + pad, y + 6 * s);
    ctx.textAlign = "right";
    ctx.fillText("|R|", x + w - pad, y + 6 * s);

    // Centered value + (layer, q) context.
    const centerY = y + h * 0.28;
    ctx.textAlign = "center";
    ctx.textBaseline = "top";
    ctx.font = `${20 * s}px ${mono}`;
    ctx.fillStyle = "rgba(205,205,205,1)";
    ctx.fillText(`|R| = ${this.rEma.toFixed(4)}`, x + w / 2, centerY);

    ctx.font = `${12 * s}px ${mono}`;
    ctx.fillStyle = "rgba(89,89,89,1)";
    ctx.fillText(`(L${pad2(this.zmqLayer)}, q=${this.zmqQPos})`, x + w / 2, centerY + 24 * s);

    // Magnitude bar.
    const norm = this.rPeak > 0 ? Math.min(1, this.rEma / this.rPeak) : 0;
    const barY = centerY + 50 * s;
    const barMargin = w * 0.08;
    const barX0 = x + barMargin;
    const barW = w - barMargin * 2;
    const barH = 3 * s;
    ctx.fillStyle = "rgba(20,20,20,1)";
    ctx.fillRect(barX0, barY, barW, barH);
    if (norm > 0) {
      ctx.fillStyle = "rgba(220,220,220,1)";
      ctx.fillRect(barX0, barY, barW * norm, barH);
    }

    // Sparkline.
    const sparkY0 = barY + 18 * s;
    const sparkY1 = y + h - 22 * s;
    const sparkH = sparkY1 - sparkY0;
    if (this.history.length > 0 && sparkH > 4) {
      const n = this.history.length;
      const step = barW / Math.max(n - 1, 1);
      ctx.fillStyle = "rgba(200,200,200,0.86)";
      for (let i = 0; i < n; i++) {
        const sx = barX0 + i * step;
        const sy = sparkY1 - this.history[i] * sparkH;
        ctx.fillRect(sx - 1 * s, sy - 1 * s, 2 * s, 2 * s);
      }
    }

    // Bottom axis label.
    ctx.font = `${12 * s}px ${mono}`;
    ctx.fillStyle = "rgba(46,46,46,1)";
    ctx.textAlign = "center";
    ctx.textBaseline = "alphabetic";
    ctx.fillText("|R|  computation magnitude", x + w / 2, y + h - 6 * s);
    ctx.textAlign = "left";
  }
}

function pad2(n: number): string {
  return String(n).padStart(2, "0");
}
