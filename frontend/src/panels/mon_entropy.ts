import type { OscEvent, OpRec } from "../stream/types";

// Monitor F — Entropy breath. Port of graphics_consumer/src/screens/mon_entropy.cpp
// (pre-attention mode, reworked 2026-06-05 per university feedback F-1).
//
// WHY THE REWORK: the canonical pre-attention fallback showed |R| (a slow EMA of
// the result magnitude) as one scalar + a 10s sparkline. With the Phase-1 captures
// stuck in layer-0 QKV (attention never reached), that scalar barely moved, so the
// panel read as static and its "entropy" name showed no entropy at all. The att_w
// attention-entropy H = -Sigma w*log w needs attention weights this capture never
// produces.
//
// THE FIX (fully real data, no synthetic fill): build a live distribution of the
// real per-op result r and show its Shannon entropy. The horizontal axis is the
// SAME symMap(r) sign-log used by Screen 0, so the install reads coherently across
// panels (Screen 0 = 2D scatter, F = 1D distribution of the same quantity). Each op
// drops into a symlog bin; the histogram decays every frame so it is a smooth
// running distribution; H = -Sigma p*log p over those bins, normalized to [0,1], is
// shown as both a number and a bar. The distribution's shape morphs as the op stream
// rotates through the forward-pass stages (embedding/gamma = sharp/low H, LayerNorm
// = broad/high H), so H visibly breathes (~0.17..0.75) on a ~5s rhythm tied to the
// computation structure. Validated on the captures (mousse_irigaray, 2M recs).
//
// The att_w post-attention entropy branch is left for captures that reach attention.

const NB = 48; // histogram bins over symMap(r) in [-1, 1]
const SYMLOG_LIN = 0.4; // near-linear region of the sym-log (matches Screen 0)
const SYM_KMAX = Math.log1p(35 / SYMLOG_LIN);

// BREATH_DECAY: per-frame histogram memory. Higher = calmer/slower/smoother breath,
// lower = snappier with more amplitude. Visual tuning knob (user reviews live).
const BREATH_DECAY = 0.94;
const H_SMOOTH = 0.1; // EMA on the displayed H so the digits do not twitch at 60fps

const LAYER_NA = 0xff;
const POS_NA = 0xffff;
const REF_H = 600;

// symMap(v): sign(v)*log1p(|v|/LIN) normalized to [-1,1] over |v| <= ~35. Identical
// to Screen 0's mapping so the two panels share one horizontal meaning.
function symMap(v: number): number {
  const s = v > 0 ? 1 : v < 0 ? -1 : 0;
  let x = (s * Math.log1p(Math.abs(v) / SYMLOG_LIN)) / SYM_KMAX;
  if (x > 1) x = 1;
  else if (x < -1) x = -1;
  return x;
}

// Normalized Shannon entropy of a histogram, in [0, 1] (1 = uniform across all bins).
function entropyNorm(hist: Float32Array): number {
  let tot = 0;
  for (let i = 0; i < hist.length; i++) tot += hist[i];
  if (tot <= 0) return 0;
  let h = 0;
  for (let i = 0; i < hist.length; i++) {
    const p = hist[i] / tot;
    if (p > 1e-12) h -= p * Math.log(p);
  }
  return h / Math.log(hist.length);
}

export class MonEntropy {
  private hist = new Float32Array(NB); // decaying symMap(r) accumulator
  private hDisp = 0; // smoothed normalized entropy actually shown

  // Context read off the most recent op (layer / q position).
  private zmqLayer = 0;
  private zmqQPos = 0;

  update(_events: OscEvent[], ops: OpRec[], dt: number): void {
    for (const rec of ops) {
      const r = rec.r;
      if (!Number.isFinite(r)) continue;
      let b = Math.floor((symMap(r) * 0.5 + 0.5) * NB);
      if (b < 0) b = 0;
      else if (b >= NB) b = NB - 1;
      this.hist[b] += 1;
      if (rec.layer !== LAYER_NA) this.zmqLayer = rec.layer;
      if ((rec.flags & 1) !== 0 && rec.qPos !== POS_NA) this.zmqQPos = rec.qPos;
    }

    // Frame-paced decay: gated on dt>0 so a paused timeline freezes the breath
    // instead of fading to flat. The op-type rotation through the stream drives the
    // entropy oscillation; decay sets how much the window remembers.
    if (dt > 0) {
      for (let i = 0; i < NB; i++) this.hist[i] *= BREATH_DECAY;
      const H = entropyNorm(this.hist);
      this.hDisp += (H - this.hDisp) * H_SMOOTH;
    }
  }

  render(ctx: CanvasRenderingContext2D, x: number, y: number, w: number, h: number): void {
    ctx.fillStyle = "#000";
    ctx.fillRect(x, y, w, h);

    const s = h / REF_H;
    const mono = "JetBrains Mono, ui-monospace, monospace";
    const pad = 10 * s;

    // Top labels: "F  entropy" left, "H" right.
    ctx.textBaseline = "top";
    ctx.font = `${12 * s}px ${mono}`;
    ctx.fillStyle = "rgba(51,51,51,1)";
    ctx.textAlign = "left";
    ctx.fillText("F  entropy", x + pad, y + 6 * s);
    ctx.textAlign = "right";
    ctx.fillText("H", x + w - pad, y + 6 * s);

    // Centered H value + (layer, q) context.
    const centerY = y + h * 0.16;
    ctx.textAlign = "center";
    ctx.font = `${20 * s}px ${mono}`;
    ctx.fillStyle = "rgba(205,205,205,1)";
    ctx.fillText(`H = ${this.hDisp.toFixed(3)}`, x + w / 2, centerY);

    ctx.font = `${12 * s}px ${mono}`;
    ctx.fillStyle = "rgba(89,89,89,1)";
    ctx.fillText(`(L${pad2(this.zmqLayer)}, q=${this.zmqQPos})`, x + w / 2, centerY + 24 * s);

    // Entropy bar (the scalar H, shown alongside the number).
    const barY = centerY + 48 * s;
    const barMargin = w * 0.08;
    const barX0 = x + barMargin;
    const barW = w - barMargin * 2;
    const barH = 3 * s;
    ctx.fillStyle = "rgba(20,20,20,1)";
    ctx.fillRect(barX0, barY, barW, barH);
    if (this.hDisp > 0) {
      ctx.fillStyle = "rgba(220,220,220,1)";
      ctx.fillRect(barX0, barY, barW * Math.min(1, this.hDisp), barH);
    }

    // Live distribution histogram (the breath itself).
    const histTop = barY + 18 * s;
    const histBottom = y + h - 22 * s;
    const histH = histBottom - histTop;
    if (histH > 4) {
      let maxv = 0;
      for (let i = 0; i < NB; i++) if (this.hist[i] > maxv) maxv = this.hist[i];
      const bw = barW / NB;
      // Faint centre line at symMap=0 to anchor the sign axis.
      ctx.fillStyle = "rgba(40,40,40,1)";
      ctx.fillRect(barX0 + barW * 0.5, histTop, 1, histH);
      if (maxv > 0) {
        ctx.fillStyle = "rgba(200,200,200,0.9)";
        for (let i = 0; i < NB; i++) {
          const v = this.hist[i] / maxv;
          if (v <= 0) continue;
          const bh = v * histH;
          ctx.fillRect(barX0 + i * bw, histBottom - bh, Math.max(1, bw - 1 * s), bh);
        }
      }
    }

    // Bottom axis label.
    ctx.font = `${12 * s}px ${mono}`;
    ctx.fillStyle = "rgba(46,46,46,1)";
    ctx.textAlign = "center";
    ctx.textBaseline = "alphabetic";
    ctx.fillText("H = -Σ p·log p   ·   result distribution (symlog r)", x + w / 2, y + h - 6 * s);
    ctx.textAlign = "left";
  }
}

function pad2(n: number): string {
  return String(n).padStart(2, "0");
}
