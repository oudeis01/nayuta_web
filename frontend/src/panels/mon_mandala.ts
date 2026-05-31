import type { OscEvent, OpRec } from "../stream/types";

// Monitor E — Head Mandala / Computation Topology. Port of
// graphics_consumer/src/screens/mon_mandala.cpp.
//
// The canonical panel is DUAL-MODE: a pre-attention "ZMQ topology" view driven by
// the OpRec (a, r) phase space, which then switches to an att_w heatmap once
// post-softmax attention weights arrive. At install pace a 2-hour window never
// reaches the attention stage (see screen_rain.ts for the bert.c arithmetic), so
// /bert/att_w never appears in the Phase-1 captures — exactly the regime where the
// install also shows the topology view. We therefore port the pre-attention mode
// faithfully (it runs on the real captured operands, no synthetic data) and leave
// the att_w heatmap branch for captures that actually reach attention.
//
// Topology math (verbatim from the C++ pre-attention branch):
//   r_ema_abs = r_ema_abs*0.9999 + |r|*0.0001        (per-rec, slow magnitude EMA)
//   scale     = 0.3 / max(r_ema_abs, 1e-6)
//   x = a*scale*(N/2) + N/2 ,  y = r*scale*(N/2) + N/2   (clamped to grid)
//   cell      = cell*0.97 + min(1, |r|*scale)*0.03    (per-rec accumulate)
//   cell     *= 0.998 each frame                       (slow decay)
// The *(N/2) factor makes the spread resolution-independent, so the small web grid
// reads the same shape as the install's 512² window.

const N = 64;
const TOPO_DECAY = 0.998;
const GAMMA = 0.6; // canonical frag: pow(v, 0.6) lifts dim cells
const REF_H = 600;

export class MonMandala {
  // cells[x*N+y], x = a-axis, y = r-axis (same convention as screen_rain.ts).
  private cells = new Float32Array(N * N);
  private rEmaAbs = 0.01; // seed matches the C++ default

  update(_events: OscEvent[], ops: OpRec[], _dt: number): void {
    const cells = this.cells;

    // Pre-attention topology: plot each op's (a, r) into phase space.
    for (const rec of ops) {
      this.rEmaAbs = this.rEmaAbs * 0.9999 + Math.abs(rec.r) * 0.0001;
      const scale = 0.3 / Math.max(this.rEmaAbs, 1e-6);
      if (!Number.isFinite(rec.a) || !Number.isFinite(rec.r)) continue;
      const x = clampI(rec.a * scale * (N * 0.5) + N * 0.5, 0, N - 1);
      const y = clampI(rec.r * scale * (N * 0.5) + N * 0.5, 0, N - 1);
      const idx = x * N + y;
      cells[idx] = cells[idx] * 0.97 + Math.min(1, Math.abs(rec.r) * scale) * 0.03;
    }

    // Slow decay so the topology breathes rather than freezes.
    for (let i = 0; i < cells.length; i++) {
      const v = cells[i] * TOPO_DECAY;
      cells[i] = v < 1e-5 ? 0 : v;
    }
  }

  // Offscreen N×N buffer, nearest-neighbor upscaled (mirrors GL_NEAREST), same as
  // ScreenRain so the two heatmaps render identically.
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
        const raw = cells[gx * N + gy];
        const v = raw <= 0 ? 0 : Math.pow(Math.min(1, raw), GAMMA);
        const c = Math.round(v * 255);
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

    // Faint corner label, matching the aggregate convention.
    const s = h / REF_H;
    ctx.font = `${13 * s}px JetBrains Mono, ui-monospace, monospace`;
    ctx.textBaseline = "top";
    ctx.textAlign = "left";
    ctx.fillStyle = "rgba(31,31,31,1)";
    ctx.fillText("E  head mandala  ·  topology", x + 10 * s, y + 6 * s);
  }
}

function clampI(v: number, lo: number, hi: number): number {
  const i = Math.floor(v);
  return i < lo ? lo : i > hi ? hi : i;
}
