// Monitor B — residual-stream double helix. Replaces the i-index sweep ribbon
// (user decision 2026-05-31). This is the FAITHFUL analogue of the residual
// "manifold" helices in transformer-circuits' counting-task paper: per token
// position, the 768-d residual is averaged over the corpus and projected onto a
// fixed per-layer PCA basis -> 3 coords (the paper's denoising). Two readouts
// per position form the two strands:
//   strand A = post-attention residual
//   strand F = post-FFN residual (the layer's output state)
// Both share one PCA space per layer, so they sit nearly parallel and twist
// differently: an open, irregular double coil, not a clean DNA helix. The data
// is the precomputed mean manifold (residual_means.bin, exported from the HF
// bert-base-uncased pilot == the install weights), shipped as one global asset.
//
// Why precomputed means, not the live op-stream capture: a residual readout
// happens once per token position, but each position costs millions of op
// records, and the install rate-limits the op stream to ~1,092 ops/s. So a
// capture-length window advances the forward by only a token or two, far too
// little to trace a coil. The mean manifold gives the full L0..L11 development
// faithfully and on its own clock; the bert.c --residual-basis live path is kept
// for the physical install / future long captures (user decision 2026-05-31).
//
// Animation (user revision 2026-05-31): ONE segment at a time. The layer's coil
// is divided into SEG_COUNT contiguous segments; each is drawn ALONE in its true
// position within the (whole-coil) frame, held still for a long beat, then
// erased, then the next segment. Only ~30 samples are ever on screen at once, so
// each piece can be read on its own rather than as a tangle. After the last
// segment the layer steps L0 -> L11 and back (ping-pong) and segment 0 of the
// next layer begins. Camera scale is fixed across a layer so each piece sits
// where it belongs in the coil; the window appears to walk the coil piece by
// piece.
//
// Position 0 is dropped: BERT's first token is an attention-sink / massive-
// activation outlier whose axis coord (c0) is several times larger than the rest
// (e.g. L0 strand A pos0 c0=35.6 vs pos1 c0=3.4), which otherwise throws a long
// straight spike off one end. It is not part of the smooth positional coil, so
// the coil renders from position 1.
//
// Camera: slow 3D spin + fixed tilt, orthographic, with a spin-stable smoothed
// auto-fit over the whole current-layer coil. The spin FREEZES while a segment
// is actively drawing (so a stroke lands still), and turns only during the hold
// and erase.

const MAGIC = 0x52484c58;
// c0 (PC1, the position axis) spans a wider range than c1/c2; scale the axis
// down so the coil reads as an open helix rather than a thin needle. Cross-
// section coords pass at gain 1; overall size is handled by auto-fit.
const AXIS_GAIN = 0.26;
// The two strands are pulled toward their per-position midpoint by this factor
// (1 = true gap, <1 = tighter) so the double coil reads closer together without
// distorting either strand's own shape.
const STRAND_GAP = 0.7;
const ALPHA_BANDS = 12; // gradient quantised into N strokes (cf. ribbon)

// Segmented reveal. SEG_COUNT is the only knob for how the coil is chunked; it
// is read live each frame, so it can be retuned with no other changes.
const SEG_COUNT = 17; // segments per layer (~30 of ~509 points each)

// Timeline (seconds). All driven by the panel's playback dt, independent of the
// op-stream timeline. Per segment: SEG_DRAW_S to stroke it on (spin frozen),
// SEG_HOLD_S to hold it still (the long study beat, >= 10s per the user), then
// ERASE_S to recede it before the next segment. Tunable; visual review is the
// user's.
const SEG_DRAW_S = 0.5;
const SEG_HOLD_S = 10.0;
const ERASE_S = 0.8;

export interface HelixData {
  nLayers: number;
  nPos: number;
  positions: Int32Array; // token-position index of each sample
  a: Float32Array; // post-attn, layer-major [nLayers*nPos*3]
  f: Float32Array; // post-ffn
}

// Parse residual_means.bin (see export_means_asset.py). Returns null on a bad
// magic so a missing/old asset just leaves the panel blank rather than throwing.
export function parseHelixData(buf: ArrayBuffer): HelixData | null {
  const dv = new DataView(buf);
  if (buf.byteLength < 16 || dv.getUint32(0, true) !== MAGIC) return null;
  const nLayers = dv.getUint32(8, true);
  const nPos = dv.getUint32(12, true);
  let off = 16;
  const positions = new Int32Array(buf.slice(off, off + nPos * 4));
  off += nPos * 4;
  const span = nLayers * nPos * 3;
  const a = new Float32Array(buf.slice(off, off + span * 4));
  off += span * 4;
  const f = new Float32Array(buf.slice(off, off + span * 4));
  return { nLayers, nPos, positions, a, f };
}

type Phase = "draw" | "hold" | "erase";

export class ResidualHelix {
  spinVel = 0.035; // slow turn so the coil reads as 3D (rad/s); frozen while drawing
  tiltX = 0.34; // fixed camera tilt (rad)

  private data: HelixData | null = null;
  private spinAng = 0;
  private scaleSmooth = 0;

  // Animation clock.
  private layer = 0; // integer layer being shown
  private depthDir = 1; // ping-pong direction through layers
  private seg = 0; // current segment index 0..SEG_COUNT-1
  private phase: Phase = "draw";
  private phaseTimer = 0; // clock within the current phase
  private segFrac = 0; // 0..1 reveal of the current segment (draw up, erase down)

  setData(d: HelixData | null): void {
    this.data = d;
  }

  update(dt: number): void {
    const d = this.data;
    if (!d) return;
    const NL = d.nLayers;
    if (NL < 1 || d.nPos < 3) return;

    let drawing = false;
    if (this.phase === "draw") {
      drawing = true;
      this.phaseTimer += dt;
      this.segFrac = Math.min(1, this.phaseTimer / SEG_DRAW_S);
      if (this.segFrac >= 1) {
        this.phase = "hold";
        this.phaseTimer = 0;
      }
    } else if (this.phase === "hold") {
      this.segFrac = 1;
      this.phaseTimer += dt;
      if (this.phaseTimer >= SEG_HOLD_S) {
        this.phase = "erase";
        this.phaseTimer = 0;
      }
    } else {
      // erase: recede this segment, then advance to the next (or next layer).
      this.phaseTimer += dt;
      this.segFrac = Math.max(0, 1 - this.phaseTimer / ERASE_S);
      if (this.phaseTimer >= ERASE_S) {
        this.seg++;
        if (this.seg >= SEG_COUNT) {
          this.seg = 0;
          this.layer += this.depthDir;
          if (this.layer >= NL - 1) {
            this.layer = NL - 1;
            this.depthDir = -1;
          } else if (this.layer <= 0) {
            this.layer = 0;
            this.depthDir = 1;
          }
        }
        this.phase = "draw";
        this.phaseTimer = 0;
        this.segFrac = 0;
      }
    }

    if (!drawing) this.spinAng += this.spinVel * dt; // spin only when not stroking
  }

  // Coords of strand (0=a,1=f) at layer L, position index pi.
  private coord(strand: 0 | 1, pi: number, L: number): [number, number, number] {
    const d = this.data!;
    const arr = strand === 0 ? d.a : d.f;
    const o = (L * d.nPos + pi) * 3;
    return [arr[o], arr[o + 1], arr[o + 2]];
  }

  // Project a coil point (axis = vertical Y) with spin + tilt, orthographic.
  private project(c: [number, number, number], yCenter: number): [number, number] {
    const X = c[1]; // c1
    const Yc = c[0] * AXIS_GAIN - yCenter; // c0 = axis
    const Z = c[2]; // c2
    const cs = Math.cos(this.spinAng);
    const sn = Math.sin(this.spinAng);
    const x1 = X * cs + Z * sn;
    const z1 = -X * sn + Z * cs;
    const ct = Math.cos(this.tiltX);
    const st = Math.sin(this.tiltX);
    const y2 = Yc * ct - z1 * st;
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
    const d = this.data;
    if (!d) return;
    const P = d.nPos;
    if (P < 3) return;
    const L = this.layer;

    // Current segment's position span (position 0 dropped, so pi runs 1..P-1).
    const Ne = P - 1;
    const segSize = Math.max(1, Math.ceil(Ne / SEG_COUNT));
    const segStartPi = 1 + this.seg * segSize;
    const segEndPi = Math.min(P - 1, this.seg * segSize + segSize);
    if (segStartPi > P - 1) return;
    const span = segEndPi - segStartPi;
    // Draw grows the head forward (start -> end). Erase eats from the START
    // forward, leaving the END point as the anchor the next segment grows from,
    // so the active point always moves forward and never backtracks.
    let fromPi: number;
    let toPi: number;
    if (this.phase === "erase") {
      fromPi = segStartPi + Math.floor((1 - this.segFrac) * span);
      toPi = segEndPi;
    } else {
      fromPi = segStartPi;
      toPi = segStartPi + Math.floor(this.segFrac * span);
    }

    // Contract both strands toward their per-position midpoint (STRAND_GAP) so
    // the coils sit closer, and track the axial centre over the WHOLE current-
    // layer coil so the frame sits still as the window walks it piece by piece.
    const ca3: [number, number, number][] = new Array(P);
    const cf3: [number, number, number][] = new Array(P);
    let ymin = Infinity;
    let ymax = -Infinity;
    for (let pi = 1; pi < P; pi++) {
      const a3 = this.coord(0, pi, L);
      const f3 = this.coord(1, pi, L);
      const mx = (a3[0] + f3[0]) * 0.5;
      const my = (a3[1] + f3[1]) * 0.5;
      const mz = (a3[2] + f3[2]) * 0.5;
      const ai: [number, number, number] = [
        mx + (a3[0] - mx) * STRAND_GAP,
        my + (a3[1] - my) * STRAND_GAP,
        mz + (a3[2] - mz) * STRAND_GAP,
      ];
      const fi: [number, number, number] = [
        mx + (f3[0] - mx) * STRAND_GAP,
        my + (f3[1] - my) * STRAND_GAP,
        mz + (f3[2] - mz) * STRAND_GAP,
      ];
      ca3[pi] = ai;
      cf3[pi] = fi;
      const ya = ai[0] * AXIS_GAIN;
      const yf = fi[0] * AXIS_GAIN;
      if (ya < ymin) ymin = ya;
      if (ya > ymax) ymax = ya;
      if (yf < ymin) ymin = yf;
      if (yf > ymax) ymax = yf;
    }
    const yCenter = (ymin + ymax) * 0.5;

    // Project the whole coil (stable whole-coil auto-fit), keep for rendering.
    const pa: [number, number][] = new Array(P);
    const pf: [number, number][] = new Array(P);
    let halfw = 1e-3;
    let halfh = 1e-3;
    for (let pi = 1; pi < P; pi++) {
      pa[pi] = this.project(ca3[pi], yCenter);
      pf[pi] = this.project(cf3[pi], yCenter);
      halfw = Math.max(halfw, Math.abs(pa[pi][0]), Math.abs(pf[pi][0]));
      halfh = Math.max(halfh, Math.abs(pa[pi][1]), Math.abs(pf[pi][1]));
    }

    const fill = 0.9; // bolder use of the right-half region
    const targetScale = fill * Math.min((w * 0.5) / halfw, (h * 0.5) / halfh);
    this.scaleSmooth =
      this.scaleSmooth <= 0
        ? targetScale
        : this.scaleSmooth + (targetScale - this.scaleSmooth) * 0.08;
    const scale = this.scaleSmooth;
    const cx0 = ox + w * 0.5;
    const cy0 = oy + h * 0.5;
    const mapX = (x: number): number => cx0 + x * scale;
    const mapY = (y: number): number => cy0 + y * scale;

    // Device-ish scale so weight tracks panel size. Lines are 2x the old 1px;
    // every revealed sample gets a small dot to show the discretisation.
    const dpx = Math.max(1, h / 520);
    ctx.lineWidth = 1; // back to the original thin stroke (user 2026-05-31)
    const dotR = 1.3 * dpx;
    const headR = 2.2 * dpx;
    const denom = P - 1;

    // Stroke + square-mark the revealed range [lo..hi]. Squares (not discs) to
    // match the project's wireframe vocabulary.
    const strokeStrand = (
      proj: [number, number][],
      aGain: number,
      lo: number,
      hi: number,
    ): void => {
      let i = lo + 1;
      while (i <= hi) {
        const band = Math.min(ALPHA_BANDS - 1, Math.floor(((i - 1) / denom) * ALPHA_BANDS));
        const a = (0.1 + 0.62 * ((band + 0.5) / ALPHA_BANDS)) * aGain;
        ctx.strokeStyle = `rgba(225,225,225,${a})`;
        ctx.beginPath();
        ctx.moveTo(mapX(proj[i - 1][0]), mapY(proj[i - 1][1]));
        let j = i;
        while (
          j <= hi &&
          Math.min(ALPHA_BANDS - 1, Math.floor(((j - 1) / denom) * ALPHA_BANDS)) === band
        ) {
          ctx.lineTo(mapX(proj[j][0]), mapY(proj[j][1]));
          j++;
        }
        ctx.stroke();
        i = j;
      }
      // A small square at every revealed sample.
      for (let pi = lo; pi <= hi; pi++) {
        const band = Math.min(ALPHA_BANDS - 1, Math.floor((pi / denom) * ALPHA_BANDS));
        const a = (0.1 + 0.62 * ((band + 0.5) / ALPHA_BANDS)) * aGain;
        ctx.fillStyle = `rgba(225,225,225,${Math.min(1, a * 1.3)})`;
        ctx.fillRect(mapX(proj[pi][0]) - dotR, mapY(proj[pi][1]) - dotR, dotR * 2, dotR * 2);
      }
      // Brighter square head at the forward-most sample (toPi). During erase toPi
      // stays at the segment end, so the leading marker never backtracks and the
      // next segment continues from there.
      const head = proj[toPi];
      ctx.fillStyle = `rgba(245,245,245,${0.95 * aGain})`;
      ctx.fillRect(mapX(head[0]) - headR, mapY(head[1]) - headR, headR * 2, headR * 2);
    };
    // Strand F (layer output) a touch brighter so the coils read apart.
    strokeStrand(pa, 0.78 * baseAlpha, fromPi, toPi);
    strokeStrand(pf, 1.0 * baseAlpha, fromPi, toPi);
  }
}
