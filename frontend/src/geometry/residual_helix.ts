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
// Animation (render model X): an intro DRAW-ON sweeps the coil into existence at
// layer 0 (position 0..N), then the fully-revealed coil MORPHS through depth
// L0 -> L11 and back, each point migrating in place to the next layer's
// geometry. First/last layers separate the two strands; mid layers (5-7) nearly
// merge them (the measured strand-separation profile), so divergence is data.
//
// Camera (spin, tilt, orthographic, smoothed auto-fit) mirrors geometry/ribbon
// so the panel reads as a sibling of the other monochrome wireframes.

const MAGIC = 0x52484c58;
// c0 (PC1, the position axis) spans a wider range than c1/c2; scale the axis
// down so the coil reads as an open helix rather than a thin needle. Cross-
// section coords pass at gain 1; overall size is handled by auto-fit.
const AXIS_GAIN = 0.26;
const ALPHA_BANDS = 12; // gradient quantised into N strokes (cf. ribbon)

// Timeline (seconds). All driven by the panel's playback dt, independent of the
// op-stream timeline (which barely advances). Tunable; visual review is the
// user's. INTRO reveals positions; then DEVELOP morphs one layer step per
// (MORPH + HOLD); at the top it ping-pongs back down so the coil always breathes
// through depth without a hard reset.
const INTRO_S = 6.0;
const MORPH_S = 2.6;
const HOLD_S = 0.8;

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

function lerp(p: number, q: number, t: number): number {
  return p + (q - p) * t;
}

export class ResidualHelix {
  spinVel = 0.14; // slow turn so the coil reads as 3D (rad/s), sibling of Ribbon
  tiltX = 0.34; // fixed camera tilt (rad)

  private data: HelixData | null = null;
  private spinAng = 0;
  private scaleSmooth = 0;

  // Animation clock.
  private introT = 0; // 0..INTRO_S draw-on progress
  private layer = 0; // integer base layer (explicit so morph frac never rounds it)
  private depth = 0; // float layer position for rendering, 0..nLayers-1
  private depthDir = 1; // ping-pong direction through depth
  private stepT = 0; // 0..(MORPH_S+HOLD_S) within the current layer step

  setData(d: HelixData | null): void {
    this.data = d;
  }

  update(dt: number): void {
    this.spinAng += this.spinVel * dt;
    if (!this.data) return;

    if (this.introT < INTRO_S) {
      this.introT += dt;
      return; // hold depth at 0 until the coil has drawn on
    }

    // Advance the per-layer step clock; each completed step commits one integer
    // layer of travel, ping-ponging at the ends so the coil migrates up then
    // back down. `layer` is kept as an explicit integer so the morph fraction
    // can never round it forward mid-step.
    const NL = this.data.nLayers;
    if (NL <= 1) return;
    this.stepT += dt;
    const step = MORPH_S + HOLD_S;
    while (this.stepT >= step) {
      this.stepT -= step;
      this.layer += this.depthDir;
      if (this.layer >= NL - 1) {
        this.layer = NL - 1;
        this.depthDir = -1;
      } else if (this.layer <= 0) {
        this.layer = 0;
        this.depthDir = 1;
      }
    }
    // depth = base layer + eased morph toward the next layer over the MORPH
    // portion of the step (the HOLD portion sits flat at the integer layer).
    const frac = Math.min(1, this.stepT / MORPH_S);
    const target = Math.max(0, Math.min(NL - 1, this.layer + this.depthDir));
    this.depth = this.layer + (target - this.layer) * frac;
  }

  // Coords of strand (0=a,1=f) at layer L, position index pi, morphed to L+dir.
  private coord(
    strand: 0 | 1,
    pi: number,
    L0: number,
    L1: number,
    frac: number,
  ): [number, number, number] {
    const d = this.data!;
    const arr = strand === 0 ? d.a : d.f;
    const o0 = (L0 * d.nPos + pi) * 3;
    const o1 = (L1 * d.nPos + pi) * 3;
    return [
      lerp(arr[o0], arr[o1], frac),
      lerp(arr[o0 + 1], arr[o1 + 1], frac),
      lerp(arr[o0 + 2], arr[o1 + 2], frac),
    ];
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

    // Draw-on cursor: positions 0..reveal are shown (full once intro completes).
    const reveal =
      this.introT >= INTRO_S
        ? P
        : Math.max(2, Math.floor((this.introT / INTRO_S) * P));
    const last = Math.min(reveal, P) - 1;
    if (last < 1) return;

    const L0 = Math.floor(this.depth);
    const L1 = Math.min(d.nLayers - 1, L0 + 1);
    const frac = this.depth - L0;

    // Build both strands' 3D coords, find axial centre, then project.
    const A: [number, number, number][] = new Array(last + 1);
    const F: [number, number, number][] = new Array(last + 1);
    let ymin = Infinity;
    let ymax = -Infinity;
    for (let i = 0; i <= last; i++) {
      A[i] = this.coord(0, i, L0, L1, frac);
      F[i] = this.coord(1, i, L0, L1, frac);
      const ya = A[i][0] * AXIS_GAIN;
      const yf = F[i][0] * AXIS_GAIN;
      if (ya < ymin) ymin = ya;
      if (ya > ymax) ymax = ya;
      if (yf < ymin) ymin = yf;
      if (yf > ymax) ymax = yf;
    }
    const yCenter = (ymin + ymax) * 0.5;

    const pa: [number, number][] = new Array(last + 1);
    const pf: [number, number][] = new Array(last + 1);
    let halfw = 1e-3;
    let halfh = 1e-3;
    for (let i = 0; i <= last; i++) {
      pa[i] = this.project(A[i], yCenter);
      pf[i] = this.project(F[i], yCenter);
      halfw = Math.max(halfw, Math.abs(pa[i][0]), Math.abs(pf[i][0]));
      halfh = Math.max(halfh, Math.abs(pa[i][1]), Math.abs(pf[i][1]));
    }

    const fill = 0.8;
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

    // Two open strands. Alpha ramps low-pos (dim) -> high-pos (bright) in
    // ALPHA_BANDS contiguous strokes. Strand F (layer output) is a touch brighter
    // so the coils read apart in the monochrome palette.
    ctx.lineWidth = 1;
    const strokeStrand = (proj: [number, number][], aGain: number): void => {
      let i = 1;
      while (i <= last) {
        const band = Math.min(ALPHA_BANDS - 1, Math.floor((i / last) * ALPHA_BANDS));
        const a = (0.08 + 0.62 * ((band + 0.5) / ALPHA_BANDS)) * aGain;
        ctx.strokeStyle = `rgba(225,225,225,${a})`;
        ctx.beginPath();
        ctx.moveTo(mapX(proj[i - 1][0]), mapY(proj[i - 1][1]));
        let j = i;
        while (
          j <= last &&
          Math.min(ALPHA_BANDS - 1, Math.floor((j / last) * ALPHA_BANDS)) === band
        ) {
          ctx.lineTo(mapX(proj[j][0]), mapY(proj[j][1]));
          j++;
        }
        ctx.stroke();
        i = j;
      }
      const head = proj[last];
      ctx.fillStyle = `rgba(245,245,245,${0.95 * aGain})`;
      ctx.beginPath();
      ctx.arc(mapX(head[0]), mapY(head[1]), 1.8, 0, Math.PI * 2);
      ctx.fill();
    };
    strokeStrand(pa, 0.78 * baseAlpha);
    strokeStrand(pf, 1.0 * baseAlpha);
  }
}
