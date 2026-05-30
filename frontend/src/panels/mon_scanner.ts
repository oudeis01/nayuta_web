import type { OscEvent, OpRec } from "../stream/types";

// Monitor A — Token Scanner. Faithful port of graphics_consumer/src/screens/
// mon_scanner.cpp. The top third shows the sentence being processed (BERT
// wordpieces, the current query token inverted); the bottom two thirds is a
// sampled op log. Tokens arrive via /bert/embed (pos, token_id) resolved
// through the BERT vocab; op records drive the query position, sub-progress
// bar, and cadence/anomaly log lines.
//
// Data-availability notes for the Phase-1 captures (cf. the panel data split):
//   - /bert/sequence_start is absent, so n_tokens is inferred from the highest
//     embed pos seen (the sentence fills in over ~42 min at 1x as embeds
//     trickle).
//   - /bert/att is absent, so the k_pos outline and the "q()*k()" attention
//     pair log lines never fire. To keep the query highlight from freezing on
//     the first token (the C++ seeds q_pos from ZMQ only once, then relies on
//     Att), this port tracks the live query position from the op stream. That
//     is the one intentional deviation from the C++ source.

const REF_H = 600;
const MAX_SEQ = 512;
const LOG_MAX = 300;
const LOG_SAMPLE_N = 300;
const LH_NA = 0xff;
const POS_NA = 0xffff;
const DEFAULT_TOKEN_OPS = 1769472; // ops per token before the first measured span

const OP_TYPE_NAMES = [
  "EMB_POS  ", "EMB_TT   ", "LN_MEAN  ", "LN_VAR   ", "LN_RSQRT ",
  "LN_NORM  ", "LN_GAMMA ", "LN_BETA  ", "MUL_QKV  ", "ADD_BIAS ",
  "MUL_QK   ", "ATT_SCALE", "SOFTM_EXP", "SOFTM_SUM", "SOFTM_DIV",
  "MUL_AV   ", "MUL_OUT  ", "BIAS_OUT ", "RES_ATTN ", "MUL_FFN_U",
  "BIAS_FFNU", "GELU     ", "MUL_FFND ", "BIAS_FFND", "RES_FFN  ",
];

function fmtF4(n: number): string {
  return (n >= 0 ? "+" : "-") + Math.abs(n).toFixed(4);
}
function fmtF5(n: number): string {
  return (n >= 0 ? "+" : "-") + Math.abs(n).toFixed(5);
}

interface TokenBox {
  disp: string;
  special: boolean; // raw started with '[' (e.g. [CLS]) -> extra dim
  x: number;
  y: number;
  boxW: number;
}

export class MonScanner {
  private vocab: string[] = [];
  private seqWords = new Array<string>(MAX_SEQ).fill("");
  private nTokens = 0;
  private qPos = -1;
  private kPos = -1;
  private layer = 0;
  private head = 0;

  private log: string[] = [];

  // op-stream sub-progress
  private zmqQPos = -1;
  private opsSinceQ = 0;
  private prevTokenOps = DEFAULT_TOKEN_OPS;
  private subProgress = 0;

  // op-stream log sampling + anomaly stats (Welford)
  private logSampleCounter = 0;
  private rMean = 0;
  private rM2 = 0;
  private rCount = 0;

  // highlight oscillation
  private rEma = 0;
  private rEmaPeak = 0.001;

  private time = 0;

  // sentence layout cache (rebuilt only when content / font / width change)
  private layout: TokenBox[] = [];
  private layoutDirty = true;
  private layoutFs = 0;
  private layoutW = 0;

  setVocab(vocab: string[]): void {
    this.vocab = vocab;
  }

  private tokWord(id: number): string {
    const w = this.vocab[id];
    return w && w.length ? w : `[${id}]`;
  }

  private pushLog(text: string): void {
    this.log.push(text);
    if (this.log.length > LOG_MAX) this.log.shift();
  }

  update(events: OscEvent[], ops: OpRec[], dt: number): void {
    this.time += dt;

    for (const ev of events) {
      switch (ev.path) {
        case "/bert/sequence_start":
          this.nTokens = ev.args[2];
          this.qPos = -1;
          this.kPos = -1;
          this.seqWords.fill("");
          this.log.length = 0;
          this.zmqQPos = -1;
          this.opsSinceQ = 0;
          this.rCount = 0;
          this.rMean = 0;
          this.rM2 = 0;
          this.layoutDirty = true;
          break;
        case "/bert/embed": {
          const pos = ev.args[0];
          if (pos >= 0 && pos < MAX_SEQ) {
            this.seqWords[pos] = this.tokWord(ev.args[1]);
            // Infer sentence length from the highest position seen, since
            // sequence_start (which would set n_tokens) is not captured.
            if (pos + 1 > this.nTokens) this.nTokens = pos + 1;
            this.layoutDirty = true;
          }
          break;
        }
        // /bert/att is absent from the captures; kept for faithfulness/live use.
        case "/bert/att": {
          this.layer = ev.args[0];
          this.head = ev.args[1];
          this.qPos = ev.args[2];
          this.kPos = ev.args[3];
          const qw = this.qPos >= 0 && this.qPos < this.nTokens ? this.seqWords[this.qPos] : "?";
          const kw = this.kPos >= 0 && this.kPos < this.nTokens ? this.seqWords[this.kPos] : "?";
          this.pushLog(`[L${pad2(this.layer)}][H${pad2(this.head)}]  q(${qw}) * k(${kw}) = ${fmtF4(ev.args[4])}`);
          break;
        }
        default:
          break;
      }
    }

    for (const rec of ops) {
      const absR = Math.abs(rec.r);

      this.rEma = this.rEma * 0.99 + absR * 0.01;
      if (this.rEma > this.rEmaPeak) this.rEmaPeak = this.rEma;

      this.rCount++;
      const delta = absR - this.rMean;
      this.rMean += delta / this.rCount;
      this.rM2 += delta * (absR - this.rMean);

      const qValid = (rec.flags & 1) !== 0 && rec.qPos !== POS_NA;
      if (qValid && rec.qPos !== this.zmqQPos) {
        if (this.zmqQPos >= 0 && this.opsSinceQ > 100) this.prevTokenOps = this.opsSinceQ;
        this.zmqQPos = rec.qPos;
        this.opsSinceQ = 0;
      }
      this.opsSinceQ++;
      this.subProgress = Math.min(1, this.opsSinceQ / this.prevTokenOps);

      // Track the live query token from the op stream (deviation: the C++ seeds
      // this once then defers to /bert/att, which these captures lack).
      if (qValid) this.qPos = rec.qPos;
      if (rec.layer !== LH_NA) this.layer = rec.layer;
      if (rec.head !== LH_NA) this.head = rec.head;

      this.logSampleCounter++;
      if (this.logSampleCounter >= LOG_SAMPLE_N) {
        this.logSampleCounter = 0;
        const name = rec.opType < 25 ? OP_TYPE_NAMES[rec.opType] : "???";
        const l = rec.layer !== LH_NA ? rec.layer : 0;
        const q = qValid ? rec.qPos : 0;
        this.pushLog(`[${name}] L${pad2(l)} t(${pad3(q)}) ${fmtF4(rec.a)} x ${fmtF4(rec.b)} = ${fmtF5(rec.r)}`);
      }

      if (this.rCount > 1000) {
        const variance = this.rM2 / (this.rCount - 1);
        const stdDev = Math.sqrt(Math.max(0, variance));
        if (absR > this.rMean + 3 * stdDev) {
          const name = rec.opType < 25 ? OP_TYPE_NAMES[rec.opType] : "???";
          const l = rec.layer !== LH_NA ? rec.layer : 0;
          const q = qValid ? rec.qPos : 0;
          this.pushLog(`[!] R=${fmtF4(rec.r)} ${name.trim()} L${pad2(l)} t(${q})`);
        }
      }
    }
  }

  // Rebuild the wrapped token layout. Wordpiece continuations ("##xx") attach to
  // the previous token with no leading gap and the marker stripped.
  private rebuildLayout(ctx: CanvasRenderingContext2D, fs: number, ox: number, aw: number, s: number): void {
    const padX = 3 * s;
    const gap = 4 * s;
    const lineH = fs + 2 * 2 * s + 4 * s;
    const mono = "JetBrains Mono, ui-monospace, monospace";
    ctx.font = `${fs}px ${mono}`;

    this.layout = [];
    let x = ox + 8 * s;
    let y = 4 * s;
    for (let i = 0; i < this.nTokens && i < MAX_SEQ; i++) {
      const raw = this.seqWords[i] || "?";
      const isWp = raw.startsWith("##");
      const disp = isWp ? raw.slice(2) : raw;
      const boxW = ctx.measureText(disp).width + padX * 2;
      let before = isWp ? 0 : gap;
      if (x + before + boxW > ox + aw - 8 * s && i > 0) {
        x = ox + 8 * s;
        y += lineH;
        before = 0;
      }
      x += before;
      this.layout.push({ disp, special: raw[0] === "[", x, y, boxW });
      x += boxW;
    }
    this.layoutDirty = false;
    this.layoutFs = fs;
    this.layoutW = aw;
  }

  render(ctx: CanvasRenderingContext2D, x: number, y: number, w: number, h: number): void {
    const s = h / REF_H;
    const fontSM = 13 * s;
    const fontMD = 13 * s;
    const padY = 2 * s;
    const mono = "JetBrains Mono, ui-monospace, monospace";

    ctx.fillStyle = "#000";
    ctx.fillRect(x, y, w, h);
    ctx.textBaseline = "top";
    ctx.textAlign = "left";

    // Top label + axis readout
    ctx.font = `${fontSM}px ${mono}`;
    ctx.fillStyle = "rgba(51,51,51,1)";
    ctx.fillText("A  scanner", x + 8 * s, y + 6 * s);
    const axis = `L${pad2(this.layer)} H${pad2(this.head)}`;
    const aw = ctx.measureText(axis).width;
    ctx.fillText(axis, x + w - aw - 8 * s, y + 6 * s);

    // ── Sentence area (top 35%) ───────────────────────────────────────────────
    const sentTop = y + 22 * s;
    const sentH = h * 0.35;
    const dividerY = sentTop + sentH;

    if (this.nTokens > 0) {
      if (this.layoutDirty || this.layoutFs !== fontMD || this.layoutW !== w) {
        this.rebuildLayout(ctx, fontMD, x, w, s);
      }
      ctx.font = `${fontMD}px ${mono}`;
      const rNorm = this.rEmaPeak > 0 ? this.rEma / this.rEmaPeak : 0;
      const boxH = fontMD + padY * 2;
      for (let i = 0; i < this.layout.length; i++) {
        const tb = this.layout[i];
        const tx = tb.x;
        const ty = sentTop + tb.y;
        if (ty + boxH > dividerY) break; // clip to the sentence area
        if (i === this.qPos) {
          const breath = 0.88 + 0.12 * (0.5 + 0.5 * Math.sin(this.time * 1.8 + rNorm * 6.28));
          const fill = Math.round(255 * breath);
          ctx.fillStyle = `rgb(${fill},${fill},${fill})`;
          ctx.fillRect(tx, ty, tb.boxW, boxH);
          ctx.fillStyle = "#000";
          ctx.fillText(tb.disp, tx + 3 * s, ty + padY);
        } else if (i === this.kPos) {
          ctx.strokeStyle = "rgba(255,255,255,0.78)";
          ctx.lineWidth = 1;
          ctx.strokeRect(tx + 0.5, ty + 0.5, tb.boxW, boxH);
          ctx.fillStyle = "rgba(220,220,220,1)";
          ctx.fillText(tb.disp, tx + 3 * s, ty + padY);
        } else {
          const b = tb.special ? 60 : 140;
          ctx.fillStyle = `rgb(${b},${b},${b})`;
          ctx.fillText(tb.disp, tx + 3 * s, ty + padY);
        }
      }
    }

    // Sub-progress bar (full width, just above the divider)
    if (this.subProgress > 0.005) {
      const barH = 3 * s;
      const m = w * 0.03;
      const x0 = x + m;
      const fullW = w - m * 2;
      const by = dividerY - barH - 2 * s;
      ctx.fillStyle = "rgba(20,20,20,1)";
      ctx.fillRect(x0, by, fullW, barH);
      ctx.fillStyle = "rgba(180,180,180,1)";
      ctx.fillRect(x0, by, fullW * this.subProgress, barH);
    }

    // Divider line
    ctx.fillStyle = "rgba(30,30,30,1)";
    ctx.fillRect(x + 8 * s, Math.round(dividerY) + 0.5, w - 16 * s, 1);

    // ── Log area (bottom 65%) — newest at the bottom ──────────────────────────
    const logTop = dividerY + 6 * s;
    const logBottom = y + h - 6 * s;
    const lineH = fontMD * 1.4;
    const rows = Math.max(0, Math.floor((logBottom - logTop) / lineH));

    ctx.font = `${fontMD}px ${mono}`;
    if (this.log.length === 0) {
      if (this.time % 0.9 < 0.45) {
        ctx.fillStyle = "rgba(115,115,115,1)";
        ctx.fillText("_", x + 8 * s, logTop);
      }
    } else {
      ctx.fillStyle = "rgba(153,153,153,1)";
      const n = Math.min(rows, this.log.length);
      for (let i = 0; i < n; i++) {
        const line = this.log[this.log.length - 1 - i];
        ctx.fillText(line, x + 8 * s, logBottom - lineH * (i + 1));
      }
    }
  }
}

function pad2(n: number): string {
  return String(n).padStart(2, "0");
}
function pad3(n: number): string {
  return String(n).padStart(3, " ");
}
