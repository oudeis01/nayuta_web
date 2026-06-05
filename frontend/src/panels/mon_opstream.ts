import type { OscEvent, OpRec } from "../stream/types";
import { ResidualHelix, type HelixData } from "../geometry/residual_helix";
import { LogQueue } from "./log_queue";

// Monitor B — Op Stream. Faithful port of graphics_consumer/src/screens/
// mon_opstream.cpp. Every captured OpRec (~1,092/s, already rate-limited at the
// source) becomes one scrolling log line; the newest sits at the bottom. Axis
// OSC events insert separators: TokenAtt -> "-- HEAD COMPLETE --", Layer ->
// "=== LAYER n COMPLETE ===" then a brief pause + clear. Those axis paths are
// absent from the Phase-1 captures, so that logic stays dormant here.
//
// The right half holds the residual-stream double helix (post-attn vs
// post-FFN PCA coils — see geometry/residual_helix.ts); the scrolling log
// occupies the left column. (This margin held the i-index sweep ribbon, and
// before that the structural Helix Side view; the Helix now lives only on
// Monitor A. Ribbon -> residual helix is the user decision 2026-05-31.)

const REF_H = 600;
const LH_NA = 0xff;
const POS_NA = 0xffff;
// Visible scrollback. The C++ deque keeps 2000 for its own scrollbar, but this
// port only paints the tail that fits, so a smaller cap is plenty and keeps
// high-speed replay from formatting lines that would scroll straight off.
const LOG_CAP = 512;
const PAUSE_AFTER_LAYER = 0.5;
// Paced-reveal tuning (plan B-1; first-pass defaults, user fine-tunes visually).
// The ceiling is deliberately well below the ~1,092 lines/s arrival rate so the
// reveal stays a calm few-lines-per-frame scroll that samples the stream, rather
// than catching all the way up (which would just reproduce the wholesale-replace
// feel B-1 reported). The flush sheds the rest and bounds latency.
const PACE_RATE = 60; // baseline reveal floor, lines/s (~one line/frame @ 60fps)
const PACE_DRAIN = 0.5; // catch-up time constant, s (reveal ~ backlog/drain)
const PACE_MAXRATE = 240; // real-time reveal ceiling, lines/s (~4/frame)
const PACE_FLUSHAT = 480; // pending cap; oldest beyond it are dropped (~2s latency)

// 9-char fixed-width names, indexed by OpType (must match op_types.h).
const OP_TYPE_NAMES = [
  "EMB_POS  ", "EMB_TT   ", "LN_MEAN  ", "LN_VAR   ", "LN_RSQRT ",
  "LN_NORM  ", "LN_GAMMA ", "LN_BETA  ", "MUL_QKV  ", "ADD_BIAS ",
  "MUL_QK   ", "ATT_SCALE", "SOFTM_EXP", "SOFTM_SUM", "SOFTM_DIV",
  "MUL_AV   ", "MUL_OUT  ", "BIAS_OUT ", "RES_ATTN ", "MUL_FFN_U",
  "BIAS_FFNU", "GELU     ", "MUL_FFND ", "BIAS_FFND", "RES_FFN  ",
];

function opSym(t: number): string {
  switch (t) {
    case 10: case 15: case 8: case 16: case 19: case 22:
      return "x";
    case 0: case 1: case 9: case 17: case 20: case 23: case 18: case 24:
      return "+";
    case 14: case 11:
      return "/";
    case 12:
      return "e^";
    case 21:
      return "~";
    default:
      return " .";
  }
}

// Queue items are stored raw and formatted only when painted, so lines that
// overflow the pace flush at high replay speed never cost a fmtRec() call.
// A RecItem is one op; a SepItem is an axis separator (weak = head complete,
// strong = layer complete).
interface RecItem {
  rec: OpRec;
}
interface SepItem {
  text: string;
  strong: boolean;
}
type Item = RecItem | SepItem;

function isSep(it: Item): it is SepItem {
  return (it as SepItem).text !== undefined;
}

// "%+8.4f": signed, 4 decimals, min field width 8.
function fmtF(n: number): string {
  const s = (n >= 0 ? "+" : "-") + Math.abs(n).toFixed(4);
  return s.padStart(8);
}
function pad2(n: number): string {
  return String(n).padStart(2, "0");
}
function pad3(n: number): string {
  return String(n).padStart(3, " ");
}

export class MonOpStream {
  // fps-paced scrolling log (plan B-1): pending ops reveal one line at a time,
  // catching up under backlog and flushing on overflow, so the stream scrolls
  // instead of being replaced wholesale each frame.
  private logq = new LogQueue<Item>({
    cap: LOG_CAP,
    pace: { rate: PACE_RATE, drain: PACE_DRAIN, maxRate: PACE_MAXRATE, flushAt: PACE_FLUSHAT },
  });
  private layer = 0;
  private paused = false;
  private pauseTimer = 0;
  private blink = 0;

  // Background residual double helix (right margin), driven by the precomputed
  // mean manifold on its own clock (see geometry/residual_helix.ts).
  private helix = new ResidualHelix();

  setHelixData(d: HelixData | null): void {
    this.helix.setData(d);
  }

  private fmtRec(r: OpRec): string {
    const lStr = r.layer !== LH_NA ? pad2(r.layer) : "--";
    const hStr = r.head !== LH_NA ? pad2(r.head) : "--";
    const qValid = (r.flags & 1) !== 0 && r.qPos !== POS_NA;
    const kValid = (r.flags & 2) !== 0 && r.kPos !== POS_NA;
    const qStr = qValid ? pad3(r.qPos) : "  -";
    const kStr = kValid ? pad3(r.kPos) : "  -";
    const name = r.opType < 25 ? OP_TYPE_NAMES[r.opType] : "UNKNOWN  ";
    const sym = opSym(r.opType);
    return `L${lStr}.H${hStr}  ${name}  t(${qStr},${kStr})  ${fmtF(r.a)} ${sym} ${fmtF(r.b)} = ${fmtF(r.r)}`;
  }

  update(events: OscEvent[], ops: OpRec[], dt: number): void {
    this.blink += dt;

    // Helix background develops on its own clock regardless of the log pause.
    this.helix.update(dt);

    if (this.paused) {
      this.pauseTimer -= dt;
      if (this.pauseTimer <= 0) {
        this.paused = false;
        this.logq.clear();
      }
    }

    for (const ev of events) {
      if (ev.path === "/bert/token_att") {
        this.logq.enqueue({ text: "-- HEAD COMPLETE --", strong: false });
      } else if (ev.path === "/bert/layer") {
        this.layer = ev.args[0];
        this.logq.enqueue({ text: `=== LAYER ${this.layer} COMPLETE ===`, strong: true });
        this.paused = true;
        this.pauseTimer = PAUSE_AFTER_LAYER;
      }
    }

    if (!this.paused) {
      // Enqueue raw ops (formatted lazily on paint). Skip the leading lines that
      // the pace flush would immediately drop when a single batch overflows.
      const start = Math.max(0, ops.length - PACE_FLUSHAT);
      for (let i = start; i < ops.length; i++) this.logq.enqueue({ rec: ops[i] });
    }

    // dt>0 gates the scroll so it holds in place while playback is paused.
    this.logq.tick(dt > 0);
  }

  render(ctx: CanvasRenderingContext2D, x: number, y: number, w: number, h: number): void {
    const s = h / REF_H;
    const fontSM = 13 * s;
    const fontMD = 12 * s;
    const mono = "JetBrains Mono, ui-monospace, monospace";

    ctx.fillStyle = "#000";
    ctx.fillRect(x, y, w, h);
    ctx.textBaseline = "top";
    ctx.textAlign = "left";

    // Top-left label
    ctx.font = `${fontSM}px ${mono}`;
    ctx.fillStyle = "rgba(51,51,51,1)";
    ctx.fillText("B  op_stream", x + 10 * s, y + 6 * s);

    const padX = 8 * s;
    const headerH = 28 * s;
    const top = y + headerH;
    const lineH = fontMD * 1.4;
    const bottom = y + h - 8 * s;
    const rows = Math.max(0, Math.floor((bottom - top) / lineH));

    // Residual double helix in the right half (isolated canvas, fits + centres
    // itself within this region). The log occupies the left column.
    const helixW = w * 0.5;
    this.helix.draw(ctx, x + w - helixW, top, helixW, h - headerH - 8 * s);

    // Scrolling log in the left column; clip so long lines don't reach the helix.
    const logW = w - helixW - 16 * s;
    ctx.save();
    ctx.beginPath();
    ctx.rect(x, top, logW, bottom - top);
    ctx.clip();

    // Paint the visible tail bottom-up so the newest line sits at the bottom.
    ctx.font = `${fontMD}px ${mono}`;
    const lines = this.logq.lines;
    const n = Math.min(rows, lines.length);
    for (let i = 0; i < n; i++) {
      const item = lines[lines.length - 1 - i];
      const ly = bottom - lineH * (i + 1);
      let text: string;
      if (isSep(item)) {
        ctx.fillStyle = item.strong ? "rgba(255,255,255,0.9)" : "rgba(89,89,89,1)";
        text = item.text;
      } else {
        ctx.fillStyle = "rgba(217,217,217,1)";
        text = this.fmtRec(item.rec);
      }
      ctx.fillText(text, x + padX, ly);
    }

    // Idle cursor — blinks at ~0.9s period when no data has arrived.
    if (lines.length === 0 && this.blink % 0.9 < 0.45) {
      ctx.fillStyle = "rgba(115,115,115,1)";
      ctx.fillText("_", x + padX, top);
    }
    ctx.restore();
  }
}
