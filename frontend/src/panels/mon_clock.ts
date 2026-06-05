import type { OscEvent, OpRec } from "../stream/types";

// Monitor C — The Clock. Faithful port of graphics_consumer/src/screens/mon_clock.cpp
// (the canonical source; action plan 8-5 names C++ as ground truth).
//
// Shows the op counter as a slot-machine scramble plus a remaining-time estimate
// rendered twice: the double-precision "main" (EMA-smoothed, ratcheted down) and
// a 32-bit "ghost" recomputed from elapsed round-tripped through float32. The
// gap between them is the representation error, made visible by the ε readout
// and the quantization gauge. cf. Counter (2024) — digital finitude.

const MAX_DIGITS = 14;
const OP_TOTAL = 5483624515190808.0;
const YR_SEC = 365.25 * 86400.0;
const REF_H = 600; // reference panel height the pixel constants were tuned for

// op_count count-up (C-2, user 2026-06-05): a monotonic ease-out toward the latest
// target — no random slot-machine scramble, no mid-pause. Speed is proportional to
// the remaining distance (ease-out); a hard cap snaps any single catch-up running
// longer than COUNT_CAP so the asymptote never lingers.
const COUNT_TAU = 0.3; // ease-out time constant (s)
const COUNT_CAP = 1.5; // max seconds for one catch-up before snapping
const BIN_BITS = 43; // remaining-seconds binary counter width (16万년 ≈ 5.0e12 s < 2^43)

function fmtDuration(sec: number): string {
  if (sec <= 0 || sec !== sec) return "--yr ---d ---h --m --s";
  const kYear = 365.25 * 86400.0;
  let yr = 0;
  if (sec >= kYear) {
    yr = Math.floor(sec / kYear);
    sec -= yr * kYear;
  }
  let s = Math.floor(sec);
  const d = Math.floor(s / 86400); s %= 86400;
  const h = Math.floor(s / 3600); s %= 3600;
  const m = Math.floor(s / 60);
  const ss = s % 60;
  const p2 = (n: number) => String(n).padStart(2, "0");
  const p3 = (n: number) => String(n).padStart(3, "0");
  if (yr > 0) return `${yr}yr ${p3(d)}d ${p2(h)}h ${p2(m)}m ${p2(ss)}s`;
  if (d > 0) return `${p3(d)}d ${p2(h)}h ${p2(m)}m ${p2(ss)}s`;
  return `${p2(h)}h ${p2(m)}m ${p2(ss)}s`;
}

// |next_float32(x) - x| — the grid step the 32-bit ghost lives on.
function float32Ulp(x: number): number {
  const f = Math.fround(x);
  if (f === 0 || !isFinite(f)) return 0;
  const buf = new ArrayBuffer(4);
  const fv = new Float32Array(buf);
  const iv = new Uint32Array(buf);
  fv[0] = f;
  iv[0] += f > 0 ? 1 : -1;
  return Math.abs(fv[0] - f);
}

export class MonClock {
  private dDisplay = new Array<number>(MAX_DIGITS).fill(0);
  private nDisplayDigits = 1;

  private displayVal = 0; // eased op_count actually shown
  private countTarget = 0; // latest op_count target
  private capClock = 0; // time spent on the current catch-up (for COUNT_CAP)
  private targetQueue: number[] = [];

  private elapsedTotal = 0;
  private lastArrivalTime = 0;
  private arrivalEma = 1.0;

  private axisFlush = false;

  private clockTick = false;
  private remainSmooth = 0;
  private remainDisplay = 0;
  private remainRaw = 0;
  private remainGhost = 0;
  private rGapEma = 0;

  private opCount = 0;
  private elapsed = 0;
  private seqIdx = 0;
  private sourceLabel = "";
  private nTokens = 0;
  private layer = 0;

  private axis1Timer = 0;
  private axis3Timer = 0;
  private axis3Msg = "";

  // Seed footer provenance from the capture manifest's corpus_sidecar. The
  // install's sequence_start OSC fires at op_count ~0, before the tap is
  // listening, so it is absent from these captures; a live sequence_start (if
  // ever captured) still overrides these values via update().
  setSeqMeta(seqIdx: number, sourceLabel: string, nTokens: number): void {
    this.seqIdx = seqIdx;
    this.sourceLabel = sourceLabel;
    this.nTokens = nTokens;
  }

  private decompose(val: number): { digits: number[]; n: number } {
    if (val < 0) val = 0;
    const digits = new Array<number>(MAX_DIGITS).fill(0);
    let n = 0;
    for (let i = 0; i < MAX_DIGITS; i++) {
      digits[i] = val % 10;
      val = Math.floor(val / 10);
      if (digits[i] || val) n = i + 1;
    }
    if (n === 0) n = 1;
    return { digits, n };
  }

  update(events: OscEvent[], _ops: OpRec[], dt: number): void {
    this.elapsedTotal += dt;
    this.axis1Timer = Math.max(0, this.axis1Timer - dt);
    this.axis3Timer = Math.max(0, this.axis3Timer - dt);

    for (const ev of events) {
      switch (ev.path) {
        case "/bert/clock": {
          this.opCount = ev.args[0];
          this.elapsed = ev.args[1];
          this.targetQueue.push(this.opCount);
          this.clockTick = true;
          const interval = this.elapsedTotal - this.lastArrivalTime;
          this.lastArrivalTime = this.elapsedTotal;
          if (interval > 0.001 && interval < 10.0)
            this.arrivalEma = this.arrivalEma * 0.9 + interval * 0.1;
          break;
        }
        case "/bert/sequence_start":
          this.seqIdx = ev.args[0];
          this.sourceLabel = String(ev.args[1]);
          this.nTokens = ev.args[2];
          break;
        case "/bert/sequence_end":
          this.seqIdx = ev.args[0];
          break;
        case "/bert/token_att":
          this.axis1Timer = 0.6;
          this.axisFlush = true;
          break;
        case "/bert/layer":
          this.layer = ev.args[0];
          this.axis3Msg = `LAYER ${this.layer} COMPLETE`;
          this.axis3Timer = 2.0;
          this.axisFlush = true;
          break;
        default:
          break;
      }
    }

    // Adopt the latest op_count target (monotonic) and ease the display toward it.
    if (this.targetQueue.length > 0) {
      const latest = this.targetQueue[this.targetQueue.length - 1];
      if (latest > this.countTarget) {
        this.countTarget = latest;
        this.capClock = 0; // restart the cap window for this new gap
      }
      this.targetQueue.length = 0;
    }
    if (this.axisFlush) {
      this.capClock = 0; // a real axis boundary re-anchors the catch-up window
      this.axisFlush = false;
    }

    // Ease-out: speed proportional to the remaining distance; COUNT_CAP snaps any
    // single catch-up that runs too long so the asymptote never lingers. Monotonic,
    // no randomness, no mid-pause.
    const remaining = this.countTarget - this.displayVal;
    if (remaining > 0.5) {
      this.capClock += dt;
      if (this.capClock >= COUNT_CAP) {
        this.displayVal = this.countTarget;
      } else {
        this.displayVal += remaining * (1 - Math.exp(-dt / COUNT_TAU));
        if (this.countTarget - this.displayVal < 0.5) this.displayVal = this.countTarget;
      }
    } else {
      this.displayVal = this.countTarget;
    }
    const dec = this.decompose(Math.floor(this.displayVal));
    this.dDisplay = dec.digits;
    this.nDisplayDigits = dec.n;

    // Remaining-time estimate: re-anchor on clock tick (rate-based, EMA-smoothed,
    // ratcheted down); free-run countdown 1s/wall-second between ticks.
    if (this.clockTick && this.opCount > 0 && this.elapsed > 0) {
      const rate = this.opCount / this.elapsed;
      this.remainRaw = (OP_TOTAL - this.opCount) / rate;
      // Ghost: what a 32-bit system computes from the same clock — elapsed
      // round-trips through float32, isolating representation loss.
      const elapsedF = Math.fround(this.elapsed);
      const rateF = this.opCount / elapsedF;
      this.remainGhost = (OP_TOTAL - this.opCount) / rateF;

      if (this.remainSmooth <= 0) this.remainSmooth = this.remainRaw;
      else this.remainSmooth = this.remainSmooth * 0.95 + this.remainRaw * 0.05;
      if (this.remainDisplay <= 0 || this.remainSmooth < this.remainDisplay)
        this.remainDisplay = this.remainSmooth;

      const gapYr = (this.remainGhost - this.remainSmooth) / YR_SEC;
      this.rGapEma =
        this.rGapEma <= 0 ? Math.abs(gapYr) : this.rGapEma * 0.9 + Math.abs(gapYr) * 0.1;
      this.clockTick = false;
    }
    if (this.remainDisplay > 0)
      this.remainDisplay = Math.max(0, this.remainDisplay - dt);
  }

  render(ctx: CanvasRenderingContext2D, x: number, y: number, w: number, h: number): void {
    const s = h / REF_H; // proportional scale to viewport
    const fontXLBase = 48 * s;
    const fontLG = 24 * s;
    const fontSM = 13 * s;
    const mono = "JetBrains Mono, ui-monospace, monospace";
    const cx = x + w / 2;

    ctx.fillStyle = "#000";
    ctx.fillRect(x, y, w, h);
    ctx.textBaseline = "top";
    ctx.textAlign = "left";

    const grey = (v: number) => {
      const c = Math.round(v * 255);
      return `rgb(${c},${c},${c})`;
    };
    const centerText = (text: string, cy: number, font: string, fill: string) => {
      ctx.font = font;
      ctx.fillStyle = fill;
      const tw = ctx.measureText(text).width;
      ctx.fillText(text, cx - tw / 2, cy);
    };

    // Top-left label
    ctx.font = `${fontSM}px ${mono}`;
    ctx.fillStyle = grey(0.12);
    ctx.fillText("C  clock", x + 10 * s, y + 6 * s);

    const centerY = y + h * 0.42;

    // op_count string with thousands separators
    let countStr = "";
    for (let i = this.nDisplayDigits - 1; i >= 0; i--) {
      countStr += String(this.dDisplay[i]);
      if (i > 0 && i % 3 === 0) countStr += ",";
    }
    if (countStr === "") countStr = "0";

    // Large op_count number (auto-shrink to width)
    const padX = 12 * s;
    const availW = w - padX * 2;
    let fs = fontXLBase;
    ctx.font = `${fs}px ${mono}`;
    let tw = ctx.measureText(countStr).width;
    if (tw > availW && availW > 0) fs = (fs * availW) / tw;
    ctx.font = `${fs}px ${mono}`;
    tw = ctx.measureText(countStr).width;
    const th = fs; // approx cap+ascent height for mono at px size
    const numTop = centerY - th;
    ctx.fillStyle = grey(1);
    ctx.fillText(countStr, cx - tw / 2, numTop);

    // dim labels
    centerText("op_count", numTop - fontSM - 2 * s, `${fontSM}px ${mono}`, grey(0.22));
    centerText("remaining", centerY + 42 * s, `${fontSM}px ${mono}`, grey(0.22));

    // main remaining time (ratcheted EMA)
    const remStr = fmtDuration(this.remainDisplay);
    centerText(remStr, centerY + 56 * s, `${fontLG}px ${mono}`, grey(0.8));

    // ── Binary counter: remaining seconds as 43 bits (cf. Counter 2024) ──────────
    // One square per bit, MSB left, row width = the ETA text width above. 0 = black
    // borderless (blends into the bg), 1 = white (alpha 100%). Digital finitude made
    // literal: the whole work's length counted down in binary.
    ctx.font = `${fontLG}px ${mono}`;
    const etaW = ctx.measureText(remStr).width;
    const binTop = centerY + 56 * s + fontLG + 6 * s;
    const binLeft = cx - etaW / 2;
    const cellW = etaW / BIN_BITS;
    const binBottom = binTop + cellW;
    if (this.remainDisplay > 0 && etaW > 0) {
      const remSec = Math.floor(this.remainDisplay);
      ctx.fillStyle = "rgba(255,255,255,1)";
      const ch = Math.max(1, Math.round(cellW));
      const ty = Math.round(binTop);
      for (let i = 0; i < BIN_BITS; i++) {
        // bit (BIN_BITS-1-i): MSB at i=0. 43 bits exceed 32, so extract by division.
        const bit = Math.floor(remSec / Math.pow(2, BIN_BITS - 1 - i)) % 2;
        if (bit !== 1) continue; // 0 = borderless black = draw nothing
        const x0 = Math.round(binLeft + i * cellW);
        const x1 = Math.round(binLeft + (i + 1) * cellW);
        ctx.fillRect(x0, ty, Math.max(1, x1 - x0), ch);
      }
    }

    // Ghost + ε + quantization gauge
    if (this.remainGhost > 0 && this.remainDisplay > 0) {
      const deltaYr = (this.remainGhost - this.remainSmooth) / YR_SEC;
      const ghostStr = fmtDuration(this.remainGhost);
      const rawY = binBottom + 14 * s;
      centerText(ghostStr, rawY, `${fontLG}px ${mono}`, grey(0.25));

      const ulp = float32Ulp(this.elapsed);
      const epsY = rawY + 30 * s;
      centerText(`ε = ${ulp.toExponential(2)} s`, epsY, `${fontSM}px ${mono}`, "rgb(178,31,31)");

      // Gauge: 1px-crisp line + ticks + even-sided square indicator.
      const LX = Math.round(cx);
      const GTOP = Math.round(epsY + 20 * s);
      const GH = Math.round(80 * s);
      const GMID = GTOP + GH / 2;
      const gaugeRangeYr = Math.max(0.5, this.rGapEma * 2.5);
      const tickCol = "rgb(35,35,35)";

      ctx.fillStyle = tickCol;
      ctx.fillRect(LX, GTOP, 1, GH);
      for (let i = 0; i <= 10; i++) {
        const ty = GTOP + Math.round((i * GH) / 10);
        const halfW = i === 5 ? Math.round(10 * s) : Math.round(4 * s);
        ctx.fillRect(LX - halfW, ty, halfW * 2 + 1, 1);
      }

      const SQ = Math.max(3, Math.round(4 * s));
      const dotNorm = Math.max(-1, Math.min(1, deltaYr / gaugeRangeYr));
      let dotCy = GMID - dotNorm * (GH * 0.5);
      dotCy = Math.max(GTOP + SQ / 2, Math.min(GTOP + GH - SQ / 2, dotCy));
      ctx.fillStyle = "rgba(255,255,255,0.86)";
      ctx.fillRect(LX - Math.floor(SQ / 2), Math.round(dotCy) - Math.floor(SQ / 2), SQ, SQ);
    }

    // Axis 3 message
    if (this.axis3Timer > 0 && this.axis3Msg) {
      const alpha = Math.min(1, this.axis3Timer / 0.3) * 0.7;
      centerText(this.axis3Msg, centerY + h * 0.18, `${fontSM}px ${mono}`, `rgba(255,255,255,${alpha})`);
    }

    // Footer
    centerText(
      `seq ${this.seqIdx}  src ${this.sourceLabel}  N=${this.nTokens}`,
      y + h - fontSM - 8 * s,
      `${fontSM}px ${mono}`,
      grey(0.18),
    );

    // Axis 1 edge flash
    if (this.axis1Timer > 0) {
      const alpha = this.axis1Timer / 0.6;
      ctx.strokeStyle = `rgba(255,255,255,${alpha * 0.47})`;
      ctx.lineWidth = 1.5 * s;
      ctx.strokeRect(x + 1, y + 1, w - 2, h - 2);
    }
  }
}
