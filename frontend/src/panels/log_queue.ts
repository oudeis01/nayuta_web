// Shared scrolling-log buffer for the op-log monitors (A scanner, B op_stream)
// and, after the desktop backport, their C++ counterparts.
//
// The op stream arrives at ~1,092 lines/s at 1x, so a naive "push every line
// that arrived this frame, paint the tail" loop shoves ~18 lines in at once each
// frame and reads as the whole block being replaced rather than a scroll
// (mon B feedback B-1, 2026-06-05). This buffer fixes that with two layers:
//
//   pending  — lines enqueued but not yet shown.
//   visible  — the bounded tail the panel paints (newest last).
//
// tick() reveals pending into visible at an fps-paced rate measured from the
// wall clock (NOT the playback dt), so the on-screen scroll stays smooth no
// matter the replay speed. Per the agreed model (plan B-1):
//   (b) baseline ~one line per frame,
//   (c) when the backlog grows the reveal rate rises to catch up, capped at a
//       real-time ceiling so it never blurs,
//   (d) when the backlog overflows the memory cap the oldest pending lines are
//       dropped (overflow flush) so it can refill from the newest.
// The concrete numbers below are first-pass defaults; the exact feel is the
// user's visual call to fine-tune (plan D-B1 (1)).
//
// A queue WITHOUT a pace config is a passthrough: tick() reveals everything at
// once. Monitor A uses that mode so its existing slow 1/300 op sampling keeps
// pacing the log and its current feel is preserved; only mon B drives the paced
// reveal.

export interface PaceOpts {
  rate: number; // baseline reveal floor, lines/sec (~fps gives one line/frame)
  drain: number; // catch-up time constant, s: reveal ~ backlog/drain when behind
  maxRate: number; // real-time reveal ceiling, lines/sec (caps catch-up speed)
  flushAt: number; // pending cap; beyond it, drop the oldest pending lines
}

export interface LogQueueOpts {
  cap: number; // max retained visible lines (older ones drop off the top)
  pace?: PaceOpts; // omit for passthrough (reveal all each tick)
}

export class LogQueue<T> {
  private visible: T[] = [];
  private pending: T[] = [];
  private acc = 0; // fractional lines carried between ticks
  private lastNow = 0; // wall-clock seconds at the previous tick
  private opts: LogQueueOpts;

  constructor(opts: LogQueueOpts) {
    this.opts = opts;
  }

  enqueue(item: T): void {
    this.pending.push(item);
  }

  // Advance the reveal. `advancing` is the playback gate: when the wall is
  // paused (playback dt == 0) the scroll holds in place, matching the frozen
  // stream, and we still refresh the wall-clock baseline so resuming after a
  // long pause reveals a clamped burst rather than a giant catch-up.
  tick(advancing: boolean): void {
    const now = performance.now() / 1000;
    const dt = this.lastNow > 0 ? Math.min(0.1, now - this.lastNow) : 0;
    this.lastNow = now;
    if (!advancing) return;

    const pace = this.opts.pace;
    if (!pace) {
      // Passthrough: reveal everything, then trim to the visible cap.
      if (this.pending.length) {
        for (let i = 0; i < this.pending.length; i++) this.visible.push(this.pending[i]);
        this.pending.length = 0;
        this.trim();
      }
      return;
    }

    // Overflow flush: keep only the newest `flushAt` pending lines.
    if (this.pending.length > pace.flushAt) {
      this.pending.splice(0, this.pending.length - pace.flushAt);
    }
    const backlog = this.pending.length;
    if (backlog === 0) {
      this.acc = 0;
      return;
    }

    // Reveal rate: baseline floor, raised toward backlog/drain to catch up, but
    // never above the real-time ceiling so a deep backlog scrolls fast yet stays
    // legible (the rest is shed by the flush above).
    let rate = Math.max(pace.rate, backlog / pace.drain);
    if (rate > pace.maxRate) rate = pace.maxRate;

    this.acc += rate * dt;
    let n = Math.floor(this.acc);
    if (n <= 0) return;
    this.acc -= n;
    if (n > backlog) n = backlog;

    for (let i = 0; i < n; i++) this.visible.push(this.pending[i]);
    this.pending.splice(0, n);
    this.trim();
  }

  private trim(): void {
    const over = this.visible.length - this.opts.cap;
    if (over > 0) this.visible.splice(0, over);
  }

  clear(): void {
    this.visible.length = 0;
    this.pending.length = 0;
    this.acc = 0;
  }

  // The visible tail the panel paints (oldest first, newest last).
  get lines(): readonly T[] {
    return this.visible;
  }
}
