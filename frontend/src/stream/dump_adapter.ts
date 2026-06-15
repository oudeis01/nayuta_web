import type { EventStream, OscEvent, OpRec, ResidualRec } from "./types";

// Replays a captured session from its zstd-compressed event log.
//
// The capture preserves real wall-clock timing in each event's ts_ns. Playback
// maps that capture timeline onto frames: each advance(realDt) moves a playback
// clock forward by realDt * speed and fires every event whose timestamp has
// been reached. This keeps the install's true pacing (action plan 11) while
// letting the viewer scrub speed.
//
// Pull model: the host owns the rAF loop and calls advance() once per frame.
// onOsc/onOpRec handlers fire synchronously inside advance().
//
// Streaming ops: the web full-dump's ops.bin.zst is ~1.79 GB compressed /
// ~3.79 GB decompressed — far too large to hold resident or even allocate as one
// ArrayBuffer. So decode_worker.ts streams it as whole-frame "slabs" (~4 MB
// each) under credit-based flow control; this adapter keeps only the slabs near
// the playback cursor, fires their records as the timeline reaches them, then
// frees each consumed slab and returns a credit (cmd:"ack") so the worker
// streams the next. Resident decompressed bytes stay bounded (≈ 96 MB) no matter
// how long the capture is. Playback is strictly forward (the host never seeks
// back or loops), which is what lets a one-pass stream suffice.
//
// ops.bin framing (written by the Rust tap): a sequence of
//   [ts_ns: u64 LE][len: u32 LE][payload: len bytes]
// where each payload is one ZMQ OpMsg:
//   [base_op_count: u64][count: u16][version: u16][reserved: u32] then count×OpRec(24B)
// `reserved` is a msg_type discriminator: 0 = OpMsg (24B records), 1 = ResidualMsg
// (16B ResidualRec records). Residual frames appear only in captures made with
// bert --residual-basis; older captures contain only OpMsg frames (reserved 0).
const MSG_RESIDUAL = 1;

// Worker → adapter messages.
type WorkerResp =
  | { type: "progress"; frac: number; stage: string }
  | { type: "events"; events: OscEvent[] }
  | { type: "slab"; buf: ArrayBuffer }
  | { type: "end" }
  | { type: "error"; message: string };

interface OpFrame {
  tsNs: number;
  recOff: number; // byte offset of records[0] within the slab
  count: number; // valid records in this frame
  base: number; // global op_count of records[0]
}

// Residual frames carry no op_count semantics (timeline only), so we keep just
// the offset and count; records are a flat 16-byte layout.
interface ResFrame {
  tsNs: number;
  recOff: number;
  count: number;
}

// One decompressed slab resident on the main thread, with its frames indexed and
// two cursors tracking how far playback has consumed each frame type.
interface Slab {
  raw: Uint8Array;
  view: DataView;
  opFrames: OpFrame[];
  resFrames: ResFrame[];
  opCur: number;
  resCur: number;
}

export class DumpAdapter implements EventStream {
  private oscHandlers: ((ev: OscEvent) => void)[] = [];
  private opHandlers: ((rec: OpRec) => void)[] = [];
  private resHandlers: ((rec: ResidualRec) => void)[] = [];

  private events: OscEvent[] = [];
  private idx = 0;
  private t0 = 0; // ts_ns of first event
  private playT = 0; // playback seconds elapsed since t0
  private speed = 1;
  private paused = false;
  private baseUrl: string;

  // ops.bin streaming state (populated only when load({ ops: true })).
  private worker?: Worker;
  private slabs: Slab[] = [];
  private opsEnded = false;

  constructor(baseUrl: string) {
    this.baseUrl = baseUrl;
  }

  onOsc(handler: (ev: OscEvent) => void): void {
    this.oscHandlers.push(handler);
  }
  onOpRec(handler: (rec: OpRec) => void): void {
    this.opHandlers.push(handler);
  }
  onResidualRec(handler: (rec: ResidualRec) => void): void {
    this.resHandlers.push(handler);
  }
  setPlaybackSpeed(x: number): void {
    this.speed = Math.max(0, x);
  }
  pause(): void {
    this.paused = true;
  }
  resume(): void {
    this.paused = false;
  }

  get duration(): number {
    if (this.events.length === 0) return 0;
    return (this.events[this.events.length - 1].tsNs - this.t0) / 1e9;
  }
  get position(): number {
    return this.playT;
  }
  get done(): boolean {
    return this.idx >= this.events.length && this.opsEnded && this.slabs.length === 0;
  }

  // The main (lid,vidx) of the first `limit` /bert/whisper events, in fire order.
  // Used to prefetch the opening utterances' audio (action plan §7-6). Neighbor
  // voices are omitted — they trail the main voice by an IOI, so lazy-loading
  // them adds no perceptible stall.
  firstWhisperPairs(limit: number): [number, number][] {
    const out: [number, number][] = [];
    for (const ev of this.events) {
      if (ev.path !== "/bert/whisper") continue;
      out.push([ev.args[0] | 0, ev.args[1] | 0]);
      if (out.length >= limit) break;
    }
    return out;
  }

  // Start the worker stream. Resolves early — once the OSC timeline is in and the
  // first ops slab (or end-of-stream) has landed — so the wall can begin while the
  // rest of ops.bin keeps streaming in the background. Rejects only if the worker
  // errors before that first resolve.
  load(
    opts: { ops?: boolean } = {},
    onProgress?: (frac: number) => void,
  ): Promise<void> {
    return new Promise<void>((resolve, reject) => {
      const w = new Worker(new URL("./decode_worker.ts", import.meta.url), { type: "module" });
      this.worker = w;
      let resolved = false;
      let gotEvents = false;
      const finish = () => {
        if (resolved) return;
        resolved = true;
        onProgress?.(1);
        resolve();
      };
      w.onmessage = (e: MessageEvent<WorkerResp>) => {
        const m = e.data;
        switch (m.type) {
          case "progress":
            onProgress?.(m.frac);
            break;
          case "events":
            this.events = m.events;
            this.t0 = m.events.length ? m.events[0].tsNs : 0;
            this.idx = 0;
            this.playT = 0;
            gotEvents = true;
            if (!opts.ops) finish();
            break;
          case "slab":
            this.slabs.push(this.indexSlab(m.buf));
            if (gotEvents) finish(); // first slab in -> safe to start the wall
            break;
          case "end":
            this.opsEnded = true;
            finish(); // tiny/empty ops streams resolve here
            break;
          case "error":
            if (!resolved) reject(new Error(m.message));
            else console.error("[dump] ops stream error:", m.message);
            break;
        }
      };
      w.onerror = (e) => {
        if (!resolved) reject(new Error(e.message || "decode worker error"));
      };
      w.postMessage({ cmd: "start", base: this.baseUrl, ops: !!opts.ops });
    });
  }

  // Stop the worker and release its stream. Called when a corpus switch supersedes
  // this adapter, so the old 1.79 GB download/decompress is aborted promptly.
  dispose(): void {
    if (!this.worker) return;
    this.worker.postMessage({ cmd: "stop" });
    this.worker.terminate();
    this.worker = undefined;
    this.slabs = [];
  }

  // Index one decompressed slab's frames (cheap: reads frame headers only). The
  // worker guarantees slabs are cut on frame boundaries, so no frame straddles
  // two slabs. Frame offsets are relative to this slab's buffer.
  private indexSlab(buffer: ArrayBuffer): Slab {
    const raw = new Uint8Array(buffer);
    const dv = new DataView(buffer);
    const opFrames: OpFrame[] = [];
    const resFrames: ResFrame[] = [];
    let off = 0;
    while (off + 12 <= raw.length) {
      const tsNs = dv.getUint32(off, true) + dv.getUint32(off + 4, true) * 4294967296;
      const len = dv.getUint32(off + 8, true);
      off += 12;
      if (off + len > raw.length || len < 16) break;
      const base = dv.getUint32(off, true) + dv.getUint32(off + 4, true) * 4294967296;
      const declared = dv.getUint16(off + 8, true);
      const msgType = dv.getUint32(off + 12, true);
      if (msgType === MSG_RESIDUAL) {
        const capacity = Math.floor((len - 16) / 16);
        resFrames.push({ tsNs, recOff: off + 16, count: Math.min(declared, capacity) });
      } else {
        // MSG_OP (0) or any unknown type defaults to the op layout, matching
        // older captures whose reserved field was always 0.
        const capacity = Math.floor((len - 16) / 24);
        opFrames.push({ tsNs, recOff: off + 16, count: Math.min(declared, capacity), base });
      }
      off += len;
    }
    return { raw, view: dv, opFrames, resFrames, opCur: 0, resCur: 0 };
  }

  // Decode one 24-byte OpRec at byte offset p within a slab, reconstructing the
  // record's global op_count from the frame base (op_count_lo is its low 32 bits).
  private readOp(s: Slab, p: number, base: number): OpRec {
    const dv = s.view;
    const lo = dv.getUint32(p, true);
    let opCount = Math.floor(base / 4294967296) * 4294967296 + lo;
    if (opCount < base) opCount += 4294967296;
    return {
      opCount,
      opType: s.raw[p + 4],
      layer: s.raw[p + 5],
      head: s.raw[p + 6],
      flags: s.raw[p + 7],
      qPos: dv.getUint16(p + 8, true),
      kPos: dv.getUint16(p + 10, true),
      a: dv.getFloat32(p + 12, true),
      b: dv.getFloat32(p + 16, true),
      r: dv.getFloat32(p + 20, true),
    };
  }

  // Decode one 16-byte ResidualRec at byte offset p within a slab.
  //   layer u8 | strand u8 | q_pos u16 | c0 f32 | c1 f32 | c2 f32
  private readResidual(s: Slab, p: number): ResidualRec {
    const dv = s.view;
    return {
      layer: s.raw[p],
      strand: s.raw[p + 1],
      qPos: dv.getUint16(p + 2, true),
      c0: dv.getFloat32(p + 4, true),
      c1: dv.getFloat32(p + 8, true),
      c2: dv.getFloat32(p + 12, true),
    };
  }

  // Advance playback by one frame; returns the playback dt (seconds) so the
  // host can drive panel animations at the same rate as event delivery.
  advance(realDt: number): number {
    if (this.paused) return 0;
    const pdt = realDt * this.speed;
    this.playT += pdt;
    const cutoff = this.t0 + this.playT * 1e9;

    // OSC first, matching the C++ panels that process axis events before the op
    // batch.
    while (this.idx < this.events.length && this.events[this.idx].tsNs <= cutoff) {
      const ev = this.events[this.idx++];
      for (const h of this.oscHandlers) h(ev);
    }

    // Op records across the resident slabs, in stream order (slabs and the frames
    // within them are ts-ascending). Stop at the first slab still holding a future
    // op frame — all later slabs are later still.
    for (const s of this.slabs) {
      while (s.opCur < s.opFrames.length && s.opFrames[s.opCur].tsNs <= cutoff) {
        const fr = s.opFrames[s.opCur++];
        let p = fr.recOff;
        for (let i = 0; i < fr.count; i++) {
          const rec = this.readOp(s, p, fr.base);
          for (const h of this.opHandlers) h(rec);
          p += 24;
        }
      }
      if (s.opCur < s.opFrames.length) break;
    }

    // Residual records: same one-pass-per-cutoff walk, fired after all due ops to
    // preserve the original (all OSC, then all ops, then all residual) ordering.
    for (const s of this.slabs) {
      while (s.resCur < s.resFrames.length && s.resFrames[s.resCur].tsNs <= cutoff) {
        const fr = s.resFrames[s.resCur++];
        let p = fr.recOff;
        for (let i = 0; i < fr.count; i++) {
          const rec = this.readResidual(s, p);
          for (const h of this.resHandlers) h(rec);
          p += 16;
        }
      }
      if (s.resCur < s.resFrames.length) break;
    }

    // Drop fully-consumed front slabs and return a credit per slab so the worker
    // streams more. A slab is done only when both cursors are exhausted.
    while (
      this.slabs.length > 0 &&
      this.slabs[0].opCur >= this.slabs[0].opFrames.length &&
      this.slabs[0].resCur >= this.slabs[0].resFrames.length
    ) {
      this.slabs.shift();
      this.worker?.postMessage({ cmd: "ack" });
    }

    return pdt;
  }
}
