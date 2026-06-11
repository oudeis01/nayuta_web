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
// ops.bin.zst is loaded lazily (load({ ops: true })) only when a consuming
// panel is mounted, since it is ~97 MB compressed / ~180 MB decompressed. The
// stream is rate-limited at the source to ~1,092 recs/s, so replay is gentle;
// the cost is the resident buffer, which we keep as the raw decompressed bytes
// plus a lightweight frame index and walk with a cursor (never exploding all
// ~8M records into JS objects up front).
//
// ops.bin framing (written by the Rust tap): a sequence of
//   [ts_ns: u64 LE][len: u32 LE][payload: len bytes]
// where each payload is one ZMQ OpMsg:
//   [base_op_count: u64][count: u16][version: u16][reserved: u32] then count×OpRec(24B)
// `reserved` is a msg_type discriminator: 0 = OpMsg (24B records), 1 = ResidualMsg
// (16B ResidualRec records). Residual frames appear only in captures made with
// bert --residual-basis; older captures contain only OpMsg frames (reserved 0).
const MSG_RESIDUAL = 1;

// The fetch + zstd decompress lives in a shared Web Worker (decode_worker.ts) so
// the synchronous ~180 MB decompress never blocks the main thread / animation.
// One worker serves the whole page; jobs are keyed by id and the latest caller
// wins (DumpAdapter awaits its own job's result). The worker reports download
// progress and finally returns the parsed OscEvent[] plus the decompressed ops
// buffer as a Transferable.
type DonePayload = { events: OscEvent[]; opsBuffer: ArrayBuffer | null };
type WorkerResp =
  | { id: number; type: "progress"; frac: number; stage: string }
  | { id: number; type: "done"; events: OscEvent[]; opsBuffer: ArrayBuffer | null }
  | { id: number; type: "error"; message: string };

interface Job {
  resolve: (p: DonePayload) => void;
  reject: (e: Error) => void;
  onProgress?: (frac: number) => void;
}

let _worker: Worker | null = null;
let _jobSeq = 0;
const _jobs = new Map<number, Job>();

function getWorker(): Worker {
  if (_worker) return _worker;
  const w = new Worker(new URL("./decode_worker.ts", import.meta.url), { type: "module" });
  w.onmessage = (e: MessageEvent<WorkerResp>) => {
    const m = e.data;
    const job = _jobs.get(m.id);
    if (!job) return;
    if (m.type === "progress") {
      job.onProgress?.(m.frac);
      return;
    }
    _jobs.delete(m.id);
    if (m.type === "error") job.reject(new Error(m.message));
    else job.resolve({ events: m.events, opsBuffer: m.opsBuffer });
  };
  _worker = w;
  return w;
}

// Run one fetch+decompress job in the worker. The progress fraction is absolute
// (events 0..0.08, ops 0.08..1), matching the bar split the host expects.
function decodeJob(
  base: string,
  ops: boolean,
  onProgress?: (frac: number) => void,
): Promise<DonePayload> {
  const id = ++_jobSeq;
  const w = getWorker();
  return new Promise<DonePayload>((resolve, reject) => {
    _jobs.set(id, { resolve, reject, onProgress });
    w.postMessage({ id, base, ops });
  });
}

interface OpFrame {
  tsNs: number;
  recOff: number; // byte offset of records[0] within opsRaw
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

  // ops.bin replay state (populated only when load({ ops: true })).
  private opsRaw?: Uint8Array;
  private opsView?: DataView;
  private frames: OpFrame[] = [];
  private frameIdx = 0;
  private resFrames: ResFrame[] = [];
  private resFrameIdx = 0;

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
    return this.idx >= this.events.length;
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

  async load(
    opts: { ops?: boolean } = {},
    onProgress?: (frac: number) => void,
  ): Promise<void> {
    // Fetch + decompress run in the worker (off the main thread); we get back the
    // parsed events and the decompressed ops buffer (Transferable). Progress is
    // already absolute (events 0..0.08, ops 0.08..1).
    const { events, opsBuffer } = await decodeJob(this.baseUrl, !!opts.ops, onProgress);
    this.events = events;
    this.t0 = events.length ? events[0].tsNs : 0;
    this.idx = 0;
    this.playT = 0;
    if (opsBuffer) this.indexOps(opsBuffer);
    onProgress?.(1);
  }

  // Index the decompressed ops buffer's frames without materializing records.
  // Cheap (proportional to frame count, reads frame headers only); the heavy
  // fetch+decompress already ran in the worker. Each frame stores its capture ts
  // (same clock as the OSC events, so the two share one playback timeline) and
  // where its records live in the resident buffer.
  private indexOps(buffer: ArrayBuffer): void {
    const raw = new Uint8Array(buffer);
    const dv = new DataView(buffer);

    const frames: OpFrame[] = [];
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
        frames.push({ tsNs, recOff: off + 16, count: Math.min(declared, capacity), base });
      }
      off += len;
    }
    frames.sort((a, b) => a.tsNs - b.tsNs);
    resFrames.sort((a, b) => a.tsNs - b.tsNs);
    this.opsRaw = raw;
    this.opsView = dv;
    this.frames = frames;
    this.frameIdx = 0;
    this.resFrames = resFrames;
    this.resFrameIdx = 0;
  }

  // Decode one 24-byte OpRec at byte offset p, reconstructing the record's
  // global op_count from the frame base (op_count_lo is its low 32 bits).
  private readOp(p: number, base: number): OpRec {
    const dv = this.opsView!;
    const lo = dv.getUint32(p, true);
    let opCount = Math.floor(base / 4294967296) * 4294967296 + lo;
    if (opCount < base) opCount += 4294967296;
    return {
      opCount,
      opType: this.opsRaw![p + 4],
      layer: this.opsRaw![p + 5],
      head: this.opsRaw![p + 6],
      flags: this.opsRaw![p + 7],
      qPos: dv.getUint16(p + 8, true),
      kPos: dv.getUint16(p + 10, true),
      a: dv.getFloat32(p + 12, true),
      b: dv.getFloat32(p + 16, true),
      r: dv.getFloat32(p + 20, true),
    };
  }

  // Decode one 16-byte ResidualRec at byte offset p.
  //   layer u8 | strand u8 | q_pos u16 | c0 f32 | c1 f32 | c2 f32
  private readResidual(p: number): ResidualRec {
    const dv = this.opsView!;
    return {
      layer: this.opsRaw![p],
      strand: this.opsRaw![p + 1],
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
    while (this.idx < this.events.length && this.events[this.idx].tsNs <= cutoff) {
      const ev = this.events[this.idx++];
      for (const h of this.oscHandlers) h(ev);
    }
    // Fire op frames on the same timeline (OSC first, matching the C++ panels
    // that process axis events before the op batch).
    while (this.frameIdx < this.frames.length && this.frames[this.frameIdx].tsNs <= cutoff) {
      const fr = this.frames[this.frameIdx++];
      let p = fr.recOff;
      for (let i = 0; i < fr.count; i++) {
        const rec = this.readOp(p, fr.base);
        for (const h of this.opHandlers) h(rec);
        p += 24;
      }
    }
    // Residual frames share the same playback timeline; fire after op records.
    while (
      this.resFrameIdx < this.resFrames.length &&
      this.resFrames[this.resFrameIdx].tsNs <= cutoff
    ) {
      const fr = this.resFrames[this.resFrameIdx++];
      let p = fr.recOff;
      for (let i = 0; i < fr.count; i++) {
        const rec = this.readResidual(p);
        for (const h of this.resHandlers) h(rec);
        p += 16;
      }
    }
    return pdt;
  }
}
