// Common stream contract shared by every data source (dump replay, live WS).
// See action plan 8-2.

// One decoded OSC event as captured by the Rust tap (events.jsonl line).
// args hold the OSC arguments in wire order; interpret them per `path`
// (the emitter-side arg order lives in bert.c osc_* functions).
export interface OscEvent {
  // Capture timestamp in ns. Stored as a JS double, so resolution degrades to
  // ~microseconds above 2^53; fine for frame-rate replay (deltas only).
  tsNs: number;
  path: string;
  types: string;
  args: number[];
}

// One floating-point op record (ops.bin OpRec, 24 bytes on the wire).
export interface OpRec {
  opCount: number;
  opType: number;
  layer: number;
  head: number;
  flags: number;
  qPos: number;
  kPos: number;
  a: number;
  b: number;
  r: number;
}

export interface EventStream {
  onOsc(handler: (ev: OscEvent) => void): void;
  onOpRec(handler: (rec: OpRec) => void): void;
  // Dump adapters honor these; live adapters may no-op.
  setPlaybackSpeed?(x: number): void;
  pause?(): void;
  resume?(): void;
}
