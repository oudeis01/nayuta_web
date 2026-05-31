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

// One residual-stream PCA record (ResidualRec, 16 bytes on the wire). Emitted by
// bert.c only when run with --residual-basis: at each residual readout the 768-d
// stream is projected onto a fixed per-layer PCA basis -> 3 coords. Carried over
// the same ZMQ socket as OpRec; the frame header's `reserved` field is the
// msg_type discriminator (0 = OpMsg, 1 = ResidualMsg).
export interface ResidualRec {
  layer: number; // 0..n_layers-1
  strand: number; // 0 = post-attn, 1 = post-ffn
  qPos: number; // token position
  c0: number; // PC1 coord (position axis, dominant)
  c1: number; // PC2 coord
  c2: number; // PC3 coord
}

export interface EventStream {
  onOsc(handler: (ev: OscEvent) => void): void;
  onOpRec(handler: (rec: OpRec) => void): void;
  // Residual records are present only in captures made with --residual-basis;
  // adapters without them simply never call the handler.
  onResidualRec?(handler: (rec: ResidualRec) => void): void;
  // Dump adapters honor these; live adapters may no-op.
  setPlaybackSpeed?(x: number): void;
  pause?(): void;
  resume?(): void;
}
