// decode_worker.ts — off-main-thread fetch + streaming zstd decompress for a
// capture dump.
//
// The web full-dump (Nayuta: The Transformer) replaced the old ~97 MB / ~180 MB
// pre-dump with a lossless K=8 capture: ops.bin.zst is ~1.79 GB compressed /
// ~3.79 GB decompressed. That no longer fits the old "decompress it all, transfer
// one buffer" model — a single 3.79 GB ArrayBuffer exceeds the browser's per-
// buffer ceiling and could never stay resident. So this worker now STREAMS:
//
//   fetch body -> fzstd.Decompress (chunked) -> cut whole-frame slabs (~4 MB) ->
//   transfer each slab to the main thread.
//
// Flow control: the worker holds a credit count. Each slab posted spends a
// credit; the main thread returns a credit (cmd:"ack") once it has fired and
// dropped a slab. When credits run out the worker stops pulling the fetch body,
// so resident decompressed bytes stay bounded (≈ INITIAL_CREDITS × SLAB_TARGET)
// on both sides regardless of how large the capture is.
//
// events.jsonl.zst is small (a few MB) and is still decoded in one shot, posted
// up front so the adapter can start as soon as the first ops slab lands.
//
// ops.bin framing (written by the Rust tap): a sequence of
//   [ts_ns: u64 LE][len: u32 LE][payload: len bytes]
// Slabs are always cut on frame boundaries, so the main thread never sees a
// frame split across two slabs. The worker only needs each frame's `len` (at
// header offset +8) to find boundaries; it does not parse payloads.
import { decompress, Decompress } from "fzstd";
import type { OscEvent } from "./types";

// ~4 MB per slab keeps each transfer well under any ArrayBuffer limit and small
// enough that the main thread frees memory promptly as playback advances.
const SLAB_TARGET = 4 * 1024 * 1024;
// 24 slabs ≈ 96 MB of decompressed lookahead held resident at once.
const INITIAL_CREDITS = 24;

type StartReq = { cmd: "start"; base: string; ops: boolean };
type AckReq = { cmd: "ack" };
type StopReq = { cmd: "stop" };
type Req = StartReq | AckReq | StopReq;

// The worker global, typed narrowly so we can postMessage with a transfer list
// without pulling the full "webworker" lib into the DOM-targeted tsconfig.
const ctx = self as unknown as {
  postMessage(message: unknown, transfer?: Transferable[]): void;
  onmessage: ((e: MessageEvent<Req>) => void) | null;
};

let credits = INITIAL_CREDITS;
let creditWaiter: (() => void) | null = null;
let aborted = false;
const abort = new AbortController();

function wakeCreditWaiter(): void {
  const w = creditWaiter;
  creditWaiter = null;
  w?.();
}

// Resolve once a credit is available (or immediately if one already is). Lets the
// fetch-pull loop park between chunks when the main thread is behind.
function awaitCredit(): Promise<void> {
  if (credits > 0 || aborted) return Promise.resolve();
  return new Promise<void>((res) => {
    creditWaiter = res;
  });
}

ctx.onmessage = (e: MessageEvent<Req>) => {
  const m = e.data;
  if (m.cmd === "ack") {
    credits++;
    wakeCreditWaiter();
  } else if (m.cmd === "stop") {
    aborted = true;
    abort.abort();
    wakeCreditWaiter();
  } else if (m.cmd === "start") {
    run(m.base, m.ops).catch((err) => {
      if (!aborted) ctx.postMessage({ type: "error", message: String(err) });
    });
  }
};

const post = (frac: number, stage: string) =>
  ctx.postMessage({ type: "progress", frac, stage });

// Fetch a URL into a Uint8Array, reporting download progress over [lo, hi].
async function fetchBytes(
  url: string,
  lo: number,
  hi: number,
  stage: string,
): Promise<Uint8Array> {
  const res = await fetch(url, { signal: abort.signal });
  if (!res.ok) throw new Error(`fetch ${url}: ${res.status}`);
  const total = Number(res.headers.get("content-length")) || 0;
  if (!res.body || !total) {
    const buf = new Uint8Array(await res.arrayBuffer());
    post(hi, stage);
    return buf;
  }
  const reader = res.body.getReader();
  const chunks: Uint8Array[] = [];
  let loaded = 0;
  for (;;) {
    const { done, value } = await reader.read();
    if (done) break;
    chunks.push(value);
    loaded += value.length;
    post(lo + (hi - lo) * Math.min(1, loaded / total), stage);
  }
  const out = new Uint8Array(loaded);
  let off = 0;
  for (const c of chunks) {
    out.set(c, off);
    off += c.length;
  }
  return out;
}

async function run(base: string, ops: boolean): Promise<void> {
  // events.jsonl.zst is small; decode it whole and post it first so the adapter
  // has the full OSC timeline (and firstWhisperPairs) before any ops arrive.
  const evComp = await fetchBytes(`${base}/events.jsonl.zst`, 0, 0.45, "loading events");
  if (aborted) return;
  const evRaw = decompress(evComp);
  const text = new TextDecoder().decode(evRaw);
  const events: OscEvent[] = [];
  for (const line of text.split("\n")) {
    if (!line) continue;
    const o = JSON.parse(line);
    events.push({ tsNs: o.ts_ns, path: o.path, types: o.types, args: o.args });
  }
  events.sort((a, b) => a.tsNs - b.tsNs);
  ctx.postMessage({ type: "events", events });

  if (!ops) {
    ctx.postMessage({ type: "end" });
    return;
  }

  // Stream ops.bin.zst: pull the compressed body in chunks, decompress
  // incrementally, and cut whole-frame slabs out of the decompressed stream.
  const res = await fetch(`${base}/ops.bin.zst`, { signal: abort.signal });
  if (!res.ok) throw new Error(`fetch ${base}/ops.bin.zst: ${res.status}`);
  const total = Number(res.headers.get("content-length")) || 0;
  const reader = res.body!.getReader();

  // Slicer state over the not-yet-emitted decompressed bytes.
  let acc = new Uint8Array(1 << 20);
  let accLen = 0; // valid bytes in acc
  let scanOff = 0; // next unparsed frame header within acc
  let cutStart = 0; // start of the current (un-emitted) slab region
  let lastBoundary = 0; // end of the last complete frame seen

  const ensureCap = (extra: number) => {
    if (accLen + extra <= acc.length) return;
    let cap = acc.length;
    while (cap < accLen + extra) cap *= 2;
    const next = new Uint8Array(cap);
    next.set(acc.subarray(0, accLen));
    acc = next;
  };

  const emitSlab = (s: number, e: number) => {
    const slab = acc.slice(s, e); // owns its buffer -> transferable
    credits--;
    ctx.postMessage({ type: "slab", buf: slab.buffer }, [slab.buffer]);
  };

  // Advance over complete frames, emitting a slab whenever enough whole frames
  // have accumulated, then compact away everything already emitted.
  const sliceFrames = (flush: boolean) => {
    while (scanOff + 12 <= accLen) {
      const o = scanOff;
      const len = acc[o + 8] | (acc[o + 9] << 8) | (acc[o + 10] << 16) | acc[o + 11] * 16777216;
      const frameEnd = o + 12 + len;
      if (frameEnd > accLen) break; // frame not fully arrived yet
      scanOff = frameEnd;
      lastBoundary = frameEnd;
      if (lastBoundary - cutStart >= SLAB_TARGET) {
        emitSlab(cutStart, lastBoundary);
        cutStart = lastBoundary;
      }
    }
    if (flush && lastBoundary > cutStart) {
      emitSlab(cutStart, lastBoundary);
      cutStart = lastBoundary;
    }
    if (cutStart > 0) {
      acc.copyWithin(0, cutStart, accLen);
      accLen -= cutStart;
      scanOff -= cutStart;
      lastBoundary -= cutStart;
      cutStart = 0;
    }
  };

  const dec = new Decompress((chunk) => {
    ensureCap(chunk.length);
    acc.set(chunk, accLen);
    accLen += chunk.length;
    sliceFrames(false);
  });

  let loaded = 0;
  for (;;) {
    await awaitCredit();
    if (aborted) return;
    const { done, value } = await reader.read();
    if (done) break;
    loaded += value.length;
    if (total) post(0.45 + 0.5 * Math.min(1, loaded / total), "loading op stream");
    dec.push(value, false);
  }
  dec.push(new Uint8Array(0), true);
  sliceFrames(true); // emit any trailing whole frames
  ctx.postMessage({ type: "end" });
}
