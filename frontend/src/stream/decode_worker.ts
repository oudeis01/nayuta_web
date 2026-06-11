// decode_worker.ts — off-main-thread fetch + zstd decompress for a capture dump.
//
// The dump's ops.bin.zst is ~93 MB compressed / ~180 MB decompressed; calling
// fzstd's synchronous decompress() on the main thread froze the whole UI for the
// duration of a corpus switch (DumpAdapter used to do exactly that). This worker
// moves the fetch + decompress (and the small events.jsonl parse) off the main
// thread, so the loader spinner and bar keep animating while a switch loads. The
// decompressed ops buffer is handed back as a Transferable (zero-copy when the
// fzstd output buffer is already tight, one copy otherwise); the main thread then
// runs the cheap frame-index walk in DumpAdapter.indexOps.
import { decompress } from "fzstd";
import type { OscEvent } from "./types";

interface LoadReq {
  id: number;
  base: string;
  ops: boolean;
}

// The worker global, typed narrowly so we can postMessage with a transfer list
// without pulling the full "webworker" lib into the DOM-targeted tsconfig.
const ctx = self as unknown as {
  postMessage(message: unknown, transfer?: Transferable[]): void;
  onmessage: ((e: MessageEvent<LoadReq>) => void) | null;
};

// Fetch a URL into a Uint8Array, reporting download progress (0..1) as bytes
// arrive (moved here from DumpAdapter). Falls back to a plain arrayBuffer read
// when the body cannot be streamed or content-length is missing.
async function fetchBytes(url: string, onProgress: (frac: number) => void): Promise<Uint8Array> {
  const res = await fetch(url);
  if (!res.ok) throw new Error(`fetch ${url}: ${res.status}`);
  const total = Number(res.headers.get("content-length")) || 0;
  if (!res.body || !total) {
    const buf = new Uint8Array(await res.arrayBuffer());
    onProgress(1);
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
    onProgress(Math.min(1, loaded / total));
  }
  const out = new Uint8Array(loaded);
  let off = 0;
  for (const c of chunks) {
    out.set(c, off);
    off += c.length;
  }
  return out;
}

// Return a tight ArrayBuffer for the bytes so it can be transferred. fzstd's
// output is usually exactly sized (offset 0, buffer length == byte length), in
// which case we hand the buffer straight across; otherwise copy once.
function tightBuffer(u: Uint8Array): ArrayBuffer {
  if (u.byteOffset === 0 && u.buffer.byteLength === u.byteLength) return u.buffer as ArrayBuffer;
  return u.slice().buffer;
}

ctx.onmessage = async (e: MessageEvent<LoadReq>) => {
  const { id, base, ops } = e.data;
  const post = (frac: number, stage: string) =>
    ctx.postMessage({ id, type: "progress", frac, stage });
  try {
    // events.jsonl.zst is small next to ops.bin.zst, so it gets the first 8% of
    // the bar; the heavy ops fetch (when mounted) drives the remaining 92%.
    const evComp = await fetchBytes(`${base}/events.jsonl.zst`, (p) => post(p * 0.08, "loading events"));
    const evRaw = decompress(evComp);
    const text = new TextDecoder().decode(evRaw);
    const events: OscEvent[] = [];
    for (const line of text.split("\n")) {
      if (!line) continue;
      const o = JSON.parse(line);
      events.push({ tsNs: o.ts_ns, path: o.path, types: o.types, args: o.args });
    }
    events.sort((a, b) => a.tsNs - b.tsNs);

    let opsBuffer: ArrayBuffer | null = null;
    if (ops) {
      const opComp = await fetchBytes(`${base}/ops.bin.zst`, (p) => post(0.08 + p * 0.92, "loading op stream"));
      opsBuffer = tightBuffer(decompress(opComp));
    }

    ctx.postMessage(
      { id, type: "done", events, opsBuffer },
      opsBuffer ? [opsBuffer] : [],
    );
  } catch (err) {
    ctx.postMessage({ id, type: "error", message: String(err) });
  }
};
