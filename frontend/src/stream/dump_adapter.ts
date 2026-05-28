import { decompress } from "fzstd";
import type { EventStream, OscEvent, OpRec } from "./types";

// Replays a captured session from its zstd-compressed event log.
//
// The capture preserves real wall-clock timing in each event's ts_ns. Playback
// maps that capture timeline onto frames: each advance(realDt) moves a playback
// clock forward by realDt * speed and fires every event whose timestamp has
// been reached. This keeps the install's true pacing (action plan 11) while
// letting the viewer scrub speed.
//
// Pull model: the host owns the rAF loop and calls advance() once per frame.
// onOsc handlers fire synchronously inside advance(). ops.bin (the high-rate
// OpRec stream) is not loaded here yet; panels that need it (Monitor B,
// Screen 0) will extend this adapter.
export class DumpAdapter implements EventStream {
  private oscHandlers: ((ev: OscEvent) => void)[] = [];
  private opHandlers: ((rec: OpRec) => void)[] = [];

  private events: OscEvent[] = [];
  private idx = 0;
  private t0 = 0; // ts_ns of first event
  private playT = 0; // playback seconds elapsed since t0
  private speed = 1;
  private paused = false;
  private baseUrl: string;

  constructor(baseUrl: string) {
    this.baseUrl = baseUrl;
  }

  onOsc(handler: (ev: OscEvent) => void): void {
    this.oscHandlers.push(handler);
  }
  onOpRec(handler: (rec: OpRec) => void): void {
    this.opHandlers.push(handler);
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

  async load(): Promise<void> {
    const res = await fetch(`${this.baseUrl}/events.jsonl.zst`);
    if (!res.ok) throw new Error(`fetch ${this.baseUrl}: ${res.status}`);
    const comp = new Uint8Array(await res.arrayBuffer());
    const raw = decompress(comp);
    const text = new TextDecoder().decode(raw);

    const evs: OscEvent[] = [];
    for (const line of text.split("\n")) {
      if (!line) continue;
      const o = JSON.parse(line);
      evs.push({ tsNs: o.ts_ns, path: o.path, types: o.types, args: o.args });
    }
    evs.sort((a, b) => a.tsNs - b.tsNs);
    this.events = evs;
    this.t0 = evs.length ? evs[0].tsNs : 0;
    this.idx = 0;
    this.playT = 0;
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
    return pdt;
  }
}
