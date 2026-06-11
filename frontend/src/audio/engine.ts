import type { OscEvent } from "../stream/types";

// Web Audio port of the SuperCollider whisper cloud engine
// (supercollider/whisper_cloud_engine.scd). The install routes two whisper
// triggers; this mirrors both faithfully (action plan §7-7..7-9):
//
//   /bert/whisper       — a lemma's attention crossed threshold. Play the word,
//                         then its nearest graph neighbors as a delayed cloud,
//                         with discourse echoes for core/marker lemmas.
//   /bert/word_trigger  — accumulated |r| crossed the word threshold. Same voice,
//                         layer-scaled amp. (Absent from Phase-1 captures; dormant
//                         but wired so live mode just works.)
//
// The install's third trigger, /bert/op_flow (a 1/sec compute pulse that drove a
// continuous ambient drone; see docs/20260524-op-flow-ambient-audio.md), is
// intentionally dropped on the web (project decision 2026-06-03). The drone is
// not reproduced here; op_flow events are ignored by handle().
//
// Accent axis (the new dimension over the install's single neutral voice): every
// utterance picks 1 of 7 accents at trigger time (project decision 2026-05-30,
// "pure random per utterance"). An echo is the same utterance repeating, so it
// reuses its parent's accent; each neighbor is its own utterance and picks fresh.
// Assets are keyed in the generation namespace: {lid:06d}-{vidx:02d}__{accent}.opus.

// The 7 accents, in the generation order (mi300x_accent README). Index is only
// used to pick uniformly; the string is what keys the asset.
const ACCENTS = ["korean", "japanese", "arabic", "russian", "ukrainian", "american", "british"];

// Engine constants — ported 1:1 from whisper_cloud_engine.scd defaults so the web
// cloud reads identically to the install's.
const MAX_NBRS = 12;
const BASE_IOI = 0.2; // neighbor inter-onset base (s)
const DIST_IOI_SCALE = 0.5; // + dist * this (s)
const PAN_JITTER = 0.25;
const TRIG_AMP = 0.8; // main voice + discourse-boosted neighbors + echoes
const NBR_AMP_BASE = 0.45;
const NBR_AMP_DIST_SC = 0.3;
const NBR_AMP_MIN = 0.15;
const CACHE_SIZE = 256; // LRU AudioBuffer entries (~wh_cacheSize)

interface Discourse {
  core: Set<number>;
  marker: Set<number>;
  ids: Set<number>; // union, for the neighbor amp boost
}

export class AudioEngine {
  private ctx: AudioContext | null = null;
  private master!: GainNode;

  // LRU AudioBuffer cache, keyed by stem ("LLLLLL-VV__accent"). Mirrors
  // ~wh_getOrLoad: hit → reuse + bump recency; miss → fetch+decode once
  // (dedup in-flight), evicting the oldest when full.
  private cache = new Map<string, AudioBuffer>();
  private loading = new Set<string>();

  // Outstanding neighbor/echo timeouts, so a corpus switch can cancel the cloud
  // tail that the old corpus scheduled (reset()).
  private pendingTimers: ReturnType<typeof setTimeout>[] = [];

  private discourse: Discourse = { core: new Set(), marker: new Set(), ids: new Set() };

  // Base URL for opus assets. Cloudflare R2 public bucket (prefix "audio/") in
  // production; overridable for a local dir during dev before the R2 upload.
  private audioBase: string;

  constructor(audioBase = "/audio") {
    this.audioBase = audioBase.replace(/\/$/, "");
  }

  // Prefetch set (action plan §7-6): the (lid,vidx) pairs the loaded tag fires
  // first, so the demo's opening utterances have no network latency. Stored even
  // while disabled, then warmed once audio is enabled (decode needs the context).
  private prefetchPairs: [number, number][] = [];
  private static readonly PREFETCH_CAP = 64;

  get enabled(): boolean {
    return this.ctx !== null && this.ctx.state === "running";
  }

  // Register the pairs to warm. If audio is already running, warm immediately;
  // otherwise enable() will pick them up. A random accent per pair is enough —
  // it primes the network/decoder for that lemma; a different accent at trigger
  // time is a cheap miss, not a stall.
  setPrefetch(pairs: [number, number][]): void {
    this.prefetchPairs = pairs;
    if (this.enabled) this.warmPrefetch();
  }

  private warmPrefetch(): void {
    const n = Math.min(this.prefetchPairs.length, AudioEngine.PREFETCH_CAP);
    for (let i = 0; i < n; i++) {
      const [lid, vidx] = this.prefetchPairs[i];
      // Decode into the LRU; the result is discarded here but cached for reuse.
      this.getOrLoad(lid, vidx, this.pickAccent(), () => {});
    }
  }

  // Must be called from a user gesture (browser autoplay policy). Idempotent:
  // creates the context the first time, resumes it thereafter.
  async enable(): Promise<void> {
    if (!this.ctx) {
      this.ctx = new AudioContext();
      this.master = this.ctx.createGain();
      this.master.gain.value = 1;
      this.master.connect(this.ctx.destination);
      await this.loadDiscourse();
    }
    if (this.ctx.state !== "running") await this.ctx.resume();
    // Warm the opening utterances now that the context exists (idempotent: hits
    // already-cached entries cheaply on re-enable).
    this.warmPrefetch();
  }

  disable(): void {
    void this.ctx?.suspend();
  }

  // Hard reset for a corpus switch: cut every voice currently sounding and cancel
  // the scheduled cloud tail, so the old corpus goes fully silent before the new
  // one loads. Keeps the context running and the decoded LRU (opus assets are
  // keyed by lemma, corpus-independent) so the new corpus restarts cleanly. New
  // whisper events after the swap rebuild the cloud from scratch.
  reset(): void {
    for (const id of this.pendingTimers) clearTimeout(id);
    this.pendingTimers = [];
    if (!this.ctx) return;
    // Replacing the master node detaches every source feeding it, silencing all
    // in-flight buffers at once; reconnect a fresh master for the new corpus.
    try {
      this.master.disconnect();
    } catch {
      // already disconnected — nothing to do
    }
    this.master = this.ctx.createGain();
    this.master.gain.value = 1;
    this.master.connect(this.ctx.destination);
  }

  // Track a scheduled voice so reset() can cancel it. Prunes the id when it fires.
  private later(fn: () => void, ms: number): void {
    const id = setTimeout(() => {
      const i = this.pendingTimers.indexOf(id);
      if (i >= 0) this.pendingTimers.splice(i, 1);
      fn();
    }, ms);
    this.pendingTimers.push(id);
  }

  private async loadDiscourse(): Promise<void> {
    try {
      const d = await fetch("/discourse_lemma_ids_v3.json").then((r) => (r.ok ? r.json() : null));
      if (!d) return;
      for (const id of d.core ?? []) {
        this.discourse.core.add(id);
        this.discourse.ids.add(id);
      }
      for (const id of d.marker ?? []) {
        this.discourse.marker.add(id);
        this.discourse.ids.add(id);
      }
    } catch {
      // No discourse file → echoes simply never fire; cloud still plays.
    }
  }

  // ── Asset loading (LRU) ──────────────────────────────────────────────────
  private stem(lid: number, vidx: number, accent: string): string {
    const l = String(lid).padStart(6, "0");
    const v = String(vidx).padStart(2, "0");
    return `${l}-${v}__${accent}`;
  }

  private pickAccent(): string {
    return ACCENTS[(Math.random() * ACCENTS.length) | 0];
  }

  // Fetch+decode once, cache with LRU eviction, then invoke onReady. Re-getting a
  // cached stem bumps its recency. In-flight stems are not re-fetched (no pile-up).
  private getOrLoad(lid: number, vidx: number, accent: string, onReady: (b: AudioBuffer) => void): void {
    if (!this.ctx) return;
    const key = this.stem(lid, vidx, accent);
    const hit = this.cache.get(key);
    if (hit) {
      this.cache.delete(key);
      this.cache.set(key, hit); // Map keeps insertion order → re-insert = most recent
      onReady(hit);
      return;
    }
    if (this.loading.has(key)) return;
    this.loading.add(key);
    fetch(`${this.audioBase}/${key}.opus`)
      .then((r) => {
        if (!r.ok) throw new Error(`audio ${key}: ${r.status}`);
        return r.arrayBuffer();
      })
      .then((buf) => this.ctx!.decodeAudioData(buf))
      .then((audio) => {
        if (this.cache.size >= CACHE_SIZE) {
          const oldest = this.cache.keys().next().value;
          if (oldest !== undefined) this.cache.delete(oldest);
        }
        this.cache.set(key, audio);
        this.loading.delete(key);
        onReady(audio);
      })
      .catch(() => {
        // Missing/unsupported asset: drop this voice silently (logged once below).
        this.loading.delete(key);
        this.warnMissingOnce();
      });
  }

  private warned = false;
  private warnMissingOnce(): void {
    if (this.warned) return;
    this.warned = true;
    console.warn(`[audio] asset fetch failed under ${this.audioBase}/ — opus not uploaded yet?`);
  }

  // One whisper voice: PlayBuf → amp → stereo pan → master (SynthDef \whisperVoice).
  private playVoice(buf: AudioBuffer, amp: number, pan: number): void {
    const ctx = this.ctx!;
    const src = ctx.createBufferSource();
    src.buffer = buf;
    const g = ctx.createGain();
    g.gain.value = amp;
    const p = ctx.createStereoPanner();
    p.pan.value = Math.max(-1, Math.min(1, pan));
    src.connect(g).connect(p).connect(this.master);
    src.start();
  }

  // core → 3 echoes @1.5s; marker → 2 echoes @2.0s; all at trigAmp, jittered pan.
  // Echoes reuse the parent utterance's accent (same voice repeating).
  private scheduleEchoes(lid: number, vidx: number, accent: string, basePan: number, n: number, interval: number): void {
    for (let k = 0; k < n; k++) {
      const delay = (k + 1) * interval + (Math.random() - 0.5) * 0.3;
      const ePan = Math.max(-1, Math.min(1, basePan + (Math.random() * 2 - 1) * PAN_JITTER * 2));
      this.later(() => {
        this.getOrLoad(lid, vidx, accent, (b) => this.playVoice(b, TRIG_AMP, ePan));
      }, delay * 1000);
    }
  }

  private maybeEcho(lid: number, vidx: number, accent: string, pan: number): void {
    if (this.discourse.core.has(lid)) this.scheduleEchoes(lid, vidx, accent, pan, 3, 1.5);
    else if (this.discourse.marker.has(lid)) this.scheduleEchoes(lid, vidx, accent, pan, 2, 2.0);
  }

  // ── Event entry point ──────────────────────────────────────────────────────
  // Fed the same OSC events the panels see. No-op until enable()d, so it is safe
  // to call every frame regardless of audio state.
  handle(events: OscEvent[]): void {
    if (!this.enabled) return;
    for (const ev of events) {
      switch (ev.path) {
        case "/bert/whisper":
          this.onWhisper(ev.args);
          break;
        case "/bert/word_trigger":
          this.onWordTrigger(ev.args);
          break;
        default:
          break;
      }
    }
  }

  // /bert/whisper arg layout (0-indexed, no path; cf. mon_whisper + bert.c):
  //   [0] lid  [1] vidx  [2] is_bridge  [3..10] affinity[8]  [11] n_nbrs
  //   then n_nbrs × { lid, vidx, dist }
  private onWhisper(a: number[]): void {
    const lid = a[0] | 0;
    const vidx = a[1] | 0;
    const n = a[11] | 0;
    const pan = Math.random() * 2 - 1; // main voice: random pan
    const accent = this.pickAccent();

    this.getOrLoad(lid, vidx, accent, (b) => this.playVoice(b, TRIG_AMP, pan));
    this.maybeEcho(lid, vidx, accent, pan);

    const kn = Math.min(n, MAX_NBRS);
    for (let j = 0; j < kn; j++) {
      const base = 12 + j * 3;
      const nLid = a[base] | 0;
      const nVidx = a[base + 1] | 0;
      const dist = a[base + 2];
      const delay = (j + 1) * BASE_IOI + dist * DIST_IOI_SCALE;
      const nAmp = this.discourse.ids.has(nLid)
        ? TRIG_AMP
        : Math.max(NBR_AMP_MIN, NBR_AMP_BASE - dist * NBR_AMP_DIST_SC);
      const nPan = Math.max(-1, Math.min(1, pan + (Math.random() * 2 - 1) * PAN_JITTER));
      const nAccent = this.pickAccent(); // each neighbor is its own utterance

      this.later(() => {
        this.getOrLoad(nLid, nVidx, nAccent, (b) => this.playVoice(b, nAmp, nPan));
        this.maybeEcho(nLid, nVidx, nAccent, nPan);
      }, delay * 1000);
    }
  }

  // /bert/word_trigger: [0] lid  [1] layer. Same voice, layer-scaled amp, vidx 0.
  private onWordTrigger(a: number[]): void {
    const lid = a[0] | 0;
    const layer = a[1] | 0;
    const pan = Math.random() * 2 - 1;
    const amp = 0.35 + Math.max(0, layer) / 36; // layers 0-11 → 0.35-0.66
    const accent = this.pickAccent();
    this.getOrLoad(lid, 0, accent, (b) => this.playVoice(b, amp, pan));
  }
}
