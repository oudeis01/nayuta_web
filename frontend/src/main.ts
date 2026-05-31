import "./style.css";
import { DumpAdapter } from "./stream/dump_adapter";
import { MonClock } from "./panels/mon_clock";
import { MonWhisper } from "./panels/mon_whisper";
import { MonOpStream } from "./panels/mon_opstream";
import { MonScanner } from "./panels/mon_scanner";
import { ScreenRain } from "./panels/screen_rain";
import { MonMandala } from "./panels/mon_mandala";
import { MonEntropy } from "./panels/mon_entropy";
import { AudioEngine } from "./audio/engine";
import type { OscEvent, OpRec } from "./stream/types";

const SESSION = "session_001";
// Asset bases (action plan §3-1, §12): the static frontend ships on Cloudflare
// Pages, but the heavy assets — opus audio and the per-tag ops.bin.zst dumps,
// which exceed the Pages 25 MB/file limit — live in the public R2 bucket. Both
// default to a same-origin path so local dev still serves from public/.
const CAPTURE_BASE = (import.meta.env.VITE_CAPTURE_BASE ?? "/captures").replace(/\/$/, "");
const AUDIO_BASE = import.meta.env.VITE_AUDIO_BASE; // undefined → AudioEngine default
const TAGS = [
  "tate_nnn",
  "mousse_irigaray",
  "artforum_identity",
  "ars_baecker",
  "ars_bias",
];
const SPEEDS = [1, 2, 5, 10, 30, 60];

// Every panel speaks this contract: digest the frame's events, then paint into
// a sub-rect of the shared canvas. render(x,y,w,h) is already the C++ panel
// signature, so the compositor only has to hand each panel its rectangle.
interface Panel {
  update(events: OscEvent[], ops: OpRec[], dt: number): void;
  render(ctx: CanvasRenderingContext2D, x: number, y: number, w: number, h: number): void;
}
interface Slot {
  panel: Panel;
  weight: number; // horizontal share within the monitor row
  ops?: boolean; // true if this panel consumes the OpRec stream
}

// View modes (action plan 8-3): the install's physical multi-monitor wall folds
// into one browser page. Two modes share one data state machine — every panel
// still update()s each frame; only render() differs. AGGREGATE composites the
// whole wall; a single-view index 0..6 full-bleeds one panel at the fidelity of
// the original monitor. Keys 1-7 pick a single view, 0 or ` returns to aggregate.
const AGGREGATE = -1;
let viewMode = AGGREGATE;
// Display labels for the single-view order, matching keys 1-7.
const VIEW_LABELS = [
  "0  computation rain",
  "A  token scanner",
  "B  op stream",
  "C  the clock",
  "D  whisper",
  "E  head mandala",
  "F  entropy breath",
];

// Seam between panels (device px) so adjacent screens read as distinct monitors.
const GAP = 2;
// Canonical aggregate layout (design doc §0-1, mirroring the HTML mockup's
// `grid-template-rows: 1fr 34vh`): Screen 0 fills the top band, the A–F monitor
// strip takes the bottom 34% of the viewport height.
const STRIP_FRAC = 0.34;

const app = document.querySelector<HTMLDivElement>("#app")!;
const canvas = document.createElement("canvas");
app.appendChild(canvas);
const ctx = canvas.getContext("2d")!;

// HUD: session picker + playback controls (dev-time; the install pace is 1x).
const hud = document.createElement("div");
hud.id = "hud";
const sessionSel = document.createElement("select");
for (const t of TAGS) {
  const o = document.createElement("option");
  o.value = t;
  o.textContent = t;
  sessionSel.appendChild(o);
}
const pauseBtn = document.createElement("button");
pauseBtn.textContent = "pause";
const slowBtn = document.createElement("button");
slowBtn.textContent = "-";
const fastBtn = document.createElement("button");
fastBtn.textContent = "+";
const speedLbl = document.createElement("span");
speedLbl.className = "speed";
// Audio is off until a user gesture (browser autoplay policy); this toggle both
// creates/resumes the AudioContext and reflects state.
const audioBtn = document.createElement("button");
audioBtn.textContent = "audio off";
// Info trigger lives in the HUD row so it shares the button styling and alignment.
const infoBtn = document.createElement("button");
infoBtn.textContent = "info";
hud.append(sessionSel, slowBtn, speedLbl, fastBtn, pauseBtn, audioBtn, infoBtn);
document.body.appendChild(hud);

// The whisper cloud / ambient drone engine (action plan §7). Fed the same OSC
// events as the panels; dormant until the viewer enables audio.
const audio = new AudioEngine(AUDIO_BASE);
audioBtn.addEventListener("click", async () => {
  if (audio.enabled) {
    audio.disable();
    audioBtn.textContent = "audio off";
  } else {
    await audio.enable();
    audioBtn.textContent = "audio on";
  }
});

// Persistent provenance line (action plan §10): single bottom-right line, always
// above the canvas so panel rendering can never bury it. The honest capture
// framing is part of the work, so it stays visible in every view.
const metaLine = document.createElement("div");
metaLine.id = "meta-line";
document.body.appendChild(metaLine);

// Info modal: the fuller provenance and the view controls.
const modalOverlay = document.createElement("div");
modalOverlay.id = "modal-overlay";
modalOverlay.className = "hidden";
const modal = document.createElement("div");
modal.id = "modal";
modalOverlay.appendChild(modal);
document.body.appendChild(modalOverlay);

function openModal() {
  modal.innerHTML = modalHtml();
  modal.querySelector<HTMLButtonElement>(".close")!.addEventListener("click", closeModal);
  modalOverlay.classList.remove("hidden");
}
function closeModal() {
  modalOverlay.classList.add("hidden");
}
infoBtn.addEventListener("click", openModal);
modalOverlay.addEventListener("click", (e) => {
  if (e.target === modalOverlay) closeModal();
});

function resize() {
  const dpr = window.devicePixelRatio || 1;
  canvas.width = Math.floor(canvas.clientWidth * dpr);
  canvas.height = Math.floor(canvas.clientHeight * dpr);
}
window.addEventListener("resize", resize);

// The lemma_id -> surface dictionary and the BERT vocab (token_id -> wordpiece,
// dense, indexed by line) are corpus-wide, so load each once and share it with
// every freshly built panel.
let lemmaDict: Record<string, string> = {};
let bertVocab: string[] = [];

let adapter: DumpAdapter;
let topPanel: Panel | null = null; // Screen 0, the full-width top band
let topOps = false;
let slots: Slot[] = []; // A–F monitor strip (bottom)

// UI meta-info (action plan §10): the capture's honest provenance, shown as a
// persistent corner label. The honest framing — length, corpus, data mode — is
// part of the work's meaning, not chrome. Populated from each tag's manifest.
interface MetaInfo {
  corpus: string; // source corpus name (e.g. "e-flux #14203")
  durationS: number; // capture length in seconds
}
let meta: MetaInfo | null = null;
// These captures are the pre-show local tap dump (action plan §3-1). Audio is the
// 2-channel Opus downmix (§2, §7). Both are fixed for Phase 1.
const DATA_MODE = "PRE-SHOW DUMP";
const AUDIO_FMT = "Opus 32kbps mono · stereo downmix";
let pending: OscEvent[] = [];
let pendingOps: OpRec[] = [];
let speedIdx = 0;
let paused = false;

function applySpeed() {
  adapter?.setPlaybackSpeed(SPEEDS[speedIdx]);
  speedLbl.textContent = `${SPEEDS[speedIdx]}x`;
}

async function loadTag(tag: string) {
  const base = `${CAPTURE_BASE}/${SESSION}/${tag}`;
  const a = new DumpAdapter(base);

  // Fresh panels per tag so replay state (counters, logs) resets cleanly.
  const rain = new ScreenRain();
  const scanner = new MonScanner();
  scanner.setVocab(bertVocab);
  const opstream = new MonOpStream();
  const clock = new MonClock();
  const whisper = new MonWhisper();
  whisper.setLemmaDict(lemmaDict);

  // Canonical aggregate layout (design doc §0-1): Screen 0 across the top band,
  // then the A–F monitor strip below at 17:17:9:17:17:9. Monitors E and F run the
  // canonical pre-attention mode (E = (a,r) topology heatmap, F = |R| magnitude),
  // both fed by the OpRec stream — the same view the install shows until the run
  // reaches the attention stage (which these captures never do).
  const mandala = new MonMandala();
  const entropy = new MonEntropy();
  topPanel = rain;
  topOps = true;
  slots = [
    { panel: scanner, weight: 17, ops: true },
    { panel: opstream, weight: 17, ops: true },
    { panel: clock, weight: 9 },
    { panel: whisper, weight: 17 },
    { panel: mandala, weight: 17, ops: true },
    { panel: entropy, weight: 9, ops: true },
  ];

  // Only pay the ops.bin fetch/decompress when a mounted panel consumes it.
  const needOps = topOps || slots.some((s) => s.ops);
  const [, manifest] = await Promise.all([
    a.load({ ops: needOps }),
    fetch(`${base}/manifest.json`).then((r) => (r.ok ? r.json() : null)),
  ]);
  adapter = a;

  const cs = manifest?.corpus_sidecar;
  if (cs) clock.setSeqMeta(cs.source_seq_idx, cs.source_name, cs.seq_len);

  // Provenance for the §10 meta label. Prefer the sidecar's human corpus name;
  // fall back to the capture tag so the label never reads empty.
  meta = {
    corpus: cs?.source_name ?? manifest?.tag ?? tag,
    durationS: manifest?.duration_s ?? manifest?.duration_cap_s ?? 0,
  };

  // Prime the opening utterances' audio (action plan §7-6). No-op until the
  // viewer enables audio; enable() will replay this set.
  audio.setPrefetch(adapter.firstWhisperPairs(64));

  pending = [];
  pendingOps = [];
  adapter.onOsc((ev) => pending.push(ev));
  adapter.onOpRec((rec) => pendingOps.push(rec));
  applySpeed();
  if (paused) adapter.pause();
}

sessionSel.addEventListener("change", () => loadTag(sessionSel.value));
slowBtn.addEventListener("click", () => {
  speedIdx = Math.max(0, speedIdx - 1);
  applySpeed();
});
fastBtn.addEventListener("click", () => {
  speedIdx = Math.min(SPEEDS.length - 1, speedIdx + 1);
  applySpeed();
});
function togglePause() {
  paused = !paused;
  pauseBtn.textContent = paused ? "resume" : "pause";
  if (paused) adapter?.pause();
  else adapter?.resume();
}
pauseBtn.addEventListener("click", togglePause);
window.addEventListener("keydown", (e) => {
  if (e.code === "Escape") {
    closeModal();
    return;
  }
  if (e.code === "Space") {
    e.preventDefault();
    togglePause();
    return;
  }
  // View switch (action plan 8-3): 1-7 full-bleed a single panel, 0 or ` return
  // to the aggregate wall. Digit codes are layout-independent (Digit1..Digit7).
  if (e.code === "Digit0" || e.code === "Backquote") {
    viewMode = AGGREGATE;
    return;
  }
  const m = /^Digit([1-7])$/.exec(e.code);
  if (m) viewMode = Number(m[1]) - 1;
});

// The panel at a single-view index: 0 = Screen 0 (top band), 1-6 = the A–F strip.
function viewPanel(i: number): Panel | null {
  if (i === 0) return topPanel;
  return slots[i - 1]?.panel ?? null;
}

// Render the active view. AGGREGATE is the two-tier wall: Screen 0 across the top
// band, the A–F strip splitting the bottom STRIP_FRAC of height by weight, each
// rect inset by GAP so adjacent screens read as separate monitors. A single view
// full-bleeds one panel at full fidelity. The meta label (§10) overlays both.
function renderLayout() {
  const W = canvas.width;
  const H = canvas.height;

  if (viewMode === AGGREGATE) {
    const stripH = Math.round(H * STRIP_FRAC);
    const topH = H - stripH;

    if (topPanel) topPanel.render(ctx, GAP, GAP, W - GAP * 2, topH - GAP * 2);

    const total = slots.reduce((sum, s) => sum + s.weight, 0);
    let x = 0;
    for (const s of slots) {
      const w = Math.round((W * s.weight) / total);
      s.panel.render(ctx, x + GAP, topH + GAP, w - GAP * 2, stripH - GAP * 2);
      x += w;
    }
  } else {
    const panel = viewPanel(viewMode);
    if (panel) panel.render(ctx, GAP, GAP, W - GAP * 2, H - GAP * 2);
  }
}

// Current capture position as m:ss (— before a tag is loaded).
function posStr(): string {
  if (!adapter) return "0:00";
  return `${Math.floor(adapter.position / 60)}:${String(Math.floor(adapter.position % 60)).padStart(2, "0")}`;
}
function durStr(): string {
  return meta ? `${Math.round(meta.durationS / 60)} min` : "—";
}
function viewStr(): string {
  return viewMode === AGGREGATE ? "AGGREGATE" : VIEW_LABELS[viewMode];
}

// Refresh the persistent bottom-right line each frame (cheap text assignment).
function updateMetaLine() {
  metaLine.textContent =
    `${DATA_MODE} · ${viewStr()} · ${meta?.corpus ?? "—"} · ${posStr()} / ${durStr()} @ ${SPEEDS[speedIdx]}x`;
}

// Modal body: the fuller §10 provenance plus the view controls. Built fresh on
// open so it reflects live state (position, speed, view).
function modalHtml(): string {
  const views = VIEW_LABELS.map((label, i) => `<kbd>${i + 1}</kbd> ${label}`).join("<br>");
  return `
    <h2>namedrop · bert wall</h2>
    <h3>capture</h3>
    <dl>
      <dt>data</dt><dd>${DATA_MODE}</dd>
      <dt>corpus</dt><dd>${meta?.corpus ?? "—"}</dd>
      <dt>length</dt><dd>${durStr()} · ${posStr()} @ ${SPEEDS[speedIdx]}x</dd>
      <dt>audio</dt><dd>${AUDIO_FMT}</dd>
    </dl>
    <h3>views</h3>
    <dl>
      <dt><kbd>0</kbd> / <kbd>\`</kbd></dt><dd>aggregate wall</dd>
      <dt><kbd>space</kbd></dt><dd>pause / resume</dd>
    </dl>
    <p style="margin-top:6px">${views}</p>
    <button class="close">close</button>
  `;
}

let last = performance.now();
function frame(now: number) {
  const realDt = Math.min(0.1, (now - last) / 1000);
  last = now;
  if (adapter && slots.length) {
    const pdt = adapter.advance(realDt);
    const evs = pending;
    const ops = pendingOps;
    pending = [];
    pendingOps = [];
    audio.handle(evs);
    ctx.fillStyle = "#000";
    ctx.fillRect(0, 0, canvas.width, canvas.height);
    topPanel?.update(evs, ops, pdt);
    for (const s of slots) s.panel.update(evs, ops, pdt);
    renderLayout();
    updateMetaLine();
  }
  requestAnimationFrame(frame);
}

resize();
// Pull the corpus-wide lookups before the first tag so Monitor D resolves
// whisper words and Monitor A resolves tokens from the very first event.
Promise.all([
  fetch("/lemma_surface.json")
    .then((r) => (r.ok ? r.json() : {}))
    .then((d) => (lemmaDict = d)),
  fetch("/bert_vocab.txt")
    .then((r) => (r.ok ? r.text() : ""))
    .then((t) => (bertVocab = t.split("\n"))),
])
  .then(() => loadTag(TAGS[0]))
  .then(() => requestAnimationFrame(frame));
