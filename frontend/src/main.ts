import "./style.css";
import { DumpAdapter } from "./stream/dump_adapter";
import { MonClock } from "./panels/mon_clock";
import { MonWhisper } from "./panels/mon_whisper";
import { MonOpStream } from "./panels/mon_opstream";
import { MonScanner } from "./panels/mon_scanner";
import { ScreenRain } from "./panels/screen_rain";
import { MonPlaceholder } from "./panels/mon_placeholder";
import type { OscEvent, OpRec } from "./stream/types";

const SESSION = "session_001";
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
hud.append(sessionSel, slowBtn, speedLbl, fastBtn, pauseBtn);
document.body.appendChild(hud);

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
let pending: OscEvent[] = [];
let pendingOps: OpRec[] = [];
let speedIdx = 0;
let paused = false;

function applySpeed() {
  adapter?.setPlaybackSpeed(SPEEDS[speedIdx]);
  speedLbl.textContent = `${SPEEDS[speedIdx]}x`;
}

async function loadTag(tag: string) {
  const base = `/captures/${SESSION}/${tag}`;
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
  // then the A–F monitor strip below at 17:17:9:17:17:9. Monitors E and F hold
  // placeholder slots — their /bert/att_w source never appears in these captures
  // (the run never reaches the attention stage), so they stay dark but keep the
  // grid proportions faithful.
  topPanel = rain;
  topOps = true;
  slots = [
    { panel: scanner, weight: 17, ops: true },
    { panel: opstream, weight: 17, ops: true },
    { panel: clock, weight: 9 },
    { panel: whisper, weight: 17 },
    { panel: new MonPlaceholder("E", "head mandala", "att_w 부재 · 정적"), weight: 17 },
    { panel: new MonPlaceholder("F", "entropy breath", "att_w 부재 · 정적"), weight: 9 },
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
  if (e.code === "Space") {
    e.preventDefault();
    togglePause();
  }
});

// Two-tier aggregate: Screen 0 spans the top band, the A–F strip splits the
// bottom STRIP_FRAC of height by weight. Each rect is inset by GAP so adjacent
// screens read as separate monitors.
function renderLayout() {
  const W = canvas.width;
  const H = canvas.height;
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
    ctx.fillStyle = "#000";
    ctx.fillRect(0, 0, canvas.width, canvas.height);
    topPanel?.update(evs, ops, pdt);
    for (const s of slots) s.panel.update(evs, ops, pdt);
    renderLayout();
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
