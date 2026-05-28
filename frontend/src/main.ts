import "./style.css";
import { DumpAdapter } from "./stream/dump_adapter";
import { MonClock } from "./panels/mon_clock";
import type { OscEvent } from "./stream/types";

const SESSION = "session_001";
const TAGS = [
  "tate_nnn",
  "mousse_irigaray",
  "artforum_identity",
  "ars_baecker",
  "ars_bias",
];
const SPEEDS = [1, 2, 5, 10, 30, 60];

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

let adapter: DumpAdapter;
let clock: MonClock;
let pending: OscEvent[] = [];
let speedIdx = 0;
let paused = false;

function applySpeed() {
  adapter?.setPlaybackSpeed(SPEEDS[speedIdx]);
  speedLbl.textContent = `${SPEEDS[speedIdx]}x`;
}

async function loadTag(tag: string) {
  const base = `/captures/${SESSION}/${tag}`;
  const a = new DumpAdapter(base);
  const [, manifest] = await Promise.all([
    a.load(),
    fetch(`${base}/manifest.json`).then((r) => (r.ok ? r.json() : null)),
  ]);
  adapter = a;
  clock = new MonClock();
  const cs = manifest?.corpus_sidecar;
  if (cs) clock.setSeqMeta(cs.source_seq_idx, cs.source_name, cs.seq_len);
  pending = [];
  adapter.onOsc((ev) => pending.push(ev));
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

let last = performance.now();
function frame(now: number) {
  const realDt = Math.min(0.1, (now - last) / 1000);
  last = now;
  if (adapter && clock) {
    const pdt = adapter.advance(realDt);
    const evs = pending;
    pending = [];
    clock.update(evs, pdt);
    clock.render(ctx, 0, 0, canvas.width, canvas.height);
  }
  requestAnimationFrame(frame);
}

resize();
loadTag(TAGS[0]).then(() => requestAnimationFrame(frame));
