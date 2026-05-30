import type { OscEvent, OpRec } from "../stream/types";
import { Nebula } from "../geometry/nebula";

// Monitor D — Whisper Log. Faithful port of the graphics_consumer whisper panel
// (action plan 8-4; design doc "Monitor D"). Each /bert/whisper event fires when
// a lemma's attention crosses the trigger threshold; the panel shows the lemma
// paired with its nearest graph neighbor and the distance between them. Pairs
// accumulate top-to-bottom and fade out as the column fills, a blinking cursor
// trailing the newest line. Bridge triggers (cross-discourse) carry a marker.
//
// The /bert/whisper arg layout (cf. bert.c osc_whisper):
//   [0] triggered_lemma_id  [1] triggered_variant_idx  [2] is_bridge
//   [3..10] affinity[8]     [11] n_nbrs
//   then n_nbrs x { lemma_id:int, variant_idx:int, dist:float }
// We display the triggered lemma and the first (closest) neighbor.

const REF_H = 600; // reference panel height the pixel constants were tuned for
const MAX_LIVE = 9; // visible pairs before the oldest fades out
const FADE_IN = 0.25; // seconds for a new line to ramp to full brightness
const FADE_OUT = 0.6; // seconds for an overflowed line to fade away
const BLINK_PERIOD = 1.1; // cursor blink cycle (seconds), on for the first half

type Kind = "pair" | "sep";

interface Entry {
  kind: Kind;
  src: string;
  dst: string;
  dist: number;
  bridge: boolean;
  fadeIn: number; // 0..1 ramps up on birth
  dying: boolean;
  death: number; // seconds since marked dying
}

type LemmaDict = Record<string, string>;

export class MonWhisper {
  private entries: Entry[] = [];
  private dict: LemmaDict = {};
  private blink = 0;

  // Background discourse nebula (semantic neighbor graph) — sibling form to the
  // structural Helix, fed the same whisper events as the log.
  private nebula = new Nebula();

  // Resolve a lemma_id to its display surface. Unknown ids (out of corpus)
  // degrade to a bracketed numeric so the panel never renders a blank pair.
  private surface(lemmaId: number): string {
    return this.dict[String(lemmaId)] ?? `[${lemmaId}]`;
  }

  setLemmaDict(dict: LemmaDict): void {
    this.dict = dict;
  }

  private push(e: Omit<Entry, "fadeIn" | "dying" | "death">): void {
    this.entries.push({ ...e, fadeIn: 0, dying: false, death: 0 });
  }

  update(events: OscEvent[], _ops: OpRec[], dt: number): void {
    this.blink += dt;

    for (const ev of events) {
      switch (ev.path) {
        case "/bert/whisper": {
          const a = ev.args;
          const srcId = a[0];
          const bridge = a[2] === 1;
          const nNbr = a[11] | 0;
          if (nNbr > 0) {
            const nbrId = a[12];
            const dist = a[14];
            this.push({ kind: "pair", src: this.surface(srcId), dst: this.surface(nbrId), dist, bridge });

            // Feed the discourse nebula: src + up to 4 nearest neighbors.
            const nbrs: [string, number][] = [];
            const kn = Math.min(nNbr, 4);
            for (let i = 0; i < kn; i++) {
              nbrs.push([this.surface(a[12 + i * 3]), a[14 + i * 3]]);
            }
            this.nebula.addWhisper(this.surface(srcId), nbrs, bridge);
          }
          break;
        }
        // Axis 1: a token finished attention across all heads. Marks a boundary
        // between whisper bursts. (Absent from the Phase-1 captures; dormant.)
        case "/bert/token_att":
          this.push({ kind: "sep", src: "", dst: "", dist: 0, bridge: false });
          break;
        // Axis 3: a layer completed over the whole sequence. Clear and restart.
        case "/bert/layer":
          for (const e of this.entries) e.dying = true;
          break;
        default:
          break;
      }
    }

    // Age entries: ramp births in, overflowed oldest out, then reap the dead.
    let live = 0;
    for (const e of this.entries) if (!e.dying) live++;
    for (const e of this.entries) {
      if (live > MAX_LIVE && !e.dying) {
        e.dying = true;
        live--;
      }
      if (e.dying) e.death += dt;
      else e.fadeIn = Math.min(1, e.fadeIn + dt / FADE_IN);
    }
    this.entries = this.entries.filter((e) => !(e.dying && e.death >= FADE_OUT));

    this.nebula.update(dt);
  }

  render(ctx: CanvasRenderingContext2D, x: number, y: number, w: number, h: number): void {
    const s = h / REF_H;
    const fontSM = 13 * s;
    const fontMD = 15 * s;
    const mono = "JetBrains Mono, ui-monospace, monospace";

    ctx.fillStyle = "#000";
    ctx.fillRect(x, y, w, h);

    // Discourse nebula spans the panel as a background layer; the log feed draws
    // on top at the top-left (faithful to the panel's "background nebula" intent).
    const footerH = 20 * s;
    this.nebula.draw(ctx, x + 8 * s, y + 24 * s, w - 16 * s, h - 24 * s - footerH, 11 * s);

    ctx.textBaseline = "top";
    ctx.textAlign = "left";

    const grey = (v: number, a = 1) => {
      const c = Math.round(v * 255);
      return `rgba(${c},${c},${c},${a})`;
    };

    // Top-left label
    ctx.font = `${fontSM}px ${mono}`;
    ctx.fillStyle = grey(0.12);
    ctx.fillText("D  whisper", x + 10 * s, y + 6 * s);

    const padX = 12 * s;
    const lineH = fontMD * 1.6;
    let cy = y + 34 * s;
    let lastBottom = cy;

    for (const e of this.entries) {
      const alpha = e.dying ? Math.max(0, 1 - e.death / FADE_OUT) : e.fadeIn;
      if (e.kind === "sep") {
        ctx.strokeStyle = grey(0.16, alpha);
        ctx.lineWidth = 1;
        const sy = Math.round(cy + lineH * 0.4) + 0.5;
        ctx.beginPath();
        ctx.moveTo(x + padX, sy);
        ctx.lineTo(x + w - padX, sy);
        ctx.stroke();
        cy += lineH * 0.7;
        lastBottom = cy;
        continue;
      }

      let tx = x + padX;
      if (e.bridge) {
        ctx.font = `${fontMD}px ${mono}`;
        ctx.fillStyle = grey(0.55, alpha);
        ctx.fillText("*", tx, cy);
        tx += ctx.measureText("* ").width;
      }

      const seg = (text: string, v: number, font: string) => {
        ctx.font = font;
        ctx.fillStyle = grey(v, alpha);
        ctx.fillText(text, tx, cy);
        tx += ctx.measureText(text).width;
      };
      seg(`"${e.src}"`, 0.85, `${fontMD}px ${mono}`);
      seg("  →  ", 0.3, `${fontMD}px ${mono}`);
      seg(`"${e.dst}"`, 0.85, `${fontMD}px ${mono}`);
      seg(`  (${e.dist.toFixed(2)})`, 0.32, `${fontSM}px ${mono}`);

      cy += lineH;
      lastBottom = cy;
    }

    // Blinking cursor trailing the newest line.
    if (this.blink % BLINK_PERIOD < BLINK_PERIOD / 2) {
      ctx.font = `${fontMD}px ${mono}`;
      ctx.fillStyle = grey(0.5);
      ctx.fillText("_", x + padX, lastBottom);
    }
  }
}
