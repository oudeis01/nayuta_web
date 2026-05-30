import type { OscEvent, OpRec } from "../stream/types";

// Placeholder for a monitor whose data source is absent from the current
// captures. Monitor E (Head Mandala) and Monitor F (Entropy Breath) both feed on
// post-softmax attention weights (/bert/att_w), which never appear in the Phase-1
// captures: at install pace a 2-hour window never reaches the attention stage
// (it is still projecting the first few tokens into Q/K/V space). Rather than
// fake the panel, we hold its slot in the canonical 6-column grid and name why it
// is dark, so the aggregate layout stays faithful and honest.

const REF_H = 600;

export class MonPlaceholder {
  private id: string;
  private name: string;
  private note: string;

  constructor(id: string, name: string, note: string) {
    this.id = id;
    this.name = name;
    this.note = note;
  }

  update(_events: OscEvent[], _ops: OpRec[], _dt: number): void {
    // No data source in these captures; nothing to advance.
  }

  render(ctx: CanvasRenderingContext2D, x: number, y: number, w: number, h: number): void {
    const s = h / REF_H;
    const mono = "JetBrains Mono, ui-monospace, monospace";

    ctx.fillStyle = "#000";
    ctx.fillRect(x, y, w, h);
    ctx.textBaseline = "top";
    ctx.textAlign = "left";

    // Top-left id label, same convention as the live panels.
    ctx.font = `${13 * s}px ${mono}`;
    ctx.fillStyle = "rgba(31,31,31,1)";
    ctx.fillText(`${this.id}  ${this.name}`, x + 10 * s, y + 6 * s);

    // Centered note explaining the silence (very dim so it reads as intentional).
    ctx.font = `${12 * s}px ${mono}`;
    ctx.fillStyle = "rgba(64,64,64,1)";
    ctx.textAlign = "center";
    ctx.textBaseline = "middle";
    ctx.fillText(this.note, x + w / 2, y + h / 2);
    ctx.textAlign = "left";
  }
}
