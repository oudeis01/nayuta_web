// Shared runtime aspect-ratio infra for the monitor panels.
//
// Several panels (Monitor A scanner, Monitor B op_stream) re-flow their internal
// layout depending on whether their rect is wide or tall: the install wall is
// fixed 1920x1080 landscape, but the web wall folds the same panels into one
// browser page, where a single-view full-bleed can be portrait when the window
// is taller than wide. The decision (2026-06-05, D-A3 / D-B3) is to switch
// layout purely on the rect's own aspect, recomputed every frame from the live
// (w, h) the compositor hands render(). Because render() runs each frame with the
// current rect, calling orient(w, h) there makes the switch runtime-reactive: a
// mid-runtime resize flips the layout on the very next frame, no state needed.
//
// Threshold (user decision 2026-06-05): w/h < 1.0 is portrait; w/h >= 1.0 keeps
// the current landscape layout, and that explicitly includes the exact-square
// w/h == 1.0 case (it stays on the landscape branch).

export type Orient = "landscape" | "portrait";

// A rect is portrait only when strictly taller than wide. Equal sides (and any
// degenerate h <= 0) fall through to landscape so the square case keeps the
// current layout per the decision above.
export function orient(w: number, h: number): Orient {
  return h > 0 && w / h < 1.0 ? "portrait" : "landscape";
}
