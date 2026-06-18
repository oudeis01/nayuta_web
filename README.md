# Nayuta: The Transformer — web

The browser version of the installation [*Nayuta: The
Transformer*](https://github.com/oudeis01/nayuta). It is a faithful alternative
experience, not a synthetic simulation: every frame is driven by captured or
replayed real `bert.c` output, and the installation pace is preserved.

Live at [nayuta.choiharam.com](https://nayuta.choiharam.com).

## Layout

- `frontend/` — the browser app (TypeScript + Vite, built with bun) plus the
  `/about` hub landing. Deployed to Cloudflare Pages.
- `tap/` — Rust adapter that taps the engine's ZMQ/OSC output and writes the
  capture format, including the streaming full-dump used by the web player.
- `audio_pipeline/` — opus encoding of the accent voice assets and playable
  subset computation. Encoded audio is served from Cloudflare R2.
- `corpus_curation/` — scoring and selection of corpus sequences for the web
  demo (seed corpora live under `seeds/`).

## Build

```sh
cd frontend
bun install
bun run build      # tsc && vite build  ->  dist/
bun run dev        # local dev server
```

`tsc` runs with strict `noUnusedLocals`, so the build fails on unused symbols.

## Data

Large assets (captured op dumps, opus audio) are not committed; they are hosted
on Cloudflare R2 and addressed at runtime via `VITE_AUDIO_BASE` and the capture
base. Deployment config (bucket URLs, Pages project) lives in a local,
uncommitted `frontend/deploy.env` (see `frontend/deploy.env.example`).

## Related repositories

- [`nayuta`](https://github.com/oudeis01/nayuta) — the physical installation.
- [`nayuta_bert`](https://github.com/oudeis01/nayuta_bert) — the pure-C engine.
- [`nayuta_crossword`](https://github.com/oudeis01/nayuta_crossword) — the crossword companion.

## License

Code is MIT. Non-code material (text, design, documentation) is CC BY-NC.
