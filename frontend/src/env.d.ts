/// <reference types="vite/client" />

// Build-time asset bases (set in the Pages build env). Both fall back to a
// same-origin path for local dev, where `public/captures` and a local `audio`
// dir are served by Vite; in production they point at the public R2 bucket.
interface ImportMetaEnv {
  readonly VITE_AUDIO_BASE?: string; // opus assets, e.g. https://<r2>/audio
  readonly VITE_CAPTURE_BASE?: string; // dump files, e.g. https://<r2>/captures
}

interface ImportMeta {
  readonly env: ImportMetaEnv;
}
