#!/usr/bin/env python3
"""Multilingual accent whisper generation for the full word list, MI300X + QA.

Marries two existing pipelines:
  - transliteration/qa_pipeline.py  (voice CLONING from 7 accent references
    + acoustic QA + retry-until-pass), the canonical accent approach.
  - OmniVoice/generate_namedrop_whisper_mi300x.py (ROCm env, batched
    model.generate, resume), the throughput approach.

Output identity is baked in at generation time as the install/web namespace
    {lemma_id:06d}-{variant_idx:02d}__{accent}.wav
so there is no fragile surface-text join afterwards. surface <-> (lid,vidx)
is a bijection in whisper_audio_words_v2.csv (all 50011 surfaces unique).

Per accent we precompute one reusable VoiceClonePrompt and batch all words
against it, then re-batch the QA failures with fresh seeds up to --max-retries.
QA runs on the in-memory waveform (no temp-file round-trip).

Usage (on the instance, inside the ROCm container, omnivoice installed):
    python generate_accent_qa_mi300x.py --limit 100        # throughput probe
    python generate_accent_qa_mi300x.py --resume           # full run
    python generate_accent_qa_mi300x.py --auto-batch --resume
    python generate_accent_qa_mi300x.py --accents korean,japanese --resume
"""

import argparse
import csv
import json
import multiprocessing as mp
import os
import random
import time
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import librosa
import numpy as np
import parselmouth
import soundfile as sf
import torch
from parselmouth.praat import call

# ── Config ───────────────────────────────────────────────────────────────────
HERE     = Path(__file__).parent
CSV_PATH = HERE / "whisper_audio_words_v2.csv"
REF_DIR  = HERE / "reference_audio"
OUT_DIR  = HERE / "out_accent"
LOG_PATH = HERE / "accent_qa_log.json"

NUM_STEP      = 64
LANGUAGE      = "en"
DEFAULT_BATCH = 512
MAX_RETRIES   = 10
MIN_BATCH     = 16
VRAM_HEADROOM = 0.10
LOG_FLUSH_EVERY = 500
# QA is CPU-bound (parselmouth) and the real bottleneck, not the GPU. The probe
# showed batch 100 -> 1024 barely moved throughput while VRAM stayed at ~2%, so
# QA runs across a process pool. Leave a couple cores for the main + GPU feed.
QA_WORKERS    = max(1, min(18, (os.cpu_count() or 2) - 2))

# QA thresholds (identical to transliteration/qa_pipeline.py)
QA = {
    "min_duration_s": 0.3,
    "min_rms": 5e-4,
    "hnr_max": 5.0,
    "voiced_frac_max": 0.12,
    "f0_natural_max": 250,
    "f0_natural_min": 75,
}

# Trim config (top_db=35 confirmed safe via listening test)
TRIM_TOP_DB = 35
TRIM_PAD_MS = 60
TRIM_FADE_MS = 25

# 7 accents. Reference files are flattened to ref_{id}.wav by pack_transfer.sh.
# ref_text transcripts carried over verbatim from qa_pipeline.py.
REFERENCES = [
    {"id": "korean",
     "text": "오늘 하루도 정말 수고했어요. 따뜻한 차 한 잔 마시면서 편안하게 쉬어가요."},
    {"id": "japanese",
     "text": "そっと目を閉じて、ゆったりとした気持ちになってください。"},
    {"id": "arabic",
     "text": "الليلة هادئة جميلة، والنجوم تضيء السماء بضوئها الناعم."},
    {"id": "russian",
     "text": "Сегодня тихий вечер. Давай я расскажу тебе спокойную историю перед сном."},
    {"id": "ukrainian",
     "text": "Сьогодні такий спокійний вечір. Хочу розповісти тобі тиху казочку."},
    {"id": "american",
     "text": "Close your eyes and take a deep breath. Everything is calm and peaceful right now."},
    {"id": "british",
     "text": "Close your eyes and take a deep breath. Everything is calm and peaceful right now."},
]
# ─────────────────────────────────────────────────────────────────────────────


def _apply_rocm_env() -> None:
    """ROCm environment for MI300X (borrowed from the neutral mi300x script)."""
    for k, v in {
        "PYTORCH_ALLOC_CONF": "expandable_segments:True",
        "HSA_FORCE_FINE_GRAIN_PCIE": "1",
        "HSA_ENABLE_SDMA": "0",
    }.items():
        os.environ.setdefault(k, v)


# ── QA (in-memory variants of qa_pipeline.py) ────────────────────────────────

def apply_trim(y: np.ndarray, sr: int) -> np.ndarray:
    y = np.asarray(y, dtype=np.float32)
    y_trimmed, (start, end) = librosa.effects.trim(y, top_db=TRIM_TOP_DB)
    pad_n = int(TRIM_PAD_MS / 1000 * sr)

    pre  = y[max(0, start - pad_n):start]
    post = y[end:min(len(y), end + pad_n)]
    pre  = np.pad(pre,  (max(0, pad_n - len(pre)),  0))
    post = np.pad(post, (0, max(0, pad_n - len(post))))

    y_out = np.concatenate([pre, y_trimmed, post])

    fade_n = min(int(TRIM_FADE_MS / 1000 * sr), len(y_out) // 4)
    if fade_n > 0:
        t = np.linspace(0, np.pi / 2, fade_n)
        y_out[:fade_n]  *= np.sin(t)
        y_out[-fade_n:] *= np.cos(t)
    return y_out


def qa_check_array(y: np.ndarray, sr: int) -> tuple[bool, str]:
    """Acoustic QA on an in-memory waveform. Same checks/order as qa_pipeline."""
    y = np.asarray(y, dtype=np.float32)
    if y.ndim > 1:
        y = y[:, 0]
    duration = len(y) / sr
    rms = float(np.sqrt(np.mean(y ** 2))) if len(y) else 0.0

    if duration < QA["min_duration_s"]:
        return False, f"TOO_SHORT({duration:.2f}s)"
    if rms < QA["min_rms"]:
        return False, f"SILENT(rms={rms:.6f})"

    snd = parselmouth.Sound(values=y.astype(np.float64), sampling_frequency=sr)

    harmonicity = call(snd, "To Harmonicity (cc)", 0.01, 75, 0.1, 1.0)
    hnr = call(harmonicity, "Get mean", 0, 0)
    hnr = float(hnr) if not np.isnan(hnr) else -999.0
    if hnr > QA["hnr_max"]:
        return False, f"HNR_HIGH({hnr:.2f})"

    pitch = call(snd, "To Pitch", 0.0, 75, 600)
    total = call(pitch, "Get number of frames")
    voiced, f0_vals = 0, []
    for i in range(1, int(total) + 1):
        f0 = call(pitch, "Get value in frame", i, "Hertz")
        if f0 and not np.isnan(f0) and f0 > 0:
            voiced += 1
            f0_vals.append(f0)
    voiced_frac = voiced / total if total > 0 else 0.0
    f0_mean = float(np.mean(f0_vals)) if f0_vals else 0.0

    if (voiced_frac > QA["voiced_frac_max"]
            and QA["f0_natural_min"] < f0_mean < QA["f0_natural_max"]):
        return False, f"VOICED_F0(vf={voiced_frac:.3f},f0={f0_mean:.1f})"

    return True, ""


# ── QA worker (CPU process pool) ─────────────────────────────────────────────

def _qa_init() -> None:
    """Pin each worker to a single BLAS thread (avoid oversubscription)."""
    for v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS",
              "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
        os.environ[v] = "1"


def qa_worker(payload):
    """Trim + QA one generated waveform; on pass, write the wav. Runs in a
    child process. Returns (stem, passed, reason)."""
    stem, raw, sr, out_path = payload
    try:
        y = apply_trim(np.asarray(raw, dtype=np.float32), sr)
        ok, reason = qa_check_array(y, sr)
        if ok:
            sf.write(out_path, y, sr)
        return stem, ok, reason
    except Exception as e:  # never let one clip kill the pool
        return stem, False, f"ERROR({type(e).__name__})"


# ── Helpers ──────────────────────────────────────────────────────────────────

def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def chunks(seq, n):
    for i in range(0, len(seq), n):
        yield seq[i:i + n]


def as_array(audio) -> np.ndarray:
    if torch.is_tensor(audio):
        audio = audio.cpu().numpy()
    audio = np.asarray(audio, dtype=np.float32)
    if audio.ndim == 2:
        audio = audio[0]
    return audio


def flush_log(log: dict, path: Path) -> None:
    tmp = path.with_suffix(".tmp")
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(log, f, ensure_ascii=False)
    tmp.replace(path)


def load_tasks(csv_path: Path):
    """Return list of (lid, vidx, surface) from the canonical CSV."""
    rows = []
    with open(csv_path, newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            surface = (row.get("surface") or "").strip()
            if not surface:
                continue
            rows.append((int(row["lemma_id"]), int(row["variant_idx"]), surface))
    return rows


# ── Main ─────────────────────────────────────────────────────────────────────

def main() -> int:
    _apply_rocm_env()

    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--csv", default=str(CSV_PATH))
    ap.add_argument("--refs-dir", default=str(REF_DIR))
    ap.add_argument("--out-dir", default=str(OUT_DIR))
    ap.add_argument("--log", default=str(LOG_PATH))
    ap.add_argument("--batch-size", type=int, default=DEFAULT_BATCH)
    ap.add_argument("--qa-workers", type=int, default=QA_WORKERS,
                    help=f"CPU processes for QA (default: {QA_WORKERS})")
    ap.add_argument("--auto-batch", action="store_true",
                    help="Calibrate VRAM per item and pick batch size")
    ap.add_argument("--num-step", type=int, default=NUM_STEP)
    ap.add_argument("--max-retries", type=int, default=MAX_RETRIES)
    ap.add_argument("--accents", default="",
                    help="Comma list subset of accent ids (default: all 7)")
    ap.add_argument("--limit", type=int, default=0,
                    help="Cap words per accent (throughput probe; 0 = all)")
    ap.add_argument("--resume", action="store_true",
                    help="Skip (lid,vidx,accent) whose wav already exists")
    ap.add_argument("--device", default=None)
    ap.add_argument("--vram-headroom", type=float, default=VRAM_HEADROOM)
    args = ap.parse_args()

    device = args.device or ("cuda:0" if torch.cuda.is_available() else "cpu")
    dtype  = torch.float16 if "cuda" in device else torch.float32
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    refs_dir = Path(args.refs_dir)
    log_path = Path(args.log)

    refs = REFERENCES
    if args.accents:
        want = {a.strip() for a in args.accents.split(",") if a.strip()}
        refs = [r for r in REFERENCES if r["id"] in want]
        if not refs:
            print(f"No matching accents in {sorted(want)}")
            return 1

    for r in refs:
        p = refs_dir / f"ref_{r['id']}.wav"
        if not p.exists():
            raise FileNotFoundError(f"Reference audio not found: {p}")

    tasks = load_tasks(Path(args.csv))
    if args.limit:
        tasks = tasks[:args.limit]
    print(f"CSV:     {args.csv}")
    print(f"Words:   {len(tasks):,}   Accents: {[r['id'] for r in refs]}")
    print(f"Targets: {len(tasks) * len(refs):,}  "
          f"(batch={args.batch_size}  max_retries={args.max_retries})")

    # ── Model ────────────────────────────────────────────────────────────────
    from omnivoice import OmniVoice
    print(f"\nLoading OmniVoice on {device}  dtype={dtype} ...")
    t_load = time.time()
    model = OmniVoice.from_pretrained(
        "k2-fsa/OmniVoice", device_map=device, dtype=dtype
    )
    sr = model.sampling_rate
    print(f"Model loaded in {time.time() - t_load:.1f}s  sr={sr}")

    batch_size = args.batch_size
    if args.auto_batch and "cuda" in device:
        batch_size = _auto_batch(model, device, refs, refs_dir, args.vram_headroom)
        print(f"[auto-batch] batch_size = {batch_size}")

    # Precompute one reusable clone prompt per accent.
    prompts = {}
    for r in refs:
        prompts[r["id"]] = model.create_voice_clone_prompt(
            ref_audio=str(refs_dir / f"ref_{r['id']}.wav"),
            ref_text=r["text"],
        )

    log = {}
    if log_path.exists():
        with open(log_path, encoding="utf-8") as f:
            log = json.load(f)
        print(f"Resuming log ({len(log)} entries)")

    # QA process pool. spawn (not fork) so children do not inherit the HIP
    # context. Workers only do CPU trim+QA, no model.
    pool = ProcessPoolExecutor(
        max_workers=max(1, args.qa_workers),
        mp_context=mp.get_context("spawn"),
        initializer=_qa_init,
    )
    print(f"QA workers: {args.qa_workers}  (cpu_count={os.cpu_count()})")

    rng = random.Random()  # unseeded: retry diversity across runs
    done_since_flush = 0
    t0 = time.time()
    n_pass = n_unres = n_skip = 0

    for r in refs:
        acc = r["id"]
        prompt = prompts[acc]

        pending = []
        for lid, vidx, surface in tasks:
            stem = f"{lid:06d}-{vidx:02d}__{acc}"
            out_path = out_dir / f"{stem}.wav"
            if args.resume and out_path.exists():
                n_skip += 1
                continue
            pending.append((stem, surface, out_path))

        print(f"\n[{acc}] pending {len(pending):,} / {len(tasks):,}")

        for attempt in range(1, args.max_retries + 1):
            if not pending:
                break
            next_pending = []
            for batch in chunks(pending, batch_size):
                set_seed(rng.randint(0, 999_999))
                texts = [surface for (_stem, surface, _p) in batch]
                try:
                    with torch.no_grad():
                        audios = model.generate(
                            text=texts,
                            voice_clone_prompt=[prompt] * len(texts),
                            language=LANGUAGE,
                            num_step=args.num_step,
                        )
                except RuntimeError as e:
                    if "out of memory" in str(e).lower():
                        torch.cuda.empty_cache()
                        print(f"  OOM at batch={len(batch)} -> halving")
                        half = max(MIN_BATCH, len(batch) // 2)
                        for sub in chunks(batch, half):
                            next_pending.extend(sub)
                        continue
                    raise

                payloads = [
                    (stem, as_array(audio), sr, str(out_path))
                    for (stem, _surface, out_path), audio in zip(batch, audios)
                ]
                results = pool.map(qa_worker, payloads, chunksize=4)
                for (stem, surface, out_path), (_s, ok, reason) in zip(batch, results):
                    if ok:
                        log[stem] = {"status": "passed", "attempts": attempt}
                        n_pass += 1
                    else:
                        next_pending.append((stem, surface, out_path))
                        log[stem] = {"status": "retry", "attempts": attempt,
                                     "last": reason}
                    done_since_flush += 1
                    if done_since_flush >= LOG_FLUSH_EVERY:
                        flush_log(log, log_path)
                        done_since_flush = 0

            done = len(pending) - len(next_pending)
            rate = (n_pass) / max(1e-9, time.time() - t0)
            print(f"  attempt {attempt}: +{done} pass  "
                  f"{len(next_pending)} left  ({rate:.1f} pass/s)")
            pending = next_pending

        for stem, surface, _p in pending:
            log[stem] = {"status": "unresolved",
                         "attempts": args.max_retries}
            n_unres += 1
        flush_log(log, log_path)

    pool.shutdown(wait=True)
    flush_log(log, log_path)
    dt = time.time() - t0
    print(f"\n{'=' * 60}")
    print(f"Done in {dt / 60:.1f} min")
    print(f"  passed:     {n_pass:,}")
    print(f"  skipped:    {n_skip:,} (already existed)")
    print(f"  unresolved: {n_unres:,}")
    print(f"Out: {out_dir}/   Log: {log_path}")
    if n_unres:
        print(f"\nRe-run the unresolved sweep with a higher cap, e.g.:")
        print(f"  python {Path(__file__).name} --resume --max-retries 20")
    return 0


def _auto_batch(model, device, refs, refs_dir, headroom) -> int:
    """Calibrate VRAM per item using a real clone prompt, then size the batch."""
    idx = int(device.split(":")[-1]) if ":" in device else 0
    total_mb = torch.cuda.get_device_properties(idx).total_memory / 1024**2
    prompt = model.create_voice_clone_prompt(
        ref_audio=str(refs_dir / f"ref_{refs[0]['id']}.wav"),
        ref_text=refs[0]["text"],
    )
    deltas = []
    for n in (4, 8, 16):
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()
        before = torch.cuda.memory_allocated() / 1024**2
        try:
            with torch.no_grad():
                model.generate(text=["calibration"] * n,
                               voice_clone_prompt=[prompt] * n,
                               language=LANGUAGE, num_step=2)
        except RuntimeError as e:
            if "out of memory" in str(e).lower():
                torch.cuda.empty_cache()
                continue
            raise
        after = torch.cuda.max_memory_allocated() / 1024**2
        deltas.append((n, max(after - before, 1.0)))
        torch.cuda.empty_cache()
    if not deltas:
        return DEFAULT_BATCH
    per_item = sum(d / n for n, d in deltas) / len(deltas)
    used_mb = torch.cuda.memory_allocated() / 1024**2
    usable = (total_mb - used_mb) * (1.0 - headroom)
    bs = int(usable / max(per_item, 0.5))
    bs = (bs // 64) * 64
    return max(bs, MIN_BATCH)


if __name__ == "__main__":
    raise SystemExit(main())
