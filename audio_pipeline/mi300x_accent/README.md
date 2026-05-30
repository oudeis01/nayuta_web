# mi300x_accent - multilingual accent whisper generation

Generates the full word list in 7 accents on an MI300X (ROCm) instance, with
voice cloning + acoustic QA. This is the GPU-side branch that feeds the web
audio engine and the SuperCollider install (both pick 1 of 7 accents at trigger
time).

## What this is

- Canonical accent method: voice CLONING from 7 references + QA
  (ported from `voice_test/transliteration/qa_pipeline.py`).
- Throughput method: ROCm env + batched `model.generate` + resume
  (borrowed from `voice_test/OmniVoice/generate_namedrop_whisper_mi300x.py`).
- Output identity is the install/web namespace, baked in at generation time:
  `{lemma_id:06d}-{variant_idx:02d}__{accent}.wav`. No surface-text join later.

## The 7 accents

korean, japanese, arabic, russian, ukrainian, american, british.
References are flattened to `reference_audio/ref_{accent}.wav` by the packer.
japanese and ukrainian use the selected non-00 takes; the rest use the 00 set.

## Scope and invariant

- Full CSV: 50,011 rows (surface <-> (lid,vidx) is a bijection, all unique).
- 50,011 x 7 = 350,077 target files.
- Invariant assumed downstream: every word exists in every accent. To uphold it,
  the unresolved QA leftovers are swept at the end with more retries.

## On the instance

```bash
# inside the ROCm container, with this pack unpacked as CWD
./setup_instance.sh                 # clone+install omnivoice upstream, deps, model
./run_accent_gen.sh --limit 100     # throughput probe: read the pass/s and ETA
./run_accent_gen.sh --resume        # full run, background, resumable
# later, to enforce the all-accents invariant:
./run_accent_gen.sh --resume --max-retries 20
```

`--resume` skips any `(lid,vidx,accent)` whose wav already exists, so the run is
safe to stop and restart. Progress + per-target QA status land in
`accent_qa_log.json` (flushed every 500 targets).

## Build the pack (on this workstation)

```bash
./pack_transfer.sh
# -> dist/namedrop_accent_pack/  and  dist/namedrop_accent_pack.tar.gz
scp dist/namedrop_accent_pack.tar.gz <instance>:~/
```

## After generation (recovery, this repo)

1. Pull `out_accent/` back + `accent_qa_log.json`.
2. Integrity: count == 350,077 minus final-unresolved; spot-check QA log.
3. Opus encode: reuse `../encode_opus.sh` extended for the `__{accent}` stem.
4. R2 upload: `../upload_r2.sh`, paired with the frontend visual test (deferred).

## Measured bottleneck (single MI300X, probes 2026-05-30)

The bottleneck is GPU generation (the 64 sequential flow-matching steps), NOT
QA and NOT VRAM:

- batch 100 -> 1024 barely moved throughput (3.5 -> 4.1 pass/s) while VRAM
  scaled with batch (6% -> 56%). Compute-bound: throughput is flat in batch.
- QA across 18 CPU workers gave only ~+12% (4.1 -> 4.6 pass/s), confirming QA
  was never the constraint. CPU stayed near 200% throughout.
- num_step 64 -> 32 gave 1.67x (4.6 -> 7.7 pass/s); not 2x because of fixed
  per-item overhead (QA, trim, IO).

Throughput at num_step 64: ~4.5 pass/s -> ~22 h for the full 350,077.

## Knobs

- `--num-step` is the real throughput lever, but it is a QUALITY knob. 64 is the
  proven accent-pipeline setting and the chosen baseline. Lower = faster, but
  A/B-listen first (QA passes either way; it does not measure naturalness).
- `--batch-size` (default 512, ~28% VRAM) does not change throughput; keep it
  comfortable. `--auto-batch` calibrates from free VRAM if needed.
- `--qa-workers` (default cpu_count-2) parallelizes QA off the main thread. Kept
  because it is harmless, though the win is marginal on a single GPU.
- More GPUs would be near-linear; this driver is single-GPU. To shard across N
  GPUs, run N copies over disjoint word ranges (a `--shard i/n` split would be
  the clean way to add it).
