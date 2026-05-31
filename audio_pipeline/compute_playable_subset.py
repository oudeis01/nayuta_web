#!/usr/bin/env python3
"""compute_playable_subset.py — derive the exact opus asset set the web audio
engine can actually request, from the captured event logs.

The engine (audio/engine.ts) plays a voice for:
  - every /bert/whisper main lemma           (args[0]=lid, args[1]=vidx)
  - each of its <=12 neighbors               (args[12 + j*3], +1 = vidx)
  - echoes (same lid/vidx, no new asset)     — already covered
  - /bert/word_trigger main lemma, vidx 0    (captures have 0 of these)

Accent is chosen at random per utterance from 7 accents, so EVERY accent of a
reachable (lid,vidx) is a potential request. The deployable asset set is thus
{ (lid,vidx) reachable } x 7 accents.

This walks all events.jsonl.zst under a captures dir, collects the reachable
(lid,vidx) pairs, expands to opus stems, checks them against the encoded opus
dir, and writes an rclone --files-from list (basenames) plus a size report.

Usage:
  compute_playable_subset.py --captures <dir> --opus <dir> --out <files_from.txt>
"""
import argparse
import json
import os
import sys

import zstandard as zstd

ACCENTS = ["korean", "japanese", "arabic", "russian", "ukrainian", "american", "british"]


def iter_events(path):
    dctx = zstd.ZstdDecompressor()
    with open(path, "rb") as fh:
        with dctx.stream_reader(fh) as reader:
            buf = b""
            while True:
                chunk = reader.read(1 << 20)
                if not chunk:
                    break
                buf += chunk
                *lines, buf = buf.split(b"\n")
                for line in lines:
                    if line:
                        yield json.loads(line)
            if buf:
                yield json.loads(buf)


def collect(captures_dir):
    pairs = set()  # (lid, vidx)
    n_whisper = 0
    n_word = 0
    for root, _dirs, files in os.walk(captures_dir):
        for fn in files:
            if fn != "events.jsonl.zst":
                continue
            fp = os.path.join(root, fn)
            for ev in iter_events(fp):
                path = ev.get("path")
                a = ev.get("args", [])
                if path == "/bert/whisper":
                    n_whisper += 1
                    lid = int(a[0]); vidx = int(a[1])
                    pairs.add((lid, vidx))
                    n = int(a[11]) if len(a) > 11 else 0
                    for j in range(min(n, 12)):
                        base = 12 + j * 3
                        if base + 1 < len(a):
                            pairs.add((int(a[base]), int(a[base + 1])))
                elif path == "/bert/word_trigger":
                    n_word += 1
                    pairs.add((int(a[0]), 0))
    return pairs, n_whisper, n_word


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--captures", required=True)
    ap.add_argument("--opus", required=True)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    pairs, n_whisper, n_word = collect(args.captures)
    print(f"events: whisper={n_whisper} word_trigger={n_word}")
    print(f"reachable (lid,vidx) pairs = {len(pairs)}")

    present, missing_pairs, total_bytes = [], set(), 0
    for (lid, vidx) in sorted(pairs):
        had_any = False
        for acc in ACCENTS:
            stem = f"{lid:06d}-{vidx:02d}__{acc}.opus"
            fp = os.path.join(args.opus, stem)
            if os.path.exists(fp):
                present.append(stem)
                total_bytes += os.path.getsize(fp)
                had_any = True
        if not had_any:
            missing_pairs.add((lid, vidx))

    with open(args.out, "w") as fh:
        for stem in present:
            fh.write(stem + "\n")

    print(f"opus files in subset = {len(present)}  ({total_bytes/1e6:.1f} MB)")
    print(f"pairs with NO opus on disk = {len(missing_pairs)}"
          + (f"  e.g. {sorted(missing_pairs)[:5]}" if missing_pairs else ""))
    print(f"--files-from written: {args.out}")


if __name__ == "__main__":
    main()
