#!/usr/bin/env python3
# ============================================================================
# gen_lemma_dict.py - emit the lemma_id -> surface lookup the frontend needs.
#
# Monitor D (whisper log) receives /bert/whisper OSC events whose args carry
# lemma_ids (triggered + neighbors), not text. The browser resolves each id to
# a display word through this dictionary. We key on the canonical variant
# (variant_idx == 0); every lemma_id in the corpus has one, so coverage is full.
#
#   python3 scripts/gen_lemma_dict.py
#
# Source CSV lives outside web/ (the whisper-tree build product). Output lands
# in public/ so Vite serves it at /lemma_surface.json (committed; ~0.6 MB).
# ============================================================================
import csv
import json
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.abspath(os.path.join(HERE, "..", "..", ".."))
SRC = os.path.join(REPO, "whisper_tree", "whisper_audio_words_v2.csv")
OUT = os.path.join(HERE, "..", "public", "lemma_surface.json")


def main() -> int:
    if not os.path.exists(SRC):
        print(f"source CSV not found: {SRC}", file=sys.stderr)
        return 1

    table: dict[str, str] = {}
    with open(SRC, newline="") as f:
        for row in csv.DictReader(f):
            if int(row["variant_idx"]) != 0:
                continue
            table[row["lemma_id"]] = row["surface"]

    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    with open(OUT, "w") as f:
        json.dump(table, f, separators=(",", ":"), ensure_ascii=False)

    print(f"wrote {len(table)} lemmas -> {os.path.relpath(OUT, REPO)} "
          f"({os.path.getsize(OUT)} bytes)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
