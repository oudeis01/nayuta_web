#!/usr/bin/env python3
"""Slice a single sequence out of corpus.bin into a self-contained corpus_demo.

Per action plan §5-3 step 3. One demo corpus per selected sequence: each output
contains exactly n_seqs=1 so bert_install processes that single sequence end to
end during a capture session. The matching corpus_words_demo file mirrors the
v2 lemma_id stream for the same sequence so Monitor A/D word lookup keeps
working.

Both output files preserve the original binary layouts:

    corpus_demo.bin       — NSBR v1 (CORPUS_MAGIC)
    corpus_words_demo.bin — NWBR v2 (WORDS_MAGIC, payload=lemma_id)

A small JSON sidecar (corpus_demo_<tag>.json) records the source seq_idx,
source name, sequence length, and a sha256 of the token payload so we can
trace captures back to the exact bytes that produced them.

Usage:
    ~/miniconda3/bin/python web/corpus_curation/build_corpus_demo.py \\
        --seq-idx 156743 --tag tate_nnn
    ~/miniconda3/bin/python web/corpus_curation/build_corpus_demo.py \\
        --seq-idx 128508 --tag mousse_irigaray --out-dir captures/seeds
"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import struct
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
DEFAULT_CORPUS  = REPO / "bert_inference" / "corpus.bin"
DEFAULT_WORDS   = REPO / "bert_inference" / "corpus_words_v2.bin"
DEFAULT_OUT_DIR = REPO / "web" / "corpus_curation" / "seeds"
DEFAULT_META    = REPO / "bert_inference" / "corpus.json"

CORPUS_MAGIC = 0x4E534252        # "NSBR"
WORDS_MAGIC  = 0x4E574252        # "NWBR"
CORPUS_VERSION = 1
WORDS_VERSION  = 2

HEADER_FMT = struct.Struct("<IIIIQQ")    # 32 B
SOURCE_FMT = struct.Struct("<II56s")     # 64 B
SEQREC_FMT = struct.Struct("<IIQ")       # 16 B


def _read_corpus(path: Path):
    with open(path, "rb") as f:
        magic, version, n_sources, n_seqs, total_tokens, _ = HEADER_FMT.unpack(f.read(32))
        if magic != CORPUS_MAGIC:
            sys.exit(f"Bad corpus magic in {path}: 0x{magic:08X}")
        if version != CORPUS_VERSION:
            sys.exit(f"Unexpected corpus version {version} (want {CORPUS_VERSION})")
        sources = []
        for _ in range(n_sources):
            buf = f.read(64)
            sid, _p, nm = SOURCE_FMT.unpack(buf)
            sources.append((sid, nm))      # keep raw nm bytes so we round-trip exactly
        seqs = []
        for _ in range(n_seqs):
            sid, slen, boff = SEQREC_FMT.unpack(f.read(16))
            seqs.append((sid, slen, boff))
    return n_sources, n_seqs, total_tokens, sources, seqs


def _read_words(path: Path):
    with open(path, "rb") as f:
        magic, version, n_sources, n_seqs, total_tokens, _ = HEADER_FMT.unpack(f.read(32))
        if magic != WORDS_MAGIC:
            sys.exit(f"Bad words magic in {path}: 0x{magic:08X}")
        if version != WORDS_VERSION:
            sys.exit(f"Unexpected words version {version} (want {WORDS_VERSION})")
        seqs = []
        for _ in range(n_seqs):
            slen, _p, boff = SEQREC_FMT.unpack(f.read(16))
            seqs.append((slen, boff))
    return n_sources, n_seqs, total_tokens, seqs


def _read_slice(path: Path, slen: int, boff: int) -> bytes:
    with open(path, "rb") as f:
        f.seek(boff)
        return f.read(slen * 4)


def _safe_tag(s: str) -> str:
    s = re.sub(r"[^A-Za-z0-9_-]+", "_", s.strip())
    return s.strip("_") or "demo"


def _build_corpus_demo(out: Path, src_id: int, src_name: bytes, token_bytes: bytes,
                      slen: int):
    """Write a one-sequence corpus.bin (NSBR v1) carrying a single source row."""
    header_size = HEADER_FMT.size
    n_sources = 1
    n_seqs = 1
    # Source table immediately follows the header; seq table follows the source
    # table. Data starts right after the seq table.
    src_table_size = SOURCE_FMT.size * n_sources
    seq_table_size = SEQREC_FMT.size * n_seqs
    data_offset = header_size + src_table_size + seq_table_size

    # Renumber the embedded source to id 0 — the demo corpus has only one row.
    with open(out, "wb") as f:
        f.write(HEADER_FMT.pack(CORPUS_MAGIC, CORPUS_VERSION,
                                n_sources, n_seqs, slen, 0))
        # 56-byte name field, padded with NUL; pad/truncate defensively.
        nm = src_name[:56].ljust(56, b"\x00")
        f.write(SOURCE_FMT.pack(0, 0, nm))
        f.write(SEQREC_FMT.pack(0, slen, data_offset))
        f.write(token_bytes)


def _build_words_demo(out: Path, lemma_bytes: bytes, slen: int):
    """Write a one-sequence corpus_words_v2.bin (NWBR v2)."""
    header_size = HEADER_FMT.size
    n_seqs = 1
    seq_table_size = SEQREC_FMT.size * n_seqs
    data_offset = header_size + seq_table_size

    with open(out, "wb") as f:
        # build_corpus_words_v2.py writes n_sources field but the bert.c loader
        # only reads what NWBR records require. Mirror the upstream values:
        # n_sources=1, n_seqs=1, total_tokens=slen so the file is internally
        # consistent with its single-sequence content.
        f.write(HEADER_FMT.pack(WORDS_MAGIC, WORDS_VERSION,
                                1, n_seqs, slen, 0))
        f.write(SEQREC_FMT.pack(slen, 0, data_offset))
        f.write(lemma_bytes)


def _sha256_hex(b: bytes) -> str:
    return hashlib.sha256(b).hexdigest()


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--corpus", type=Path, default=DEFAULT_CORPUS)
    ap.add_argument("--words", type=Path, default=DEFAULT_WORDS)
    ap.add_argument("--meta", type=Path, default=DEFAULT_META,
                    help="corpus.json for human-readable source names")
    ap.add_argument("--seq-idx", type=int, required=True)
    ap.add_argument("--tag", type=str, required=True,
                    help="short identifier used in filenames (e.g. tate_nnn)")
    ap.add_argument("--out-dir", type=Path, default=DEFAULT_OUT_DIR)
    args = ap.parse_args()

    tag = _safe_tag(args.tag)
    args.out_dir.mkdir(parents=True, exist_ok=True)

    print(f"Reading corpus header from {args.corpus}", file=sys.stderr)
    n_sources, n_seqs, total_tokens, sources, seqs = _read_corpus(args.corpus)
    if args.seq_idx < 0 or args.seq_idx >= n_seqs:
        sys.exit(f"seq_idx {args.seq_idx} out of range (n_seqs={n_seqs})")

    sid, slen, boff = seqs[args.seq_idx]
    src_id_bytes = next((s[1] for s in sources if s[0] == sid), b"unknown")
    src_name = src_id_bytes.split(b"\x00")[0].decode("utf-8", errors="replace")
    print(f"  seq_idx={args.seq_idx}  src={src_name}  len={slen}", file=sys.stderr)

    token_bytes = _read_slice(args.corpus, slen, boff)
    if len(token_bytes) != slen * 4:
        sys.exit("Short read on corpus payload")

    print(f"Reading words header from {args.words}", file=sys.stderr)
    w_n_sources, w_n_seqs, w_total, w_seqs = _read_words(args.words)
    if w_n_seqs != n_seqs:
        sys.exit(f"corpus/words sequence count mismatch ({n_seqs} vs {w_n_seqs})")
    w_slen, w_boff = w_seqs[args.seq_idx]
    if w_slen != slen:
        sys.exit(f"corpus/words length mismatch for seq {args.seq_idx} "
                 f"({slen} vs {w_slen})")
    lemma_bytes = _read_slice(args.words, w_slen, w_boff)

    corpus_out = args.out_dir / f"corpus_demo_{tag}.bin"
    words_out  = args.out_dir / f"corpus_words_demo_{tag}.bin"
    meta_out   = args.out_dir / f"corpus_demo_{tag}.json"

    _build_corpus_demo(corpus_out, sid, src_id_bytes, token_bytes, slen)
    _build_words_demo(words_out, lemma_bytes, slen)

    # Round-trip verification: re-read the freshly written files and assert
    # bytes match before declaring success.
    _, _, _, vs_sources, vs_seqs = _read_corpus(corpus_out)
    assert vs_sources[0][0] == 0
    vs_sid, vs_slen, vs_boff = vs_seqs[0]
    assert vs_slen == slen, (vs_slen, slen)
    with open(corpus_out, "rb") as f:
        f.seek(vs_boff)
        assert f.read(slen * 4) == token_bytes
    _, _, _, vw_seqs = _read_words(words_out)
    vw_slen, vw_boff = vw_seqs[0]
    assert vw_slen == slen, (vw_slen, slen)
    with open(words_out, "rb") as f:
        f.seek(vw_boff)
        assert f.read(slen * 4) == lemma_bytes

    meta = {
        "tag": tag,
        "source_seq_idx": args.seq_idx,
        "source_name": src_name,
        "source_id_original": sid,
        "seq_len": slen,
        "tokens_sha256": _sha256_hex(token_bytes),
        "lemmas_sha256": _sha256_hex(lemma_bytes),
        "corpus_path": str(corpus_out.relative_to(REPO)),
        "words_path":  str(words_out.relative_to(REPO)),
    }
    meta_out.write_text(json.dumps(meta, indent=2))

    print(f"\nWrote:", file=sys.stderr)
    print(f"  {corpus_out}  ({corpus_out.stat().st_size} B)", file=sys.stderr)
    print(f"  {words_out}   ({words_out.stat().st_size} B)", file=sys.stderr)
    print(f"  {meta_out}",  file=sys.stderr)
    print(f"\nsha256 tokens={meta['tokens_sha256'][:16]}…  "
          f"lemmas={meta['lemmas_sha256'][:16]}…", file=sys.stderr)
    return 0


if __name__ == "__main__":
    sys.exit(main())
