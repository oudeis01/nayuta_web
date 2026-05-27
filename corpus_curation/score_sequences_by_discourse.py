#!/usr/bin/env python3
"""Score every corpus sequence by discourse-keyword density.

Per the web-version action plan (§5):

    weight(core_theoretical_keywords) = 3
    weight(names)                     = 2
    weight(broad_discourse_markers)   = 1

A keyword is "resolved" via:
    keyword string -> tokenize on whitespace -> for each token:
        surface_to_lemma[token] -> lemma_to_id[lemma]

A sequence's score is sum of weighted hits over all token positions where the
position's lemma_id (from corpus_words_v2.bin) appears in a target set. If a
single lemma_id belongs to multiple weight buckets, its highest weight wins.

Filter: only seq_len == 512 sequences considered.

Output:
    - JSON (default ./scores_top.json): top-N records with seq_idx, score,
      per-discourse breakdown, source name, first-tokens preview, full text.
    - JSONL of every scored sequence (optional via --all-out) for further triage.

Usage:
    ~/miniconda3/bin/python web/corpus_curation/score_sequences_by_discourse.py
    ~/miniconda3/bin/python web/corpus_curation/score_sequences_by_discourse.py \\
        --top 20 --preview-tokens 30
"""

from __future__ import annotations

import argparse
import json
import struct
import sys
from collections import Counter, defaultdict
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
DEFAULT_CORPUS       = REPO / "bert_inference" / "corpus.bin"
DEFAULT_WORDS_V2     = REPO / "bert_inference" / "corpus_words_v2.bin"
DEFAULT_DISCOURSE    = REPO / "whisper_tree" / "transfer" / "whisper_tree_bundle" / "artifacts" / "discourse_keywords_v3.json"
DEFAULT_LEMMA_INDEX  = REPO / "whisper_tree" / "artifacts" / "lemma_index.json"
DEFAULT_SURFACE_MAP  = REPO / "whisper_tree" / "artifacts" / "surface_lemma_map.json"
DEFAULT_CORPUS_META  = REPO / "bert_inference" / "corpus.json"
DEFAULT_OUT_TOP      = REPO / "web" / "corpus_curation" / "scores_top.json"

CORPUS_MAGIC  = 0x4E534252        # "NSBR"
WORDS_MAGIC   = 0x4E574252        # "NWBR"
WORDS_VERSION = 2

HEADER_FMT = struct.Struct("<IIIIQQ")    # 32 B
SOURCE_FMT = struct.Struct("<II56s")     # 64 B
SEQREC_FMT = struct.Struct("<IIQ")       # 16 B

WEIGHTS_FULL = {
    "core_theoretical_keywords": 3,
    "names":                     2,
    "broad_discourse_markers":   1,
}
WEIGHTS_CORE_ONLY = {
    "core_theoretical_keywords": 1,
    # names/broad disabled
}


def _read_corpus_header(path: Path):
    with open(path, "rb") as f:
        magic, version, n_sources, n_seqs, total_tokens, _ = HEADER_FMT.unpack(f.read(32))
        if magic != CORPUS_MAGIC:
            sys.exit(f"Bad corpus magic in {path}: 0x{magic:08X}")
        sources = []
        for _ in range(n_sources):
            sid, _p, nm = SOURCE_FMT.unpack(f.read(64))
            sources.append((sid, nm.split(b"\x00")[0].decode("utf-8")))
        seqs = []
        for _ in range(n_seqs):
            sid, slen, boff = SEQREC_FMT.unpack(f.read(16))
            seqs.append((sid, slen, boff))
    return n_sources, n_seqs, total_tokens, sources, seqs


def _read_words_header(path: Path):
    with open(path, "rb") as f:
        magic, version, n_sources, n_seqs, total_tokens, _ = HEADER_FMT.unpack(f.read(32))
        if magic != WORDS_MAGIC:
            sys.exit(f"Bad words magic in {path}: 0x{magic:08X}")
        if version != WORDS_VERSION:
            sys.exit(f"Expected words version {WORDS_VERSION}, got {version}")
        seqs = []
        for _ in range(n_seqs):
            slen, _p, boff = SEQREC_FMT.unpack(f.read(16))
            seqs.append((slen, boff))
    return n_seqs, seqs


def _load_seq_lemma_ids(path: Path, slen: int, boff: int) -> list[int]:
    with open(path, "rb") as f:
        f.seek(boff)
        return list(struct.unpack(f"<{slen}i", f.read(slen * 4)))


def _load_seq_token_ids(path: Path, slen: int, boff: int) -> list[int]:
    with open(path, "rb") as f:
        f.seek(boff)
        return list(struct.unpack(f"<{slen}i", f.read(slen * 4)))


def _resolve_keyword(keyword: str, surface_to_lemma, lemma_to_id):
    """Map a (possibly multi-word) keyword to a set of lemma_ids.

    Splits on whitespace and resolves each token through surface_to_lemma →
    lemma_to_id. Tokens that fail to resolve are reported but the keyword is
    not discarded — partial resolution still contributes.
    """
    lemma_ids: set[int] = set()
    unresolved_tokens: list[str] = []
    for tok in keyword.lower().split():
        # Strip simple punctuation tails so e.g. "post-colonial" still yields hits
        tok = tok.strip(".,;:!?'\"()[]")
        if not tok:
            continue
        lemma = surface_to_lemma.get(tok)
        if lemma is None:
            unresolved_tokens.append(tok)
            continue
        lid = lemma_to_id.get(lemma)
        if lid is None:
            unresolved_tokens.append(f"{tok}->{lemma}")
            continue
        lemma_ids.add(lid)
    return lemma_ids, unresolved_tokens


def build_weight_map(discourse_kw: dict, surface_to_lemma, lemma_to_id, weights: dict):
    """Return (weight_per_lid, discourse_per_lid, resolution_report).

    weight_per_lid: lemma_id -> max weight across all categories it appears in
    discourse_per_lid: lemma_id -> set of discourse names it belongs to
    resolution_report: { discourse: { category: { resolved: int, total: int,
                                                  unresolved: [str, ...] } } }
    """
    weight_per_lid: dict[int, int] = {}
    discourse_per_lid: dict[int, set[str]] = defaultdict(set)
    report: dict = {}

    for discourse, cats in discourse_kw.items():
        report[discourse] = {}
        for cat, kws in cats.items():
            w = weights.get(cat, 0)
            if w == 0:
                continue
            unresolved = []
            n_resolved_kw = 0
            for kw in kws:
                lids, ur = _resolve_keyword(kw, surface_to_lemma, lemma_to_id)
                if ur:
                    unresolved.append({"keyword": kw, "unresolved": ur})
                if lids:
                    n_resolved_kw += 1
                for lid in lids:
                    if weight_per_lid.get(lid, 0) < w:
                        weight_per_lid[lid] = w
                    discourse_per_lid[lid].add(discourse)
            report[discourse][cat] = {
                "weight": w,
                "n_keywords_total": len(kws),
                "n_keywords_with_any_lid": n_resolved_kw,
                "unresolved_samples": unresolved[:8],
                "n_unresolved": len(unresolved),
            }
    return weight_per_lid, discourse_per_lid, report


def score_sequences(words_path: Path, words_seqs, weight_per_lid, discourse_per_lid,
                    require_full: bool):
    n = len(words_seqs)
    scores = []
    n_skipped = 0
    f = open(words_path, "rb")
    try:
        for idx, (slen, boff) in enumerate(words_seqs):
            if require_full and slen != 512:
                n_skipped += 1
                continue
            f.seek(boff)
            arr = struct.unpack(f"<{slen}i", f.read(slen * 4))
            score = 0
            by_disc = Counter()
            hits = 0
            for lid in arr:
                if lid < 0:
                    continue
                w = weight_per_lid.get(lid)
                if w is None:
                    continue
                score += w
                hits += 1
                for d in discourse_per_lid.get(lid, ()):
                    by_disc[d] += 1
            if score == 0:
                continue
            scores.append((idx, score, hits, dict(by_disc), slen))
            if (idx + 1) % 20000 == 0:
                print(f"  scored {idx+1:>7,}/{n:,}", file=sys.stderr)
    finally:
        f.close()
    return scores, n_skipped


def decode_text(tokenizer, token_ids):
    """Best-effort surface text for human review — wordpiece-aware."""
    # Use tokenizer.decode which handles ## continuations and special tokens.
    return tokenizer.decode(token_ids, skip_special_tokens=False,
                            clean_up_tokenization_spaces=True)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--corpus", type=Path, default=DEFAULT_CORPUS)
    ap.add_argument("--words",  type=Path, default=DEFAULT_WORDS_V2)
    ap.add_argument("--discourse", type=Path, default=DEFAULT_DISCOURSE)
    ap.add_argument("--lemma-index", type=Path, default=DEFAULT_LEMMA_INDEX)
    ap.add_argument("--surface-map", type=Path, default=DEFAULT_SURFACE_MAP)
    ap.add_argument("--top", type=int, default=20)
    ap.add_argument("--preview-tokens", type=int, default=30,
                    help="How many leading tokens to render verbatim in the report")
    ap.add_argument("--full-only", action="store_true", default=True,
                    help="Restrict to seq_len==512 (default; spec §5-2)")
    ap.add_argument("--no-full-only", dest="full_only", action="store_false")
    ap.add_argument("--out", type=Path, default=DEFAULT_OUT_TOP)
    ap.add_argument("--all-out", type=Path, default=None,
                    help="If set, write per-sequence JSONL of all non-zero scores")
    ap.add_argument("--mode", choices=["full", "core-only"], default="core-only",
                    help="full: spec §5-2 weights (core=3 / names=2 / broad=1). "
                         "core-only: only core_theoretical_keywords count (weight=1).")
    args = ap.parse_args()
    weights = WEIGHTS_FULL if args.mode == "full" else WEIGHTS_CORE_ONLY
    print(f"Mode: {args.mode}  weights: {weights}", file=sys.stderr)

    print(f"Loading corpus meta from {args.corpus}", file=sys.stderr)
    n_sources, n_seqs, total_tokens, sources, c_seqs = _read_corpus_header(args.corpus)
    src_name = {sid: nm for sid, nm in sources}
    print(f"  {n_sources} sources, {n_seqs:,} seqs, {total_tokens:,} tokens",
          file=sys.stderr)

    print(f"Loading word-level lemma_ids from {args.words}", file=sys.stderr)
    w_n_seqs, w_seqs = _read_words_header(args.words)
    if w_n_seqs != n_seqs:
        sys.exit(f"corpus has {n_seqs} seqs but words file has {w_n_seqs}")

    print(f"Loading discourse keywords from {args.discourse}", file=sys.stderr)
    discourse_kw = json.loads(args.discourse.read_text())

    print(f"Loading lemma index", file=sys.stderr)
    lemma_to_id = json.loads(args.lemma_index.read_text())["lemma_to_id"]
    surface_to_lemma = json.loads(args.surface_map.read_text())

    weight_per_lid, discourse_per_lid, resolution_report = build_weight_map(
        discourse_kw, surface_to_lemma, lemma_to_id, weights)
    print(f"Resolved target lemma_ids: {len(weight_per_lid):,}", file=sys.stderr)
    # Per-discourse summary
    for d, cats in resolution_report.items():
        line = ", ".join(
            f"{c}={info['n_keywords_with_any_lid']}/{info['n_keywords_total']}"
            for c, info in cats.items())
        print(f"  {d:15s} {line}", file=sys.stderr)

    print(f"Scoring sequences (full_only={args.full_only})...", file=sys.stderr)
    scores, n_skipped = score_sequences(args.words, w_seqs, weight_per_lid,
                                        discourse_per_lid, args.full_only)
    print(f"  {len(scores):,} sequences with non-zero score "
          f"(skipped {n_skipped:,} short sequences)", file=sys.stderr)

    scores.sort(key=lambda r: r[1], reverse=True)
    top = scores[:args.top]

    # Decode tokens only for the top entries
    print(f"Loading BERT tokenizer for top-{args.top} text rendering...",
          file=sys.stderr)
    from transformers import BertTokenizer
    tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")

    top_records = []
    for seq_idx, score, hits, by_disc, slen in top:
        c_sid, c_slen, c_boff = c_seqs[seq_idx]
        assert c_slen == slen, f"length mismatch seq {seq_idx}: {c_slen} vs {slen}"
        token_ids = _load_seq_token_ids(args.corpus, c_slen, c_boff)
        leading_ids = token_ids[: args.preview_tokens]
        leading_text = decode_text(tokenizer, leading_ids)
        full_text = decode_text(tokenizer, token_ids)
        top_records.append({
            "seq_idx": seq_idx,
            "source_id": c_sid,
            "source_name": src_name.get(c_sid, "?"),
            "seq_len": slen,
            "score": score,
            "n_hits": hits,
            "score_density": round(score / slen, 4),
            "per_discourse_hits": by_disc,
            "leading_text": leading_text,
            "full_text": full_text,
        })

    out = {
        "mode": args.mode,
        "weights": weights,
        "n_seqs_considered": n_seqs,
        "n_seqs_skipped_short": n_skipped,
        "n_seqs_with_score": len(scores),
        "n_target_lemmas": len(weight_per_lid),
        "resolution_report": resolution_report,
        "top": top_records,
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(out, indent=2, ensure_ascii=False))
    print(f"\nWrote {args.out}", file=sys.stderr)

    if args.all_out is not None:
        args.all_out.parent.mkdir(parents=True, exist_ok=True)
        with open(args.all_out, "w") as f:
            for seq_idx, score, hits, by_disc, slen in scores:
                c_sid, _, _ = c_seqs[seq_idx]
                f.write(json.dumps({
                    "seq_idx": seq_idx,
                    "source_name": src_name.get(c_sid, "?"),
                    "score": score,
                    "n_hits": hits,
                    "seq_len": slen,
                    "per_discourse_hits": by_disc,
                }) + "\n")
        print(f"Wrote per-sequence stream to {args.all_out}", file=sys.stderr)

    # Brief stdout summary so the operator sees ranking without opening JSON
    print("\n" + "=" * 72)
    print(f"TOP {args.top} sequences  mode={args.mode}  weights={weights}")
    print("=" * 72)
    for i, r in enumerate(top_records, 1):
        disc_str = ", ".join(f"{d}:{n}" for d, n in
                             sorted(r["per_discourse_hits"].items(),
                                    key=lambda kv: -kv[1]))
        print(f"\n[{i:>2}] seq_idx={r['seq_idx']}  score={r['score']}  "
              f"hits={r['n_hits']}  density={r['score_density']}  "
              f"src={r['source_name']}")
        print(f"     discourse: {disc_str}")
        print(f"     leading:   {r['leading_text'][:200]}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
