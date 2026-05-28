#!/usr/bin/env python3
"""Extract the set of (lemma_id, variant_idx) audio keys referenced by captured
/bert/whisper events.

Each /bert/whisper OSC message (see bert.c:osc_whisper) carries:
    args[0]      main lemma_id (node_id)
    args[1]      main variant idx (always 0, canonical)
    args[2]      is_bridge
    args[3..10]  affinity[8] (floats)
    args[11]     n_neighbors
    args[12..]   n triples of (neighbor_lemma_id, variant_idx, dist)

The web demo must play the main voice plus every neighbor in the cloud, so the
audio keys it needs are: the main (lid, 0) plus each (neighbor_lid, var_idx).
This is exactly the prefetch set (action plan 7-6) and the minimal encode set.

The keyword index is stable, so these keys do not change when audio is
re-synthesized; only the audio content behind a key changes. Re-run is cheap
and deterministic.

Usage:
    extract_keyset.py SESSION_DIR [SESSION_DIR ...] --out keyset.json
    extract_keyset.py web/captures/session_001/*/ --out keyset.json --per-session

A SESSION_DIR is any directory containing events.jsonl.zst. Keys are emitted as
filename stems "LLLLLL-VV" (zero-padded), matching the wav/opus naming.
"""
import argparse
import json
import sys
import zstandard
from pathlib import Path


def iter_events(jsonl_zst: Path):
    dctx = zstandard.ZstdDecompressor()
    with open(jsonl_zst, "rb") as fh:
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
            if buf.strip():
                yield json.loads(buf)


def keys_from_whisper(args):
    """Yield (lid, vidx) for the main voice and every neighbor in one message."""
    if len(args) < 12:
        return
    main_lid = int(args[0])
    main_var = int(args[1])
    yield main_lid, main_var
    n = int(args[11])
    base = 12
    for i in range(n):
        off = base + 3 * i
        if off + 1 >= len(args):
            break
        yield int(args[off]), int(args[off + 1])


def scan_session(session_dir: Path):
    """Return (Counter-like dict key->count, n_whisper) for one session."""
    events = session_dir / "events.jsonl.zst"
    if not events.exists():
        raise FileNotFoundError(f"no events.jsonl.zst in {session_dir}")
    counts = {}
    n_whisper = 0
    for ev in iter_events(events):
        if ev.get("path") != "/bert/whisper":
            continue
        n_whisper += 1
        for lid, vidx in keys_from_whisper(ev.get("args", [])):
            stem = f"{lid:06d}-{vidx:02d}"
            counts[stem] = counts.get(stem, 0) + 1
    return counts, n_whisper


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("sessions", nargs="+", type=Path,
                    help="session dir(s) containing events.jsonl.zst")
    ap.add_argument("--out", required=True, type=Path,
                    help="output keyset JSON path")
    ap.add_argument("--per-session", action="store_true",
                    help="also write <out_stem>.<tag>.json prefetch list per session")
    args = ap.parse_args()

    union = {}
    per_session = {}
    total_whisper = 0
    for sd in args.sessions:
        sd = sd.resolve()
        counts, n_whisper = scan_session(sd)
        total_whisper += n_whisper
        tag = sd.name
        per_session[tag] = {
            "n_whisper": n_whisper,
            "n_keys": len(counts),
            "keys": sorted(counts),
        }
        for k, c in counts.items():
            union[k] = union.get(k, 0) + c
        print(f"  {tag:22} whisper={n_whisper:6d}  distinct_keys={len(counts):6d}",
              file=sys.stderr)

    out = {
        "sessions": [sd.name for sd in args.sessions],
        "total_whisper": total_whisper,
        "n_keys": len(union),
        "keys": sorted(union),
    }
    args.out.write_text(json.dumps(out, indent=2))
    print(f"union: {len(union)} distinct keys across {len(args.sessions)} session(s)"
          f" -> {args.out}", file=sys.stderr)

    if args.per_session:
        for tag, data in per_session.items():
            p = args.out.with_suffix(f".{tag}.json")
            p.write_text(json.dumps(data, indent=2))
            print(f"  per-session: {p}", file=sys.stderr)


if __name__ == "__main__":
    main()
