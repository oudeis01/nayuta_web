#!/usr/bin/env bash
# Reusable batch re-encoder: wav -> opus (24 kbps mono voip) for the web demo.
#
# Idempotent: a key is re-encoded only when its .opus is missing or older than
# the source .wav (or with --force). This is the path for "audio changed" — when
# OmniVoice re-synthesizes some wavs, re-run and only the changed files are
# touched. The keyword index (lemma_id <-> word) is stable, so keys never move.
#
# Usage:
#   encode_opus.sh --src DIR --dst DIR [options]
#
#   --src   DIR     source wav dir            (required)
#   --dst   DIR     output opus dir           (required, created if absent)
#   --keyset FILE   only encode these keys; FILE is either a keyset.json
#                   (uses its "keys" array) or a plain text file of stems
#                   ("LLLLLL-VV", one per line). Omit to encode every *.wav.
#   --bitrate RATE  libopus bitrate           (default 24k)
#   --jobs  N       parallel jobs             (default: nproc)
#   --force         re-encode even if up to date
#   --dry-run       list what would happen, encode nothing
#
# Examples:
#   encode_opus.sh --src ~/.../whisper_audio --dst ./opus \
#       --keyset ../captures/session_001/keyset.json
#   encode_opus.sh --src ~/.../whisper_audio --dst ./opus   # whole corpus
set -euo pipefail

SRC="" DST="" KEYSET="" BITRATE="24k" JOBS="$(nproc)" FORCE=0 DRYRUN=0
while [ $# -gt 0 ]; do
  case "$1" in
    --src)     SRC="$2"; shift 2;;
    --dst)     DST="$2"; shift 2;;
    --keyset)  KEYSET="$2"; shift 2;;
    --bitrate) BITRATE="$2"; shift 2;;
    --jobs)    JOBS="$2"; shift 2;;
    --force)   FORCE=1; shift;;
    --dry-run) DRYRUN=1; shift;;
    *) echo "unknown arg: $1" >&2; exit 2;;
  esac
done
[ -n "$SRC" ] && [ -n "$DST" ] || { echo "error: --src and --dst required" >&2; exit 2; }
[ -d "$SRC" ] || { echo "error: src not a dir: $SRC" >&2; exit 2; }
command -v ffmpeg >/dev/null || { echo "error: ffmpeg not found" >&2; exit 2; }
mkdir -p "$DST"

# Build the list of stems to consider.
stems_file="$(mktemp)"
trap 'rm -f "$stems_file"' EXIT
if [ -n "$KEYSET" ]; then
  case "$KEYSET" in
    *.json) python3 -c 'import json,sys; print("\n".join(json.load(open(sys.argv[1]))["keys"]))' "$KEYSET" > "$stems_file";;
    *)      grep -v '^[[:space:]]*$' "$KEYSET" > "$stems_file";;
  esac
else
  (cd "$SRC" && ls *.wav 2>/dev/null | sed 's/\.wav$//') > "$stems_file"
fi
total=$(wc -l < "$stems_file")
echo "candidates: $total   src=$SRC   dst=$DST   bitrate=$BITRATE   jobs=$JOBS   force=$FORCE"

# Per-stem worker: skip when up to date unless forced; encode otherwise.
encode_one() {
  local stem="$1" src="$2" dst="$3" bitrate="$4" force="$5" dry="$6"
  local in="$src/$stem.wav" out="$dst/$stem.opus"
  if [ ! -f "$in" ]; then echo "MISSING $stem" ; return 0; fi
  if [ "$force" -eq 0 ] && [ -f "$out" ] && [ "$out" -nt "$in" ]; then
    echo "skip $stem"; return 0
  fi
  if [ "$dry" -eq 1 ]; then echo "would-encode $stem"; return 0; fi
  ffmpeg -loglevel error -y -i "$in" -c:a libopus -b:a "$bitrate" \
         -application voip -ac 1 "$out" && echo "ok $stem" || echo "FAIL $stem"
}
export -f encode_one

parallel --will-cite -j "$JOBS" \
  encode_one {} "$SRC" "$DST" "$BITRATE" "$FORCE" "$DRYRUN" \
  < "$stems_file" \
  | { ok=0 skip=0 fail=0 miss=0;
      while read -r status _; do
        case "$status" in
          ok) ok=$((ok+1));; skip) skip=$((skip+1));;
          FAIL) fail=$((fail+1));; MISSING) miss=$((miss+1));;
          would-encode) ok=$((ok+1));;
        esac
      done
      echo "done: encoded/would=$ok skipped=$skip failed=$fail missing=$miss"; }
