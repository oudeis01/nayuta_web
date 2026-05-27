#!/bin/bash
# Run a single capture session: bert_install + namedrop_tap for `duration_s`
# seconds against the corpus_demo identified by `tag`.
#
# Layout (relative to this script's parent, BASE):
#   $BASE/bert_install
#   $BASE/bert_base_uncased.bin
#   $BASE/whisper_graph_v3.bin
#   $BASE/variant_lookup.bin
#   $BASE/seeds/corpus_demo_<tag>.bin
#   $BASE/seeds/corpus_words_demo_<tag>.bin
#   $BASE/seeds/corpus_demo_<tag>.json
#   $BASE/tap/namedrop_tap
#
# Output:
#   $BASE/captures/<tag>/{ops.bin.zst, events.jsonl.zst, manifest.json,
#                         bert.log, tap.log}

set -euo pipefail

TAG="${1:?usage: run_session.sh <tag> [duration_s]}"
DURATION_S="${2:-7200}"

# Capture-only port pair, chosen to avoid colliding with whatever the host
# normally runs (SC on 57120, kiosk graphics on 5555). Override via env if
# even these conflict.
ZMQ_PORT="${ZMQ_PORT:-5556}"
OSC_PORT="${OSC_PORT:-57131}"

# BASE = directory containing this script (the capture root on each host).
BASE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
OUT="$BASE/captures/$TAG"
mkdir -p "$OUT"

if [[ -f "$OUT/manifest.json" ]]; then
    echo "[$(date -Is)] $TAG already has manifest.json — skipping"
    exit 0
fi

CORPUS="$BASE/seeds/corpus_demo_${TAG}.bin"
WORDS="$BASE/seeds/corpus_words_demo_${TAG}.bin"
SIDECAR="$BASE/seeds/corpus_demo_${TAG}.json"
for f in "$CORPUS" "$WORDS" "$SIDECAR" \
         "$BASE/bert_install" "$BASE/bert_base_uncased.bin" \
         "$BASE/whisper_graph_v3.bin" "$BASE/variant_lookup.bin" \
         "$BASE/tap/namedrop_tap"; do
    [[ -f "$f" ]] || { echo "MISSING: $f" >&2; exit 1; }
done

echo "[$(date -Is)] starting tag=$TAG dur=${DURATION_S}s on $(hostname)"

"$BASE/bert_install" "$BASE/bert_base_uncased.bin" \
    --corpus "$CORPUS" \
    --whisper-map "$WORDS" \
    --whisper-graph "$BASE/whisper_graph_v3.bin" \
    --whisper-lookup "$BASE/variant_lookup.bin" \
    --whisper-threshold 0.001 \
    --whisper-debounce-ms 120000 \
    --whisper-min-interval-ms 250 \
    --rate-limit 1092 \
    -z "$ZMQ_PORT" \
    --osc-host "127.0.0.1:${OSC_PORT}" \
    > "$OUT/bert.log" 2>&1 &
BERT_PID=$!
echo "$BERT_PID" > "$OUT/bert.pid"

# Let bert_install bind its ZMQ PUB before tap connects.
sleep 1

cleanup() {
    if kill -0 "$BERT_PID" 2>/dev/null; then
        echo "[$(date -Is)] stopping bert (pid $BERT_PID)"
        kill -TERM "$BERT_PID" 2>/dev/null || true
        for _ in 1 2 3 4 5; do
            kill -0 "$BERT_PID" 2>/dev/null || break
            sleep 1
        done
        kill -KILL "$BERT_PID" 2>/dev/null || true
    fi
    rm -f "$OUT/bert.pid"
}
trap cleanup EXIT INT TERM

"$BASE/tap/namedrop_tap" \
    --out-dir "$OUT" \
    --tag "$TAG" \
    --duration-s "$DURATION_S" \
    --zmq-endpoint "tcp://127.0.0.1:${ZMQ_PORT}" \
    --osc-bind "127.0.0.1:${OSC_PORT}" \
    --corpus-sidecar "$SIDECAR" \
    --bert-install "$BASE/bert_install" \
    2>&1 | tee "$OUT/tap.log"

echo "[$(date -Is)] session $TAG complete"
ls -la "$OUT/"
