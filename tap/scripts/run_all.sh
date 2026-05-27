#!/bin/bash
# Run multiple capture sessions sequentially. Each session is a self-contained
# bert_install + namedrop_tap pair, so we can safely tear one down before
# starting the next. If any session aborts mid-flight the script continues
# with the next tag rather than leaving captures half-done.
#
# Usage:
#   ./run_all.sh <tag1> <tag2> ...
#   DURATION_S=1800 ./run_all.sh <tag1>     # override per invocation

set -uo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
DURATION_S="${DURATION_S:-7200}"

if [[ $# -eq 0 ]]; then
    echo "usage: $0 <tag1> <tag2> ..." >&2
    exit 1
fi

START=$(date +%s)
echo "[$(date -Is)] run_all.sh starting on $(hostname) — tags: $*  duration=${DURATION_S}s each"

for TAG in "$@"; do
    echo
    echo "==================== $TAG ===================="
    if ! "$SCRIPT_DIR/run_session.sh" "$TAG" "$DURATION_S"; then
        echo "[$(date -Is)] session $TAG FAILED — continuing"
    fi
done

END=$(date +%s)
echo
echo "[$(date -Is)] run_all.sh complete — wall-clock $((END - START)) s"
