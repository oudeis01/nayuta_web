#!/usr/bin/env bash
# ============================================================================
# run_accent_gen.sh - launch accent generation on the instance.
#
# Mirrors the overnight-capture nohup pattern: detached, resumable, logged.
# All args pass through to generate_accent_qa_mi300x.py.
#
#   ./run_accent_gen.sh --limit 100      # foreground throughput probe
#   ./run_accent_gen.sh --resume         # background full run
#   ./run_accent_gen.sh --resume --max-retries 20   # unresolved sweep
#
# Watch:   tail -f run_accent_gen.log
# ============================================================================
set -euo pipefail
HERE="$(cd "$(dirname "$0")" && pwd)"
cd "$HERE"

LOG="run_accent_gen.log"
DRIVER="generate_accent_qa_mi300x.py"

# A probe (--limit) runs in the foreground so you see the ETA immediately.
is_probe=false
for a in "$@"; do [[ "$a" == "--limit" ]] && is_probe=true; done

if $is_probe; then
  exec python3 "$DRIVER" "$@"
fi

echo "Launching full run in background -> $LOG"
nohup python3 "$DRIVER" "$@" >>"$LOG" 2>&1 &
disown
echo "PID $!   (tail -f $LOG)"
