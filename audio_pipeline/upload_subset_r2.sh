#!/usr/bin/env bash
# upload_subset_r2.sh — push the deployable assets to the R2 bucket via rclone.
#
# Two payloads, two layouts:
#   audio/                 the playable opus subset (rclone --files-from the list
#                          produced by compute_playable_subset.py), pulled from
#                          the full local opus dir without copying the 2.7G whole.
#   captures/session_001/  events.jsonl.zst + manifest.json + ops.bin.zst, taken
#                          from the frontend public tree with --copy-links so the
#                          symlinked ops.bin.zst resolves to its real 93 MB file.
#
# rclone "copy" (not sync) never deletes remote objects, so a re-run only fills
# gaps / updates changed files. The S3 credentials live in the rclone remote
# (configured out-of-tree), never in this script.
#
# Prereq: an rclone remote of type s3 / provider Cloudflare pointing at the R2
# S3 endpoint. Create it (one-time) with the access key + secret from an R2 API
# token (Cloudflare dashboard -> R2 -> Manage R2 API Tokens):
#   rclone config create <remote> s3 provider=Cloudflare \
#     access_key_id=<KEY> secret_access_key=<SECRET> \
#     endpoint=https://<account_id>.r2.cloudflarestorage.com region=auto
#
# Usage:
#   upload_subset_r2.sh <remote> <bucket> [--dry-run]
#   e.g. upload_subset_r2.sh r2aftertheory aftertheory-web
set -euo pipefail

REMOTE="${1:?rclone remote name}"
BUCKET="${2:?r2 bucket name}"
DRY=""
[ "${3:-}" = "--dry-run" ] && DRY="--dry-run"

HERE="$(cd "$(dirname "$0")" && pwd)"        # web/audio_pipeline
WEB="$(dirname "$HERE")"                       # web
OPUS_DIR="$HERE/mi300x_accent/out_accent_opus"
FILES_FROM="$HERE/subset_files.txt"
CAPTURES="$WEB/frontend/public/captures/session_001"

[ -d "$OPUS_DIR" ]   || { echo "missing opus dir: $OPUS_DIR" >&2; exit 1; }
[ -f "$FILES_FROM" ] || { echo "missing file list: $FILES_FROM (run compute_playable_subset.py)" >&2; exit 1; }
[ -d "$CAPTURES" ]   || { echo "missing captures: $CAPTURES" >&2; exit 1; }
command -v rclone >/dev/null || { echo "rclone not found" >&2; exit 1; }

COMMON=(--transfers 32 --checkers 64 --s3-no-check-bucket --progress $DRY)

echo "==> audio/  ($(wc -l < "$FILES_FROM") files) -> $REMOTE:$BUCKET/audio"
rclone copy "${COMMON[@]}" --files-from "$FILES_FROM" "$OPUS_DIR" "$REMOTE:$BUCKET/audio"

echo "==> captures/session_001/  (events + manifest + ops.bin.zst) -> $REMOTE:$BUCKET/captures/session_001"
rclone copy "${COMMON[@]}" --copy-links "$CAPTURES" "$REMOTE:$BUCKET/captures/session_001"

echo "done."
