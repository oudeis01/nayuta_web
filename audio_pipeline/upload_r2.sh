#!/usr/bin/env bash
# Reusable uploader: sync local opus dir -> Cloudflare R2 bucket via rclone.
#
# rclone "sync" mirrors the local dir into the remote prefix: changed files are
# re-uploaded, files absent locally are deleted remotely. This pairs with
# encode_opus.sh idempotency — after re-synth + re-encode, this pushes only the
# changed objects. R2 egress is free; uploads (class-A ops) are the only cost.
#
# Prereq (one-time): configure an rclone remote of type s3 with provider=Cloudflare
# pointing at your R2 endpoint + access keys. Then pass --remote <name>:<bucket>.
#
# Usage:
#   upload_r2.sh --src ./opus --remote r2:nayuta-web --prefix audio [--dry-run]
#
#   --src    DIR        local opus dir            (required)
#   --remote NAME:BUCKET rclone remote + bucket   (required)
#   --prefix PATH       key prefix in bucket      (default: audio)
#   --dry-run           show transfer plan, change nothing
#
# This script intentionally does NOT hardcode credentials or a bucket name.
set -euo pipefail

SRC="" REMOTE="" PREFIX="audio" DRYRUN=""
while [ $# -gt 0 ]; do
  case "$1" in
    --src)     SRC="$2"; shift 2;;
    --remote)  REMOTE="$2"; shift 2;;
    --prefix)  PREFIX="$2"; shift 2;;
    --dry-run) DRYRUN="--dry-run"; shift;;
    *) echo "unknown arg: $1" >&2; exit 2;;
  esac
done
[ -n "$SRC" ] && [ -n "$REMOTE" ] || { echo "error: --src and --remote required" >&2; exit 2; }
[ -d "$SRC" ] || { echo "error: src not a dir: $SRC" >&2; exit 2; }
command -v rclone >/dev/null || { echo "error: rclone not found" >&2; exit 2; }

echo "sync $SRC -> $REMOTE/$PREFIX ${DRYRUN:+(dry-run)}"
rclone sync $DRYRUN \
  --transfers 16 --checkers 32 \
  --s3-no-check-bucket \
  --progress \
  "$SRC" "$REMOTE/$PREFIX"
