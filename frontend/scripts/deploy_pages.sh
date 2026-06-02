#!/usr/bin/env bash
# deploy_pages.sh — build the static frontend and publish it to Cloudflare Pages.
#
# What ships is the build, not the source: `bun run build` runs `tsc && vite
# build` into dist/, baking VITE_CAPTURE_BASE / VITE_AUDIO_BASE into the bundle
# so the live site fetches the heavy capture dumps + opus audio from R2. Pages
# only carries the static shell (see docs/20260527-web-version-action-plan.md
# §7-4: Pages for the site, R2 for the >25 MB ops.bin.zst and the opus pack).
#
# The opus audio + capture dumps go to R2 separately via
# web/audio_pipeline/upload_subset_r2.sh — this script never touches them.
#
# Config lives in an env file (default web/frontend/deploy.env, gitignored) so
# the bucket URLs + Pages project are stored once instead of retyped each run.
# Copy deploy.env.example -> deploy.env and fill it in. Keys:
#   VITE_CAPTURE_BASE   capture dump base URL on R2   (required)
#   VITE_AUDIO_BASE     opus audio base URL on R2     (required)
#   CF_PAGES_PROJECT    existing Cloudflare Pages project name (required to publish)
#   CF_PAGES_BRANCH     deploy branch, prod vs preview (optional)
# Override the file location with DEPLOY_ENV=/path. The flags below still win
# over whatever the file sets.
#
# NOTE: CF_PAGES_PROJECT must already exist. `wrangler pages deploy` does not
# create projects; make it once with `wrangler pages project create <name>`
# (R2 bucket name and Pages project name are independent namespaces).
#
# Prereq (one-time): `wrangler login`, or for non-interactive / CI export
# CLOUDFLARE_API_TOKEN + CLOUDFLARE_ACCOUNT_ID. Credentials never live here.
#
# Usage:
#   deploy_pages.sh                      # all config from deploy.env
#   deploy_pages.sh --build-only         # build dist/ without publishing
#   deploy_pages.sh --project NAME ...   # override a single value ad hoc
#
#   --project NAME       override CF_PAGES_PROJECT
#   --capture-base URL   override VITE_CAPTURE_BASE
#   --audio-base URL     override VITE_AUDIO_BASE
#   --branch NAME        override CF_PAGES_BRANCH
#   --build-only         build dist/ but do not publish
#   --skip-install       skip `bun install` (deps already present)
#   --dry-run            build, then print the wrangler command instead of running it
set -euo pipefail

HERE="$(cd "$(dirname "$0")" && pwd)"   # web/frontend/scripts
FRONTEND="$(dirname "$HERE")"            # web/frontend

# Load the deploy config first so its values act as defaults; `set -a` exports
# every key it sets, so the VITE_* vars reach the vite build automatically.
ENV_FILE="${DEPLOY_ENV:-$FRONTEND/deploy.env}"
if [ -f "$ENV_FILE" ]; then
  echo "==> config: $ENV_FILE"
  set -a; . "$ENV_FILE"; set +a
fi

PROJECT="${CF_PAGES_PROJECT:-}"
BRANCH="${CF_PAGES_BRANCH:-}"
CAPTURE_BASE="${VITE_CAPTURE_BASE:-}"
AUDIO_BASE="${VITE_AUDIO_BASE:-}"
BUILD_ONLY="" SKIP_INSTALL="" DRYRUN=""
while [ $# -gt 0 ]; do
  case "$1" in
    --project)      PROJECT="$2"; shift 2;;
    --capture-base) CAPTURE_BASE="$2"; shift 2;;
    --audio-base)   AUDIO_BASE="$2"; shift 2;;
    --branch)       BRANCH="$2"; shift 2;;
    --build-only)   BUILD_ONLY=1; shift;;
    --skip-install) SKIP_INSTALL=1; shift;;
    --dry-run)      DRYRUN=1; shift;;
    *) echo "unknown arg: $1" >&2; exit 2;;
  esac
done

[ -n "$CAPTURE_BASE" ] || { echo "error: VITE_CAPTURE_BASE not set (deploy.env or --capture-base)" >&2; exit 2; }
[ -n "$AUDIO_BASE" ]   || { echo "error: VITE_AUDIO_BASE not set (deploy.env or --audio-base)" >&2; exit 2; }
if [ -z "$BUILD_ONLY" ]; then
  [ -n "$PROJECT" ] || { echo "error: CF_PAGES_PROJECT not set (deploy.env or --project), or pass --build-only" >&2; exit 2; }
  command -v wrangler >/dev/null || { echo "error: wrangler not found (npm i -g wrangler)" >&2; exit 2; }
fi
command -v bun >/dev/null || { echo "error: bun not found" >&2; exit 2; }

# Re-export the resolved values so CLI overrides (not just file keys) reach vite.
export VITE_CAPTURE_BASE="$CAPTURE_BASE" VITE_AUDIO_BASE="$AUDIO_BASE"

cd "$FRONTEND"
[ -n "$SKIP_INSTALL" ] || bun install

echo "==> build  (CAPTURE_BASE=$CAPTURE_BASE  AUDIO_BASE=$AUDIO_BASE)"
bun run build

if [ -n "$BUILD_ONLY" ]; then
  echo "build-only: dist/ ready at $FRONTEND/dist (not published)."
  exit 0
fi

DEPLOY=(wrangler pages deploy dist --project-name "$PROJECT")
[ -n "$BRANCH" ] && DEPLOY+=(--branch "$BRANCH")

if [ -n "$DRYRUN" ]; then
  echo "dry-run: ${DEPLOY[*]}"
  exit 0
fi

echo "==> publish -> Cloudflare Pages project '$PROJECT'${BRANCH:+ (branch $BRANCH)}"
"${DEPLOY[@]}"
echo "done."
