#!/usr/bin/env bash
# ============================================================================
# setup_instance.sh - prepare an MI300X (ROCm) instance for accent generation.
#
# omnivoice is NOT shipped in this pack. It is cloned fresh from upstream here
# (faster than transferring the local tree) and the model weights download from
# HuggingFace at first use.
#
# Run inside the ROCm container, e.g.:
#   docker run -it --device /dev/kfd --device /dev/dri --group-add render \
#     -v "$PWD":/workspace -w /workspace rocm/pytorch:latest
#   ./setup_instance.sh
#
#   ./setup_instance.sh --no-model   # skip model pre-download
# ============================================================================
set -euo pipefail

REPO_URL="https://github.com/k2-fsa/OmniVoice.git"
MODEL_ID="k2-fsa/OmniVoice"
REPO_DIR="$HOME/OmniVoice"
SKIP_MODEL=false
HERE="$(cd "$(dirname "$0")" && pwd)"

while [[ $# -gt 0 ]]; do
  case "$1" in
    --no-model) SKIP_MODEL=true; shift ;;
    -h|--help)  head -n 18 "$0" | sed 's/^# \?//'; exit 0 ;;
    *) echo "Unknown arg: $1"; exit 1 ;;
  esac
done

echo "[1/4] GPU check"
python3 -c "
import torch
assert torch.cuda.is_available(), 'GPU not visible to torch'
for i in range(torch.cuda.device_count()):
    p = torch.cuda.get_device_properties(i)
    print(f'  GPU {i}: {p.name}  VRAM={p.total_memory/1024**3:.1f} GiB')
"

echo "[2/4] Clone + install OmniVoice (upstream)"
if [ -d "$REPO_DIR/.git" ]; then
  git -C "$REPO_DIR" pull --ff-only || true
else
  git clone "$REPO_URL" "$REPO_DIR"
fi
pip install -e "$REPO_DIR"

echo "[3/4] Install QA dependencies"
pip install -r "$HERE/requirements_qa.txt"

if [ "$SKIP_MODEL" = false ]; then
  echo "[4/4] Pre-download model weights ($MODEL_ID)"
  python3 -c "
from huggingface_hub import snapshot_download
snapshot_download('$MODEL_ID')
print('  model cached')
"
else
  echo "[4/4] Skipped model pre-download"
fi

echo
echo "Ready. Probe first, then full run:"
echo "  ./run_accent_gen.sh --limit 100      # throughput probe"
echo "  ./run_accent_gen.sh --resume         # full run (background)"
