#!/usr/bin/env bash
# Tiny-H3, one command, start to finish (single 5090):
#
#   bash scripts/all.sh --fast    # small slice, ~15 min, asserts the plumbing
#   bash scripts/all.sh           # full corpus pass (~hours)
# Steps: env check -> model components -> corpus slice -> latents -> train -> demos.
set -euo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "$HERE/env.sh"

FAST=0
for arg in "$@"; do
  case "$arg" in
    --fast) FAST=1 ;;
    *) echo "unknown argument: $arg" >&2; exit 2 ;;
  esac
done

SEG_DIR="${TINY_H3_SEG_DIR:-$TINY_H3_DATA/h3selfgen_seg384}"
RUN_DIR="${RUN_DIR:-$TINY_H3_ROOT/runs/tiny_h3}"
STEPS="${STEPS:-9000}"
if [[ "$FAST" == "1" ]]; then
  MAX_CLIPS="${MAX_CLIPS:-24}"
  STEPS=300
else
  MAX_CLIPS="${MAX_CLIPS:-0}"
fi
mkdir -p "$RUN_DIR"
echo "run dir: $RUN_DIR"

step() { echo; echo "==================== $* ===================="; }

step "1/6 environment"
python - <<'PY'
import torch
print(f"torch {torch.__version__}, cuda={torch.cuda.is_available()}"
      + (f", gpu={torch.cuda.get_device_name(0)}" if torch.cuda.is_available() else "  <-- CPU only!"))
from diffusers import MiniMaxH3Transformer3DModel  # noqa: F401
print("diffusers H3 classes ok")
PY

step "2/6 model components (H3 VAEs + Qwen3-0.6B; skipped when cached)"
bash "$HERE/download.sh" --group model

step "3/6 corpus segments (H3-SelfGen -> 22-frame segments @ 384x384)"
bash "$HERE/data.sh" download
python "$TINY_H3_ROOT/tools/segment_clips.py" \
  --metadata "${TINY_H3_DATASET_DIR:-$TINY_H3_DATA/h3_selfgen}/metadata.jsonl" \
  --video-root "${TINY_H3_DATASET_DIR:-$TINY_H3_DATA/h3_selfgen}" \
  --out "$SEG_DIR" --segment-frames 22 --size 384 --workers 24 --max-clips "$MAX_CLIPS"

step "4/6 latent cache (official VAEs + Qwen3-0.6B, 256-token prompts)"
python "$TINY_H3_ROOT/tools/prepare_latents.py" --data-dir "$SEG_DIR" \
  --out "$SEG_DIR/latents" --device cuda --size 384 --text-tokens 256 \
  --batch-size 2 --split train

step "5/6 train (1.23B DiT, configs/config.json)"
python -m tiny_h3.train.train_full \
  --latents "$SEG_DIR/latents" \
  --preset configs/config.json \
  --out "$RUN_DIR" --steps "$STEPS" --batch-size 8 --grad-accum 3 --lr 2e-4 \
  --device cuda

step "6/6 demos (7 preset corpus prompts)"
bash "$HERE/demo.sh" "$RUN_DIR/final"

echo
echo "all done."
echo "  demo videos : $TINY_H3_DATA/../outputs (see demo.sh output)"
echo "  checkpoints : $RUN_DIR"
