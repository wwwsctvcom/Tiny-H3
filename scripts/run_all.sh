#!/usr/bin/env bash
# Tiny-H3, one command, start to finish:
#
#   bash scripts/run_all.sh                 # full loop on one GPU (~1-2 h on a 5090)
#   bash scripts/run_all.sh --fast          # smoke-sized loop (~15 min), asserts the plumbing
#   bash scripts/run_all.sh --skip-data     # reuse an existing latent cache
#
# Steps: env check -> download assets (if missing) -> synth data -> latent cache ->
#        full fine-tune -> text-to-AV demo -> report.
set -euo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "$HERE/env.sh"

FAST=0
SKIP_DATA=0
SKIP_DOWNLOAD=0
for arg in "$@"; do
  case "$arg" in
    --fast) FAST=1 ;;
    --skip-data) SKIP_DATA=1 ;;
    --skip-download) SKIP_DOWNLOAD=1 ;;
    *) echo "unknown argument: $arg" >&2; exit 2 ;;
  esac
done

if [[ "$FAST" == "1" ]]; then
  export TINY_H3_SYNTH_COUNT="${TINY_H3_SYNTH_COUNT:-64}"
  export TINY_H3_SYNTH_VAL="${TINY_H3_SYNTH_VAL:-8}"
  export TINY_H3_DATA_SIZE="${TINY_H3_DATA_SIZE:-128}"
  PRESET="smoke"
  STEPS="${STEPS:-300}"
else
  PRESET="tiny_h3_5090"
  STEPS="${STEPS:-4000}"
fi
RUN_DIR="${RUN_DIR:-$TINY_H3_ROOT/runs/$(date +%Y%m%d-%H%M%S)}"
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

if [[ "$SKIP_DOWNLOAD" == "0" ]]; then
  step "2/6 assets (H3 VAEs, Qwen3-0.6B; skipped automatically when cached)"
  python "$TINY_H3_ROOT/tools/download_assets.py" --group model
else
  step "2/6 assets (skipped)"
fi

if [[ "$SKIP_DATA" == "0" ]]; then
  step "3/6 synthetic clips"
  python "$TINY_H3_ROOT/tools/make_synth_data.py" \
    --out "$TINY_H3_DATA/synth" \
    --count "${TINY_H3_SYNTH_COUNT:-512}" --val-count "${TINY_H3_SYNTH_VAL:-64}" \
    --size "${TINY_H3_DATA_SIZE:-256}" --frames "${TINY_H3_FRAMES:-22}" \
    --workers "${TINY_H3_WORKERS:-8}" --preview

  step "4/6 latent cache (frozen H3 VAEs + Qwen3)"
  python "$TINY_H3_ROOT/tools/prepare_latents.py" \
    --data-dir "$TINY_H3_DATA/synth" --out "$TINY_H3_DATA/synth/latents" \
    --device cuda
else
  step "3-4/6 data (skipped)"
fi

step "5/6 full fine-tune ($PRESET, $STEPS steps)"
python -m tiny_h3.train.train_full \
  --latents "$TINY_H3_DATA/synth/latents" \
  --preset "$PRESET" \
  --steps "$STEPS" \
  --out "$RUN_DIR/train_full" \
  --device cuda

step "6/6 text-to-audio-video demo"
python -m tiny_h3.pipeline \
  --checkpoint "$RUN_DIR/train_full/final" \
  --prompts \
    "a red circle bouncing on a dark background, with rhythmic thumps" \
    "two squares, one cyan and one yellow, pulsing on a purple background, with steady beats at a fast tempo" \
    "three rings in purple, white and yellow, orbiting on a blue background, with a rising and falling tone" \
  --steps 24 \
  --out "$RUN_DIR/demo" \
  --device cuda

echo
echo "all done."
echo "  demo videos : $RUN_DIR/demo"
echo "  checkpoints : $RUN_DIR/train_full"
