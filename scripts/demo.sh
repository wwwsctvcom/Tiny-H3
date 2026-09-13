#!/usr/bin/env bash
# Generate demo videos from a trained checkpoint.
#
#   bash scripts/demo.sh runs/mem8_overfit/final                    # preset corpus prompts
#   bash scripts/demo.sh runs/mem8_overfit/final "integrated_multimodal_description: [Shot 1] Live-action, cinematic, ..."
set -euo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "$HERE/env.sh"

CKPT="${1:?usage: $0 <checkpoint> [prompt ...]}"
shift || true
OUT="${TINY_H3_DEMO_OUT:-$TINY_H3_ROOT/outputs/demo_$(date +%H%M%S)}"

if [[ "$#" -gt 0 ]]; then
  PROMPTS=("$@")
else
  PROMPTS=(
    "a red circle bouncing on a dark background, with rhythmic thumps"
    "two squares, one cyan and one yellow, pulsing on a purple background, with steady beats at a fast tempo"
    "three rings in purple, white and yellow, orbiting on a blue background, with a rising and falling tone"
    "a white ring swinging on a warm background, with soft ticks at a slow tempo"
  )
fi

python -m tiny_h3.pipeline \
  --checkpoint "$CKPT" \
  --prompts "${PROMPTS[@]}" \
  --out "$OUT" \
  --height "${TINY_H3_HEIGHT:-384}" --width "${TINY_H3_WIDTH:-384}" \
  --frames "${TINY_H3_FRAMES:-22}" --steps "${TINY_H3_STEPS:-24}" \
  --device "${TINY_H3_DEVICE:-cuda}"
echo "videos written to $OUT"
ls -la "$OUT"
