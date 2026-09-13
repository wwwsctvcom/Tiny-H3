#!/usr/bin/env bash
# Prepare training data.
#
#   bash scripts/data.sh synth              # procedural clips only (CPU, minutes)
#   bash scripts/data.sh latents            # encode clips -> latent cache (needs the H3 VAEs; GPU)
#   bash scripts/data.sh all                # synth + latents
#
# Real footage is optional: after `scripts/download.sh --group data`, point
# tools/prepare_latents.py at --manifest <dataset>/train.jsonl (see docs/design.md).
set -euo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "$HERE/env.sh"

STEP="${1:-all}"
SIZE="${TINY_H3_DATA_SIZE:-256}"
FRAMES="${TINY_H3_FRAMES:-22}"
COUNT="${TINY_H3_SYNTH_COUNT:-512}"
VAL_COUNT="${TINY_H3_SYNTH_VAL:-64}"
WORKERS="${TINY_H3_WORKERS:-8}"

synth() {
  echo "== synthetic clips (${COUNT} train / ${VAL_COUNT} val, ${SIZE}x${SIZE}, ${FRAMES} frames) =="
  python "$TINY_H3_ROOT/tools/make_synth_data.py" \
    --out "$TINY_H3_DATA/synth" --count "$COUNT" --val-count "$VAL_COUNT" \
    --size "$SIZE" --frames "$FRAMES" --workers "$WORKERS" --preview
}

latents() {
  echo "== encoding clips into the latent cache =="
  local extra=()
  if [[ -n "${TINY_H3_LATENT_SPLIT:-}" ]]; then extra+=(--split "$TINY_H3_LATENT_SPLIT"); fi
  python "$TINY_H3_ROOT/tools/prepare_latents.py" \
    --data-dir "$TINY_H3_DATA/synth" --out "$TINY_H3_DATA/synth/latents" \
    --device "${TINY_H3_DEVICE:-cuda}" "${extra[@]}"
}

case "$STEP" in
  synth) synth ;;
  latents) latents ;;
  all) synth; latents ;;
  *) echo "usage: $0 [synth|latents|all]" >&2; exit 2 ;;
esac
echo "done: $STEP"
