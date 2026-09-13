#!/usr/bin/env bash
# Prepare the Tiny-H3 training corpus.
#
#   bash scripts/data.sh download   # fetch the H3-SelfGen dataset from ModelScope (full 8,560 clips)
#   bash scripts/data.sh slice      # slice clips into 22-frame segments @ 384x384 (CPU, minutes)
#   bash scripts/data.sh latents    # encode segments with the official VAEs (GPU)
#   bash scripts/data.sh all        # download + slice + latents
set -euo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "$HERE/env.sh"

DATASET_DIR="${TINY_H3_DATASET_DIR:-$TINY_H3_DATA/h3_selfgen}"
SEG_DIR="${TINY_H3_SEG_DIR:-$TINY_H3_DATA/h3selfgen_seg384}"
STEP="${1:-all}"

download() {
  python - <<'PY'
from modelscope.hub.snapshot_download import dataset_snapshot_download
dataset_snapshot_download("DiffSynth-Studio/MiniMax-H3-Self-Generated-Dataset",
                          local_dir="$DATASET_DIR")
PY
}

slice() {
  python "$TINY_H3_ROOT/tools/segment_clips.py" \
    --metadata "$DATASET_DIR/metadata.jsonl" --video-root "$DATASET_DIR" \
    --out "$SEG_DIR" --segment-frames 22 --size 384 --workers 24
}

latents() {
  python "$TINY_H3_ROOT/tools/prepare_latents.py" --data-dir "$SEG_DIR" \
    --out "$SEG_DIR/latents" --device "${TINY_H3_DEVICE:-cuda}" \
    --size 384 --text-tokens 256 --batch-size 2 --split train
}

case "$STEP" in
  download) download ;;
  slice) slice ;;
  latents) latents ;;
  all) download; slice; latents ;;
  *) echo "usage: $0 [download|slice|latents|all]" >&2; exit 2 ;;
esac
echo "done: $STEP"
