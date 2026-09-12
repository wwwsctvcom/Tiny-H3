#!/usr/bin/env bash
# Install Tiny-H3's Python environment (run once per instance).
#
#   bash scripts/setup_env.sh              # pip + diffusers from the vendored clone
#   bash scripts/setup_env.sh --with-torch # also install torch (only for a bare Python env)
#
# Notes
#  * ``diffusers``: MiniMax-H3 support is on ``main`` (0.41.0.dev0), not in any release yet,
#    so it is installed from the pinned commit in ``scripts/pins.env``.
#  * pip uses the AutoDL aliyun mirror (``/etc/pip.conf``).  Only the optional GitHub clone
#    needs the academic proxy; the script turns it on for that step and off again.

set -euo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT="$(dirname "$HERE")"
# shellcheck disable=SC1091
source "$HERE/env.sh"
# shellcheck disable=SC1091
source "$HERE/pins.env"

WITH_TORCH=0
for arg in "$@"; do
  case "$arg" in
    --with-torch) WITH_TORCH=1 ;;
    *) echo "unknown argument: $arg" >&2; exit 2 ;;
  esac
done

echo "== Tiny-H3 env =="
echo "project      : $PROJECT"
echo "refs         : $TINY_H3_REF"
echo "HF_HOME      : $HF_HOME"
echo "python       : $(python -V 2>&1)"

if [[ "$WITH_TORCH" == "1" ]]; then
  echo "== installing torch (cu128 wheels for Blackwell/5090) =="
  pip install --index-url https://download.pytorch.org/whl/cu128 "torch==$TORCH_VERSION" "torchvision"
fi

echo "== installing python dependencies =="
pip install -r "$PROJECT/requirements.txt"

echo "== installing pinned diffusers ($DIFFUSERS_COMMIT) =="
DIFFUSERS_DIR="$TINY_H3_REF/diffusers"
if [[ -d "$DIFFUSERS_DIR/.git" ]]; then
  git -C "$DIFFUSERS_DIR" fetch --depth 1 origin "$DIFFUSERS_COMMIT" 2>/dev/null || true
  git -C "$DIFFUSERS_DIR" checkout -q "$DIFFUSERS_COMMIT" 2>/dev/null || true
  pip install -e "$DIFFUSERS_DIR"
else
  echo "  vendored clone missing -- cloning (China mirror first, GitHub fallback)"
  mkdir -p "$TINY_H3_REF"
  if ! git clone "${TINY_H3_GIT_MIRROR:-https://gitclone.com/github.com/huggingface/diffusers.git}" "$DIFFUSERS_DIR"; then
    rm -rf "$DIFFUSERS_DIR"
    echo "  mirror failed -- falling back to github (academic proxy)"
    # shellcheck disable=SC1091
    source /etc/network_turbo 2>/dev/null || true
    git clone https://github.com/huggingface/diffusers.git "$DIFFUSERS_DIR"
  fi
  git -C "$DIFFUSERS_DIR" checkout -q "$DIFFUSERS_COMMIT"
  pip install -e "$DIFFUSERS_DIR"
fi

echo "== installing tiny-h3 itself (gives `tiny-h3-full`, `tiny-h3-generate`, ...) =="
pip install -e "$PROJECT"

echo "== sanity check =="
python - <<'PY'
import importlib
mods = ["torch", "diffusers", "transformers", "accelerate", "peft", "soundfile", "imageio_ffmpeg", "scipy"]
for name in mods:
    try:
        mod = importlib.import_module(name)
        print(f"  ok   {name:16s} {getattr(mod, '__version__', '?')}")
    except Exception as exc:  # noqa: BLE001
        print(f"  FAIL {name:16s} {type(exc).__name__}: {exc}")
from diffusers import MiniMaxH3Transformer3DModel, AutoencoderKLMiniMaxH3, AutoencoderKLMiniMaxH3Audio
print("  ok   MiniMax-H3 classes importable")
import torch
print(f"  torch cuda available: {torch.cuda.is_available()}"
      + (f" ({torch.cuda.get_device_name(0)})" if torch.cuda.is_available() else " (CPU-only box)"))
PY

echo
echo "done.  next: bash scripts/download_assets.sh && bash scripts/prepare_data.sh"
