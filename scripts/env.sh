# Tiny-H3 environment.  Source this before running anything:  source scripts/env.sh
#
# Every artefact lives on the AutoDL data disk so instance reboots and image swaps keep it:
#
#   /root/autodl-tmp/hf_cache      HuggingFace cache (H3 weights, t5-small)
#   /root/autodl-tmp/refs          vendored diffusers (pinned commit) + reference repos
#   /root/autodl-tmp/tiny-h3       this project
#   /root/autodl-tmp/datasets      downloaded real datasets (WISA-254, concerts)
#
# Override any of them by exporting the variable before sourcing.

# Project root is derived from this file's location, so a fresh clone works as-is.
export TINY_H3_ROOT="${TINY_H3_ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}"
export TINY_H3_REF="${TINY_H3_REF:-/root/autodl-tmp/refs}"
export TINY_H3_DATA="${TINY_H3_DATA:-$TINY_H3_ROOT/data}"
export TINY_H3_MODELS="${TINY_H3_MODELS:-/root/autodl-tmp/models}"
export TINY_H3_DATASETS="${TINY_H3_DATASETS:-/root/autodl-tmp/datasets}"

# HuggingFace: mirror by default (measured ~40x faster than the academic proxy here),
# plain HTTP only -- the mirror rejects the Xet CAS handshake with 401.
export HF_HOME="${HF_HOME:-/root/autodl-tmp/hf_cache}"
export HF_ENDPOINT="${HF_ENDPOINT:-https://hf-mirror.com}"
export HF_HUB_DISABLE_XET="${HF_HUB_DISABLE_XET:-1}"
export HF_HUB_ENABLE_HF_TRANSFER="${HF_HUB_ENABLE_HF_TRANSFER:-0}"

export TOKENIZERS_PARALLELISM=false
# The tiny_h3 package (sources under src/, import name kept via pyproject package-dir)
# is provided by `pip install -e .` (scripts/setup_env.sh).  tools/*.py and pytest also
# work from a bare checkout through their own bootstrap mapping.

# Uncomment to silence the "tensorflow/ flax not installed" noise from transformers.
# export TRANSFORMERS_NO_ADVISORY_WARNINGS=1

mkdir -p "$TINY_H3_DATA" "$TINY_H3_MODELS" "$TINY_H3_DATASETS" "$HF_HOME"
