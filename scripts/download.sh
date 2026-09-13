#!/usr/bin/env bash
# Download every asset Tiny-H3 needs (see tools/download_assets.py for the details).
#
#   bash scripts/download.sh                 # models + data
#   bash scripts/download.sh --group model    # H3 configs + VAEs only
#   bash scripts/download.sh --group data --num-concert-clips 60
set -euo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "$HERE/env.sh"
exec python "$TINY_H3_ROOT/tools/download_assets.py" "$@"
