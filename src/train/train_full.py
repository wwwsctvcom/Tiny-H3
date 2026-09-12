#!/usr/bin/env python
"""Full-parameter Tiny-H3 training on one GPU.

The only model is diffusers' ``MiniMaxH3Transformer3DModel`` with the
``tiny_h3_5090`` dimensions (see ``tiny_h3/model.py``); every parameter is trainable.

Quick start::

    source scripts/env.sh
    python -m tiny_h3.train.train_full \
      --latents $TINY_H3_DATA/synth/latents \
      --out runs/full --preset tiny_h3_5090

The defaults target a 32 GB RTX 5090: bf16, gradient checkpointing, batch 2 x accumulation 4.
Use ``--preset smoke --batch-size 1 --steps 20`` for a plumbing test.
"""

from __future__ import annotations

from .core import build_common_parser, config_from_args, run_standard_training


def main() -> None:
    parser = build_common_parser(__doc__)
    args = parser.parse_args()
    cfg = config_from_args(args)
    run_standard_training(cfg, mode="full")


if __name__ == "__main__":
    main()
