#!/usr/bin/env python
"""PEFT LoRA training for Tiny-H3.

Targets match the H3 recipe in miles-diffusion: packed self-attention Q/K/V/output and the
SwiGLU feed-forward projections.  The frozen random Tiny-H3 base is saved next to the adapter,
so inference never depends on an implicit seed or external checkpoint.

Quick start::

    python -m tiny_h3.train.train_lora \
      --latents $TINY_H3_DATA/synth/latents --out runs/lora \
      --rank 32 --lora-alpha 32 --lr 3e-4

This is primarily an educational LoRA path: for a model trained from scratch, Full is the
quality baseline; LoRA shines when continuing from a Full checkpoint (``--base``).
"""

from __future__ import annotations

import json
import os

import torch

from ..model import ModelSpec, build_dit, save_checkpoint
from .core import build_common_parser, config_from_args, run_standard_training

LORA_TARGETS = [
    "attn.to_q",
    "attn.to_k",
    "attn.to_v",
    "attn.to_out.0",
    "ff.net.0.proj",
    "ff.net.2",
]


def main() -> None:
    parser = build_common_parser(__doc__)
    parser.set_defaults(lr=3e-4)
    parser.add_argument("--base", default="", help="Full Tiny-H3 checkpoint to continue from (default: fresh deterministic base)")
    parser.add_argument("--rank", type=int, default=32)
    parser.add_argument("--lora-alpha", type=int, default=32)
    parser.add_argument("--lora-dropout", type=float, default=0.0)
    args = parser.parse_args()
    cfg = config_from_args(args)

    base_path = args.base
    rank = args.rank
    alpha = args.lora_alpha
    dropout = args.lora_dropout

    def model_builder(spec: ModelSpec, dtype: torch.dtype):
        if base_path:
            from diffusers import MiniMaxH3Transformer3DModel

            model = MiniMaxH3Transformer3DModel.from_pretrained(base_path, torch_dtype=torch.float32)
        else:
            # Make the frozen base reproducible.  Adapters alone are meaningless unless every
            # user starts from these exact base weights, so save_checkpoint() below persists it.
            torch.manual_seed(cfg.seed)
            model = build_dit(spec, dtype=torch.float32)
        return model

    def add_adapter(model):
        from peft import LoraConfig

        config = LoraConfig(
            r=rank,
            lora_alpha=alpha,
            lora_dropout=dropout,
            target_modules=LORA_TARGETS,
            init_lora_weights="gaussian",
            bias="none",
        )
        # MiniMaxH3Transformer3DModel inherits diffusers' PeftAdapterMixin.
        model.add_adapter(config)
        return model

    def save_adapter(model, out_dir: str, spec: ModelSpec, step: int) -> None:
        os.makedirs(out_dir, exist_ok=True)
        # Adapter-only files (adapter_model.safetensors + adapter_config.json).
        model.save_lora_adapter(out_dir)
        # Also persist the base once at the run root; if --base was supplied, record its path.
        base_dir = os.path.join(cfg.out, "base")
        if not base_path and not os.path.exists(os.path.join(base_dir, "config.json")):
            # Reconstruct the identical seed-created base and save it as a normal diffusers
            # checkpoint, so the adapter stays usable without the training run directory.
            torch.manual_seed(cfg.seed)
            base_model = build_dit(spec, dtype=torch.float32)
            save_checkpoint(base_dir, base_model, spec, step=0, extra={"mode": "lora_base", "seed": cfg.seed})
            del base_model
        with open(os.path.join(out_dir, "tiny_h3_meta.json"), "w", encoding="utf-8") as f:
            json.dump({
                **spec.to_dict(), "step": step, "mode": "lora", "base": base_path or os.path.relpath(base_dir, out_dir),
                "rank": rank, "lora_alpha": alpha, "targets": LORA_TARGETS,
            }, f, indent=2)

    run_standard_training(
        cfg,
        mode="lora",
        model_builder=model_builder,
        model_postprocess=add_adapter,
        save_adapter=save_adapter,
    )


if __name__ == "__main__":
    main()
