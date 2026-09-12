#!/usr/bin/env python
"""PyTorch FSDP training for Tiny-H3 (single GPU works; multi-GPU actually shards).

Launch::

    # Required acceptance path: one 5090.  FSDP wrapper/checkpoint plumbing is exercised;
    # world_size=1 uses NO_SHARD because there is nothing to distribute.
    torchrun --standalone --nproc_per_node=1 -m tiny_h3.train.train_fsdp \
      --latents $TINY_H3_DATA/synth/latents --out runs/fsdp

    # The same command scales without code changes:
    torchrun --standalone --nproc_per_node=4 -m tiny_h3.train.train_fsdp ...

FSDP auto-wraps each diffusers ``MiniMaxH3TransformerBlock`` / token-refiner block, keeps
optimizer state local, and gathers a full state dict on rank 0 for a normal diffusers-format
checkpoint.  Mixed precision is bf16 by default; gradient checkpointing is enabled before
wrapping so activations, not parameters, dominate far less memory.
"""

from __future__ import annotations

import contextlib
import json
import math
import os
import time
from dataclasses import asdict

import torch
import torch.distributed as dist

from ..data.dataset import build_loader
from ..model import ModelSpec, build_dit, count_parameters, save_checkpoint
from .core import (
    append_jsonl,
    build_common_parser,
    config_from_args,
    cosine_warmup,
    cycle,
    dtype_from_name,
    flow_matching_loss,
    move_batch,
    seed_everything,
)


def setup_distributed() -> tuple[int, int, int, torch.device]:
    world = int(os.environ.get("WORLD_SIZE", "1"))
    rank = int(os.environ.get("RANK", "0"))
    local_rank = int(os.environ.get("LOCAL_RANK", "0"))
    if not dist.is_initialized():
        # FSDP needs the default process group even at world_size=1 (NO_SHARD).  torchrun
        # exports the MASTER_* variables; plain `python -m` falls back to localhost.
        os.environ.setdefault("MASTER_ADDR", "127.0.0.1")
        os.environ.setdefault("MASTER_PORT", "29517")
        dist.init_process_group(backend="nccl" if torch.cuda.is_available() else "gloo")
    if torch.cuda.is_available():
        torch.cuda.set_device(local_rank)
        device = torch.device("cuda", local_rank)
    else:
        device = torch.device("cpu")
    return rank, world, local_rank, device


def cleanup() -> None:
    if dist.is_initialized():
        dist.barrier()
        dist.destroy_process_group()


def save_fsdp_checkpoint(model, out_dir: str, spec: ModelSpec, rank: int, world: int, step: int) -> None:
    """Gather one normal full state dict on rank 0; users never need an FSDP-specific loader."""
    from torch.distributed.fsdp import FullStateDictConfig, FullyShardedDataParallel as FSDP, StateDictType

    config = FullStateDictConfig(offload_to_cpu=True, rank0_only=True)
    with FSDP.state_dict_type(model, StateDictType.FULL_STATE_DICT, config):
        state = model.state_dict()
    if rank == 0:
        # Rebuild the unwrapped diffusers module and save in its native directory format.
        plain = build_dit(spec, dtype=torch.float32)
        missing, unexpected = plain.load_state_dict(state, strict=False)
        if missing or unexpected:
            raise RuntimeError(f"FSDP gather mismatch: missing={missing[:5]} unexpected={unexpected[:5]}")
        save_checkpoint(out_dir, plain, spec, step=step, extra={"mode": "fsdp", "world_size": world})
        del plain


def patch_diffusers_get_parameter_dtype() -> None:
    """Work around a diffusers bug that only FSDP reaches.

    ``diffusers.models.modeling_utils.get_parameter_dtype`` reuses the builtin name ``tuple``
    as a loop variable.  The buggy branch is dead code for plain modules (an earlier return
    always fires), but FSDP collects submodule parameters into the FlatParameter, so the H3
    AdaLN's ``self.linear`` has an empty parameter list there and evaluation reaches the
    nested ``def``, whose ``list[tuple[str, Tensor]]`` annotation then raises
    ``UnboundLocalError`` on Python 3.12.  Install a copy whose loop variable doesn't shadow
    the builtin, in both the defining module and the H3 transformer module that imported it.
    """
    import torch.nn as nn
    from torch import Tensor

    import diffusers.models.modeling_utils as modeling_utils
    import diffusers.models.transformers.transformer_minimax_h3 as transformer_h3

    def get_parameter_dtype(parameter) -> torch.dtype:
        last_dtype = None
        for name, param in parameter.named_parameters():
            last_dtype = param.dtype
            if (
                hasattr(parameter, "_keep_in_fp32_modules")
                and parameter._keep_in_fp32_modules
                and any(m in name for m in parameter._keep_in_fp32_modules)
            ):
                continue
            if param.is_floating_point():
                return param.dtype
        for buffer in parameter.buffers():
            last_dtype = buffer.dtype
            if buffer.is_floating_point():
                return buffer.dtype
        if last_dtype is not None:
            return last_dtype

        def find_tensor_attributes(module: nn.Module) -> list[tuple[str, Tensor]]:
            found = [(k, v) for k, v in module.__dict__.items() if torch.is_tensor(v)]
            return found

        gen = parameter._named_members(get_members_fn=find_tensor_attributes)
        last_item = None
        for item in gen:
            last_item = item
            if item[1].is_floating_point():
                return item[1].dtype
        if last_item is not None:
            return last_item[1].dtype
        for buffer in parameter.buffers():
            last_dtype = buffer.dtype
            if buffer.is_floating_point():
                return buffer.dtype
        if last_dtype is not None:
            return last_dtype
        for param in parameter.parameters():
            return param.dtype
        return torch.float32

    modeling_utils.get_parameter_dtype = get_parameter_dtype
    if hasattr(transformer_h3, "get_parameter_dtype"):
        transformer_h3.get_parameter_dtype = get_parameter_dtype


def main() -> None:
    parser = build_common_parser(__doc__)
    args = parser.parse_args()
    cfg = config_from_args(args)
    patch_diffusers_get_parameter_dtype()
    rank, world, local_rank, device = setup_distributed()
    seed_everything(cfg.seed, rank)
    dtype = dtype_from_name(cfg.precision) if device.type == "cuda" else torch.float32
    is_main = rank == 0

    try:
        loader, cache = build_loader(
            cfg.latents,
            batch_size=cfg.batch_size,
            num_workers=cfg.num_workers,
            max_samples=cfg.max_samples or None,
            dtype=torch.float32,
        )
        # DistributedSampler is intentionally explicit: every rank sees different cached clips.
        sampler = torch.utils.data.distributed.DistributedSampler(
            loader.dataset, num_replicas=world, rank=rank, shuffle=True, seed=cfg.seed
        ) if world > 1 else None
        if sampler is not None:
            loader = torch.utils.data.DataLoader(
                loader.dataset, batch_size=cfg.batch_size, sampler=sampler,
                num_workers=cfg.num_workers, collate_fn=loader.collate_fn,
                drop_last=True, persistent_workers=cfg.num_workers > 0,
            )
        layout = cache.layout()
        spec = ModelSpec(
            preset=cfg.preset,
            text_encoder=cache.text_encoder or None,
            text_dim=cache.text_dim,
            text_tokens=cache.text_tokens,
            text_layer=cache.text_layer,
        )
        model = build_dit(spec, dtype=torch.float32)
        if cfg.gradient_checkpointing:
            model.enable_gradient_checkpointing()

        from torch.distributed.fsdp import (
            FullyShardedDataParallel as FSDP,
            MixedPrecision,
            ShardingStrategy,
        )
        from torch.distributed.fsdp.wrap import transformer_auto_wrap_policy
        from diffusers.models.transformers.transformer_minimax_h3 import (
            MiniMaxH3TokenRefinerBlock,
            MiniMaxH3TransformerBlock,
        )
        from functools import partial

        auto_wrap = partial(
            transformer_auto_wrap_policy,
            transformer_layer_cls={MiniMaxH3TransformerBlock, MiniMaxH3TokenRefinerBlock},
        )
        mixed = MixedPrecision(param_dtype=dtype, reduce_dtype=dtype, buffer_dtype=dtype)
        strategy = ShardingStrategy.FULL_SHARD if world > 1 else ShardingStrategy.NO_SHARD
        model = FSDP(
            model,
            auto_wrap_policy=auto_wrap,
            mixed_precision=mixed,
            sharding_strategy=strategy,
            device_id=device if device.type == "cuda" else None,
            use_orig_params=True,
            limit_all_gathers=True,
        )
        trainable = [p for p in model.parameters() if p.requires_grad]
        optimizer = torch.optim.AdamW(trainable, lr=cfg.lr, betas=(0.9, 0.95), weight_decay=cfg.weight_decay)
        scheduler = torch.optim.lr_scheduler.LambdaLR(
            optimizer, lambda s: cosine_warmup(s, cfg.steps, cfg.warmup_steps)
        )
        if is_main:
            os.makedirs(cfg.out, exist_ok=True)
            with open(os.path.join(cfg.out, "train_config.json"), "w", encoding="utf-8") as f:
                json.dump({**asdict(cfg), "mode": "fsdp", "world_size": world,
                           "sharding": strategy.name, "parameters": count_parameters(model)}, f, indent=2)
            print(f"FSDP world={world} strategy={strategy.name} device={device} dtype={dtype} "
                  f"params={count_parameters(model)/1e6:.2f}M seq={layout.seq_len}")

        iterator = cycle(loader)
        optimizer.zero_grad(set_to_none=True)
        running = {"loss": 0.0, "video": 0.0, "audio": 0.0}
        started = time.time()
        for step in range(1, cfg.steps + 1):
            if sampler is not None and step % max(1, len(loader)) == 1:
                sampler.set_epoch(step // max(1, len(loader)))
            for micro in range(cfg.grad_accum):
                batch = move_batch(next(iterator), device, dtype)
                no_sync = model.no_sync() if micro < cfg.grad_accum - 1 else contextlib.nullcontext()
                with no_sync, torch.autocast(device_type=device.type, dtype=dtype,
                                             enabled=device.type == "cuda" and dtype != torch.float32):
                    out = flow_matching_loss(
                        model, layout, batch, audio_weight=cfg.audio_loss_weight,
                        video_shift=cfg.video_flow_shift, audio_shift=cfg.audio_flow_shift,
                        min_sigma=cfg.min_sigma, max_sigma=cfg.max_sigma,
                    )
                    loss = out.loss / cfg.grad_accum
                loss.backward()
                running["loss"] += float(out.loss.detach())
                running["video"] += float(out.video_loss.detach())
                running["audio"] += float(out.audio_loss.detach())

            grad_norm = model.clip_grad_norm_(cfg.max_grad_norm)
            optimizer.step()
            optimizer.zero_grad(set_to_none=True)
            scheduler.step()

            if is_main and (step == 1 or step % cfg.log_every == 0):
                denom = cfg.grad_accum if step == 1 else cfg.grad_accum * cfg.log_every
                row = {"step": step, "loss": running["loss"] / denom,
                       "video_loss": running["video"] / denom, "audio_loss": running["audio"] / denom,
                       "lr": scheduler.get_last_lr()[0], "grad_norm": float(grad_norm),
                       "seconds": time.time() - started,
                       "max_cuda_gb": torch.cuda.max_memory_allocated(device) / 2**30 if device.type == "cuda" else 0.0}
                print(" ".join(f"{k}={v:.5g}" if isinstance(v, float) else f"{k}={v}" for k, v in row.items()))
                append_jsonl(os.path.join(cfg.out, "metrics.jsonl"), row)
                running = {"loss": 0.0, "video": 0.0, "audio": 0.0}

            if step % cfg.save_every == 0:
                save_fsdp_checkpoint(model, os.path.join(cfg.out, f"step_{step:07d}"), spec, rank, world, step)

        save_fsdp_checkpoint(model, os.path.join(cfg.out, "final"), spec, rank, world, cfg.steps)
        if is_main:
            print(f"training complete -> {cfg.out}/final")
    finally:
        cleanup()


if __name__ == "__main__":
    main()
