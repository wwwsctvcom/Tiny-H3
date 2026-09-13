"""Shared flow-matching training core for Full, LoRA and FSDP.

All three modes call the same :func:`flow_matching_loss`; only model wrapping and which
parameters are trainable differ.  This keeps the educational comparison honest:

* **Full** -- every DiT parameter updates.
* **LoRA** -- PEFT adapters on H3 attention/FFN projections update.
* **FSDP** -- same full loss, transformer blocks wrapped/sharded by PyTorch FSDP2.

H3 convention (important): ``x_t = t*x0 + (1-t)*noise`` and ``v_target = x0-noise``.
Video and audio draw different sigmas and use different exponential shifts, but one packed
forward handles both by assigning a timestep index to every sequence row.
"""

from __future__ import annotations

import argparse
import contextlib
import json
import math
import os
import random
import time
from dataclasses import asdict, dataclass
from typing import Iterator

import numpy as np
import torch
import torch.nn.functional as F

from ..data.dataset import LatentCache, build_loader
from ..model import DEFAULT_TEXT_ENCODER, ModelSpec, build_dit, count_parameters, load_dit, save_checkpoint
from ..sampler import AUDIO_FLOW_SHIFT, VIDEO_FLOW_SHIFT, build_schedule, packed_velocity


@dataclass
class TrainConfig:
    latents: str
    out: str
    preset: str = "tiny_h3_5090"
    steps: int = 4000
    batch_size: int = 2
    grad_accum: int = 4
    lr: float = 1e-4
    weight_decay: float = 0.01
    warmup_steps: int = 200
    max_grad_norm: float = 1.0
    seed: int = 42
    device: str = "cuda"
    precision: str = "bf16"
    gradient_checkpointing: bool = True
    num_workers: int = 2
    log_every: int = 10
    save_every: int = 500
    audio_loss_weight: float = 1.0
    video_flow_shift: float = VIDEO_FLOW_SHIFT
    audio_flow_shift: float = AUDIO_FLOW_SHIFT
    min_sigma: float = 0.02
    max_sigma: float = 0.98
    compile: bool = False
    max_samples: int = 0
    init: str = ""
    batched: bool = True
    eval_every: int = 0
    eval_batches: int = 32
    patience: int = 0


def dtype_from_name(name: str) -> torch.dtype:
    return {"fp32": torch.float32, "fp16": torch.float16, "bf16": torch.bfloat16}[name]


def seed_everything(seed: int, rank: int = 0) -> None:
    seed = int(seed) + int(rank)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def shifted_sigma(batch: int, shift: float, device: torch.device, lo: float, hi: float) -> torch.Tensor:
    """Uniform base sigma pushed through H3's exponential shift and clipped away from endpoints."""
    base = torch.rand(batch, device=device, dtype=torch.float32)
    sigma = shift * base / (1.0 + (shift - 1.0) * base)
    return sigma.clamp(lo, hi)


def add_noise(x0: torch.Tensor, sigma: torch.Tensor, noise: torch.Tensor) -> torch.Tensor:
    """H3 forward process: ``x_t = (1-sigma)*x0 + sigma*noise``."""
    while sigma.ndim < x0.ndim:
        sigma = sigma.unsqueeze(-1)
    return (1.0 - sigma) * x0 + sigma * noise


@dataclass
class LossOutput:
    loss: torch.Tensor
    video_loss: torch.Tensor
    audio_loss: torch.Tensor
    sigma_video: torch.Tensor
    sigma_audio: torch.Tensor


def flow_matching_loss(
    model: torch.nn.Module,
    layout,
    batch: dict,
    *,
    audio_weight: float = 1.0,
    video_shift: float = VIDEO_FLOW_SHIFT,
    audio_shift: float = AUDIO_FLOW_SHIFT,
    min_sigma: float = 0.02,
    max_sigma: float = 0.98,
    batched: bool = True,
) -> LossOutput:
    """Joint H3 video+audio flow loss over one cached latent batch."""
    x0_video = batch["video_latents"]
    x0_audio = batch["audio_latents"]
    text = batch["text_embed"]
    bsz = x0_video.shape[0]
    device = x0_video.device

    if batched:
        # One sigma pair per micro-batch: the H3 transformer indexes its AdaLN modulation
        # per sequence row (not per batch item), so a batched forward must share timesteps
        # and therefore a shared sigma.  Sigma diversity across the optimizer step comes
        # from grad_accum independent micro-batches; the gradient expectation is unchanged.
        # packed_velocity already treats the batch axis as pure replication, so this is a
        # single big forward instead of bsz tiny ones.
        sigma_v = shifted_sigma(1, video_shift, device, min_sigma, max_sigma)[0]
        sigma_a = shifted_sigma(1, audio_shift, device, min_sigma, max_sigma)[0]
        noise_v = torch.randn_like(x0_video)
        noise_a = torch.randn_like(x0_audio)
        xt_video = add_noise(x0_video, sigma_v, noise_v)
        xt_audio = add_noise(x0_audio, sigma_a, noise_a)
        target_v = x0_video - noise_v
        target_a = x0_audio - noise_a
        pred_v, pred_a = packed_velocity(
            model, layout, text, xt_video, xt_audio, float(sigma_v), float(sigma_a),
        )
        sigma_v, sigma_a = sigma_v.unsqueeze(0), sigma_a.unsqueeze(0)
    else:
        # Per-sample sigma pairs (the original faithful path): one packed forward each.
        sigma_v = shifted_sigma(bsz, video_shift, device, min_sigma, max_sigma)
        sigma_a = shifted_sigma(bsz, audio_shift, device, min_sigma, max_sigma)
        noise_v = torch.randn_like(x0_video)
        noise_a = torch.randn_like(x0_audio)
        xt_video = add_noise(x0_video, sigma_v, noise_v)
        xt_audio = add_noise(x0_audio, sigma_a, noise_a)
        target_v = x0_video - noise_v
        target_a = x0_audio - noise_a
        pred_v, pred_a = [], []
        for i in range(bsz):
            pv, pa = packed_velocity(
                model,
                layout,
                text[i : i + 1],
                xt_video[i : i + 1],
                xt_audio[i : i + 1],
                float(sigma_v[i]),
                float(sigma_a[i]),
            )
            pred_v.append(pv)
            pred_a.append(pa)
        pred_v = torch.cat(pred_v)
        pred_a = torch.cat(pred_a)

    video_loss = F.mse_loss(pred_v.float(), target_v.float())
    audio_loss = F.mse_loss(pred_a.float(), target_a.float())
    return LossOutput(
        loss=video_loss + float(audio_weight) * audio_loss,
        video_loss=video_loss,
        audio_loss=audio_loss,
        sigma_video=sigma_v.mean(),
        sigma_audio=sigma_a.mean(),
    )


def cycle(loader) -> Iterator[dict]:
    while True:
        yield from loader


def cosine_warmup(step: int, total_steps: int, warmup: int) -> float:
    if step < warmup:
        return float(step + 1) / max(1, warmup)
    progress = (step - warmup) / max(1, total_steps - warmup)
    return 0.5 * (1.0 + math.cos(math.pi * min(1.0, progress)))


def move_batch(batch: dict, device: torch.device, dtype: torch.dtype) -> dict:
    return {
        **batch,
        "video_latents": batch["video_latents"].to(device=device, dtype=dtype, non_blocking=True),
        "audio_latents": batch["audio_latents"].to(device=device, dtype=dtype, non_blocking=True),
        "text_embed": batch["text_embed"].to(device=device, dtype=dtype, non_blocking=True),
    }


def unwrap_model(model: torch.nn.Module) -> torch.nn.Module:
    """Get the diffusers model below DDP / torch.compile wrappers."""
    while hasattr(model, "module"):
        model = model.module
    if hasattr(model, "_orig_mod"):
        model = model._orig_mod
    return model


def append_jsonl(path: str, obj: dict) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "a", encoding="utf-8") as f:
        f.write(json.dumps(obj, ensure_ascii=False) + "\n")


def build_common_parser(description: str) -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(description=description, formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    ap.add_argument("--latents", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--preset", default="stage4_xlarge",
                    help="built-in name (stage4_xlarge / smoke for CPU tests) or a model config JSON path, e.g. configs/stage4_xlarge.json")
    ap.add_argument("--steps", type=int, default=4000)
    ap.add_argument("--batch-size", type=int, default=2)
    ap.add_argument("--grad-accum", type=int, default=4)
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--weight-decay", type=float, default=0.01)
    ap.add_argument("--warmup-steps", type=int, default=200)
    ap.add_argument("--max-grad-norm", type=float, default=1.0)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--precision", choices=["fp32", "fp16", "bf16"], default="bf16")
    ap.add_argument("--no-gradient-checkpointing", action="store_true")
    ap.add_argument("--num-workers", type=int, default=2)
    ap.add_argument("--log-every", type=int, default=10)
    ap.add_argument("--save-every", type=int, default=500)
    ap.add_argument("--audio-loss-weight", type=float, default=1.0)
    ap.add_argument("--compile", action="store_true")
    ap.add_argument("--max-samples", type=int, default=0)
    ap.add_argument("--init", default="", help="initialize the DiT from a trained checkpoint dir (full mode)")
    ap.add_argument("--eval-every", type=int, default=0,
                    help="evaluate the val split every N steps (0 = off)")
    ap.add_argument("--eval-batches", type=int, default=32, help="val batches per evaluation (2 x 32 = 64 clips)")
    ap.add_argument("--patience", type=int, default=0,
                    help="early stop after N evaluations without val improvement (0 = off)")
    ap.add_argument("--no-batched", action="store_true",
                    help="per-sample forwards (numerically equivalent, far slower; A/B testing only)")
    return ap


def config_from_args(args) -> TrainConfig:
    values = vars(args).copy()
    values["gradient_checkpointing"] = not values.pop("no_gradient_checkpointing", False)
    values["batched"] = not values.pop("no_batched", False)
    # Mode-specific flags are not TrainConfig fields.
    allowed = TrainConfig.__dataclass_fields__
    return TrainConfig(**{k: v for k, v in values.items() if k in allowed})


def run_standard_training(
    cfg: TrainConfig,
    *,
    mode: str,
    model_builder=None,
    model_postprocess=None,
    save_adapter=None,
) -> None:
    """Single-process trainer used by Full and LoRA."""
    os.makedirs(cfg.out, exist_ok=True)
    device = torch.device(cfg.device if cfg.device != "cuda" or torch.cuda.is_available() else "cpu")
    dtype = dtype_from_name(cfg.precision) if device.type == "cuda" else torch.float32
    seed_everything(cfg.seed)

    loader, cache = build_loader(
        cfg.latents,
        batch_size=cfg.batch_size,
        num_workers=cfg.num_workers,
        max_samples=cfg.max_samples or None,
        dtype=torch.float32,
    )
    layout = cache.layout()
    spec = ModelSpec(
        preset=cfg.preset,
        text_encoder=cache.text_encoder or DEFAULT_TEXT_ENCODER,
        text_dim=cache.text_dim,
        text_tokens=cache.text_tokens,
        text_layer=cache.text_layer,
    )
    model = model_builder(spec, dtype) if model_builder else build_dit(spec, dtype=torch.float32)
    if cfg.init:
        if model_builder is not None:
            raise SystemExit("--init is only supported for full-parameter training (train_full/train_fsdp)")
        init_sd = load_dit(cfg.init, device="cpu", dtype=torch.float32)[0].state_dict()
        model.load_state_dict(init_sd)
        del init_sd
        print(f"init weights <- {cfg.init}")
    if cfg.gradient_checkpointing:
        model.enable_gradient_checkpointing()
    model = model.to(device=device, dtype=dtype)
    if model_postprocess:
        model = model_postprocess(model)
    if cfg.compile:
        model = torch.compile(model)

    trainable = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.AdamW(trainable, lr=cfg.lr, betas=(0.9, 0.95), eps=1e-8, weight_decay=cfg.weight_decay)
    scheduler = torch.optim.lr_scheduler.LambdaLR(
        optimizer, lambda s: cosine_warmup(s, cfg.steps, cfg.warmup_steps)
    )
    scaler = torch.amp.GradScaler("cuda", enabled=dtype == torch.float16 and device.type == "cuda")
    total, trainable_n = count_parameters(model), sum(p.numel() for p in trainable)
    print(f"mode={mode} device={device} dtype={dtype} params={total/1e6:.2f}M trainable={trainable_n/1e6:.2f}M")
    print(f"data={len(cache)} samples batch={cfg.batch_size} x accum={cfg.grad_accum}; packed seq={layout.seq_len}")
    with open(os.path.join(cfg.out, "train_config.json"), "w", encoding="utf-8") as f:
        json.dump({**asdict(cfg), "mode": mode, "parameters": total, "trainable_parameters": trainable_n}, f, indent=2)

    iterator = cycle(loader)
    model.train()
    optimizer.zero_grad(set_to_none=True)
    running = {"loss": 0.0, "video": 0.0, "audio": 0.0}
    started = time.time()

    # Early stopping: every cfg.eval_every steps, score the held-out val split with a
    # deterministic sigma stream.  The RNG state is saved/restored around each evaluation
    # so val scoring never perturbs the training randomness; stop when cfg.patience
    # consecutive evaluations fail to improve the best val loss.
    best_val, evals_without_improvement = float("inf"), 0
    val_loader = None
    if cfg.eval_every:
        val_loader, _ = build_loader(
            cfg.latents, splits=("val",), batch_size=2, shuffle=False, num_workers=0, dtype=torch.float32,
        )
        print(f"early stop: eval every {cfg.eval_every} steps on {len(val_loader.dataset)} val clips, "
              f"patience={cfg.patience or 'off'}")

    def evaluate_val(step: int) -> float:
        nonlocal best_val, evals_without_improvement
        cpu_state = torch.get_rng_state()
        cuda_states = torch.cuda.get_rng_state_all() if device.type == "cuda" else None
        seed_everything(1234)
        model.eval()
        total, count = 0.0, 0
        with torch.no_grad():
            for i, batch in enumerate(cycle(val_loader)):
                if i >= cfg.eval_batches:
                    break
                batch = move_batch(batch, device, torch.bfloat16)
                with torch.autocast(device_type=device.type, dtype=dtype,
                                    enabled=device.type == "cuda" and dtype != torch.float32):
                    out = flow_matching_loss(
                        model, layout, batch,
                        audio_weight=cfg.audio_loss_weight,
                        video_shift=cfg.video_flow_shift,
                        audio_shift=cfg.audio_flow_shift,
                        min_sigma=cfg.min_sigma, max_sigma=cfg.max_sigma,
                        batched=cfg.batched,
                    )
                total += float(out.loss.detach())
                count += 1
        torch.set_rng_state(cpu_state)
        if cuda_states is not None:
            torch.cuda.set_rng_state_all(cuda_states)
        model.train()
        val = total / max(1, count)
        if val < best_val - 1e-4:
            best_val, evals_without_improvement = val, 0
        else:
            evals_without_improvement += 1
        print(f"val_step={step} val_loss={val:.5f} best={best_val:.5f} "
              f"no_improve={evals_without_improvement}" + (f"/{cfg.patience}" if cfg.patience else ""))
        append_jsonl(os.path.join(cfg.out, "metrics.jsonl"),
                     {"step": step, "val_loss": val, "best_val": best_val})
        return val

    for step in range(1, cfg.steps + 1):
        for micro in range(cfg.grad_accum):
            batch = move_batch(next(iterator), device, dtype)
            sync_context = contextlib.nullcontext()
            with sync_context, torch.autocast(device_type=device.type, dtype=dtype, enabled=device.type == "cuda" and dtype != torch.float32):
                output = flow_matching_loss(
                    model, layout, batch,
                    audio_weight=cfg.audio_loss_weight,
                    video_shift=cfg.video_flow_shift,
                    audio_shift=cfg.audio_flow_shift,
                    min_sigma=cfg.min_sigma,
                    max_sigma=cfg.max_sigma,
                    batched=cfg.batched,
                )
                loss = output.loss / cfg.grad_accum
            scaler.scale(loss).backward()
            running["loss"] += float(output.loss.detach())
            running["video"] += float(output.video_loss.detach())
            running["audio"] += float(output.audio_loss.detach())

        scaler.unscale_(optimizer)
        grad_norm = torch.nn.utils.clip_grad_norm_(trainable, cfg.max_grad_norm)
        scaler.step(optimizer)
        scaler.update()
        optimizer.zero_grad(set_to_none=True)
        scheduler.step()

        if step % cfg.log_every == 0 or step == 1:
            denom = cfg.log_every * cfg.grad_accum if step > 1 else cfg.grad_accum
            row = {
                "step": step,
                "loss": running["loss"] / denom,
                "video_loss": running["video"] / denom,
                "audio_loss": running["audio"] / denom,
                "lr": scheduler.get_last_lr()[0],
                "grad_norm": float(grad_norm),
                "seconds": time.time() - started,
                "max_cuda_gb": torch.cuda.max_memory_allocated() / 2**30 if device.type == "cuda" else 0.0,
            }
            print(" ".join(f"{k}={v:.5g}" if isinstance(v, float) else f"{k}={v}" for k, v in row.items()))
            append_jsonl(os.path.join(cfg.out, "metrics.jsonl"), row)
            running = {"loss": 0.0, "video": 0.0, "audio": 0.0}

        if step % cfg.save_every == 0:
            ckpt = os.path.join(cfg.out, f"step_{step:07d}")
            if save_adapter:
                save_adapter(unwrap_model(model), ckpt, spec, step)
            else:
                save_checkpoint(ckpt, unwrap_model(model), spec, step=step, extra={"mode": mode})

        if cfg.eval_every and step % cfg.eval_every == 0:
            evaluate_val(step)
            if cfg.patience and evals_without_improvement >= cfg.patience:
                print(f"early stop at step {step}: {cfg.patience} evaluations without val improvement")
                break

    final = os.path.join(cfg.out, "final")
    if save_adapter:
        save_adapter(unwrap_model(model), final, spec, cfg.steps)
    else:
        save_checkpoint(final, unwrap_model(model), spec, step=cfg.steps, extra={"mode": mode})
    print(f"training complete -> {final}")
