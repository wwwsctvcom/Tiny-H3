#!/usr/bin/env python
"""Single-GPU Flow-GRPO for Tiny-H3 -- no serving engine, Ray or SGLang.

This is the complete algorithm in one readable local loop:

1. sample ``N`` audio-video trajectories for one prompt with the current policy and the
   reverse-SDE sampler from ``tiny_h3.sampler``;
2. decode the final latents with diffusers' official H3 video/audio VAEs;
3. compute offline rewards (prompt adherence + audio-visual synchronization + basic quality);
4. normalize rewards within the prompt group to GRPO advantages;
5. replay recorded ``(x_t, x_next, old_log_prob)`` transitions, recompute the new transition
   probability through the DiT, and optimize the PPO clipped objective;
6. optionally penalize departure from the rollout behaviour mean (a sampled trust-region KL).

The policy is a LoRA adapter on a Full checkpoint, matching miles-diffusion's H3 strategy but
keeping rollout in-process: on a single 5090 there is one DiT copy, the two frozen VAEs are
only used at the reward boundary, and train/rollout cannot silently disagree.

Example::

    python -m tiny_h3.train.train_flow_grpo \
      --base runs/full/final --out runs/grpo \
      --prompt-data data/synth/val.jsonl \
      --group-size 4 --rollouts 100
"""

from __future__ import annotations

import argparse
import json
import math
import os
import random
import time
from contextlib import nullcontext

import numpy as np
import torch

from ..media import AUDIO_SAMPLE_RATE
from ..model import TextConditioner, load_dit
from ..packing import build_t2va_layout
from ..rewards.prompt_match import prompt_match_reward
from ..rewards.sync import av_sync_reward
from ..sampler import (
    AUDIO_FLOW_SHIFT,
    VIDEO_FLOW_SHIFT,
    SDEConfig,
    packed_velocity,
    sample_av,
    sde_transition_stats,
    transition_log_prob,
)
from ..vae import decode_audio, decode_video, load_vaes, num_samples_for_frames, video_latent_frames
from .core import append_jsonl, dtype_from_name, seed_everything

LORA_TARGETS = ["attn.to_q", "attn.to_k", "attn.to_v", "attn.to_out.0", "ff.net.0.proj", "ff.net.2"]


def load_prompts(path: str) -> list[str]:
    if not os.path.exists(path):
        raise FileNotFoundError(path)
    prompts = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            row = json.loads(line)
            prompts.append(str(row.get("prompt", row.get("text", ""))).strip())
    prompts = [p for p in prompts if p]
    if not prompts:
        raise ValueError(f"no prompts found in {path}")
    return prompts


def decode_for_reward(vaes, video_latents, audio_latents, num_frames: int, fps: float):
    video = decode_video(vaes, video_latents.to(vaes.device))[0, :, :num_frames]
    wave = decode_audio(vaes, audio_latents[0].to(vaes.device))
    num_samples = num_samples_for_frames(fps, num_frames)
    frames = (video.permute(1, 2, 3, 0).clamp(0, 1) * 255).round().byte().cpu().numpy()
    return frames, wave[:, :num_samples].float().cpu().numpy()


def quality_reward(frames: np.ndarray, wave: np.ndarray) -> float:
    """Reject collapsed/invalid outputs; broad by design so it does not dictate aesthetics."""
    pixels = frames.astype(np.float32) / 255.0
    contrast = float(pixels.std())
    motion = float(np.abs(np.diff(pixels, axis=0)).mean()) if len(pixels) > 1 else 0.0
    peak = float(np.abs(wave).max()) if wave.size else 0.0
    rms = float(np.sqrt(np.mean(wave**2))) if wave.size else 0.0
    image_ok = np.clip(contrast / 0.18, 0, 1) * 0.5 + np.clip(motion / 0.03, 0, 1) * 0.5
    audio_ok = np.clip(rms / 0.08, 0, 1) * (1.0 if peak < 0.995 else 0.5)
    return float(0.7 * image_ok + 0.3 * audio_ok)


def reward(frames, wave, prompt, fps) -> tuple[float, dict]:
    prompt_score, prompt_parts = prompt_match_reward(frames, wave, fps, AUDIO_SAMPLE_RATE, prompt)
    sync_score = av_sync_reward(frames, wave, fps, AUDIO_SAMPLE_RATE)
    q_score = quality_reward(frames, wave)
    # Prompt fidelity is the task reward; sync and basic validity prevent the policy from
    # gaming it with static coloured frames or arbitrary loud noise.
    total = 0.55 * prompt_score + 0.30 * sync_score + 0.15 * q_score
    return float(total), {"prompt": prompt_score, "sync": sync_score, "quality": q_score, **prompt_parts}


def normalize_advantages(rewards: list[float], eps: float = 1e-4) -> torch.Tensor:
    x = torch.tensor(rewards, dtype=torch.float32)
    return (x - x.mean()) / (x.std(unbiased=False) + eps)


def grpo_transition_loss(
    model,
    layout,
    text,
    transition: dict,
    advantage: torch.Tensor,
    *,
    noise_level: float,
    clip_range: float,
    kl_beta: float,
    dtype: torch.dtype,
    device: torch.device,
) -> tuple[torch.Tensor, dict]:
    """PPO-clipped Flow-GRPO loss for one recorded joint video+audio transition."""
    v = transition["video"]
    a = transition["audio"]
    video_latent = v["latent"][None].to(device=device, dtype=dtype)
    audio_latent = a["latent"][None].to(device=device, dtype=dtype)
    next_video = v["next_latent"][None].to(device=device, dtype=torch.float32)
    next_audio = a["next_latent"][None].to(device=device, dtype=torch.float32)
    sv, svn = float(v["sigma"]), float(v["sigma_next"])
    sa, san = float(a["sigma"]), float(a["sigma_next"])

    velocity_v, velocity_a = packed_velocity(model, layout, text, video_latent, audio_latent, sv, sa)
    # sigma_max is the first non-unit value of the relevant schedule.  At the first step the
    # rollout stores sigma==1; use sigma_next as the finite denominator exactly as miles does.
    mean_v, std_v = sde_transition_stats(
        velocity_v, video_latent, sv, svn, sigma_max=svn if abs(sv - 1) < 1e-6 else sv, noise_level=noise_level
    )
    mean_a, std_a = sde_transition_stats(
        velocity_a, audio_latent, sa, san, sigma_max=san if abs(sa - 1) < 1e-6 else sa, noise_level=noise_level
    )
    log_new_v = transition_log_prob(next_video, mean_v, std_v)
    log_new_a = transition_log_prob(next_audio, mean_a, std_a)
    log_new = 0.5 * (log_new_v + log_new_a)
    log_old = 0.5 * (
        v["log_prob"].to(device=device, dtype=torch.float32).reshape_as(log_new)
        + a["log_prob"].to(device=device, dtype=torch.float32).reshape_as(log_new)
    )
    ratio = torch.exp((log_new - log_old).clamp(-20, 20))
    adv = advantage.to(device).reshape_as(ratio)
    unclipped = -adv * ratio
    clipped = -adv * torch.clamp(ratio, 1.0 - clip_range, 1.0 + clip_range)
    policy_loss = torch.maximum(unclipped, clipped).mean()

    # Trust region to the behaviour policy that produced this rollout.  This is cheap and
    # uses the stored transition means; LoRA's frozen base is also persisted for inference.
    kl = policy_loss.new_zeros(())
    if kl_beta > 0:
        old_mean_v = v["mean_old"][None].to(device=device, dtype=torch.float32)
        old_mean_a = a["mean_old"][None].to(device=device, dtype=torch.float32)
        kl_v = ((mean_v - old_mean_v) ** 2 / (2.0 * std_v**2)).flatten(1).mean()
        kl_a = ((mean_a - old_mean_a) ** 2 / (2.0 * std_a**2)).flatten(1).mean()
        kl = 0.5 * (kl_v + kl_a)
    loss = policy_loss + kl_beta * kl
    metrics = {
        "policy_loss": float(policy_loss.detach()), "kl": float(kl.detach()),
        "ratio": float(ratio.detach().mean()),
        "clipfrac": float((torch.abs(ratio.detach() - 1.0) > clip_range).float().mean()),
        "log_prob_delta": float((log_new.detach() - log_old).abs().mean()),
    }
    return loss, metrics


def save_grpo_adapter(model, out_dir: str, meta: dict) -> None:
    os.makedirs(out_dir, exist_ok=True)
    model.save_lora_adapter(out_dir)
    with open(os.path.join(out_dir, "tiny_h3_meta.json"), "w", encoding="utf-8") as f:
        json.dump(meta, f, ensure_ascii=False, indent=2)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    ap.add_argument("--base", required=True, help="Full/FSDP Tiny-H3 checkpoint")
    ap.add_argument("--out", required=True)
    ap.add_argument("--prompt-data", required=True)
    ap.add_argument("--h3-model", default="MiniMaxAI/MiniMax-H3")
    ap.add_argument("--rollouts", type=int, default=100)
    ap.add_argument("--group-size", type=int, default=4)
    ap.add_argument("--sample-steps", type=int, default=12)
    ap.add_argument("--train-epochs", type=int, default=2, help="passes over each rollout group's transitions")
    ap.add_argument("--transition-stride", type=int, default=2, help="train every Nth denoising transition")
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--rank", type=int, default=32)
    ap.add_argument("--lora-alpha", type=int, default=32)
    ap.add_argument("--clip-range", type=float, default=0.2)
    ap.add_argument("--kl-beta", type=float, default=0.01)
    ap.add_argument("--noise-level", type=float, default=0.7)
    ap.add_argument("--max-grad-norm", type=float, default=1.0)
    ap.add_argument("--height", type=int, default=256)
    ap.add_argument("--width", type=int, default=256)
    ap.add_argument("--frames", type=int, default=22)
    ap.add_argument("--fps", type=float, default=24.0)
    ap.add_argument("--save-every", type=int, default=10)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--vae-device", default="", help="defaults to device; cpu saves VRAM but is slow")
    ap.add_argument("--precision", choices=["fp32", "fp16", "bf16"], default="bf16")
    args = ap.parse_args()

    os.makedirs(args.out, exist_ok=True)
    seed_everything(args.seed)
    device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("Flow-GRPO requires the requested CUDA GPU, but CUDA is unavailable")
    dtype = dtype_from_name(args.precision) if device.type == "cuda" else torch.float32
    vae_device = torch.device(args.vae_device or args.device)

    model, spec = load_dit(args.base, device=device, dtype=dtype)
    from peft import LoraConfig

    model.add_adapter(LoraConfig(
        r=args.rank, lora_alpha=args.lora_alpha, target_modules=LORA_TARGETS,
        lora_dropout=0.0, bias="none", init_lora_weights="gaussian",
    ))
    model.enable_gradient_checkpointing()
    model.train()
    trainable = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.AdamW(trainable, lr=args.lr, betas=(0.9, 0.95), weight_decay=0.0)

    text_encoder = TextConditioner(spec.text_encoder, spec.text_tokens, device=device, dtype=dtype)
    vaes = load_vaes(args.h3_model, device=vae_device)
    prompts = load_prompts(args.prompt_data)
    latent_t = video_latent_frames(args.frames)
    latent_h, latent_w = args.height // 16, args.width // 16
    num_samples = num_samples_for_frames(args.fps, args.frames)
    audio_t = math.ceil(num_samples / 800)
    layout = build_t2va_layout(
        spec.text_tokens, audio_t, latent_t, latent_h, latent_w,
        patch_size=spec.patch_size, in_channels=spec.in_channels,
        audio_channels_dim=spec.audio_in_channels,
    )
    meta = {**spec.to_dict(), "mode": "lora", "algorithm": "flow_grpo", "base": os.path.abspath(args.base),
            "rank": args.rank, "lora_alpha": args.lora_alpha, "targets": LORA_TARGETS}
    with open(os.path.join(args.out, "grpo_config.json"), "w", encoding="utf-8") as f:
        json.dump(vars(args), f, ensure_ascii=False, indent=2)

    rng = random.Random(args.seed)
    started = time.time()
    for rollout_id in range(1, args.rollouts + 1):
        prompt = prompts[(rollout_id - 1) % len(prompts)]
        text = text_encoder.encode([prompt]).to(device=device, dtype=dtype)
        samples, rewards, reward_parts = [], [], []

        model.eval()
        for sample_id in range(args.group_size):
            result = sample_av(
                model, layout, text,
                video_shape=(1, spec.in_channels, latent_t, latent_h, latent_w),
                audio_shape=(1, 2, spec.audio_in_channels, audio_t),
                num_steps=args.sample_steps,
                seed=args.seed + rollout_id * 10_000 + sample_id,
                sde=SDEConfig(noise_level=args.noise_level, record=True),
                device=device, dtype=dtype,
            )
            frames, wave = decode_for_reward(vaes, result.video_latents, result.audio_latents, args.frames, args.fps)
            rew, parts = reward(frames, wave, prompt, args.fps)
            samples.append(result)
            rewards.append(rew)
            reward_parts.append(parts)
        advantages = normalize_advantages(rewards)

        model.train()
        order = [(s, t) for s in range(args.group_size)
                 for t in range(0, len(samples[s].trajectory), args.transition_stride)]
        metrics_acc = []
        for _epoch in range(args.train_epochs):
            rng.shuffle(order)
            for sample_id, trans_id in order:
                optimizer.zero_grad(set_to_none=True)
                with torch.autocast(device_type=device.type, dtype=dtype,
                                    enabled=device.type == "cuda" and dtype != torch.float32):
                    loss, metrics = grpo_transition_loss(
                        model, layout, text, samples[sample_id].trajectory[trans_id], advantages[sample_id],
                        noise_level=args.noise_level, clip_range=args.clip_range, kl_beta=args.kl_beta,
                        dtype=dtype, device=device,
                    )
                loss.backward()
                grad_norm = torch.nn.utils.clip_grad_norm_(trainable, args.max_grad_norm)
                optimizer.step()
                metrics["loss"] = float(loss.detach())
                metrics["grad_norm"] = float(grad_norm)
                metrics_acc.append(metrics)

        row = {
            "rollout": rollout_id, "prompt": prompt,
            "reward_mean": float(np.mean(rewards)), "reward_std": float(np.std(rewards)),
            "reward_min": float(np.min(rewards)), "reward_max": float(np.max(rewards)),
            "parts": {key: float(np.mean([p[key] for p in reward_parts if isinstance(p.get(key), (int, float))]))
                      for key in ("prompt", "sync", "quality")},
            "loss": float(np.mean([m["loss"] for m in metrics_acc])),
            "policy_loss": float(np.mean([m["policy_loss"] for m in metrics_acc])),
            "kl": float(np.mean([m["kl"] for m in metrics_acc])),
            "ratio": float(np.mean([m["ratio"] for m in metrics_acc])),
            "clipfrac": float(np.mean([m["clipfrac"] for m in metrics_acc])),
            "seconds": time.time() - started,
            "max_cuda_gb": torch.cuda.max_memory_allocated(device) / 2**30 if device.type == "cuda" else 0.0,
        }
        print(" ".join(f"{k}={v:.5g}" if isinstance(v, float) else f"{k}={v}" for k, v in row.items() if k != "parts"))
        append_jsonl(os.path.join(args.out, "metrics.jsonl"), row)

        if rollout_id % args.save_every == 0:
            save_grpo_adapter(model, os.path.join(args.out, f"rollout_{rollout_id:06d}"), {**meta, "step": rollout_id})

    save_grpo_adapter(model, os.path.join(args.out, "final"), {**meta, "step": args.rollouts})
    print(f"Flow-GRPO complete -> {args.out}/final")


if __name__ == "__main__":
    main()
