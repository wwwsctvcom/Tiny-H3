#!/usr/bin/env python
"""Standalone inference: one prompt -> one MP4 with synchronized sound.

Every stage of the Tiny-H3 / MiniMax-H3 inference pipeline, in order, so the
whole flow can be read top to bottom:

    1. load the trained DiT checkpoint
    2. build the packed-sequence layout (text | audio | video rows) for the target grid
    3. encode the prompt with the frozen text encoder
    4. start from gaussian noise shaped like video + audio latents
    5. run N Euler steps through the DiT (H3 predicts v = x0 - noise)
    6. decode the denoised latents with the official frozen H3 VAEs
    7. mux frames + stereo waveform into a playable MP4

Usage::

    python tools/inference.py --checkpoint runs/full/final \
        --prompt "a red circle bouncing on a dark background, with rhythmic thumps" \
        --out outputs/inference

Repeat ``--prompt`` to render several clips in one run.
"""

from __future__ import annotations

import argparse
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import _bootstrap  # noqa: E402  (maps the tiny_h3 package name onto ../src)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    ap.add_argument("--checkpoint", required=True, help="trained checkpoint dir (full/FSDP/LoRA/GRPO)")
    ap.add_argument("--prompt", action="append", required=True, help="text prompt; repeat for multiple clips")
    ap.add_argument("--out", default="outputs/inference", help="output directory")
    ap.add_argument("--steps", type=int, default=24, help="denoising steps")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--size", type=int, default=256, help="square canvas size, multiple of 32")
    ap.add_argument("--frames", type=int, default=22, help="pixel frames, on the 17n+5 grid")
    ap.add_argument("--fps", type=float, default=24.0)
    args = ap.parse_args()

    import numpy as np
    import torch

    from tiny_h3.media import write_mp4
    from tiny_h3.model import load_dit
    from tiny_h3.packing import build_t2va_layout
    from tiny_h3.sampler import sample_av
    from tiny_h3.vae import (
        audio_latent_frames,
        decode_audio,
        decode_video,
        load_vaes,
        num_samples_for_frames,
        video_latent_frames,
    )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dtype = torch.bfloat16 if device.type == "cuda" else torch.float32
    os.makedirs(args.out, exist_ok=True)

    # 1. the trained DiT (63.7M by default)
    print(f"[1/5] loading DiT from {args.checkpoint}")
    model, spec = load_dit(args.checkpoint, device=device, dtype=dtype)

    # 2. packed-sequence layout for the target grid
    latent_t = video_latent_frames(args.frames)
    latent_h, latent_w = args.size // 16, args.size // 16
    audio_t = audio_latent_frames(num_samples_for_frames(args.fps, args.frames))
    layout = build_t2va_layout(
        num_text_tokens=spec.text_tokens,
        num_audio_latents=audio_t,
        latent_frames=latent_t,
        latent_height=latent_h,
        latent_width=latent_w,
        patch_size=spec.patch_size,
        in_channels=spec.in_channels,
        audio_channels_dim=spec.audio_in_channels,
    )
    print(f"[2/5] grid {args.size}x{args.size} @ {args.fps} fps, {args.frames} frames -> "
          f"packed sequence of {layout.seq_len} rows")

    # 3. frozen text encoder (Qwen3-0.6B)
    from tiny_h3.model import TextConditioner

    conditioner = TextConditioner(spec.text_encoder, spec.text_tokens, device=device, text_layer=spec.text_layer)
    print(f"[3/5] text encoder: {spec.text_encoder} ({spec.text_dim}-dim x {spec.text_tokens} tokens)")

    # 4. flow: gaussian noise -> 24 Euler steps through the packed DiT
    vaes = load_vaes("MiniMaxAI/MiniMax-H3", device=device)
    print(f"[4/5] denoising {args.steps} steps (seed={args.seed})")
    for index, prompt in enumerate(args.prompt):
        text = conditioner.encode([prompt]).to(device, dtype)
        result = sample_av(
            model,
            layout,
            text,
            video_shape=(1, spec.in_channels, latent_t, latent_h, latent_w),
            audio_shape=(1, 2, spec.audio_in_channels, audio_t),
            num_steps=args.steps,
            seed=args.seed + index,
            device=device,
            dtype=dtype,
        )

        # 5. decode with the official VAEs and mux to MP4
        video = decode_video(vaes, result.video_latents)[0]
        wave = decode_audio(vaes, result.audio_latents[0])
        frames = (video[:, : args.frames].permute(1, 2, 3, 0).clamp(0, 1) * 255).round().byte().cpu().numpy()
        num_samples = num_samples_for_frames(args.fps, args.frames)
        path = os.path.join(args.out, f"{index:02d}_" + "_".join(prompt.split())[:48] + ".mp4")
        write_mp4(path, frames, fps=args.fps, wave=wave[:, :num_samples].float().cpu().numpy(),
                  sample_rate=32000)
        print(f"      saved {path}")
    print("[5/5] done")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
