"""Tiny-H3 text-to-audio-video inference pipeline and CLI.

This is intentionally a small, readable orchestration layer; all heavy model classes are from
``diffusers``:

* ``MiniMaxH3Transformer3DModel`` -- the trained tiny joint DiT;
* ``AutoencoderKLMiniMaxH3`` -- official real-video decoder;
* ``AutoencoderKLMiniMaxH3Audio`` -- official 32 kHz audio decoder.

Example::

    python -m tiny_h3.pipeline --checkpoint runs/full/final \
      --prompts "a red circle bouncing on a dark background, with rhythmic thumps" \
      --out outputs/demo
"""

from __future__ import annotations

import argparse
import json
import os
import re

import numpy as np
import torch

from .media import write_mp4
from .model import TextConditioner, load_dit
from .packing import build_t2va_layout
from .sampler import sample_av
from .vae import (
    AUDIO_LATENTS_PER_SECOND,
    decode_audio,
    decode_video,
    load_vaes,
    num_samples_for_frames,
    video_latent_frames,
)


def _slug(text: str, length: int = 48) -> str:
    value = re.sub(r"[^a-zA-Z0-9_-]+", "_", text.strip()).strip("_").lower()
    return value[:length] or "sample"


def _load_model(checkpoint: str, device: torch.device, dtype: torch.dtype):
    """Load Full/FSDP output or a LoRA adapter directory."""
    meta_path = os.path.join(checkpoint, "tiny_h3_meta.json")
    with open(meta_path, encoding="utf-8") as f:
        meta = json.load(f)
    if meta.get("mode") != "lora":
        model, spec = load_dit(checkpoint, device=device, dtype=dtype)
        return model, spec

    base = meta["base"]
    if not os.path.isabs(base):
        base = os.path.normpath(os.path.join(checkpoint, base))
    model, spec = load_dit(base, device=device, dtype=dtype)
    # save_adapter writes the diffusers-convention safetensors name; without an explicit
    # weight_name diffusers probes for pytorch_lora_weights.bin first and raises on this dir.
    model.load_lora_adapter(checkpoint, weight_name="pytorch_lora_weights.safetensors", prefix=None)
    model.set_adapters(["default"])
    return model, spec


class TinyH3Pipeline:
    """A minimal pipeline with the same component split as diffusers' full H3 pipeline."""

    def __init__(
        self,
        checkpoint: str,
        h3_model: str = "MiniMaxAI/MiniMax-H3",
        device: str | torch.device = "cuda",
        dtype: torch.dtype = torch.bfloat16,
        vae_device: str | torch.device | None = None,
    ) -> None:
        self.device = torch.device(device)
        if self.device.type == "cuda" and not torch.cuda.is_available():
            raise RuntimeError("CUDA was requested but is unavailable. Run on a GPU instance or pass --device cpu.")
        self.dtype = dtype if self.device.type == "cuda" else torch.float32
        self.dit, self.spec = _load_model(checkpoint, self.device, self.dtype)
        self.dit.eval()
        self.text = TextConditioner(
            self.spec.text_encoder, self.spec.text_tokens, device=self.device,
            dtype=self.dtype, text_layer=self.spec.text_layer,
        )
        self.vae_device = torch.device(vae_device or device)
        self.vaes = load_vaes(h3_model, device=self.vae_device)

    @torch.no_grad()
    def __call__(
        self,
        prompt: str,
        *,
        height: int = 256,
        width: int = 256,
        num_frames: int = 22,
        fps: float = 24.0,
        num_steps: int = 24,
        seed: int = 0,
    ) -> tuple[np.ndarray, np.ndarray]:
        """Generate ``(uint8 frames [T,H,W,3], float32 stereo wave [2,N])``."""
        if height % 32 or width % 32:
            raise ValueError(f"height/width must be multiples of 32, got {height}x{width}")
        latent_frames = video_latent_frames(num_frames)
        latent_h, latent_w = height // 16, width // 16
        num_samples = num_samples_for_frames(fps, num_frames)
        audio_frames = int(np.ceil(num_samples / 800))

        layout = build_t2va_layout(
            num_text_tokens=self.spec.text_tokens,
            num_audio_latents=audio_frames,
            latent_frames=latent_frames,
            latent_height=latent_h,
            latent_width=latent_w,
            patch_size=self.spec.patch_size,
            in_channels=self.spec.in_channels,
            audio_channels_dim=self.spec.audio_in_channels,
        )
        text = self.text.encode([prompt]).to(self.device, self.dtype)
        result = sample_av(
            self.dit,
            layout,
            text,
            video_shape=(1, self.spec.in_channels, latent_frames, latent_h, latent_w),
            audio_shape=(1, 2, self.spec.audio_in_channels, audio_frames),
            num_steps=num_steps,
            seed=seed,
            device=self.device,
            dtype=self.dtype,
            progress=True,
        )

        # Official VAEs may live on another device; moving only latents keeps the DiT resident.
        video = decode_video(self.vaes, result.video_latents.to(self.vae_device))[0]
        wave = decode_audio(self.vaes, result.audio_latents[0].to(self.vae_device))
        frames = (video[:, :num_frames].permute(1, 2, 3, 0).clamp(0, 1) * 255).round().byte().cpu().numpy()
        wave = wave[:, :num_samples].float().cpu().numpy()
        return frames, wave

    def save(
        self,
        prompt: str,
        path: str,
        *,
        fps: float = 24.0,
        **generate_kwargs,
    ) -> str:
        frames, wave = self(prompt, fps=fps, **generate_kwargs)
        os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
        write_mp4(path, frames, fps=fps, wave=wave, sample_rate=32000)
        return path


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    ap.add_argument("--checkpoint", required=True)
    ap.add_argument("--prompts", nargs="+", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--h3-model", default="MiniMaxAI/MiniMax-H3")
    ap.add_argument("--height", type=int, default=256)
    ap.add_argument("--width", type=int, default=256)
    ap.add_argument("--frames", type=int, default=22)
    ap.add_argument("--fps", type=float, default=24.0)
    ap.add_argument("--steps", type=int, default=24)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--vae-device", default="", help="defaults to --device; use cpu to save VRAM (much slower)")
    ap.add_argument("--precision", choices=["fp32", "fp16", "bf16"], default="bf16")
    args = ap.parse_args()

    dtype = {"fp32": torch.float32, "fp16": torch.float16, "bf16": torch.bfloat16}[args.precision]
    pipe = TinyH3Pipeline(
        args.checkpoint,
        h3_model=args.h3_model,
        device=args.device,
        dtype=dtype,
        vae_device=args.vae_device or None,
    )
    os.makedirs(args.out, exist_ok=True)
    manifest = []
    for index, prompt in enumerate(args.prompts):
        path = os.path.join(args.out, f"{index:02d}_{_slug(prompt)}.mp4")
        pipe.save(
            prompt,
            path,
            height=args.height,
            width=args.width,
            num_frames=args.frames,
            fps=args.fps,
            num_steps=args.steps,
            seed=args.seed + index,
        )
        manifest.append({"prompt": prompt, "file": os.path.basename(path), "seed": args.seed + index})
        print(f"saved {path}")
    with open(os.path.join(args.out, "manifest.json"), "w", encoding="utf-8") as f:
        json.dump(manifest, f, ensure_ascii=False, indent=2)


if __name__ == "__main__":
    main()
