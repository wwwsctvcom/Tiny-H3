#!/usr/bin/env python
"""OPTIONAL: fine-tune the official H3 autoencoders on your own domain.

**Recommendation: don't, unless you have a reason.**  The released VAEs were trained on
large-scale real video/audio; they are the reason a 47 M DiT can produce a *real* mp4 at all.
Retraining them on a few hundred clips (or even a few thousand) lowers the quality ceiling,
and any change to them invalidates every existing latent cache and disentangles your DiT from
the official latent space.

When it *does* make sense: your data is far from natural video (unusual sensors, extreme
colour spaces, medical/depth, a very different audio bandwidth) and you can see the frozen VAE
blurring or hallucinating on reconstructions.  Verify with ``tools/reconstruct_data.py`` first.

Design used here (the only version that fits one 5090):

* the **encoder stays frozen** -- latents are computed once and cached;
* only the **decoder is trained**, in bf16 with gradient checkpointing;
* loss = L1 + 0.1 * (1 - SSIM) on pixels; the audio VAE uses L1 on the waveform;
* afterwards you must re-encode with ``tools/prepare_latents.py`` and re-train the DiT.

Example::

    python tools/finetune_vae.py --data-dir data/synth --out vae_ft --target video_decoder --steps 300
    python tools/finetune_vae.py --data-dir data/synth --out vae_ft --target audio --steps 300
"""

from __future__ import annotations

import argparse
import json
import os
import sys

import numpy as np

from _bootstrap import bootstrap  # noqa: E402  (maps the tiny_h3 package name onto ../src)

from tiny_h3.media import read_video_frames, read_wav  # noqa: E402


def ssim_like(a, b, eps: float = 1e-6):
    """Cheap global SSIM proxy (per-frame, channel-averaged); avoids adding a dependency."""
    import torch

    mu_a, mu_b = a.mean(dim=(1, 2, 3), keepdim=True), b.mean(dim=(1, 2, 3), keepdim=True)
    va = a.var(dim=(1, 2, 3), keepdim=True, unbiased=False)
    vb = b.var(dim=(1, 2, 3), keepdim=True, unbiased=False)
    cab = ((a - mu_a) * (b - mu_b)).mean(dim=(1, 2, 3), keepdim=True)
    c1, c2 = 0.01**2, 0.03**2
    score = ((2 * mu_a * mu_b + c1) * (2 * cab + c2)) / ((mu_a**2 + mu_b**2 + c1) * (va + vb + c2 + eps))
    return float(score.mean())


def iter_clips(data_dir: str, grid: dict, limit: int = 0):
    records = []
    with open(os.path.join(data_dir, "train.jsonl"), encoding="utf-8") as f:
        for line in f:
            records.append(json.loads(line))
    if limit:
        records = records[:limit]
    for rec in records:
        frames = read_video_frames(
            os.path.join(data_dir, rec["metadata"]["video"]),
            (grid["size"], grid["size"]), grid["fps"], max_frames=grid["frames"],
        )
        wave = read_wav(os.path.join(data_dir, rec["metadata"]["audio"]), sample_rate=grid["sample_rate"])
        yield rec, frames, wave


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--data-dir", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--target", choices=["video_decoder", "audio"], default="video_decoder")
    ap.add_argument("--h3-model", default="MiniMaxAI/MiniMax-H3")
    ap.add_argument("--steps", type=int, default=300)
    ap.add_argument("--lr", type=float, default=2e-5)
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--log-every", type=int, default=20)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    import torch

    from tiny_h3.vae import PIXEL_MEAN, PIXEL_STD, load_vaes

    torch.manual_seed(args.seed)
    with open(os.path.join(args.data_dir, "meta.json"), encoding="utf-8") as f:
        grid = json.load(f)
    bundle = load_vaes(args.h3_model, device=args.device)
    os.makedirs(args.out, exist_ok=True)

    if args.target == "video_decoder":
        vae = bundle.video
        vae.requires_grad_(False)
        vae.decoder.requires_grad_(True)
        vae.decoder.train()
        optimizer = torch.optim.AdamW([p for p in vae.decoder.parameters() if p.requires_grad], lr=args.lr)
        mean = torch.tensor(PIXEL_MEAN, device=args.device).view(1, -1, 1, 1, 1)
        std = torch.tensor(PIXEL_STD, device=args.device).view(1, -1, 1, 1, 1)
        running = 0.0
        for step, (_rec, frames, _wave) in enumerate(iter_clips(args.data_dir, grid, args.limit), start=1):
            pixels = torch.from_numpy(frames).permute(3, 0, 1, 2)[None].float().div(255).to(args.device)
            pixels = (pixels - mean) / std
            with torch.no_grad():
                posterior = vae.encode(pixels, return_dict=False)[0]
                latents = posterior.mode()  # frozen-encoder latents
            decoded = vae.decode(latents, return_dict=False)[0]
            loss = (decoded - pixels).abs().mean() + 0.1 * (1.0 - ssim_like(decoded, pixels))
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_([p for p in vae.decoder.parameters() if p.requires_grad], 1.0)
            optimizer.step()
            running += float(loss.detach())
            if step % args.log_every == 0 or step == 1:
                print(f"step {step:5d} loss={running / (step if step == 1 else args.log_every):.5f} "
                      f"max_cuda_gb={torch.cuda.max_memory_allocated()/2**30:.1f}" if torch.cuda.is_available() else "", flush=True)
                if step > 1:
                    running = 0.0
            if step >= args.steps:
                break
        vae.decoder.eval()
    else:
        vae = bundle.audio
        vae.requires_grad_(True)
        optimizer = torch.optim.AdamW(vae.parameters(), lr=args.lr)
        running = 0.0
        for step, (_rec, _frames, wave) in enumerate(iter_clips(args.data_dir, grid, args.limit), start=1):
            wave_t = torch.from_numpy(wave).to(args.device)  # (2, N) -> two mono batch items
            decoded, posterior = vae(wave_t[:, None, :], sample_posterior=True, return_dict=False)
            recon = (decoded.squeeze(1) - wave_t).abs().mean()
            loss = recon + 1e-4 * posterior.kl().mean()
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(vae.parameters(), 1.0)
            optimizer.step()
            running += float(loss.detach())
            if step % args.log_every == 0 or step == 1:
                print(f"step {step:5d} loss={running / (step if step == 1 else args.log_every):.5f}", flush=True)
                if step > 1:
                    running = 0.0
            if step >= args.steps:
                break
        vae.eval()

    # Save in the same layout diffusers loads (subfolder vae/ or audio_vae/).
    sub = "vae" if args.target == "video_decoder" else "audio_vae"
    out_dir = os.path.join(args.out, sub)
    vae.save_pretrained(out_dir)
    with open(os.path.join(args.out, "finetune_meta.json"), "w", encoding="utf-8") as f:
        json.dump({"target": args.target, "steps": args.steps, "lr": args.lr,
                   "data_dir": os.path.abspath(args.data_dir), "base": args.h3_model}, f, indent=2)
    print(f"saved fine-tuned {sub} -> {out_dir}")
    print("reminder: re-run tools/prepare_latents.py and re-train the DiT; the official latent "
          "space no longer applies once the VAE changes.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
