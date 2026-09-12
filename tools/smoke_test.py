#!/usr/bin/env python
"""One-shot plumbing test: pack -> train a few steps -> sample -> decode -> mp4.

Run this first on any new machine.  With ``--preset smoke`` and no H3 weights it only needs a
few hundred MB of VRAM and validates every code path except the official VAE decode:

    python tools/smoke_test.py --latents $TINY_H3_DATA/synth/latents

Add the VAEs (download ~11 GB once) to also decode a real mp4:

    python tools/smoke_test.py --latents $TINY_H3_DATA/synth/latents --with-vae
"""

from __future__ import annotations

import argparse
import os
import sys

from _bootstrap import bootstrap  # noqa: E402  (maps the tiny_h3 package name onto ../src)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--latents", required=True, help="latent cache directory (tools/prepare_latents.py output)")
    ap.add_argument("--preset", default="smoke")
    ap.add_argument("--steps", type=int, default=20)
    ap.add_argument("--sample-steps", type=int, default=4)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--out", default="outputs/smoke")
    ap.add_argument("--with-vae", action="store_true", help="also decode through the official H3 VAEs")
    ap.add_argument("--h3-model", default="MiniMaxAI/MiniMax-H3")
    args = ap.parse_args()

    import torch

    from tiny_h3.data.dataset import build_loader
    from tiny_h3.model import ModelSpec, build_dit, count_parameters
    from tiny_h3.sampler import sample_av
    from tiny_h3.train.core import flow_matching_loss, move_batch, seed_everything

    device = torch.device(args.device if args.device != "cuda" or torch.cuda.is_available() else "cpu")
    dtype = torch.float32 if device.type == "cpu" else torch.bfloat16
    seed_everything(0)
    print(f"[1/5] device={device} dtype={dtype}")

    loader, cache = build_loader(args.latents, batch_size=2, num_workers=0)
    layout = cache.layout()
    spec = ModelSpec(preset=args.preset, text_tokens=cache.text_tokens)
    model = build_dit(spec, dtype=torch.float32).to(device=device, dtype=dtype)
    print(f"[2/5] dataset={len(cache)} packed_seq={layout.seq_len} params={count_parameters(model)/1e6:.2f}M")

    opt = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=1e-4)
    it = iter(loader)
    for step in range(1, args.steps + 1):
        try:
            batch = next(it)
        except StopIteration:
            it = iter(loader)
            batch = next(it)
        batch = move_batch(batch, device, dtype)
        out = flow_matching_loss(model, layout, batch)
        opt.zero_grad(set_to_none=True)
        out.loss.backward()
        opt.step()
        if step % max(1, args.steps // 5) == 0 or step == 1:
            print(f"      step {step:4d} loss={float(out.loss):.4f} video={float(out.video_loss):.4f} "
                  f"audio={float(out.audio_loss):.4f}")
    print("[3/5] training ok")

    prompt = cache.records[0]["prompt"]
    from tiny_h3.model import TextConditioner

    text = TextConditioner(spec.text_encoder, spec.text_tokens, device=device).encode([prompt]).to(device, dtype)
    result = sample_av(
        model, layout, text,
        # sample_av works on batched latents: (1,) + the per-sample (C, T, H, W) layout shapes.
        video_shape=(1, *layout.video_shape), audio_shape=(1, *layout.audio_shape),
        num_steps=args.sample_steps, device=device, dtype=dtype, progress=True,
    )
    print(f"[4/5] sampling ok -> video{tuple(result.video_latents.shape)} audio{tuple(result.audio_latents.shape)}")

    os.makedirs(args.out, exist_ok=True)
    if args.with_vae:
        from tiny_h3.media import AUDIO_SAMPLE_RATE, write_mp4
        from tiny_h3.vae import decode_audio, decode_video, load_vaes

        vaes = load_vaes(args.h3_model, device=device)
        video = decode_video(vaes, result.video_latents)[0]
        wave = decode_audio(vaes, result.audio_latents[0])
        frames = (video.permute(1, 2, 3, 0).clamp(0, 1) * 255).round().byte().cpu().numpy()
        path = os.path.join(args.out, "smoke.mp4")
        write_mp4(path, frames, fps=cache.grid["fps"], wave=wave.float().cpu().numpy(),
                  sample_rate=AUDIO_SAMPLE_RATE)
        print(f"[5/5] wrote {path} (prompt: {prompt!r})")
    else:
        print("[5/5] skipped VAE decode (pass --with-vae to write a real mp4)")
    print("smoke test passed")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
