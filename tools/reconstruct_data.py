#!/usr/bin/env python
"""Round-trip real training samples through the official H3 VAEs and write demo mp4 files.

This is checkpoint-independent acceptance gate #1: before asking a tiny DiT to learn a latent
space, prove that the downloaded *real* H3 decoders can reconstruct our training video/audio
into a normal playable mp4 with a real AAC stereo track.

Example::

    python tools/reconstruct_data.py --data-dir data/synth --out outputs/reconstruction --count 4
"""

from __future__ import annotations

import argparse
import json
import os
import sys

import numpy as np

from _bootstrap import bootstrap  # noqa: E402  (maps the tiny_h3 package name onto ../src)

from tiny_h3.media import read_video_frames, read_wav, write_mp4  # noqa: E402


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    ap.add_argument("--data-dir", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--count", type=int, default=4)
    ap.add_argument("--h3-model", default="MiniMaxAI/MiniMax-H3")
    ap.add_argument("--device", default="cuda")
    args = ap.parse_args()

    import torch

    from tiny_h3.vae import decode_audio, decode_video, encode_audio, encode_video, load_vaes

    with open(os.path.join(args.data_dir, "meta.json"), encoding="utf-8") as f:
        meta = json.load(f)
    records = []
    with open(os.path.join(args.data_dir, "train.jsonl"), encoding="utf-8") as f:
        for line in f:
            records.append(json.loads(line))
            if len(records) >= args.count:
                break
    os.makedirs(args.out, exist_ok=True)
    bundle = load_vaes(args.h3_model, device=args.device)
    manifest = []
    for index, rec in enumerate(records):
        frames = read_video_frames(
            os.path.join(args.data_dir, rec["metadata"]["video"]),
            (int(meta["size"]), int(meta["size"])), float(meta["fps"]), max_frames=int(meta["frames"]),
        )
        wave = read_wav(os.path.join(args.data_dir, rec["metadata"]["audio"]), sample_rate=int(meta["sample_rate"]))
        pixels = torch.from_numpy(frames).permute(3, 0, 1, 2)[None].to(args.device)
        wave_t = torch.from_numpy(wave).to(args.device)
        with torch.no_grad():
            z_video = encode_video(bundle, pixels)
            z_audio = encode_audio(bundle, wave_t)
            video_out = decode_video(bundle, z_video)[0, :, : len(frames)]
            wave_out = decode_audio(bundle, z_audio)[:, : wave.shape[1]]
        frames_out = (video_out.permute(1, 2, 3, 0).clamp(0, 1) * 255).round().byte().cpu().numpy()
        wave_out = wave_out.float().cpu().numpy()
        out_path = os.path.join(args.out, f"{index:02d}_h3_vae_reconstruction.mp4")
        write_mp4(out_path, frames_out, float(meta["fps"]), wave_out, int(meta["sample_rate"]))
        manifest.append({
            "file": os.path.basename(out_path), "prompt": rec["prompt"],
            "video_latents": list(z_video.shape), "audio_latents": list(z_audio.shape),
            "pixel_mae": float(np.abs(frames_out.astype(np.float32) - frames.astype(np.float32)).mean()),
            "audio_rms": float(np.sqrt(np.mean(wave_out**2))),
        })
        print(f"saved {out_path}  video={tuple(z_video.shape)} audio={tuple(z_audio.shape)}")
    with open(os.path.join(args.out, "manifest.json"), "w", encoding="utf-8") as f:
        json.dump(manifest, f, ensure_ascii=False, indent=2)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
