#!/usr/bin/env python
"""Encode clips into Tiny-H3's latent cache (the training input).

One ``.npz`` per clip with everything the DiT trainer needs:

===============  ==========================================================
``video_latents``  ``(24, 5n+2, H/16, W/16)`` float16, H3-normalized
``audio_latents``  ``(2, 32, T)`` float16, H3-normalized (posterior mean)
``text_embed``     ``(text_tokens, 512)`` float32 from the frozen T5
``prompt``         the prompt string
===============  ==========================================================

Encoding uses the **released H3 VAEs**, so training never pays for pixels and the tiny DiT
learns exactly the latent space the official decoder inverts (see ``tiny_h3/vae.py`` for the
conventions).  Resumable: existing ``.npz`` files are skipped unless ``--overwrite``.

Example::

    source scripts/env.sh
    python tools/prepare_latents.py --data-dir $TINY_H3_DATA/synth --out $TINY_H3_DATA/synth/latents
    python tools/prepare_latents.py --data-dir $TINY_H3_DATA/concerts --out $TINY_H3_DATA/concerts/latents
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from concurrent.futures import ThreadPoolExecutor

import numpy as np

from _bootstrap import bootstrap  # noqa: E402  (maps the tiny_h3 package name onto ../src)

from tiny_h3.media import AUDIO_SAMPLE_RATE, read_audio_wave, read_video_frames, read_wav  # noqa: E402
from tiny_h3.model import DEFAULT_TEXT_ENCODER, DEFAULT_TEXT_TOKENS, TEXT_ENCODER_CHOICES  # noqa: E402
from tiny_h3.vae import (  # noqa: E402
    audio_latent_frames,
    encode_audio,
    encode_video,
    num_samples_for_frames,
    video_latent_frames,
)


def load_manifest(data_dir: str, split: str) -> tuple[list[dict], dict]:
    records: list[dict] = []
    for name in ("train", "val") if split == "both" else (split,):
        path = os.path.join(data_dir, f"{name}.jsonl")
        if not os.path.exists(path):
            raise FileNotFoundError(f"missing manifest {path}; run scripts/prepare_data.sh synth first")
        with open(path, encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line:
                    rec = json.loads(line)
                    rec["_split"] = name
                    records.append(rec)
    meta_path = os.path.join(data_dir, "meta.json")
    meta = json.load(open(meta_path, encoding="utf-8")) if os.path.exists(meta_path) else {}
    return records, meta


def fit_frames(frames: np.ndarray, num_frames: int) -> np.ndarray:
    """Pad by repeating the last frame or truncate to exactly ``num_frames``."""
    if len(frames) == num_frames:
        return frames
    if len(frames) > num_frames:
        return frames[:num_frames]
    if len(frames) == 0:
        raise ValueError("clip decoded to zero frames")
    pad = np.repeat(frames[-1:], num_frames - len(frames), axis=0)
    return np.concatenate([frames, pad], axis=0)


def fit_audio(wave: np.ndarray, num_samples: int, channels: int = 2) -> np.ndarray:
    if wave.size == 0:
        return np.zeros((channels, num_samples), dtype=np.float32)
    if wave.shape[1] >= num_samples:
        return wave[:, :num_samples]
    pad = np.zeros((wave.shape[0], num_samples - wave.shape[1]), dtype=np.float32)
    return np.concatenate([wave, pad], axis=1)


def prepare_one(rec: dict, root: str, grid: dict, channels: int = 2) -> dict:
    """Decode one clip onto the grid; returns numpy arrays (GPU work happens later)."""
    video_path = os.path.join(root, rec["metadata"]["video"])
    audio_path = rec["metadata"].get("audio")
    if audio_path:
        audio_path = os.path.join(root, audio_path)

    frames = read_video_frames(video_path, (grid["size"], grid["size"]), fps=grid["fps"])
    frames = fit_frames(frames, grid["frames"])
    if audio_path and os.path.exists(audio_path):
        wave = read_wav(audio_path, sample_rate=AUDIO_SAMPLE_RATE)
    else:
        wave = read_audio_wave(video_path, sample_rate=AUDIO_SAMPLE_RATE, channels=channels)
    wave = fit_audio(wave, num_samples_for_frames(grid["fps"], grid["frames"]))
    return {
        "frames": frames,
        "wave": wave,
        "prompt": rec.get("prompt", ""),
        "seed": rec.get("metadata", {}).get("seed"),
        "params": rec.get("params", {}),
        "split": rec.get("_split", "train"),
        "video": rec["metadata"]["video"],
    }


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--data-dir", required=True, help="directory holding train.jsonl / val.jsonl / clips")
    ap.add_argument("--out", required=True, help="latent cache directory")
    ap.add_argument("--split", default="both", choices=["train", "val", "both"])
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--h3-model", default="MiniMaxAI/MiniMax-H3", help="repo id or local snapshot dir with vae/ and audio_vae/")
    ap.add_argument("--text-encoder", default=DEFAULT_TEXT_ENCODER,
                    help=f"frozen conditioner: {', '.join(TEXT_ENCODER_CHOICES)}")
    ap.add_argument("--text-tokens", type=int, default=DEFAULT_TEXT_TOKENS)
    ap.add_argument("--text-layer", type=int, default=-1,
                    help="-1 = final hidden state; >=0 mimics official H3's intermediate-layer read")
    ap.add_argument("--size", type=int, default=0, help="canvas size (default: data-dir meta.json)")
    ap.add_argument("--fps", type=float, default=0.0)
    ap.add_argument("--frames", type=int, default=0)
    ap.add_argument("--batch-size", type=int, default=1, help="clips encoded per VAE call (1 is safest)")
    ap.add_argument("--workers", type=int, default=3, help="CPU decode prefetch threads")
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--overwrite", action="store_true")
    args = ap.parse_args()

    import torch

    from tiny_h3.model import TextConditioner
    from tiny_h3.vae import load_vaes

    records, meta = load_manifest(args.data_dir, args.split)
    grid = {
        "size": args.size or int(meta.get("size", 256)),
        "fps": args.fps or float(meta.get("fps", 24.0)),
        "frames": args.frames or int(meta.get("frames", 22)),
    }
    expected_video_t = video_latent_frames(grid["frames"])
    expected_audio_t = audio_latent_frames(num_samples_for_frames(grid["fps"], grid["frames"]))
    if args.limit:
        records = records[: args.limit]

    device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        print("warning: cuda requested but unavailable, falling back to cpu (slow)", flush=True)
        device = torch.device("cpu")

    os.makedirs(args.out, exist_ok=True)
    print(f"encoding {len(records)} clips -> {args.out} on {device}")
    print(f"grid: {grid['size']}x{grid['size']} @ {grid['fps']} fps, {grid['frames']} frames "
          f"-> video latents {expected_video_t} frames, audio latents {expected_audio_t} frames")

    bundle = load_vaes(args.h3_model, device=device)
    conditioner = TextConditioner(args.text_encoder, args.text_tokens, device=device, text_layer=args.text_layer)

    done, skipped, failed = 0, 0, 0
    # One manifest per split (latents_train.jsonl / latents_val.jsonl): that is exactly the
    # layout tiny_h3.data.dataset.LatentCache reads, and per-split npz names keep the
    # train/val seed ranges from colliding.
    manifest_splits = ("train", "val") if args.split == "both" else (args.split,)
    manifests = {
        s: open(os.path.join(args.out, f"latents_{s}.jsonl"), "w", encoding="utf-8")
        for s in manifest_splits
    }
    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        pending = {}
        it = iter(enumerate(records))

        def submit_next():
            idx, rec = next(it)
            return idx, rec, pool.submit(prepare_one, rec, args.data_dir, grid)

        try:
            for _ in range(max(1, args.workers * 2)):
                idx, rec, fut = submit_next()
                pending[idx] = (rec, fut)
        except StopIteration:
            pass

        while pending:
            idx = min(pending)
            rec, future = pending.pop(idx)
            batch = future.result()
            name = f"{batch['split']}_{rec.get('metadata', {}).get('seed', idx):06d}"
            out_path = os.path.join(args.out, f"{name}.npz")
            try:
                if os.path.exists(out_path) and not args.overwrite:
                    skipped += 1
                else:
                    # Keep uint8 so encode_video applies the ImageNet pixel normalization itself.
                    # (T, H, W, 3) -> (1, 3, T, H, W) as encode_video expects.
                    frames_t = torch.from_numpy(batch["frames"]).permute(3, 0, 1, 2)[None]
                    wave_t = torch.from_numpy(batch["wave"])
                    with torch.no_grad():
                        video_latents = encode_video(bundle, frames_t)[0]
                        audio_latents = encode_audio(bundle, wave_t)
                    text = conditioner.encode([batch["prompt"]])[0]
                    np.savez_compressed(
                        out_path,
                        video_latents=video_latents.to(torch.float16).cpu().numpy(),
                        audio_latents=audio_latents.to(torch.float16).cpu().numpy(),
                        text_embed=text.numpy().astype(np.float32),
                        prompt=np.array(batch["prompt"]),
                    )
                    if video_latents.shape[1] != expected_video_t or audio_latents.shape[2] != expected_audio_t:
                        print(f"  ! {name}: unexpected latent shapes "
                              f"{tuple(video_latents.shape)} / {tuple(audio_latents.shape)}", flush=True)
                    manifests[batch["split"]].write(json.dumps({
                        "file": os.path.basename(out_path),
                        "prompt": batch["prompt"],
                        "params": batch["params"],
                        "seed": batch["seed"],
                        "split": batch["split"],
                        "video_latents_shape": list(video_latents.shape),
                        "audio_latents_shape": list(audio_latents.shape),
                        "text_tokens": int(text.shape[0]),
                        "grid": grid,
                    }, ensure_ascii=False) + "\n")
                    done += 1
            except Exception as exc:  # noqa: BLE001
                failed += 1
                print(f"  ! failed {name}: {type(exc).__name__}: {exc}", flush=True)

            if (done + skipped + failed) % 25 == 0:
                print(f"  {done} encoded, {skipped} cached, {failed} failed "
                      f"/ {len(records)}", flush=True)
            try:
                i2, r2, f2 = submit_next()
                pending[i2] = (r2, f2)
            except StopIteration:
                pass

    for handle in manifests.values():
        handle.close()
    with open(os.path.join(args.out, "latents_meta.json"), "w", encoding="utf-8") as f:
        json.dump({"grid": grid, "count": done + skipped, "failed": failed,
                   "h3_model": args.h3_model, "text_encoder": args.text_encoder,
                   "text_tokens": args.text_tokens, "text_dim": int(conditioner.text_dim),
                   "text_layer": int(args.text_layer),
                   "video_latent_frames": expected_video_t,
                   "audio_latent_frames": expected_audio_t}, f, indent=2)
    print(f"done: {done} encoded, {skipped} already cached, {failed} failed -> "
          f"{', '.join(os.path.basename(handle.name) for handle in manifests.values())}")
    return 0 if failed == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
