#!/usr/bin/env python
"""Generate the synthetic audio-visual training set for Tiny-H3.

Writes ``clips/*.mp4`` plus ``train.jsonl`` / ``val.jsonl`` manifests in the same
``{"prompt": ..., "metadata": {"video": ...}}`` shape the MiniMax-H3 recipes use (with a few
extra fields for the verifier rewards), and a ``preview.png`` montage so you can eyeball the
data before spending GPU time.

Example::

    python tools/make_synth_data.py --out data/synth --count 480 --val-count 48 --size 128
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from concurrent.futures import ProcessPoolExecutor, as_completed

import numpy as np

from _bootstrap import bootstrap  # noqa: E402  (maps the tiny_h3 package name onto ../src)

from tiny_h3.data.synth import MOTIONS, generate_scene  # noqa: E402
from tiny_h3.media import write_clip  # noqa: E402


def _render_one(job: dict) -> dict:
    scene = generate_scene(
        seed=job["seed"],
        size=job["size"],
        fps=job["fps"],
        frames=job["frames"],
        sample_rate=job["sample_rate"],
        motion=job["motion"],
    )
    video_path = os.path.join(job["clip_dir"], f"{job['seed']:06d}.mp4")
    audio_path = os.path.join(job["clip_dir"], f"{job['seed']:06d}.wav")
    # Video and audio are stored separately: the wav keeps the waveform sample-exact, while an
    # mp4's AAC priming would delay it by ~2048 samples and quietly break audio-visual sync.
    write_clip(video_path, audio_path, scene.frames, scene.fps, scene.wave, scene.sample_rate)
    return {
        "prompt": scene.prompt,
        "metadata": {
            "video": os.path.relpath(video_path, job["root"]),
            "audio": os.path.relpath(audio_path, job["root"]),
            "seed": job["seed"],
        },
        "params": scene.params,
        "grid": {"size": job["size"], "fps": job["fps"], "frames": job["frames"], "sample_rate": job["sample_rate"]},
    }


def _montage(records: list[dict], root: str, out_path: str, size: int, cols: int = 6, rows: int = 4) -> None:
    """A small contact sheet of the first frame of `cols*rows` clips."""
    from PIL import Image

    picks = records[: cols * rows]
    sheet = Image.new("RGB", (cols * size, rows * size), (0, 0, 0))
    from tiny_h3.media import read_video_frames

    for i, rec in enumerate(picks):
        frames = read_video_frames(os.path.join(root, rec["metadata"]["video"]), (size, size), fps=24, max_frames=1)
        if not len(frames):
            continue
        tile = Image.fromarray(frames[0])
        sheet.paste(tile, ((i % cols) * size, (i // cols) * size))
    sheet.save(out_path)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out", default="data/synth")
    ap.add_argument("--count", type=int, default=480, help="training clips")
    ap.add_argument("--val-count", type=int, default=48)
    ap.add_argument("--size", type=int, default=128)
    ap.add_argument("--fps", type=float, default=24.0)
    ap.add_argument("--frames", type=int, default=22, help="must sit on H3's 17n+5 grid: 22, 39, 56, ...")
    ap.add_argument("--sample-rate", type=int, default=32000)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--workers", type=int, default=min(16, os.cpu_count() or 4))
    ap.add_argument("--preview", action="store_true", help="also write preview.png (needs ffmpeg decode)")
    args = ap.parse_args()

    if (args.frames - 5) % 17 != 0:
        print(f"warning: {args.frames} frames is not on H3's 17n+5 grid; the video VAE will pad.", flush=True)

    root = os.path.abspath(args.out)
    clip_dir = os.path.join(root, "clips")
    os.makedirs(clip_dir, exist_ok=True)

    jobs = []
    for split, count, base in (("train", args.count, args.seed), ("val", args.val_count, args.seed + 10_000_000)):
        for i in range(count):
            jobs.append(
                {
                    "seed": base + i,
                    "split": split,
                    "size": args.size,
                    "fps": args.fps,
                    "frames": args.frames,
                    "sample_rate": args.sample_rate,
                    "motion": MOTIONS[i % len(MOTIONS)],
                    "clip_dir": clip_dir,
                    "root": root,
                }
            )

    records: dict[str, list[dict]] = {"train": [], "val": []}
    done = 0
    with ProcessPoolExecutor(max_workers=args.workers) as pool:
        futures = {pool.submit(_render_one, job): job for job in jobs}
        for future in as_completed(futures):
            job = futures[future]
            records[job["split"]].append(future.result())
            done += 1
            if done % max(1, len(jobs) // 10) == 0:
                print(f"  rendered {done}/{len(jobs)}", flush=True)

    for split in ("train", "val"):
        records[split].sort(key=lambda r: r["metadata"]["seed"])
        with open(os.path.join(root, f"{split}.jsonl"), "w", encoding="utf-8") as f:
            for rec in records[split]:
                f.write(json.dumps(rec, ensure_ascii=False) + "\n")

    with open(os.path.join(root, "meta.json"), "w", encoding="utf-8") as f:
        json.dump(
            {
                "size": args.size,
                "fps": args.fps,
                "frames": args.frames,
                "sample_rate": args.sample_rate,
                "train": len(records["train"]),
                "val": len(records["val"]),
                "generator": "tiny_h3.data.synth",
            },
            f,
            ensure_ascii=False,
            indent=2,
        )

    if args.preview and records["train"]:
        try:
            _montage(records["train"], root, os.path.join(root, "preview.png"), args.size)
            print(f"wrote {os.path.join(root, 'preview.png')}")
        except Exception as exc:  # pragma: no cover
            print(f"preview skipped: {exc}")

    sample = records["train"][0] if records["train"] else {}
    print(f"done: {len(records['train'])} train + {len(records['val'])} val clips in {root}")
    if sample:
        print("sample prompt:", sample["prompt"])
    return 0


if __name__ == "__main__":
    sys.exit(main())
