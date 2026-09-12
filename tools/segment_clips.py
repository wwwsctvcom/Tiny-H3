#!/usr/bin/env python
"""Convert long external clips into Tiny-H3 training segments.

Slices every source clip into fixed-length segments on the ``17n+5`` frame grid, rescales
them to the training canvas, keeps the audio track in sync, and writes a standard Tiny-H3
data directory (``train.jsonl`` / ``meta.json`` / ``clips/``) that
``tools/prepare_latents.py`` consumes directly.

Example (MiniMax-H3 self-generated dataset, 124-frame clips -> 5 segments of 22 frames)::

    python tools/segment_clips.py \
        --metadata /path/metadata.jsonl --video-root /path/videos \
        --out $TINY_H3_DATA/h3selfgen_seg --max-clips 1100 --workers 16
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from concurrent.futures import ProcessPoolExecutor

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import _bootstrap  # noqa: E402


def segment_one(task: dict) -> dict | None:
    """Cut one (source clip, segment index) pair; returns the train.jsonl row."""
    from tiny_h3.media import ffmpeg_exe

    src, out_path, start_sec, dur_sec, size, fps = (
        task["src"], task["out_path"], task["start_sec"], task["dur_sec"], task["size"], task["fps"],
    )
    cmd = [
        ffmpeg_exe(), "-v", "error", "-y",
        "-ss", f"{start_sec:.4f}", "-i", src, "-t", f"{dur_sec:.4f}",
        "-vf", f"scale={size}:{size}:flags=lanczos", "-r", str(fps),
        "-c:v", "libx264", "-preset", "veryfast", "-crf", "22", "-pix_fmt", "yuv420p",
        "-c:a", "aac", "-b:a", "128k", "-ar", "32000", "-ac", "2",
        "-movflags", "+faststart", out_path,
    ]
    proc = subprocess.run(cmd, capture_output=True)
    if proc.returncode != 0 or not os.path.exists(out_path) or os.path.getsize(out_path) < 10_000:
        return None
    return task["row"]


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    ap.add_argument("--metadata", required=True, help="jsonl with prompt + video_path (+ height/width)")
    ap.add_argument("--video-root", required=True, help="root the video_path fields resolve against")
    ap.add_argument("--out", required=True, help="output data directory (train.jsonl/meta.json/clips)")
    ap.add_argument("--segment-frames", type=int, default=22, help="frames per segment (17n+5 grid)")
    ap.add_argument("--size", type=int, default=256, help="square training canvas")
    ap.add_argument("--fps", type=float, default=24.0)
    ap.add_argument("--max-clips", type=int, default=0, help="limit source clips (0 = all)")
    ap.add_argument("--workers", type=int, default=16)
    args = ap.parse_args()

    records = []
    with open(args.metadata, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                records.append(json.loads(line))
    if args.max_clips:
        records = records[: args.max_clips]

    seg_sec = args.segment_frames / args.fps
    os.makedirs(os.path.join(args.out, "clips"), exist_ok=True)

    tasks, skipped = [], 0
    for rec in records:
        src = os.path.join(args.video_root, rec["video_path"])
        if not os.path.exists(src):
            skipped += 1
            continue
        num_frames = int(rec.get("num_frames") or 0)
        if num_frames < args.segment_frames:
            skipped += 1
            continue
        stem = os.path.splitext(os.path.basename(rec["video_path"]))[0]
        n_seg = num_frames // args.segment_frames
        for seg in range(n_seg):
            name = f"{stem}_seg{seg}.mp4"
            tasks.append({
                "src": src,
                "out_path": os.path.join(args.out, "clips", name),
                "start_sec": seg * seg_sec,
                "dur_sec": seg_sec,
                "size": args.size,
                "fps": args.fps,
                "row": {
                    "prompt": rec.get("prompt", ""),
                    "metadata": {"video": f"clips/{name}"},
                },
            })
    print(f"{len(records)} source clips ({skipped} skipped) -> {len(tasks)} segments of "
          f"{args.segment_frames} frames @ {args.size}x{args.size}")

    done = 0
    with ProcessPoolExecutor(max_workers=args.workers) as pool:
        for i, row in enumerate(pool.map(segment_one, tasks, chunksize=8)):
            done += 1 if row else 0
            if (i + 1) % 500 == 0:
                print(f"  {i + 1}/{len(tasks)} segmented ({done} ok)", flush=True)

    # Build the manifest from segments that actually landed on disk.
    ok_rows = []
    for task in tasks:
        if os.path.exists(task["out_path"]) and os.path.getsize(task["out_path"]) >= 10_000:
            ok_rows.append(task["row"])

    with open(os.path.join(args.out, "train.jsonl"), "w", encoding="utf-8") as f:
        for row in ok_rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")
    with open(os.path.join(args.out, "meta.json"), "w", encoding="utf-8") as f:
        json.dump({"size": args.size, "fps": args.fps, "frames": args.segment_frames,
                   "sample_rate": 32000}, f, indent=2)
    print(f"done: {len(ok_rows)} segments -> {args.out}/train.jsonl")
    return 0 if ok_rows else 1


if __name__ == "__main__":
    raise SystemExit(main())
