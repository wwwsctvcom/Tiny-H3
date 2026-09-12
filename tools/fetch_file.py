#!/usr/bin/env python
"""Parallel range downloader with resume + sha256 verification.

`huggingface_hub` downloads one stream per file; when a mirror throttles a single connection this
utility splits the file into ranges and fetches them concurrently, then verifies the whole-file
sha256 (which, for HuggingFace LFS blobs, is the blob filename).

It is used by hand when the mirror is slow; the normal asset path stays `tools/download_assets.py`.

Example::

    python tools/fetch_file.py \
      --url https://hf-mirror.com/MiniMaxAI/MiniMax-H3/resolve/main/vae/diffusion_pytorch_model-00001-of-00003.safetensors \
      --out /root/autodl-tmp/hf_cache/hub/models--MiniMaxAI--MiniMax-H3/blobs/72f4c6be....incomplete \
      --resume --workers 8 --sha256 72f4c6be...
"""

from __future__ import annotations

import argparse
import concurrent.futures as futures
import hashlib
import os
import subprocess
import sys
import time


def head_size(url: str) -> int | None:
    out = subprocess.run(
        ["curl", "-sIL", "-m", "30", url], capture_output=True, text=True
    ).stdout
    for line in out.splitlines():
        if line.lower().startswith("content-length:"):
            return int(line.split(":")[1].strip())
    return None


def fetch_range(url: str, start: int, end: int, path: str, retries: int = 4) -> str:
    """Download ``[start, end]`` into ``path`` (skipped when the file already has the right size)."""
    want = end - start + 1
    if os.path.exists(path) and os.path.getsize(path) == want:
        return path
    for attempt in range(1, retries + 1):
        proc = subprocess.run(
            ["curl", "-sL", "-m", "1800", "-r", f"{start}-{end}", "-o", path, url],
            capture_output=True,
        )
        if proc.returncode == 0 and os.path.exists(path) and os.path.getsize(path) == want:
            return path
        time.sleep(min(30, 3 * attempt))
    raise RuntimeError(f"range {start}-{end} failed after {retries} attempts")


def sha256_of(path: str, chunk: int = 1 << 22) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as f:
        while True:
            block = f.read(chunk)
            if not block:
                break
            digest.update(block)
    return digest.hexdigest()


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--url", required=True)
    ap.add_argument("--out", required=True, help="destination file (may already contain a partial prefix)")
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--sha256", default="", help="expected digest; skipped when empty")
    ap.add_argument("--chunk-mb", type=int, default=256)
    ap.add_argument("--size", type=int, default=0, help="total size in bytes (else taken from the server)")
    args = ap.parse_args()

    total = args.size or head_size(args.url)
    if not total:
        print("could not determine file size", file=sys.stderr)
        return 2
    prefix = os.path.getsize(args.out) if os.path.exists(args.out) else 0
    if prefix > total:
        print(f"existing file is larger than the target ({prefix} > {total}); refusing", file=sys.stderr)
        return 2
    print(f"target {total/1e9:.3f} GB, already present {prefix/1e9:.3f} GB, workers={args.workers}", flush=True)

    chunk = args.chunk_mb * 1024 * 1024
    spans = []
    start = prefix
    while start < total:
        end = min(start + chunk, total) - 1
        spans.append((start, end))
        start = end + 1

    tmp_dir = args.out + ".parts"
    os.makedirs(tmp_dir, exist_ok=True)
    parts = [os.path.join(tmp_dir, f"{i:05d}_{s}_{e}") for i, (s, e) in enumerate(spans)]
    started = time.time()
    done_bytes = prefix
    with futures.ThreadPoolExecutor(max_workers=args.workers) as pool:
        jobs = {
            pool.submit(fetch_range, args.url, s, e, p): (s, e, p)
            for (s, e), p in zip(spans, parts)
        }
        for job in futures.as_completed(jobs):
            job.result()  # raises on failure
            s, e, p = jobs[job]
            done_bytes += e - s + 1
            speed = (done_bytes - prefix) / max(1e-6, time.time() - started)
            print(f"  {done_bytes/1e9:6.3f}/{total/1e9:.3f} GB  ({speed/1e6:.2f} MB/s)", flush=True)

    # Assemble: existing prefix + the downloaded parts, in order.
    with open(args.out, "r+b" if prefix else "wb") as target:
        target.seek(prefix)
        for part in parts:
            with open(part, "rb") as src:
                while True:
                    block = src.read(1 << 22)
                    if not block:
                        break
                    target.write(block)
    assert os.path.getsize(args.out) == total, "assembled file has the wrong size"

    if args.sha256:
        digest = sha256_of(args.out)
        if digest != args.sha256:
            print(f"sha256 mismatch:\n  expected {args.sha256}\n  got      {digest}", file=sys.stderr)
            return 1
        print(f"sha256 ok: {digest}")
    for part in parts:
        os.remove(part)
    os.rmdir(tmp_dir)
    print(f"done -> {args.out} ({time.time() - started:.0f}s)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
