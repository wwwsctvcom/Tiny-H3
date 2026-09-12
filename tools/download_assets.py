#!/usr/bin/env python
"""Download every asset Tiny-H3 needs, from one place.

Three groups:

* ``model``   -- MiniMax-H3 official configs/tokenizer/docs/assets and the two VAEs
                 (``vae/`` 10.4 GB video, ``audio_vae/`` 0.6 GB audio).  The 66 GB DiT and
                 the 66 GB Qwen3-VL text encoder are *deliberately* skipped: Tiny-H3 trains
                 its own tiny DiT and conditions on a small text encoder.
* ``data``    -- real-video datasets that fit a hobby disk: ``rockdu/WISA-80K-Practical-Dynamics-254``
                 (254 clips already on H3's serving grid) and a subset of
                 ``alejandroparedeslatorre/concerts_audiovideo_dataset`` (real footage *with*
                 audio, used for the audio-visual demo).
* ``all``     -- both groups.

The default endpoint is the hf-mirror mirror (no proxy) because it is ~40x faster than the
academic proxy in this environment.  Set ``HF_ENDPOINT`` to override.

Examples::

    python tools/download_assets.py --group all
    python tools/download_assets.py --group model --skip-video-vae
    python tools/download_assets.py --group data --num-concert-clips 120
"""

from __future__ import annotations

import argparse
import os
import sys
import time

# Must be set before importing huggingface_hub.
os.environ.setdefault("HF_ENDPOINT", "https://hf-mirror.com")
# The mirror rejects the Xet CAS handshake (401) and rate-limits its token endpoint; plain
# HTTP downloads are both simpler and, measured here, ~40x faster than the academic proxy.
os.environ.setdefault("HF_HUB_DISABLE_XET", "1")
os.environ.setdefault("HF_HUB_ENABLE_HF_TRANSFER", "0")

H3_REPO = "MiniMaxAI/MiniMax-H3"
WISA_REPO = "rockdu/WISA-80K-Practical-Dynamics-254"
CONCERTS_REPO = "alejandroparedeslatorre/concerts_audiovideo_dataset"
# Frozen conditioners, pulled through the same mirror.  Tiny-H3 defaults to the 0.6 B Qwen3
# (1024-dim) to stay in the Qwen lineage MiniMax-H3 itself uses; t5-small is the lightweight
# alternative for CPU smoke tests.
TEXT_ENCODER_REPOS = ("Qwen/Qwen3-0.6B", "t5-small")
TEXT_ENCODER_PATTERNS = ["*.json", "*.txt", "*.safetensors", "tokenizer*", "vocab*", "merges*"]

MODEL_PATTERNS = [
    "*.json",
    "*.md",
    "README.md",
    "LICENSE",
    "docs/*",
    "scripts/*",
    "assets/*",
    "scheduler/*",
    "audio_scheduler/*",
    "tokenizer/*",
    "processor/*",
    "transformer/config.json",
    "transformer/*.index.json",
    "transformer_ref/config.json",
    "vae/config.json",
    "vae/*.index.json",
    "audio_vae/*",
    "FL2VA/model_index.json",
    "FL2VA/audio_vae/*",
    "FL2VA/video_vae/*.py",
    "FL2VA/video_vae/config.json",
    "FL2VA/video_vae/source/config.json",
    "FL2VA/tokenizer/*",
    "FL2VA/processor/*",
]

VIDEO_VAE_PATTERNS = ["vae/diffusion_pytorch_model*.safetensors"]
AUDIO_VAE_PATTERNS = ["audio_vae/diffusion_pytorch_model.safetensors"]


def _hub():
    from huggingface_hub import snapshot_download

    return snapshot_download


def human(n: float) -> str:
    for unit in ("B", "KB", "MB", "GB", "TB"):
        if abs(n) < 1024 or unit == "TB":
            return f"{n:.2f} {unit}"
        n /= 1024
    return f"{n:.2f} TB"


def dir_size(path: str) -> int:
    total = 0
    for root, _dirs, files in os.walk(path):
        for name in files:
            try:
                total += os.path.getsize(os.path.join(root, name))
            except OSError:
                pass
    return total


def _download(repo: str, patterns=None, label: str = "", dry_run: bool = False):
    snapshot_download = _hub()
    if dry_run:
        print(f"[dry-run] would download {repo} patterns={patterns}")
        return None
    t0 = time.time()
    path = snapshot_download(
        repo,
        allow_patterns=patterns,
        max_workers=8,
        # hf_transfer off: the xet token endpoint is the piece hf-mirror rate-limits hardest.
    )
    print(f"[done] {label or repo} -> {path}  ({time.time() - t0:.0f}s)", flush=True)
    return path


def download_model(args) -> None:
    print(f"== MiniMax-H3 configs / tokenizer / docs / assets (endpoint={os.environ['HF_ENDPOINT']}) ==", flush=True)
    _download(H3_REPO, MODEL_PATTERNS, "H3 small files", args.dry_run)

    print("== frozen text encoders ==", flush=True)
    for repo in TEXT_ENCODER_REPOS:
        _download(repo, TEXT_ENCODER_PATTERNS, repo, args.dry_run)

    patterns = AUDIO_VAE_PATTERNS
    if not args.skip_video_vae:
        patterns = patterns + VIDEO_VAE_PATTERNS
    else:
        print("[skip] vae/diffusion_pytorch_model*.safetensors (--skip-video-vae)")

    print(f"== MiniMax-H3 VAE weights: {patterns} ==", flush=True)
    path = _download(H3_REPO, patterns, "H3 VAE weights", args.dry_run)
    if path and not args.skip_video_vae:
        print("   note: the video VAE is ~10.4 GB, the audio VAE ~0.6 GB.")


def download_data(args) -> None:
    print("== WISA-80K-Practical-Dynamics-254: 254 real clips + train.jsonl (video only) ==", flush=True)
    _download(WISA_REPO, ["clips/*", "train.jsonl", "README.md"], "WISA-254", args.dry_run)

    print(f"== concerts_audiovideo_dataset: {args.num_concert_clips} real clips WITH audio ==", flush=True)
    if args.dry_run:
        print("[dry-run] would download a concerts subset")
        return
    from huggingface_hub import list_repo_files

    files = [f for f in list_repo_files(CONCERTS_REPO, repo_type="dataset") if f.endswith(".mp4")]
    files.sort()
    # Spread the picks over the whole listing: the dataset is ordered by source video, so a
    # head-only slice would collapse to a couple of recordings.
    stride = max(1, len(files) // max(1, args.num_concert_clips))
    picked = files[::stride][: args.num_concert_clips]
    print(f"   {len(files)} clips available, picking {len(picked)} (stride {stride})", flush=True)
    _download(CONCERTS_REPO, picked + ["concert*.txt", "README.md"], "concerts subset", args.dry_run)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--group", choices=["model", "data", "all"], default="all")
    ap.add_argument("--skip-video-vae", action="store_true", help="skip the 10.4 GB video VAE")
    ap.add_argument("--num-concert-clips", type=int, default=120)
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()

    print(f"HF_HOME={os.environ.get('HF_HOME', '~/.cache/huggingface')}")
    if args.group in ("model", "all"):
        download_model(args)
    if args.group in ("data", "all"):
        download_data(args)
    print("all downloads finished", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
