#!/usr/bin/env python
"""Publish a trained Tiny-H3 checkpoint to the Hugging Face Hub.

Assembles the diffusers-standard model repo (``config.json`` + safetensors, the same
layout ``from_pretrained`` consumes) plus a generated model card, then uploads it.

    # assemble locally and print what would be uploaded
    python tools/publish_hf.py --checkpoint runs/xl_large/final \
        --repo-id <user>/Tiny-H3-XL --dry-run

    # upload (token from --token or $HF_TOKEN; needs write access to the repo)
    python tools/publish_hf.py --checkpoint runs/xl_large/final \
        --repo-id <user>/Tiny-H3-XL --token hf_xxx
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import _bootstrap  # noqa: E402


def _param_count(safetensors_path: str) -> int:
    """Exact parameter count from the safetensors header (no weight loading)."""
    from safetensors import safe_open

    total = 0
    with safe_open(safetensors_path, framework="pt", device="cpu") as f:
        for key in f.keys():
            n = 1
            for dim in f.get_slice(key).get_shape():
                n *= dim
            total += n
    return total


def build_model_card(checkpoint: str, repo_id: str, notes: str) -> str:
    with open(os.path.join(checkpoint, "tiny_h3_meta.json"), encoding="utf-8") as f:
        meta = json.load(f)
    params = meta.get("parameters") or _param_count(
        os.path.join(checkpoint, "diffusion_pytorch_model.safetensors"))
    cfg_path = os.path.join(os.path.dirname(checkpoint.rstrip("/")), "train_config.json")
    trained = {}
    if os.path.exists(cfg_path):
        with open(cfg_path, encoding="utf-8") as f:
            trained = json.load(f)
    steps = trained.get("steps", meta.get("step"))
    size = f"{params/1e6:.1f}M"

    return f"""---
license: apache-2.0
library_name: diffusers
pipeline_tag: text-to-video
tags:
- text-to-video
- text-to-audio
- audio-generation
- minimax-h3
- flow-matching
- diffusion-transformer
- tiny-h3
---

# Tiny-H3 ({size} DiT)

A [MiniMax-H3](https://huggingface.co/MiniMaxAI/MiniMax-H3)-compatible text-to-audio-video
diffusion transformer trained with the [Tiny-H3](https://github.com/wwwsctvcom/Tiny-H3)
framework: one packed sequence of text, audio and video rows, the official frozen
MiniMax-H3 video/audio VAEs on both sides, and a single flow-matching objective
(`v = x0 - noise`, H3's data-ward sign).

| | |
|---|---|
| Architecture | `MiniMaxH3Transformer3DModel` (diffusers) |
| Parameters | {size} |
| Training data | procedural geometric T2AV clips, 256x256 / 22 frames / 32 kHz stereo |
| Training steps | {steps} |

{notes}

## Usage

The H3 model classes currently live on diffusers `main`:

```bash
pip install git+https://github.com/huggingface/diffusers.git
```

```python
import torch
from diffusers import MiniMaxH3Transformer3DModel

dit = MiniMaxH3Transformer3DModel.from_pretrained(
    "{repo_id}", torch_dtype=torch.bfloat16)
```

For text -> MP4 inference (prompt in, playable H.264 + AAC out), use the Tiny-H3 repo's
inference entry point, which also loads the frozen official VAE decoders:

```bash
git clone https://github.com/wwwsctvcom/Tiny-H3 && cd Tiny-H3
pip install -e .
python tools/inference.py --checkpoint <local copy of this repo> \\
    --prompt "a red circle bouncing on a dark background, with rhythmic thumps"
```

## Limitations

* Trained on procedurally rendered geometric scenes: clean motion and synced sound,
  not photorealistic footage.
* Text conditioning comes from a frozen Qwen3-0.6B (`tiny_h3_meta.json`); English
  prompt templates work best.
"""


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    ap.add_argument("--checkpoint", required=True, help="trained checkpoint dir (config.json + safetensors)")
    ap.add_argument("--repo-id", required=True, help="HF repo, e.g. <user>/Tiny-H3-XL")
    ap.add_argument("--token", default=os.environ.get("HF_TOKEN"), help="HF write token (default $HF_TOKEN)")
    ap.add_argument("--private", action="store_true", help="create the repo as private")
    ap.add_argument("--notes", default="", help="extra paragraph for the model card (training details)")
    ap.add_argument("--staging", default="/tmp/tiny_h3_hf_stage", help="local assembly dir")
    ap.add_argument("--dry-run", action="store_true", help="assemble locally, do not upload")
    args = ap.parse_args()

    for name in ("config.json", "diffusion_pytorch_model.safetensors", "tiny_h3_meta.json"):
        path = os.path.join(args.checkpoint, name)
        if not os.path.exists(path):
            raise SystemExit(f"missing {path} -- is this a Tiny-H3 checkpoint?")

    staging = os.path.join(args.staging, args.repo_id.replace("/", "__"))
    if os.path.exists(staging):
        shutil.rmtree(staging)
    os.makedirs(staging)
    for name in ("config.json", "diffusion_pytorch_model.safetensors", "tiny_h3_meta.json"):
        shutil.copy2(os.path.join(args.checkpoint, name), staging)
    card = build_model_card(args.checkpoint, args.repo_id, args.notes)
    with open(os.path.join(staging, "README.md"), "w", encoding="utf-8") as f:
        f.write(card)
    print(f"assembled {staging}:")
    for name in sorted(os.listdir(staging)):
        print(f"  {name:36s} {os.path.getsize(os.path.join(staging, name))/1e6:8.1f} MB")

    if args.dry_run:
        print("dry run: not uploading")
        return 0

    token = args.token or os.environ.get("HF_TOKEN")
    if not token:
        raise SystemExit("no token: pass --token or set $HF_TOKEN (needs write access)")

    from huggingface_hub import HfApi

    api = HfApi(token=token)
    api.create_repo(args.repo_id, repo_type="model", private=args.private, exist_ok=True)
    api.upload_folder(folder_path=staging, repo_id=args.repo_id, repo_type="model")
    print(f"uploaded -> https://huggingface.co/{args.repo_id}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
