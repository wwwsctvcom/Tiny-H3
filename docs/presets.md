# Model presets and resource reference

Presets live in `src/model.py:TINY_H3_PRESETS` and are selected with `--preset`. All of them keep the
official latent geometry (24 video channels, 32 audio channels, `(1,2,2)` patch), so every preset can share
one official VAE and one latent cache.

## Compared with official H3

Official DiT: hidden 5376 / 50 layers / 56×128 heads / FFN 14336 / time-embed 2688 → **24.4 B params**
(66.28 GB of bf16 weights, per the checkpoint index metadata). Tiny-H3 shrinks only width, depth and FFN —
**never** the dimensions that define the latent space or the packed layout:

| Dimension | Official | Tiny-H3 |
|---|---|---|
| `hidden_size` | 5376 | 512 (default), up to 1024 |
| `num_layers` | 50 | 10 (default), up to 16 |
| `num_attention_heads × attention_head_dim` | 56×128 = 7168 | 8×64 = 512 |
| `ffn_dim` | 14336 | 1408 |
| `time_embed_dim` | 2688 | 256 |
| `num_refiner_layers` | 2 | **unchanged** |
| `text_dim` | 5120 (Qwen3-VL) | 1024 (Qwen3-0.6B, frozen) |
| `in_channels` / `audio_in_channels` / `patch_size` | 24 / 32 / (1,2,2) | **unchanged** — shares the official VAE latents |
| `rope_freq_dim` / `freq_dim` / `rope_theta` | 16 / 256 / 10000 | **unchanged** |

| Preset | hidden | layers | heads×dim | FFN | Params | Params + optimizer | Share of official |
|---|---|---|---|---|---|---|---|
| `smoke` | 256 | 6 | 4×64 | 704 | **7.3 M** | ~0.1 GiB | 0.03 % |
| `tiny_h3_5090` (default) | 512 | 10 | 8×64 | 1408 | **47.6 M** | ~0.5 GiB | 0.20 % |
| `tiny_h3_xl` | 768 | 14 | 12×64 | 2048 | **140.3 M** | ~1.6 GiB | 0.57 % |
| `tiny_h3_5090_max` | 1024 | 16 | 16×64 | 2816 | **285.1 M** | ~3.2 GiB | 1.17 % |

Frozen components, never trained: video VAE 10.4 GB, audio VAE 0.6 GB, text encoder (Qwen3-0.6B ≈ 1.2 GB
bf16, Qwen3-0.6B 1.2 GB). The official alternative, Qwen3-VL-8B, is 66 GB — which is why the conditioner is
swapped for a 0.x B Qwen.

> In other words: **VRAM is not the bottleneck**; the 5090 has room for much larger DiTs. Data volume and
> wall-clock run out first. With the 512-clip synthetic set, `tiny_h3_5090` is plenty; to raise the quality
> ceiling, add real data and/or longer clips and higher resolution before switching to a bigger preset.

## Data grids

| Canvas | Frames | FPS | Video latents | Audio latents | Packed sequence |
|---|---|---|---|---|---|
| 256×256 | 22 (`17×1+5`) | 24 | `24×7×16×16` | `2×32×37` | 48 + 74 + 448 = 570 |
| 384×384 | 22 | 24 | `24×7×24×24` | `2×32×37` | 48 + 74 + 1008 = 1130 |
| 256×256 | 39 (`17×2+5`) | 24 | `24×12×16×16` | `2×32×65` | 48 + 130 + 768 = 946 |

(Text rows are fixed at 48: prompts are padded to a constant length so a whole batch shares one packed
layout. Change it with `tools/prepare_latents.py --text-tokens`.)

## Memory and time (RTX 5090 32 GB, bf16 + gradient checkpointing)

| Stage | VRAM | Time |
|---|---|---|
| `prepare_latents.py` (official VAEs, fp32) | ~14 GB | ~1–2 s per clip |
| Full training (`tiny_h3_5090`, batch 2 × accum 4, 570 tokens) | ~6–8 GB | ~1–2 it/s |
| LoRA training | ~6 GB | same speed (identical forward) |
| FSDP, single GPU (`NO_SHARD`) | ≈ full training | small wrapping overhead |
| Flow-GRPO (group 4, 12 sampling steps) | ~10–12 GB (VAEs resident) | 30–60 s per rollout |
| Inference (24 steps, 22 frames, 256×256) | ~10 GB | 5–15 s per clip |

> If VRAM is tight: `--batch-size 1` → `--vae-device cpu` (GRPO/generation) → `--preset smoke`.
> Whenever resolution or frame count changes, keep the two grid rules: multiples of 32, and `17n+5` frames.

## How many steps?

| Goal | Data | Steps |
|---|---|---|
| Plumbing check (loss goes down) | 64 clips | 20–50 |
| Recognizable colours/shapes (blurry but readable) | 512 clips | 1–2 k |
| Clean demo (crisp shapes, synced sound) | 512 clips | 4–8 k |
| Visible Flow-GRPO reward gain | 64 validation prompts | 50–200 rollouts |
