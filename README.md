# Tiny-H3

**A small, complete text-to-audio-video (T2AV) generation project that trains and generates on a single RTX 5090.**

Tiny-H3 is a scaled-down, reproducible learning project built on top of
[MiniMax-H3](https://huggingface.co/MiniMaxAI/MiniMax-H3) — an omni-modal diffusion transformer that
denoises text, video and audio in one packed sequence. Everything model-side comes from
[`diffusers`](https://github.com/huggingface/diffusers): the H3 transformer, the real video VAE, the real
32 kHz audio VAE, the scheduler and the LoRA loading paths.

What you get end to end: procedural audio-visual data → latent cache from the official H3 VAEs →
train a small DiT (full / LoRA / FSDP / Flow-GRPO) → sample → decode → **a playable MP4 with a real
synchronized soundtrack**.

```text
prompt ──► frozen Qwen3-0.6B ──┐
                               ├──► Tiny-H3 DiT (trainable, ~48M)  ──► sample ──► frozen official VAEs ──► MP4 (H.264 + AAC stereo)
latents (official H3 VAE) ─────┘
```

| | Official MiniMax-H3 | Tiny-H3 |
|---|---|---|
| DiT | 24.4 B params, hidden 5376, 50 layers | **47.6 M** default preset (512 / 10 layers), up to 285 M |
| Text encoder | Qwen3-VL-8B (66 GB) | **Qwen3-0.6B** frozen (1024-dim), or Qwen2.5-0.5B / t5-small |
| Video / audio VAE | official | **official, frozen** (10.4 GB / 0.6 GB) — this is what makes real output possible on one GPU |
| Latent space, packing, timesteps, flow sign | — | unchanged: `in_channels=24`, `audio_in_channels=32`, patch `(1,2,2)`, `v = x0 - noise` |
| RL infrastructure | SGLang + Ray + multi-node FSDP2 | single process, PyTorch + diffusers only |

Design details live in [`docs/design.md`](docs/design.md); sizing/resource tables in
[`docs/presets.md`](docs/presets.md).

---

## Status

| Part | State |
|---|---|
| Asset download, data generator, media IO, rewards | verified (runs on CPU) |
| Packing / scheduler / sampler / model construction / trainer wiring | statically checked; needs one GPU run to confirm end to end |
| Training (full, LoRA, FSDP, Flow-GRPO) and generation | **requires a CUDA GPU** — run the acceptance steps below |

The acceptance ladder, cheapest first:

```bash
python tools/reconstruct_data.py --data-dir $TINY_H3_DATA/synth --out outputs/reconstruction --count 4   # official VAEs round-trip real data
python tools/smoke_test.py --latents $TINY_H3_DATA/synth/latents --with-vae --preset smoke               # plumbing + a real mp4
bash scripts/run_all.sh                                                                                   # train + generate demos
```

---

## Requirements

| | Minimum | Notes |
|---|---|---|
| GPU | 1× RTX 5090 (32 GB) | Blackwell sm_120 needs a CUDA 12.8+ torch build |
| System RAM | 32 GB | 64 GB recommended (VAE encode/decode in float32) |
| Disk | 60 GB | H3 VAEs ~11 GB + latent cache ~1–2 GB + checkpoints |
| Python | 3.10+ | torch, diffusers (pinned commit), transformers, peft, accelerate |

Everything is stored under `/root/autodl-tmp` by default (`scripts/env.sh`), so an AutoDL instance
reboot keeps the weights and the latent cache.

---

## Quick start

```bash
git clone <this repo> /root/autodl-tmp/tiny-h3 && cd /root/autodl-tmp/tiny-h3

# 1. environment (installs deps + pinned diffusers + this package, then prints a sanity check)
bash scripts/setup_env.sh

# 2. model downloads: H3 configs, video VAE (10.4 GB), audio VAE (0.6 GB), Qwen3-0.6B, t5-small
source scripts/env.sh
bash scripts/download_assets.sh

# 3. data: 512 train + 64 val procedural clips, then encode them with the official VAEs
bash scripts/prepare_data.sh synth
bash scripts/prepare_data.sh latents        # GPU

# 4. train (defaults: 47.6M DiT, batch 2 × accum 4, bf16 + gradient checkpointing)
python -m tiny_h3.train.train_full --latents $TINY_H3_DATA/synth/latents --out runs/full --steps 4000

# 5. generate
bash scripts/generate_demo.sh runs/full/final
# -> outputs/demo_*/00_a_red_circle_bouncing_on_a_dark_background_with_rhythmic_thumps.mp4  (+3 more)
```

Or the whole loop in one command:

```bash
bash scripts/run_all.sh          # assets → data → latents → full training → demo   (~1–2 h on a 5090)
bash scripts/run_all.sh --fast   # smoke-sized version, ~15 min, asserts the plumbing
```

---

## Model downloads

`bash scripts/download_assets.sh` (also callable as
`python tools/download_assets.py --group model|data|all`) fetches:

| Asset | Size | Why it is needed |
|---|---|---|
| `MiniMaxAI/MiniMax-H3` configs / docs / tokenizer | ~40 MB | class configs, packing constants, reference prompt guides |
| `MiniMaxAI/MiniMax-H3` → `vae/` | **10.4 GB** | official real-video decoder (frozen) |
| `MiniMaxAI/MiniMax-H3` → `audio_vae/` | **0.6 GB** | official 32 kHz stereo audio codec (frozen) |
| `Qwen/Qwen3-0.6B` | ~1.2 GB | frozen text conditioner (default; 1024-dim) |
| `t5-small` | ~240 MB | lightweight alternative conditioner |
| *(optional)* `rockdu/WISA-80K-Practical-Dynamics-254` | 1.1 GB | 254 real clips on H3's serving grid |
| *(optional)* `concerts_audiovideo_dataset` subset | ~1 GB | real footage **with** audio |

The 66 GB H3 DiT and the 66 GB Qwen3-VL-8B text encoder are **deliberately not downloaded** —
Tiny-H3 trains its own small DiT and uses a 0.x B Qwen for conditioning.

Downloads go through `hf-mirror.com` by default (measured ~40× faster than the academic proxy here),
with Xet disabled because the mirror rejects its handshake. Override with `HF_ENDPOINT` / `HF_HOME`.

If the mirror throttles a single connection, `tools/fetch_file.py` splits the file into ranges and
downloads them concurrently, resuming from a partial file and verifying the whole-file sha256::

    python tools/fetch_file.py --url <resolve url> --out <partial file> --workers 10 --sha256 <digest>

(HuggingFace LFS blob filenames *are* their sha256, so the digest comes free from the cache path.)

---

## Data

### Default: procedural clips (no dataset download)

```bash
bash scripts/prepare_data.sh synth      # 512 train + 64 val
bash scripts/prepare_data.sh latents    # official VAEs + Qwen → latent cache (GPU, minutes)
```

`src/data/synth.py` renders four scene families with **frame-accurate sound** and emits a prompt that
describes every attribute (colour, shape, count, motion, sound, tempo):

| Motion | Sound | Example prompt |
|---|---|---|
| `bouncing` | pitched impact on each contact | *a red circle bouncing on a dark background, with rhythmic thumps* |
| `pulsing` | kick on a BPM grid + pad | *two squares … pulsing … with steady beats at a fast tempo* |
| `orbiting` | tone gliding with height | *three rings … orbiting … with a rising and falling tone* |
| `swinging` | tick at each extreme + whoosh | *a white ring swinging … with soft ticks* |

Because the prompt is machine-generated, the Flow-GRPO rewards can verify it offline with classical CV
(`src/rewards/`) — no reward model needs to be downloaded.

Inspect any clip before training:

```bash
python tools/inspect_clip.py data/synth/clips/000002.mp4 --fps 24 --size 256
# frames, loudness, audio onsets vs motion peaks, and their median alignment distance
```

### Optional: your own footage

Any folder with a `train.jsonl` of `{"prompt": "...", "metadata": {"video": "clips/x.mp4"}}` works: run
`tools/prepare_latents.py --data-dir <folder>`. Frame counts must sit on the `17n+5` grid (22, 39, 56, …)
and canvas dimensions must be multiples of 32, otherwise the VAE pads and the audio/video clock drifts.

---

## Training

Four paths share **one** flow-matching loss (`src/train/core.py`), one dataset and one DiT — they only
differ in what is trainable and how the process is launched.

### Full fine-tune

```bash
python -m tiny_h3.train.train_full --latents $TINY_H3_DATA/synth/latents --out runs/full --steps 4000
```

* default preset `tiny_h3_5090` = 47.6 M params; params + optimizer states ≈ 0.5 GiB, so VRAM is never the
  constraint — raise `--batch-size` freely. Bigger presets: `tiny_h3_xl` (140 M), `tiny_h3_5090_max` (285 M).
* logs: console + `runs/full/metrics.jsonl` (`loss`, `video_loss`, `audio_loss`, `lr`, `grad_norm`, `max_cuda_gb`)
* checkpoints: diffusers-format directories (`--save-every`), loadable with `from_pretrained`

### LoRA

```bash
python -m tiny_h3.train.train_lora --latents $TINY_H3_DATA/synth/latents --out runs/lora --rank 32 --lr 3e-4
python -m tiny_h3.train.train_lora --latents $TINY_H3_DATA/synth/latents --out runs/lora_ft --base runs/full/final
```

Targets match the official H3 recipe: `attn.to_q`, `attn.to_k`, `attn.to_v`, `attn.to_out.0`,
`ff.net.0.proj`, `ff.net.2`. Without `--base` the seed-fixed base weights are saved alongside the adapter.

### FSDP (works on one GPU, shards on many)

```bash
torchrun --standalone --nproc_per_node=1 -m tiny_h3.train.train_fsdp --latents $TINY_H3_DATA/synth/latents --out runs/fsdp
torchrun --standalone --nproc_per_node=4 -m tiny_h3.train.train_fsdp --latents $TINY_H3_DATA/synth/latents --out runs/fsdp4
```

Single GPU uses `NO_SHARD` (nothing to distribute) but still exercises the full FSDP wrap/save path;
multi-GPU switches to `FULL_SHARD` automatically. Checkpoints are gathered on rank 0 into plain diffusers
directories, so inference code never sees FSDP.

### Flow-GRPO (online RL, single GPU, no serving engine)

```bash
python -m tiny_h3.train.train_flow_grpo \
  --base runs/full/final \
  --prompt-data $TINY_H3_DATA/synth/val.jsonl \
  --out runs/grpo --rollouts 100 --group-size 4 --sample-steps 12
```

The whole algorithm is in one readable file: reverse-SDE rollout with log-probabilities → decode with the
official VAEs → offline rewards (prompt adherence 0.55 + audio-visual sync 0.30 + basic quality 0.15) →
GRPO group-normalized advantages → PPO-clipped update on replayed transitions with an optional KL to the
behaviour mean. The policy is a LoRA adapter on the frozen full checkpoint, which keeps the single-GPU
memory budget comfortable.

Metrics per rollout land in `runs/grpo/metrics.jsonl` (`reward_mean/std/min/max`, `policy_loss`, `kl`,
`ratio`, `clipfrac`), with adapters saved every `--save-every` rollouts.

---

## Generation and demos

```bash
bash scripts/generate_demo.sh runs/full/final                       # 4 preset prompts
python -m tiny_h3.pipeline --checkpoint runs/full/final \
  --prompts "a cyan ring orbiting on a dark background, with a rising and falling tone" \
  --out outputs/demo --steps 24 --frames 22 --height 256 --width 256
```

Output: standard MP4 (`H.264` + `AAC` **stereo**), plus `manifest.json` next to the videos.
`--checkpoint` accepts full/FSDP checkpoints and GRPO/LoRA adapter directories.

### Demo gallery

| Prompt | Video |
|---|---|
| *a red circle bouncing on a dark background, with rhythmic thumps* | `outputs/demo_*/00_*.mp4` — place a sample under `assets/demos/` and link it here |
| *two squares, one cyan and one yellow, pulsing … with steady beats at a fast tempo* | `outputs/demo_*/01_*.mp4` |
| *three rings in purple, white and yellow, orbiting … with a rising and falling tone* | `outputs/demo_*/02_*.mp4` |

> Maintainers: after the first successful run, copy 2–3 clips into `assets/demos/` (a few hundred KB each)
> and replace the table rows with links; keep them small so the repository stays cloneable.

### What "good" looks like

* **Colours and shapes match the prompt** (the reward measures this, so training curves show it directly).
* **Motion pattern matches** (bouncing / pulsing / orbiting / swinging).
* **Sound is synchronized** with the events — impacts land on contacts, beats on beats.
* Resolution/frame count are whatever you trained on (256×256 / 22 frames ≈ 0.9 s by default).

Audio-visual content is procedural by default, so expect clean geometric animation with real, synced
sound. Train on real footage (see *Optional: your own footage*) to move toward photographic content; the
ceiling then becomes data volume and DiT capacity, not the pipeline.

---

## Project layout

```text
src/           the tiny_h3 package (import name kept via pyproject package-dir)
  packing.py   H3 packed sequence (text | audio | video rows, positions, modality tags, row timesteps)
  model.py     tiny DiT presets, Qwen/T5 conditioner, checkpoint save/load
  vae.py       official H3 video/audio VAE wrappers (encode, decode, latent normalisation)
  sampler.py   sigma schedules, ODE (diffusers MiniMaxH3Scheduler), reverse-SDE step with log-prob
  pipeline.py  text → latents → official VAEs → MP4 (CLI)
  media.py     mp4/wav IO, grid fitting, ffmpeg wrapper
  data/        procedural AV generator, latent-cache dataset
  rewards/     prompt-adherence and audio-visual-sync rewards (numpy only)
  train/       core.py + train_full / train_lora / train_fsdp / train_flow_grpo
tools/         download_assets, fetch_file, make_synth_data, prepare_latents, reconstruct_data,
               finetune_vae, inspect_clip, smoke_test
scripts/       env.sh, setup_env.sh, download_assets.sh, prepare_data.sh, run_all.sh, generate_demo.sh
docs/          design.md (packing, flow sign, SDE/GRPO maths, VAE rationale), presets.md (sizing tables)
tests/         test_data.py (numpy only), test_torch.py (packing/scheduler)
```

---

## FAQ

**Why `python -m tiny_h3.…` instead of `python src/….py`?**
Package modules use relative imports; `-m` provides the package context. The import name stays
`tiny_h3` even though the sources live flat under `src/` — pyproject wires that up with
`package-dir`, which `scripts/setup_env.sh` installs via `pip install -e .`, giving you
`tiny-h3-full`, `tiny-h3-generate`, etc. The `tools/*.py` scripts also run straight from a
checkout (they map the package name themselves).

**Is the VAE trained?**
No, and by default it should stay frozen — it is what makes real output possible on one GPU. All-parameter
training would need ~40 GB for the 10.4 GB float32 VAE, and it would invalidate every latent cache.
If your domain is far from natural video (and `tools/reconstruct_data.py` shows clear degradation),
`tools/finetune_vae.py` fine-tunes **only the decoder** with cached encoder latents (~16 GB); you must then
re-encode and re-train the DiT.

**Which text encoder should I use?**
`Qwen/Qwen3-0.6B` (default) keeps MiniMax-H3's Qwen lineage at 0.x B scale. `--text-layer N` conditions on
an intermediate hidden state, mirroring how the official model reads layer 50 instead of the last one.
Switching encoder changes `text_dim`, so re-run `tools/prepare_latents.py` (the trainer refuses mismatched
caches). `t5-small` is the lightest option for CPU smoke tests.

**Downloads are slow / HuggingFace is unreachable?**
`scripts/env.sh` defaults to `hf-mirror.com` with Xet disabled. For GitHub steps use
`source /etc/network_turbo` (AutoDL).

**Training is stable but samples look like noise.**
Check the flow sign first: H3 predicts `v = x0 - noise` (`x0_hat = x + sigma·v`), the opposite of the common
convention. Then check the VAE normalisation (`latents_mean/std`, ImageNet pixel stats) — both are handled
in `src/vae.py`, so a mismatch usually means hand-rolled code drifting from it.

---

## License and acknowledgements

Project code: **Apache-2.0** (see [`LICENSE`](LICENSE)).

Standing on the shoulders of: [MiniMax-H3](https://huggingface.co/MiniMaxAI/MiniMax-H3) (models and VAEs,
subject to its own license) · [diffusers](https://github.com/huggingface/diffusers) (model classes,
scheduler, LoRA) · [miles-diffusion](https://github.com/radixark/miles_diffusion) and
[Flow-GRPO](https://github.com/yifan123/flow_grpo) (RL formulation) · optional datasets
`rockdu/WISA-80K-Practical-Dynamics-254` and `alejandroparedeslatorre/concerts_audiovideo_dataset`.
