# Design notes

Why the code is written the way it is. Every formula follows the official MiniMax-H3 port in `diffusers`
(`modular_pipelines/minimax_h3/`, `models/transformers/transformer_minimax_h3.py`) and the H3 recipes in
miles-diffusion.

## 1. One transformer for three modalities

H3 has no cross-attention: everything is packed into **one 1-D sequence**.

```text
[ text rows (0..N-1) | audio rows (channel-major) | video rows (frame-major) ]
        tag = 1                 tag = 2                     tag = 0
```

* **Video rows** — latents `(24, T', H/16, W/16)` patchified with `patch_size=(1,2,2)` so each row carries
  `24*1*2*2 = 96` values; `(t,h,w)` positions come from `temporal_position_grid` and `frame_position_grid`
  (aspect-normalized, scaled by 32, so a square canvas spans `[0, 32)`).
* **Audio rows** — the audio VAE is **mono** and stereo is carried as two batch items. Rows are
  channel-major: the first `T_a` rows are the left channel, the next `T_a` the right, each row 32 values.
  The width coordinate is pinned to the two extremes of the video's width grid (that is how channels are
  distinguished), height is always 0.
* **Shared time axis** — one unit = one audio latent (40/s) = `24 fps × 5/3`. Video latent frames advance
  non-uniformly by `5/3 × (1,4,4,4,4)`, mirroring the VAE's 17-pixels-to-5-latents grouping.

`src/packing.py` produces `position_ids / token_tags / video_indices / audio_indices / text_indices`;
the model's `forward` scatters the three blocks into the packed buffer using those indices.

## 2. Timesteps and modulation

Each block looks up AdaLN parameters per `(timestep, modality)` pair: `adaln_index = timestep_index*3 + tag`.
Video and audio have their own sigma schedules (official `shift=12` / `3`), so one forward can serve rows at
different noise levels; text rows never reach an output head and inherit the video timestep.

Training samples sigmas from the same shifted distribution the sampler uses:

```text
sigma = shift * u / (1 + (shift-1)*u),  u ~ U(0,1)
x_t   = (1 - sigma) * x0 + sigma * noise
```

## 3. Loss: data-ward velocity

Official H3 predicts `v = x0 - noise` — the opposite sign of the usual flow-matching convention — and
recovers the clean sample as `x0_hat = x_t + sigma * v`.

```text
L = MSE(v_pred_video, x0_video - noise_video) + w * MSE(v_pred_audio, x0_audio - noise_audio)
```

`src/train/core.py:flow_matching_loss` is the single training loss; full, LoRA and FSDP differ only in
model wrapping and which parameters are trainable.

One implementation detail: H3's batch axis is a pure replication axis (all structural arguments are shared),
yet per-sample sigmas differ, so the loss runs one packed forward per sample. With a small model and short
sequences the cost is acceptable and the semantics stay exactly official.

## 4. Sampling: ODE (official scheduler) and SDE (Flow-GRPO)

* **ODE (inference default)** — diffusers' own `MiniMaxH3Scheduler`, two instances (`shift=12` video,
  `shift=3` audio), `set_timesteps(num_steps)` then repeated `step()`. Its update is `x0_hat = x + sigma*v`
  followed by the blend `x_next = r*x + (1-r)*x0_hat`, `r = sigma_next/sigma`.
* **SDE (Flow-GRPO rollouts)** — the deterministic scheduler exposes no per-step log-probability, so
  `src/sampler.py` adds Flow-GRPO's exploration step: `std_dev_t = sqrt(sigma/(1-sigma)) * eta`, with a
  drift term that adds the score while preserving the marginals:

```text
mean   = x * (1 + std^2/(2σ) * Δσ) - v * (1 + std^2 (1-σ)/(2σ)) * Δσ      # Δσ = σ_next - σ < 0
x_next = mean + std * sqrt(-Δσ) * N(0, I)
log p(x_next | x) = -||x_next - mean||^2 / (2 std^2 (-Δσ)) - log(std sqrt(-Δσ)) - 0.5 log 2π
```

At `eta=0` this degenerates to the official Euler step, so inference and rollout share one sigma grid and
one packed forward. `tiny_h3.pipeline` uses ODE; `train_flow_grpo.py` passes
`SDEConfig(noise_level=0.7)`.

## 5. Flow-GRPO

Equivalent to miles-diffusion's `loss_hub/flow_grpo.py` minus the distributed and serving layers:

1. sample `G` trajectories per prompt, recording `(x_t, x_next, log p_old)` at every step;
2. reward = `0.55 * prompt adherence + 0.30 * audio-visual sync + 0.15 * basic quality`
   (`src/rewards/`);
3. `A_i = (r_i - mean(r)) / (std(r) + eps)` — GRPO group normalization;
4. `ratio = exp(log p_new - log p_old)`,
   `L = -mean(max(A*ratio, A*clip(ratio, 1±eps)))`;
5. optional KL `((mean_new - mean_old)^2 / (2 std^2))` against the behaviour policy.

Video and audio log-probabilities are averaged, so **both modalities are optimized jointly**. (The miles H3
recipe trains the video branch only and leaves audio locked; since our reward includes audio-visual sync,
training both is the more meaningful setup.)

## 6. Why the rewards can be computed offline

The synthetic generator (`src/data/synth.py`) ties every attribute — colour, shape, count, motion, BPM,
timbre — to the prompt text. `src/rewards/prompt_match.py` parses the prompt and verifies those
attributes on the *generated* media with classical CV, so no reward model has to be downloaded and a
pretty-but-off-prompt sample cannot game the score. This is the synthetic-domain analogue of the OCR
verifier used in the SD3 recipe, and it is what makes the RL loop reproducible.

## 7. Differences from the official and reference implementations

| Aspect | Official H3 | Tiny-H3 |
|---|---|---|
| DiT | 50 layers, hidden 5376, 24.4 B params | presets: `smoke` 7.3 M / `tiny_h3_5090` 47.6 M / `xl` 140 M / `_5090_max` 285 M |
| Text encoder | Qwen3-VL-8B (66 GB, layer 50) | `Qwen/Qwen3-0.6B` frozen (1024-dim; Qwen2.5-0.5B or t5-small optional), `--text-layer` can mimic the intermediate-layer read |
| VAEs | official | **official, frozen**; optional decoder-only fine-tune in `tools/finetune_vae.py` |
| Packing / timesteps / loss sign | official | line-by-line aligned |
| Sampling | official `MiniMaxH3Scheduler` | inference uses that same diffusers class; RL adds the SDE with log-prob |
| RL infrastructure | SGLang + Ray + multi-node FSDP2 | single-process local rollout, PyTorch + diffusers only |

## 8. VAE: why it is not trained by default

The VAE sets the quality ceiling; the DiT only decides *what* to generate. The released VAEs were trained on
large-scale real video/audio, and freezing them is exactly why a 47 M DiT can decode to real, playable
pixels and sound. Retraining them on a small dataset lowers that ceiling, and all-parameter training of the
10.4 GB float32 video VAE needs roughly 40 GB of optimizer+gradient+weight memory — impossible on one 32 GB
card.

If your domain is genuinely far from natural video/audio (special sensors, extreme colour spaces, unusual
bandwidth) and `tools/reconstruct_data.py` shows clear degradation, the compromise implemented in
`tools/finetune_vae.py` is: keep the **encoder frozen** (latents are precomputed), train **only the
decoder** in bf16 with gradient checkpointing (~16 GB). The cost: you must re-encode the latent cache and
re-train the DiT, and you lose latent compatibility with the official checkpoint.
