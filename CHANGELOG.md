# Changelog

## 0.1.0 (2026-09-12)

Initial public layout.

* Procedural text-to-audio-video data generator (512 train / 64 val clips, frame-accurate sound).
* Latent cache from the official frozen MiniMax-H3 video/audio VAEs.
* Four training paths over one flow-matching loss: full fine-tune, LoRA, single/multi-GPU FSDP,
  and single-GPU Flow-GRPO with offline rewards.
* Text → latents → official VAEs → MP4 pipeline (H.264 + AAC stereo).
* Acceptance tools: VAE round-trip reconstruction, smoke test, clip inspector.
* Package sources live flat under `src/` with the `tiny_h3` import name kept via `package-dir`.
