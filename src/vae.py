"""Thin wrappers around MiniMax-H3's official VAEs.

Tiny-H3 trains a small DiT but keeps the *released* H3 autoencoders frozen, so its latents
live in the same space as the 66 GB checkpoint and the demo is decoded by a VAE trained on
real video/audio.  Everything here mirrors the reference pipeline exactly:

* video pixels are ImageNet-normalized before the encoder, the posterior is **sampled** with
  a fixed seed and rounded through float16, then normalized per channel with
  ``latents_mean`` / ``latents_std``;
* audio is mono with the two stereo channels carried as two batch items, the encoder's
  posterior **mean** is used (never sampled), then normalized the same way;
* ``17 * n + 5`` pixel frames map to ``5 * n + 2`` video latent frames.

Keeping these conventions matters: a mismatched latent space decodes to noise even when the
model itself trained fine.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch

VIDEO_LATENT_CHANNELS = 24
AUDIO_LATENT_CHANNELS = 32
AUDIO_SAMPLE_RATE = 32000
AUDIO_HOP_LENGTH = 800  # 32 kHz / 800 -> 40 audio latents per second
AUDIO_LATENTS_PER_SECOND = AUDIO_SAMPLE_RATE / AUDIO_HOP_LENGTH
PIXEL_MEAN = (0.485, 0.456, 0.406)
PIXEL_STD = (0.229, 0.224, 0.225)
VIDEO_CLIP_LENGTH = 17
VIDEO_TOKEN_DROP = 3


@dataclass
class VAEBundle:
    """The two frozen autoencoders plus the conventions they need."""

    video: torch.nn.Module
    audio: torch.nn.Module
    device: torch.device

    @property
    def video_latent_channels(self) -> int:
        return int(self.video.config.latent_channels)

    @property
    def audio_latent_channels(self) -> int:
        return int(self.audio.config.latent_channels)

    @property
    def video_spatial_compression(self) -> int:
        return int(torch.tensor(self.video.config.spatial_downsample_factors).prod().item())


def video_latent_frames(num_frames: int) -> int:
    """``17n + 5`` pixel frames -> ``5n + 2`` latent frames (H3's video grid)."""
    if num_frames < 5 or (num_frames - 5) % VIDEO_CLIP_LENGTH != 0:
        raise ValueError(f"num_frames must be 17n+5 (22, 39, 56, ...), got {num_frames}")
    return 5 * ((num_frames - 5) // VIDEO_CLIP_LENGTH) + 2


def audio_latent_frames(num_samples: int) -> int:
    return -(-int(num_samples) // AUDIO_HOP_LENGTH)  # ceil division, matching the VAE's right pad


def num_samples_for_frames(fps: float, num_frames: int, sample_rate: int = AUDIO_SAMPLE_RATE) -> int:
    return int(round(num_frames / fps * sample_rate))


def load_vaes(model_path: str, device: str | torch.device = "cpu", video_dtype: torch.dtype = torch.float32) -> VAEBundle:
    """Load both H3 VAEs.  ``model_path`` is a local snapshot dir or a HF repo id.

    The video VAE is kept in float32 (its modules are pinned by ``_keep_in_fp32_modules``);
    only the device and, for speed on GPU, autocast around the calls should vary.
    """
    from diffusers import AutoencoderKLMiniMaxH3, AutoencoderKLMiniMaxH3Audio

    device = torch.device(device)
    video = AutoencoderKLMiniMaxH3.from_pretrained(model_path, subfolder="vae", torch_dtype=video_dtype)
    audio = AutoencoderKLMiniMaxH3Audio.from_pretrained(model_path, subfolder="audio_vae", torch_dtype=torch.float32)
    video = video.to(device).eval()
    audio = audio.to(device).eval()
    video.requires_grad_(False)
    audio.requires_grad_(False)
    return VAEBundle(video=video, audio=audio, device=device)


def _norm_stats(config, channels: int, device, ndim: int) -> tuple[torch.Tensor, torch.Tensor]:
    mean = torch.tensor(list(config.latents_mean), device=device, dtype=torch.float32)
    std = torch.tensor(list(config.latents_std), device=device, dtype=torch.float32)
    shape = (1, -1, 1, 1, 1) if ndim == 5 else (1, -1, 1)
    return mean.view(*shape), std.view(*shape)


@torch.no_grad()
def encode_video(bundle: VAEBundle, pixels: torch.Tensor, seed: int = 42, sample: bool = True) -> torch.Tensor:
    """``(B, 3, T, H, W)`` pixels in ``[0, 1]`` (uint8 ``0..255`` also accepted) -> normalized latents."""
    vae = bundle.video
    if pixels.dtype == torch.uint8:
        pixels = pixels.float().div(255.0)
    else:
        pixels = pixels.float()
        if float(pixels.max()) > 1.0 + 1e-3:
            raise ValueError(
                "encode_video expects pixels in [0, 1] or uint8 in 0..255; got a float tensor with "
                f"max={float(pixels.max()):.1f}. Divide by 255 (or pass uint8) before calling."
            )
    pixels = pixels.to(device=bundle.device, dtype=torch.float32)
    mean = torch.tensor(PIXEL_MEAN, device=pixels.device).view(1, -1, 1, 1, 1)
    std = torch.tensor(PIXEL_STD, device=pixels.device).view(1, -1, 1, 1, 1)
    pixels = (pixels - mean) / std

    posterior = vae.encode(pixels, return_dict=False)[0]
    if sample:
        # `posterior.sample` needs a generator on the posterior's own device (diffusers'
        # randn_tensor rejects a CPU generator for CUDA tensors).
        generator = torch.Generator(device=posterior.mean.device).manual_seed(seed)
        latents = posterior.sample(generator=generator)
        # The reference rounds conditioning latents through float16 before normalizing.
        latents = latents.to(torch.float16).float()
    else:
        latents = posterior.mode()
    lat_mean, lat_std = _norm_stats(vae.config, VIDEO_LATENT_CHANNELS, latents.device, ndim=5)
    return ((latents - lat_mean) / lat_std).to(torch.float32)


@torch.no_grad()
def decode_video(bundle: VAEBundle, latents: torch.Tensor, autocast_fp16: bool = True) -> torch.Tensor:
    """Normalized latents -> ``(B, 3, T, H, W)`` pixels in ``[0, 1]``."""
    vae = bundle.video
    lat_mean, lat_std = _norm_stats(vae.config, latents.shape[1], latents.device, ndim=5)
    latents = latents.float() * lat_std + lat_mean
    use_amp = autocast_fp16 and latents.device.type == "cuda"
    with torch.autocast(device_type=latents.device.type, dtype=torch.float16, enabled=use_amp):
        video = vae.decode(latents, return_dict=False)[0]
    video = video.float()
    mean = torch.tensor(PIXEL_MEAN, device=video.device).view(1, -1, 1, 1, 1)
    std = torch.tensor(PIXEL_STD, device=video.device).view(1, -1, 1, 1, 1)
    return (video * std + mean).clamp(0.0, 1.0)


@torch.no_grad()
def encode_audio(bundle: VAEBundle, wave: torch.Tensor) -> torch.Tensor:
    """``(2, N)`` stereo waveform -> normalized latents ``(2, 32, T)`` (posterior mean)."""
    vae = bundle.audio
    wave = wave.to(device=bundle.device, dtype=torch.float32)
    if wave.dim() == 2:
        wave = wave[:, None, :]  # (channels, 1, samples) -- the two channels are batch items
    posterior = vae.encode(wave, return_dict=False)[0]
    latents = posterior.mode().float()  # (channels, 32, T)
    lat_mean, lat_std = _norm_stats(vae.config, AUDIO_LATENT_CHANNELS, latents.device, ndim=3)
    return ((latents - lat_mean) / lat_std).contiguous()


@torch.no_grad()
def decode_audio(bundle: VAEBundle, latents: torch.Tensor) -> torch.Tensor:
    """Normalized latents ``(2, 32, T)`` -> stereo waveform ``(2, N)``."""
    vae = bundle.audio
    lat_mean, lat_std = _norm_stats(vae.config, latents.shape[1], latents.device, ndim=3)
    latents = latents.float() * lat_std + lat_mean
    decoded = vae.decode(latents, return_dict=False)[0]  # (channels, 1, samples)
    return decoded.float().squeeze(1).clamp(-1.0, 1.0)
