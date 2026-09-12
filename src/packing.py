"""Packed-sequence layout for Tiny-H3 (text + video + audio in one transformer).

The layout mirrors the reference MiniMax-H3 port in ``diffusers``
(``modular_pipelines/minimax_h3/before_denoise.py``); Tiny-H3 only drops the
image/video/reference conditioning blocks that the t2va task does not use.

One packed 1-D sequence, in this exact order::

    [ text rows | audio rows | video rows ]
      tag 1        tag 2        tag 0

* **text rows** -- one row per text token, ``position_ids = (token_index, 0, 0)``.
* **audio rows** -- channel-major: all latents of channel 0, then all of channel 1.
  ``position_ids = (text_len + latent_index, 0, width)`` with ``width`` pinned to the
  two extremes of the video's width grid (that is how the model tells the channels apart).
* **video rows** -- frame-major, row-major inside a frame, ``patch_size`` merged into the
  channel axis.  ``position_ids = (text_len + rotary_time(frame), h, w)``.

The time axis is shared: one unit is one audio latent (40 latents/s at 32 kHz) which equals
``24 fps * 5/3``, so video latent frames advance by ``5/3 * (1, 4, 4, 4, 4)`` units --
the non-uniform pattern that mirrors the VAE's 17-pixel-frames-to-5-latent-frames grouping.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch

VIDEO_TAG = 0
TEXT_TAG = 1
AUDIO_TAG = 2

ROPE_FRAME_RESCALE = 5.0 / 3.0
ROPE_FRAMES_PER_LATENT = (1, 4, 4, 4, 4)
ROPE_SPATIAL_SCALE = 32.0


def patchify_video_latents(latents: torch.Tensor, patch_size: tuple[int, int, int]) -> torch.Tensor:
    """``(B, C, T, H, W)`` latents -> ``(B * num_patches, C * pt * ph * pw)`` rows, frame-major then row-major."""
    patch_t, patch_h, patch_w = patch_size
    batch_size, channels, num_frames, height, width = latents.shape
    if num_frames % patch_t or height % patch_h or width % patch_w:
        raise ValueError(f"Latents of shape {tuple(latents.shape)} are not divisible by the patch {patch_size}.")
    rows = latents.reshape(
        batch_size,
        channels,
        num_frames // patch_t,
        patch_t,
        height // patch_h,
        patch_h,
        width // patch_w,
        patch_w,
    )
    rows = rows.permute(0, 2, 4, 6, 1, 3, 5, 7)
    return rows.reshape(-1, channels * patch_t * patch_h * patch_w).contiguous()


def unpatchify_video_latents(
    rows: torch.Tensor,
    patch_size: tuple[int, int, int],
    channels: int,
    num_frames: int,
    height: int,
    width: int,
) -> torch.Tensor:
    """Inverse of :func:`patchify_video_latents`, returning ``(1, C, T, H, W)``."""
    patch_t, patch_h, patch_w = patch_size
    num_frames_lat = num_frames // patch_t
    height_lat = height // patch_h
    width_lat = width // patch_w
    expected = num_frames_lat * height_lat * width_lat
    if rows.shape[0] != expected:
        raise ValueError(f"Expected {expected} rows, got {rows.shape[0]}.")
    rows = rows.reshape(num_frames_lat, height_lat, width_lat, channels, patch_t, patch_h, patch_w)
    latents = rows.permute(3, 0, 4, 1, 5, 2, 6)
    return latents.reshape(1, channels, num_frames, height, width).contiguous()


def spatial_position_grid(dim: int, patch: int, sqrt_area: float) -> torch.Tensor:
    """Aspect-normalized spatial rotary axis, centred and scaled to ``[0, 32)`` on a square canvas."""
    ratio = dim / sqrt_area
    left = (1.0 - ratio) / 2.0
    steps = dim // patch
    grid = (left + torch.arange(steps, dtype=torch.float64) * (ratio / steps)) * ROPE_SPATIAL_SCALE
    return grid


def frame_position_grid(latent_height: int, latent_width: int, patch_h: int, patch_w: int):
    """``(h, w)`` rotary coordinates of one latent frame plus the raw width grid."""
    sqrt_area = (latent_height * latent_width) ** 0.5
    height_grid = spatial_position_grid(latent_height, patch_h, sqrt_area)
    width_grid = spatial_position_grid(latent_width, patch_w, sqrt_area)
    grid_h, grid_w = torch.meshgrid(height_grid, width_grid, indexing="ij")
    return torch.stack([grid_h.reshape(-1), grid_w.reshape(-1)], dim=-1), width_grid


def temporal_position_grid(num_latent_frames: int, origin: float) -> torch.Tensor:
    """Rotary time of every latent frame: non-uniform ``5/3 * (1, 4, 4, 4, 4)`` spacing."""
    spans = torch.tensor(
        [ROPE_FRAME_RESCALE * ROPE_FRAMES_PER_LATENT[i % len(ROPE_FRAMES_PER_LATENT)] for i in range(num_latent_frames)],
        dtype=torch.float64,
    )
    return origin + torch.cat([torch.zeros(1, dtype=torch.float64), spans[:-1].cumsum(0)])


@dataclass
class PackedLayout:
    """Structural description of one packed sequence; shared by every batch item."""

    position_ids: torch.Tensor  # (seq_len, 3) float32
    token_tags: torch.Tensor  # (seq_len,) long
    video_indices: torch.Tensor  # (num_video_rows,) long
    audio_indices: torch.Tensor  # (num_audio_rows,) long
    text_indices: torch.Tensor  # (num_text_rows,) long
    num_text_tokens: int
    num_audio_latents: int
    audio_channels: int
    latent_frames: int
    latent_height: int
    latent_width: int
    rows_per_frame: int
    patch_size: tuple[int, int, int] = (1, 2, 2)
    in_channels: int = 24
    audio_channels_dim: int = 32

    @property
    def seq_len(self) -> int:
        return self.position_ids.shape[0]

    @property
    def patch_volume(self) -> int:
        return int(self.patch_size[0] * self.patch_size[1] * self.patch_size[2])

    @property
    def video_shape(self) -> tuple[int, int, int, int]:
        """``(C, T, H, W)`` of the video latents this layout was built for."""
        return (self.in_channels, self.latent_frames, self.latent_height, self.latent_width)

    @property
    def audio_shape(self) -> tuple[int, int, int]:
        """``(channels, C, T)`` of the audio latents this layout was built for."""
        return (self.audio_channels, self.audio_channels_dim, self.num_audio_latents)

    def row_timesteps(self, video_timestep: float, audio_timestep: float) -> tuple[torch.Tensor, torch.Tensor]:
        """Per-row timestep values reduced to the transformer's ``(unique, inverse)`` pair.

        Text rows inherit the video timestep -- they never reach an output head.
        """
        row_t = torch.full((self.seq_len,), float(video_timestep), dtype=torch.float32)
        row_t[self.audio_indices] = float(audio_timestep)
        return torch.unique(row_t, sorted=True, return_inverse=True)


def build_t2va_layout(
    num_text_tokens: int,
    num_audio_latents: int,
    latent_frames: int,
    latent_height: int,
    latent_width: int,
    patch_size: tuple[int, int, int] = (1, 2, 2),
    audio_channels: int = 2,
    in_channels: int = 24,
    audio_channels_dim: int = 32,
    text_token_tags: torch.Tensor | None = None,
) -> PackedLayout:
    """Build the t2va packed layout. All shapes are for **one** sample (batch axis is pure replication)."""
    _, patch_h, patch_w = patch_size
    rows_per_frame = (latent_height // patch_h) * (latent_width // patch_w)
    num_audio_rows = num_audio_latents * audio_channels
    num_video_rows = latent_frames * rows_per_frame
    seq_len = num_text_tokens + num_audio_rows + num_video_rows

    audio_start = num_text_tokens
    video_start = audio_start + num_audio_rows

    position_ids = torch.zeros(seq_len, 3, dtype=torch.float64)
    position_ids[:num_text_tokens, 0] = torch.arange(num_text_tokens, dtype=torch.float64)

    frame_grid, width_grid = frame_position_grid(latent_height, latent_width, patch_h, patch_w)

    # Audio rows: channel-major, one rotary unit per latent, pinned to the width extremes.
    audio_time = float(num_text_tokens) + torch.arange(num_audio_latents, dtype=torch.float64)
    position_ids[audio_start:video_start, 0] = audio_time.repeat(audio_channels)
    position_ids[audio_start:video_start, 2] = torch.cat(
        [
            torch.full((num_audio_latents,), float(width_grid[0]), dtype=torch.float64),
            torch.full((num_audio_rows - num_audio_latents,), float(width_grid[-1]), dtype=torch.float64),
        ]
    )

    # Video rows: frame-major, row-major.
    video_positions = torch.empty(latent_frames, rows_per_frame, 3, dtype=torch.float64)
    video_positions[:, :, 0] = temporal_position_grid(latent_frames, float(num_text_tokens))[:, None]
    video_positions[:, :, 1:] = frame_grid[None]
    position_ids[video_start:] = video_positions.reshape(-1, 3)

    video_indices = torch.arange(video_start, seq_len)
    audio_indices = torch.arange(audio_start, video_start)
    text_indices = torch.arange(num_text_tokens)

    token_tags = torch.empty(seq_len, dtype=torch.long)
    token_tags[text_indices] = TEXT_TAG if text_token_tags is None else text_token_tags.to(torch.long)
    token_tags[audio_indices] = AUDIO_TAG
    token_tags[video_indices] = VIDEO_TAG

    return PackedLayout(
        position_ids=position_ids.to(torch.float32),
        token_tags=token_tags,
        video_indices=video_indices,
        audio_indices=audio_indices,
        text_indices=text_indices,
        num_text_tokens=num_text_tokens,
        num_audio_latents=num_audio_latents,
        audio_channels=audio_channels,
        latent_frames=latent_frames,
        latent_height=latent_height,
        latent_width=latent_width,
        rows_per_frame=rows_per_frame,
        patch_size=tuple(patch_size),
        in_channels=in_channels,
        audio_channels_dim=audio_channels_dim,
    )


def pack_video_rows(video_latents: torch.Tensor, patch_size: tuple[int, int, int]) -> torch.Tensor:
    """``(B, C, T, H, W)`` -> ``(B, T*H*W/patch, C*pt*ph*pw)`` keeping the batch axis."""
    batch = video_latents.shape[0]
    rows = patchify_video_latents(video_latents, patch_size)
    return rows.reshape(batch, -1, rows.shape[-1])


def pack_audio_rows(audio_latents: torch.Tensor) -> torch.Tensor:
    """``(channels, C, T)`` -> ``(1, channels * T, C)``, channel-major.

    Matches H3's mono autoencoder carried as two batch items: each row is one latent step of
    one channel, all 32 codec channels kept as the row's feature axis.
    """
    channels, channels_dim, num_latents = audio_latents.shape
    rows = audio_latents.permute(0, 2, 1)  # (channels, T, C)
    return rows.reshape(1, channels * num_latents, channels_dim).contiguous()


def unpack_audio_rows(rows: torch.Tensor, channels: int, num_latents: int) -> torch.Tensor:
    """Inverse of :func:`pack_audio_rows`: ``(1, channels*T, C)`` -> ``(channels, C, T)``."""
    channels_dim = rows.shape[-1]
    out = rows.reshape(channels, num_latents, channels_dim).permute(0, 2, 1)
    return out.contiguous()
