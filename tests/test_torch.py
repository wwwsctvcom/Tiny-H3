"""Model-side checks: packing round-trips, layout invariants, the H3 schedule.

These import torch/diffusers, so they need a machine that can hold the framework in RAM
(and the model tests additionally need the H3 weights).
"""

from __future__ import annotations

import torch

from tiny_h3.packing import (
    build_t2va_layout,
    pack_audio_rows,
    pack_video_rows,
    unpack_audio_rows,
    unpatchify_video_latents,
)
from tiny_h3.sampler import build_schedule, euler_step
from tiny_h3.vae import video_latent_frames


def test_h3_frame_grid():
    assert video_latent_frames(22) == 7
    assert video_latent_frames(39) == 12
    assert video_latent_frames(56) == 17


def test_video_patch_round_trip():
    x = torch.arange(2 * 3 * 2 * 4 * 6).view(2, 3, 2, 4, 6).float()
    rows = pack_video_rows(x, (1, 2, 2))
    rebuilt = torch.cat([
        unpatchify_video_latents(rows[i], (1, 2, 2), 3, 2, 4, 6) for i in range(2)
    ])
    torch.testing.assert_close(rebuilt, x)


def test_audio_pack_round_trip():
    x = torch.arange(2 * 32 * 37).view(2, 32, 37).float()
    rows = pack_audio_rows(x)
    torch.testing.assert_close(unpack_audio_rows(rows, 2, 37), x)


def test_layout_counts_and_tags():
    layout = build_t2va_layout(48, 37, 7, 8, 8)
    assert layout.video_indices.numel() == 7 * 4 * 4
    assert layout.audio_indices.numel() == 2 * 37
    assert layout.text_indices.numel() == 48
    assert layout.seq_len == 48 + 74 + 112
    assert set(layout.token_tags.tolist()) == {0, 1, 2}
    timesteps, inverse = layout.row_timesteps(0.2, 0.7)
    # float32 can't hold 0.2 exactly, so compare with a tolerance rather than ==.
    torch.testing.assert_close(timesteps, torch.tensor([0.2, 0.7]))
    assert inverse.numel() == layout.seq_len


def test_h3_scheduler_endpoints():
    sigmas, timesteps = build_schedule(8, shift=12)
    assert sigmas[0] == 1
    assert sigmas[-1] == 0
    assert len(timesteps) == 8
    x = torch.randn(2, 3)
    v = torch.randn(2, 3)
    out = euler_step(v, x, sigma=1.0, sigma_next=0.0)
    torch.testing.assert_close(out, x + v)
