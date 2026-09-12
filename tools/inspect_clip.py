#!/usr/bin/env python
"""Inspect one clip: grid, loudness, motion/onset alignment. Pure numpy, no torch.

Example::

    python tools/inspect_clip.py /tmp/synth_test/clips/000002.mp4 --fps 24 --size 96
"""

from __future__ import annotations

import argparse
import os
import sys

import numpy as np

from _bootstrap import bootstrap  # noqa: E402  (maps the tiny_h3 package name onto ../src)

from tiny_h3.media import AUDIO_SAMPLE_RATE, probe, read_audio_wave, read_video_frames  # noqa: E402


def onset_times(wave: np.ndarray, sample_rate: int, hop: int = 320, thresh: float = 0.35) -> list[float]:
    mono = wave.mean(axis=0)
    frame = 1024
    energies = np.array(
        [np.sqrt(np.mean(mono[i : i + frame] ** 2)) for i in range(0, max(1, len(mono) - frame), hop)]
    )
    if energies.size == 0 or energies.max() <= 1e-8:
        return []
    flux = np.diff(energies, prepend=energies[:1])
    flux = np.clip(flux, 0, None)
    if flux.max() <= 1e-8:
        return []
    peaks = (flux > thresh * flux.max()) & (flux >= np.roll(flux, 1))
    return [float(i * hop / sample_rate) for i in np.nonzero(peaks)[0]]


def motion_times(frames: np.ndarray, fps: float, thresh: float = 0.35) -> list[float]:
    if len(frames) < 2:
        return []
    gray = frames.astype(np.float32).mean(axis=-1)
    diff = np.abs(np.diff(gray, axis=0)).mean(axis=(1, 2))
    if diff.max() <= 1e-6:
        return []
    peaks = (diff > thresh * diff.max()) & (diff >= np.roll(diff, 1))
    return [float(i / fps) for i in np.nonzero(peaks)[0]]


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("clip")
    ap.add_argument("--fps", type=float, default=24.0)
    ap.add_argument("--size", type=int, default=128)
    args = ap.parse_args()

    info = probe(args.clip)
    print(f"file        : {args.clip}")
    print(f"probe       : video={info.get('has_video')} audio={info.get('has_audio')} duration={info.get('duration')}")

    frames = read_video_frames(args.clip, (args.size, args.size), fps=args.fps)
    wave = read_audio_wave(args.clip, sample_rate=AUDIO_SAMPLE_RATE, channels=2)
    print(f"frames      : {frames.shape}  dtype={frames.dtype}  max={frames.max()}")
    print(f"audio       : {wave.shape}  sr={AUDIO_SAMPLE_RATE}  rms={float(np.sqrt((wave**2).mean())):.4f} "
          f"peak={float(np.abs(wave).max()):.3f}")

    onsets = onset_times(wave, AUDIO_SAMPLE_RATE)
    motions = motion_times(frames, args.fps)
    print(f"audio onsets: {[round(t, 2) for t in onsets[:12]]}{' ...' if len(onsets) > 12 else ''}")
    print(f"motion peaks: {[round(t, 2) for t in motions[:12]]}{' ...' if len(motions) > 12 else ''}")
    if onsets and motions:
        a, m = np.array(onsets), np.array(motions)
        d = np.abs(a[:, None] - m[None, :]).min(axis=1)
        print(f"nearest audio->motion distance: median={np.median(d):.3f}s  <1 frame: {int((d < 0.9/args.fps).sum())}/{len(a)}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
