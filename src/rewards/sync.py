"""Audio-visual synchronization reward (numpy only, runs on decoded media).

Two complementary measurements, because the synthetic domain contains both event-based sound
(bouncing thumps, beats, ticks) and continuous sound (an orbiting tone):

1. **onset vs motion** -- does audio energy spike when the picture moves?
2. **pitch vs height** -- does the tone rise when the object rises?

Each is a Pearson correlation of two per-frame envelopes; the reward is the better of the two,
clipped to ``[0, 1]`` so uncorrelated or silent clips score zero.
"""

from __future__ import annotations

import numpy as np

_EPS = 1e-8


def _pearson_at_lags(a: np.ndarray, b: np.ndarray, max_lag: int = 2) -> float:
    """Max Pearson correlation over lags ``-max_lag..max_lag`` (a small alignment slack)."""
    if a.size < 3 or b.size < 3 or a.std() < _EPS or b.std() < _EPS:
        return 0.0
    a = (a - a.mean()) / (a.std() + _EPS)
    b = (b - b.mean()) / (b.std() + _EPS)
    best = -1.0
    for lag in range(-max_lag, max_lag + 1):
        if lag < 0:
            x, y = a[-lag:], b[: len(b) + lag]
        elif lag > 0:
            x, y = a[: len(a) - lag], b[lag:]
        else:
            x, y = a, b
        if x.size < 3:
            continue
        best = max(best, float(np.mean(x * y)))
    return best


def per_frame_audio_energy(wave: np.ndarray, sample_rate: int, num_frames: int, fps: float) -> np.ndarray:
    """RMS per video frame plus its positive flux (how much new energy appeared)."""
    mono = wave.mean(axis=0)
    samples_per_frame = sample_rate / fps
    energy = np.zeros(num_frames, dtype=np.float64)
    for i in range(num_frames):
        start = int(i * samples_per_frame)
        stop = min(int((i + 1) * samples_per_frame), mono.size)
        if stop > start:
            energy[i] = float(np.sqrt(np.mean(mono[start:stop] ** 2)))
    flux = np.clip(np.diff(energy, prepend=energy[:1]), 0, None)
    return energy, flux


def per_frame_motion_energy(frames: np.ndarray) -> np.ndarray:
    """Mean absolute frame difference, one value per frame."""
    if len(frames) < 2:
        return np.zeros(len(frames), dtype=np.float64)
    gray = frames.astype(np.float32).mean(axis=-1)
    diff = np.abs(np.diff(gray, axis=0)).mean(axis=(1, 2))
    return np.concatenate([[0.0], diff]).astype(np.float64)


def per_frame_spectral_centroid(wave: np.ndarray, sample_rate: int, num_frames: int, fps: float) -> np.ndarray:
    """A cheap pitch proxy: spectral centroid of each video-frame's audio window."""
    mono = wave.mean(axis=0)
    samples_per_frame = sample_rate / fps
    window = 1024
    freqs = np.fft.rfftfreq(window, d=1.0 / sample_rate)
    centroids = np.zeros(num_frames, dtype=np.float64)
    for i in range(num_frames):
        start = int(i * samples_per_frame)
        chunk = mono[start : start + window]
        if chunk.size < window:
            chunk = np.pad(chunk, (0, window - chunk.size))
        if np.abs(chunk).max() < 1e-5:
            continue
        spectrum = np.abs(np.fft.rfft(chunk * np.hanning(window)))
        total = spectrum.sum()
        if total > _EPS:
            centroids[i] = float((spectrum * freqs).sum() / total)
    return centroids


def per_frame_height(frames: np.ndarray) -> np.ndarray:
    """Normalized 1 - (centroid row) of the foreground, per frame."""
    if len(frames) == 0:
        return np.zeros(0, dtype=np.float64)
    gray = frames.astype(np.float32).mean(axis=-1)
    background = np.median(gray, axis=0)
    out = np.zeros(len(frames), dtype=np.float64)
    height = gray.shape[1]
    for i in range(len(frames)):
        mask = np.abs(gray[i] - background) > 24.0
        if mask.sum() < 4:
            out[i] = 0.5
            continue
        rows = np.nonzero(mask.any(axis=1))[0]
        out[i] = 1.0 - float(rows.mean() / max(1, height - 1))
    return out


def av_sync_reward(frames: np.ndarray, wave: np.ndarray, fps: float, sample_rate: int) -> float:
    """``0..1``; higher means sound and picture line up in time."""
    num_frames = len(frames)
    if num_frames < 4 or wave.size == 0 or float(np.abs(wave).max()) < 1e-4:
        return 0.0
    _, flux = per_frame_audio_energy(wave, sample_rate, num_frames, fps)
    motion = per_frame_motion_energy(frames)
    onset_score = max(0.0, _pearson_at_lags(flux, motion))

    centroid = per_frame_spectral_centroid(wave, sample_rate, num_frames, fps)
    height = per_frame_height(frames)
    pitch_score = max(0.0, _pearson_at_lags(centroid, height))

    return float(np.clip(max(onset_score, pitch_score), 0.0, 1.0))
