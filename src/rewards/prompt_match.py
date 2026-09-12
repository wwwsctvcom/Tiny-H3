"""Prompt-adherence reward for the synthetic domain (numpy only).

The prompts are templated (``tiny_h3.data.synth``), so the expected attributes can be parsed
straight out of the prompt text and *verified* on the decoded media with classical CV:

* colour names  -- nearest ``COLORS`` entry for the dominant saturated pixels;
* object count  -- connected components of the foreground mask;
* motion type   -- centroid/area heuristics (bounce / pulse / orbit / swing);
* tempo         -- median onset interval against the prompt's slow/medium/fast word;
* sound family  -- onset density and pitch variation against the prompt's sound phrase.

This plays the role the OCR verifier plays in the SD3 recipe: an offline, deterministic reward
that is meaningful for the data distribution actually being trained on.
"""

from __future__ import annotations

import re

import numpy as np

from ..data.synth import COLORS, TEMPOS

NUMBER_WORDS = {"a": 1, "an": 1, "one": 1, "two": 2, "three": 3}
MOTION_WORDS = ("bouncing", "pulsing", "orbiting", "swinging")
SOUND_FAMILIES = {
    "rhythmic thumps": "impacts",
    "steady beats": "beats",
    "a rising and falling tone": "tone",
    "soft ticks": "ticks",
}
_COLOR_NAMES = list(COLORS)
_RGB = np.array([COLORS[name] for name in _COLOR_NAMES], dtype=np.float32)


def parse_prompt(prompt: str) -> dict:
    lowered = prompt.lower()
    colors = [name for name in _COLOR_NAMES if re.search(rf"\b{name}\b", lowered)]
    count = 1
    match = re.search(r"\b(a|an|one|two|three)\b", lowered)
    if match:
        count = NUMBER_WORDS[match.group(1)]
    motion = next((m for m in MOTION_WORDS if m in lowered), None)
    tempo = next((word for word in TEMPOS if re.search(rf"\b{word}\b", lowered)), None)
    sound = next((family for phrase, family in SOUND_FAMILIES.items() if phrase in lowered), None)
    return {"colors": colors, "count": count, "motion": motion, "tempo": tempo, "sound": sound}


def _foreground_mask(frames: np.ndarray, threshold: float = 24.0) -> np.ndarray:
    gray = frames.astype(np.float32).mean(axis=-1)
    background = np.median(gray, axis=0)
    return np.abs(gray - background) > threshold


def _dominant_color_names(frames: np.ndarray, top_k: int = 3) -> list[str]:
    """Nearest named colour for the biggest saturated blobs (background excluded by saturation)."""
    pixels = frames.reshape(-1, 3).astype(np.float32)
    if pixels.size == 0:
        return []
    mx = pixels.max(axis=1)
    mn = pixels.min(axis=1)
    saturation = (mx - mn) / np.maximum(mx, 1e-3)
    keep = (saturation > 0.25) & (mx > 60)
    if keep.sum() < 20:
        return []
    selected = pixels[keep]
    distances = np.linalg.norm(selected[:, None, :] - _RGB[None, :, :], axis=2)
    nearest = distances.argmin(axis=1)
    counts = np.bincount(nearest, minlength=len(_COLOR_NAMES))
    order = np.argsort(-counts)
    return [_COLOR_NAMES[i] for i in order[:top_k] if counts[i] > 0]


def _count_objects(frames: np.ndarray) -> int:
    mask = _foreground_mask(frames)
    if mask.size == 0:
        return 0
    mask = mask[len(mask) // 2]  # middle frame
    try:
        from scipy import ndimage

        labels, num = ndimage.label(mask)
        if num == 0:
            return 0
        sizes = np.bincount(labels.ravel())
        sizes[0] = 0
        big = int((sizes > 0.002 * mask.size).sum())
        return max(1, big)
    except Exception:  # pragma: no cover - scipy always present in this project
        return int(mask.any())


def _classify_motion(frames: np.ndarray) -> str | None:
    """Cheap motion features from the foreground centroid and area tracks."""
    mask = _foreground_mask(frames)
    if mask.sum() == 0:
        return None
    areas = mask.reshape(len(mask), -1).sum(axis=1).astype(np.float64)
    if areas.max() < 4:
        return None
    rows = np.arange(mask.shape[1])[:, None]
    cols = np.arange(mask.shape[2])[None, :]
    total = areas + 1e-6
    cy = (mask * rows[None]).reshape(len(mask), -1).sum(axis=1) / total
    cx = (mask * cols[None]).reshape(len(mask), -1).sum(axis=1) / total

    def rel_range(track: np.ndarray) -> float:
        return float(np.ptp(track) / (np.abs(track).mean() + 1e-6))

    area_cv = float(areas.std() / (areas.mean() + 1e-6))
    range_y, range_x = rel_range(cy), rel_range(cx)

    if area_cv > 0.18 and range_y < 0.12 and range_x < 0.12:
        return "pulsing"
    if range_y > 0.35 and range_x > 0.35:
        return "bouncing"
    if range_y > 0.10 and range_x > 0.10 and area_cv <= 0.18:
        return "orbiting"
    if max(range_y, range_x) > 0.10:
        return "swinging"
    return None


def _onset_rate(wave: np.ndarray, sample_rate: int) -> tuple[float, float]:
    """``(onsets per second, pitch variation)`` from a short-time energy track."""
    if wave.size == 0:
        return 0.0, 0.0
    mono = wave.mean(axis=0)
    hop = 512
    frame = 1024
    n = max(1, (mono.size - frame) // hop)
    energy = np.array([np.sqrt(np.mean(mono[i * hop : i * hop + frame] ** 2) + 1e-12) for i in range(n)])
    flux = np.clip(np.diff(energy, prepend=energy[:1]), 0, None)
    if flux.max() <= 1e-6:
        return 0.0, 0.0
    peaks = (flux > 0.3 * flux.max()) & (flux >= np.roll(flux, 1))
    # Merge onsets closer than 150 ms: a single percussive hit rings for several frames.
    min_gap = max(1, int(0.15 * sample_rate / hop))
    kept = []
    for idx in np.nonzero(peaks)[0]:
        if not kept or idx - kept[-1] >= min_gap:
            kept.append(int(idx))
    duration = mono.size / sample_rate
    rate = len(kept) / max(duration, 1e-6)
    centroid = np.array([
        float(np.abs(np.fft.rfft(mono[i * hop : i * hop + frame] * np.hanning(frame))).argmax()) for i in range(n)
    ])
    pitch_var = float(centroid.std() / (centroid.mean() + 1e-6)) if n > 1 else 0.0
    return rate, pitch_var


def _score_colors(frames: np.ndarray, expected: list[str]) -> float | None:
    if not expected:
        return None
    found = set(_dominant_color_names(frames))
    return float(len(found & set(expected)) / len(expected))


def _score_count(frames: np.ndarray, expected: int) -> float | None:
    count = _count_objects(frames)
    if count <= 0:
        return 0.0
    return float(max(0.0, 1.0 - abs(count - expected) / max(2.0, expected)))


def _score_motion(frames: np.ndarray, expected: str | None) -> float | None:
    if expected is None:
        return None
    got = _classify_motion(frames)
    if got is None:
        return 0.0
    return 1.0 if got == expected else 0.0


def _score_tempo(wave: np.ndarray, sample_rate: int, expected: str | None, motion: str | None) -> float | None:
    if expected is None or motion not in ("pulsing", "swinging"):
        return None
    rate, _ = _onset_rate(wave, sample_rate)
    if rate <= 0:
        return 0.0
    bpm = rate * 60.0
    target = TEMPOS[expected]
    # Nearest of the three tempo buckets, using log distance.
    best = min(TEMPOS, key=lambda word: abs(np.log(max(bpm, 1e-3)) - np.log(TEMPOS[word])))
    del target
    return 1.0 if best == expected else 0.0


def _score_sound(wave: np.ndarray, sample_rate: int, expected: str | None) -> float | None:
    if expected is None:
        return None
    rate, pitch_var = _onset_rate(wave, sample_rate)
    if expected == "impacts":
        return 1.0 if 0.5 <= rate <= 6.0 else (0.5 if rate > 0 else 0.0)
    if expected == "beats":
        return 1.0 if 0.8 <= rate <= 3.0 else (0.5 if rate > 0 else 0.0)
    if expected == "ticks":
        return 1.0 if 0.5 <= rate <= 5.0 else (0.5 if rate > 0 else 0.0)
    if expected == "tone":
        return 1.0 if pitch_var > 0.02 else 0.0
    return None


def prompt_match_reward(frames: np.ndarray, wave: np.ndarray, fps: float, sample_rate: int, prompt: str) -> tuple[float, dict]:
    """``(score, parts)`` where ``parts`` keeps the per-attribute detail for logging."""
    spec = parse_prompt(prompt)
    parts = {
        "colors": _score_colors(frames, spec["colors"]),
        "count": _score_count(frames, spec["count"]),
        "motion": _score_motion(frames, spec["motion"]),
        "tempo": _score_tempo(wave, sample_rate, spec["tempo"], spec["motion"]),
        "sound": _score_sound(wave, sample_rate, spec["sound"]),
    }
    available = {k: v for k, v in parts.items() if v is not None}
    score = float(np.mean(list(available.values()))) if available else 0.0
    return score, parts
