"""Video/audio IO for Tiny-H3: decode any mp4 to a fixed grid, write mp4 with sound.

Everything goes through one ffmpeg binary (``imageio-ffmpeg`` ships a static build, so the
project has no system ffmpeg dependency) plus ``soundfile`` for wav.  Decoding always
targets an explicit canvas and frame rate, which is what the training grid needs anyway.
"""

from __future__ import annotations

import functools
import json
import os
import subprocess
import tempfile

import numpy as np

AUDIO_SAMPLE_RATE = 32000  # MiniMax-H3's audio VAE rate
AUDIO_CHANNELS = 2  # H3 carries stereo as two mono batch items


@functools.lru_cache(maxsize=1)
def ffmpeg_exe() -> str:
    """Path to a usable ffmpeg binary."""
    env = os.environ.get("TINY_H3_FFMPEG")
    if env:
        return env
    try:
        import imageio_ffmpeg

        return imageio_ffmpeg.get_ffmpeg_exe()
    except Exception:  # pragma: no cover - fallback for environments with system ffmpeg
        for candidate in ("ffmpeg", "/usr/bin/ffmpeg", "/usr/local/bin/ffmpeg"):
            if subprocess.run(["which" if "/" not in candidate else "test", candidate], capture_output=True).returncode == 0:
                return candidate
        raise RuntimeError("No ffmpeg found. Install imageio-ffmpeg or set TINY_H3_FFMPEG.")


def _run(cmd: list[str], **kwargs) -> subprocess.CompletedProcess:
    proc = subprocess.run(cmd, capture_output=True, **kwargs)
    if proc.returncode != 0:
        tail = proc.stderr.decode("utf-8", "ignore")[-2000:]
        raise RuntimeError(f"ffmpeg failed ({proc.returncode}):\n{' '.join(cmd[:8])} ...\n{tail}")
    return proc


def probe(path: str) -> dict:
    """Best-effort stream info parsed from ffmpeg's banner (no ffprobe in the static build)."""
    proc = subprocess.run([ffmpeg_exe(), "-hide_banner", "-i", path], capture_output=True)
    text = proc.stderr.decode("utf-8", "ignore")
    info: dict = {"has_video": "Video:" in text, "has_audio": "Audio:" in text, "raw": text}
    for line in text.splitlines():
        line = line.strip()
        if line.startswith("Duration:"):
            info["duration"] = line.split(",")[0].split("Duration:")[1].strip()
        if "Video:" in line:
            parts = [p.strip() for p in line.split(",")]
            for part in parts:
                if "x" in part and part.replace(".", "").replace("x", "").isdigit():
                    info["size"] = part
                    break
            for part in parts:
                if part.endswith("fps"):
                    info["fps"] = part
                    break
    return info


def read_video_frames(
    path: str,
    size: tuple[int, int],
    fps: float,
    max_frames: int | None = None,
) -> np.ndarray:
    """Decode ``path`` onto a fixed ``(width, height)`` canvas at ``fps``.

    Returns ``uint8`` frames shaped ``(T, H, W, 3)`` in RGB.  The output length is whatever
    the clip yields; callers that need exactly ``N`` frames truncate or pad explicitly.
    """
    width, height = size
    vf = f"fps={fps},scale={width}:{height}:flags=lanczos"
    cmd = [ffmpeg_exe(), "-v", "error", "-i", path, "-vf", vf, "-pix_fmt", "rgb24", "-f", "rawvideo", "-"]
    proc = subprocess.run(cmd, capture_output=True)
    if proc.returncode != 0:
        raise RuntimeError(f"ffmpeg decode failed: {proc.stderr.decode('utf-8', 'ignore')[-1500:]}")
    frame_bytes = width * height * 3
    frames = np.frombuffer(proc.stdout, dtype=np.uint8)
    count = frames.size // frame_bytes
    frames = frames[: count * frame_bytes].reshape(count, height, width, 3)
    if max_frames is not None:
        frames = frames[:max_frames]
    return np.ascontiguousarray(frames)


def read_audio_wave(
    path: str,
    sample_rate: int = AUDIO_SAMPLE_RATE,
    channels: int = AUDIO_CHANNELS,
    allow_silent: bool = True,
) -> np.ndarray:
    """Decode the audio track as float32 ``(channels, N)`` at ``sample_rate``.

    Silent (or absent) tracks become zeros so video-only datasets can still be encoded;
    set ``allow_silent=False`` to raise instead.
    """
    cmd = [
        ffmpeg_exe(), "-v", "error", "-i", path,
        "-vn", "-ac", str(channels), "-ar", str(sample_rate), "-f", "f32le", "-",
    ]
    proc = subprocess.run(cmd, capture_output=True)
    if proc.returncode != 0 or not proc.stdout:
        if not allow_silent:
            raise RuntimeError(f"no audio track in {path}")
        return np.zeros((channels, 0), dtype=np.float32)
    wave = np.frombuffer(proc.stdout, dtype=np.float32)
    wave = wave[: (wave.size // channels) * channels].reshape(-1, channels).T
    return np.ascontiguousarray(wave)


def write_wav(path: str, wave: np.ndarray, sample_rate: int = AUDIO_SAMPLE_RATE) -> None:
    import soundfile as sf

    sf.write(path, wave.T, sample_rate, subtype="FLOAT")


def write_mp4(
    path: str,
    frames: np.ndarray,
    fps: float,
    wave: np.ndarray | None = None,
    sample_rate: int = AUDIO_SAMPLE_RATE,
    crf: int = 18,
    audio_bitrate: str = "192k",
) -> None:
    """Write ``frames`` (uint8 ``(T, H, W, 3)``) and optional ``wave`` to one mp4.

    The video keeps **exactly** ``len(frames)`` frames -- no ``-shortest`` truncation, so a clip
    authored on H3's ``17n+5`` grid stays on it.  AAC priming makes the audio a few tens of
    milliseconds late on decode, so training data keeps its waveform in a sidecar wav
    (:func:`write_clip`) and only demos go through this muxer.
    """
    if frames.ndim != 4 or frames.dtype != np.uint8:
        raise ValueError(f"frames must be uint8 (T,H,W,3), got {frames.shape} {frames.dtype}")
    height, width = frames.shape[1], frames.shape[2]
    tmp_wav = None
    try:
        cmd = [ffmpeg_exe(), "-y", "-v", "error", "-f", "rawvideo", "-pix_fmt", "rgb24",
               "-s", f"{width}x{height}", "-r", str(fps), "-i", "pipe:0"]
        if wave is not None and wave.size:
            fd, tmp_wav = tempfile.mkstemp(suffix=".wav")
            os.close(fd)
            write_wav(tmp_wav, wave, sample_rate)
            cmd += ["-i", tmp_wav]
        cmd += ["-map", "0:v:0"]
        if tmp_wav:
            cmd += ["-map", "1:a:0", "-c:a", "aac", "-b:a", audio_bitrate]
        cmd += ["-c:v", "libx264", "-preset", "medium", "-crf", str(crf), "-pix_fmt", "yuv420p",
                "-movflags", "+faststart", path]
        proc = subprocess.run(cmd, input=frames.tobytes(), capture_output=True)
        if proc.returncode != 0:
            raise RuntimeError(f"ffmpeg encode failed: {proc.stderr.decode('utf-8', 'ignore')[-1500:]}")
    finally:
        if tmp_wav and os.path.exists(tmp_wav):
            os.remove(tmp_wav)


def write_clip(
    video_path: str,
    wav_path: str,
    frames: np.ndarray,
    fps: float,
    wave: np.ndarray | None,
    sample_rate: int = AUDIO_SAMPLE_RATE,
) -> None:
    """Write a training clip: silent (or muted) mp4 plus an exact float32 wav sidecar."""
    write_mp4(video_path, frames, fps, wave=None)
    if wave is not None:
        write_wav(wav_path, wave, sample_rate)


def read_wav(path: str, sample_rate: int | None = None) -> np.ndarray:
    """Read a wav as float32 ``(channels, N)``, optionally resampling."""
    import soundfile as sf

    wave, sr = sf.read(path, dtype="float32", always_2d=True)
    wave = wave.T
    if sample_rate is not None and sr != sample_rate:
        wave = resample_audio(wave, sr, sample_rate)
    return np.ascontiguousarray(wave)


def resample_audio(wave: np.ndarray, sr_in: int, sr_out: int = AUDIO_SAMPLE_RATE) -> np.ndarray:
    """Resample ``(channels, N)`` float32 audio."""
    if sr_in == sr_out:
        return wave
    try:
        from scipy.signal import resample_poly

        from math import gcd

        g = gcd(int(sr_in), int(sr_out))
        return resample_poly(wave, sr_out // g, sr_in // g, axis=-1).astype(np.float32)
    except Exception:
        n_out = int(round(wave.shape[-1] * sr_out / sr_in))
        x_old = np.linspace(0.0, 1.0, wave.shape[-1], endpoint=False)
        x_new = np.linspace(0.0, 1.0, n_out, endpoint=False)
        return np.stack([np.interp(x_new, x_old, ch) for ch in wave]).astype(np.float32)


def normalize_wave(wave: np.ndarray, peak: float = 0.9) -> np.ndarray:
    """Peak-normalize, leaving silence untouched."""
    m = float(np.abs(wave).max()) if wave.size else 0.0
    if m < 1e-6:
        return wave.astype(np.float32)
    return (wave / m * peak).astype(np.float32)


def save_json(path: str, obj) -> None:
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=False, indent=2)


def load_json(path: str):
    with open(path, encoding="utf-8") as f:
        return json.load(f)
