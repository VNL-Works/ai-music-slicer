from __future__ import annotations

import subprocess
from pathlib import Path

import numpy as np
import soundfile as sf

# Every audio file aislice writes is peak-normalized on its own to PEAK_DB. Near-silent audio (more than
# SILENT_DB below the track it was cut from) is left as is, so silence never gets boosted into hiss.
# The gain applied is recorded (report.json "normalize" + a _gains.json next to slices) so the original
# levels can be recovered, e.g. by the Picker to spot empty or quiet slices.
PEAK_DB = -0.1
SILENT_DB = -45.0
GAINS_FILE = "_gains.json"

AUDIO_EXTS = {".wav", ".mp3", ".flac", ".m4a", ".aac", ".ogg", ".aif", ".aiff", ".opus"}


def decode_to_wav(src: Path, dst: Path) -> None:
    """Decode any format to float32 WAV at its native sample rate (ffmpeg)."""
    subprocess.run(["ffmpeg", "-v", "error", "-y", "-i", str(src), "-c:a", "pcm_f32le", str(dst)], check=True)


def read(path: Path) -> tuple[np.ndarray, int]:
    y, sr = sf.read(str(path), dtype="float32", always_2d=True)
    return y, sr  # (samples, channels)


def write(path: Path, y: np.ndarray, sr: int, fmt: str = "wav", bit_depth: str = "24") -> None:
    subtype = {"16": "PCM_16", "24": "PCM_24", "32f": "FLOAT"}[bit_depth]
    sf.write(str(path), np.clip(y, -1, 1) if bit_depth != "32f" else y, sr,
             subtype=subtype, format=fmt.upper())


def normalize(y: np.ndarray, peak_db: float = PEAK_DB, floor: float | None = None) -> tuple[np.ndarray, float]:
    """Scale `y` so its peak is `peak_db` dBFS. Returns (scaled, gain in dB). Audio whose peak is below
    `floor` (linear) or is silent is returned unchanged with gain 0."""
    peak = float(np.max(np.abs(y))) if y.size else 0.0
    if peak <= 1e-9 or (floor is not None and peak < floor):
        return y, 0.0
    g = 10 ** (peak_db / 20) / peak
    return (y * g).astype(y.dtype, copy=False), round(20 * np.log10(g), 4)


def undo_gain(y: np.ndarray, gain_db: float) -> np.ndarray:
    return y if not gain_db else (y * 10 ** (-gain_db / 20)).astype(y.dtype, copy=False)


def read_gains(folder: Path) -> dict[str, float]:
    import json
    try:
        return json.loads((Path(folder) / GAINS_FILE).read_text())
    except (OSError, ValueError):
        return {}


def collect_inputs(paths: list[str]) -> list[Path]:
    out = []
    for p in map(Path, paths):
        if p.is_dir():
            out += sorted(f for f in p.iterdir() if f.suffix.lower() in AUDIO_EXTS)
        elif p.suffix.lower() in AUDIO_EXTS:
            out.append(p)
    return out
