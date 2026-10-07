from __future__ import annotations

import subprocess
from pathlib import Path

import numpy as np
import soundfile as sf

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


def collect_inputs(paths: list[str]) -> list[Path]:
    out = []
    for p in map(Path, paths):
        if p.is_dir():
            out += sorted(f for f in p.iterdir() if f.suffix.lower() in AUDIO_EXTS)
        elif p.suffix.lower() in AUDIO_EXTS:
            out.append(p)
    return out
