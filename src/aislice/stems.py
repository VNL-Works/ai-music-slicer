"""Optional stem separation: split the grid-locked song into stems that share its exact grid.

Separation runs AFTER the tempo is locked, so every stem inherits the same grid and can be
sliced with exactly the same cut points as the mix. Backend: `audio-separator`
(pip install "aislice[stems]"), which runs Demucs, RoFormer, MDX and VR models.
"""
from __future__ import annotations

import re
import tempfile
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

from . import io

MODEL_SR = 44100  # separation models run at 44.1 kHz; stems are resampled back to the song's rate

# Presets use Demucs v4 weights (MIT licensed). Any audio-separator model can be used via --stem-model,
# e.g. htdemucs_ft.yaml (best Demucs quality, ~4x slower) or a RoFormer model for vocals.
PRESETS = {
    "2": ("htdemucs.yaml", "vocals + instrumental"),
    "4": ("htdemucs.yaml", "vocals, drums, bass, other"),
    "6": ("htdemucs_6s.yaml", "vocals, drums, bass, guitar, piano, other"),
}
DEFAULT_MODEL_DIR = Path.home() / ".cache" / "aislice" / "models"
INSTALL_HINT = 'stem separation needs the optional extra: pip install "aislice[stems]"'


@dataclass
class StemResult:
    model: str
    stems: dict[str, np.ndarray]          # name -> (samples, channels) at the song's sample rate
    lag_samples: int = 0                  # alignment correction applied (song samples)
    residual_db: float | None = None      # energy of (mix - sum of stems) relative to the mix
    notes: list[str] = field(default_factory=list)


def _stem_name(filename: str) -> str:
    """audio-separator names outputs like 'song_(Vocals)_htdemucs_ft.wav'."""
    m = re.findall(r"\(([^)]+)\)", Path(filename).stem)
    name = (m[-1] if m else Path(filename).stem).strip().lower()
    return re.sub(r"[^a-z0-9]+", "_", name).strip("_") or "stem"


def run_backend(wav: Path, model: str, work_dir: Path, model_dir: Path | None = None) -> dict[str, Path]:
    """Run audio-separator on `wav`; return {stem name: output file}. Replaced by a fake in tests."""
    try:
        import logging

        from audio_separator.separator import Separator
    except ModuleNotFoundError as e:  # pragma: no cover - exercised only without the extra
        if e.name and e.name.split(".")[0] == "audio_separator":
            raise RuntimeError(INSTALL_HINT) from e
        raise RuntimeError(f"audio-separator is installed but its dependency '{e.name}' is missing: "
                           f"pip install {e.name}  (or reinstall: pip install -U \"aislice[stems]\")") from e
    kw = dict(output_dir=str(work_dir), output_format="WAV", sample_rate=MODEL_SR, use_soundfile=True,
              normalization_threshold=1.0, log_level=logging.ERROR)
    kw["model_file_dir"] = str(model_dir or DEFAULT_MODEL_DIR)
    sep = Separator(**kw)
    sep.load_model(model_filename=model)
    outs = sep.separate(str(wav))
    return {_stem_name(o): (Path(o) if Path(o).is_absolute() else work_dir / o) for o in outs}


def _align_lag(mix: np.ndarray, total: np.ndarray, sr: int, max_ms: float = 50.0) -> int:
    """Lag (samples) that best aligns sum-of-stems to the mix, searched within +-max_ms."""
    a = mix.mean(1)
    b = total.mean(1)
    n = min(len(a), len(b), sr * 60)          # first minute is plenty
    a, b = a[:n], b[:n]
    m = 1 << int(np.ceil(np.log2(2 * n)))
    xc = np.fft.irfft(np.fft.rfft(a, m) * np.conj(np.fft.rfft(b, m)), m)
    L = int(max_ms / 1000 * sr)
    lags = np.r_[np.arange(0, L + 1), np.arange(-L, 0)]
    vals = np.r_[xc[: L + 1], xc[-L:]]
    return int(lags[int(np.argmax(vals))])


def _fit(y: np.ndarray, n: int, channels: int) -> np.ndarray:
    if y.shape[1] != channels:
        y = np.repeat(y.mean(1, keepdims=True), channels, axis=1)
    if len(y) >= n:
        return y[:n]
    return np.concatenate([y, np.zeros((n - len(y), channels), y.dtype)])


def separate(mix: np.ndarray, sr: int, preset: str | None = "4", model: str | None = None,
             model_dir: Path | None = None, backend=None) -> StemResult:
    """Separate a (samples, channels) song into stems aligned sample-exactly to it."""
    import librosa

    if model is None:
        if preset not in PRESETS:
            raise ValueError(f"unknown stems preset {preset!r} (use 2, 4 or 6)")
        model = PRESETS[preset][0]
    backend = backend or run_backend
    n, ch = mix.shape
    with tempfile.TemporaryDirectory() as td:
        td = Path(td)
        src = td / "mix.wav"
        io.write(src, mix, sr, bit_depth="32f")
        files = backend(src, model, td / "out", model_dir)
        if not files:
            raise RuntimeError(f"separation with {model} produced no stems")
        stems = {}
        for name, f in files.items():
            y, fsr = io.read(f)
            if fsr != sr:
                y = librosa.resample(y.T, orig_sr=fsr, target_sr=sr, res_type="soxr_vhq").T
            stems[name] = _fit(np.ascontiguousarray(y, dtype=np.float32), n, ch)

    res = StemResult(model=model, stems=stems)
    total = sum(stems.values())
    lag = _align_lag(mix, total, sr)
    if lag:
        for k, y in stems.items():
            stems[k] = np.roll(y, lag, axis=0)
            if lag > 0:
                stems[k][:lag] = 0
            else:
                stems[k][lag:] = 0
        res.lag_samples = lag
        res.notes.append(f"stems shifted by {lag} samples to line up with the mix")
        total = sum(stems.values())
    e_mix = float(np.mean(mix.astype(np.float64) ** 2)) + 1e-20
    res.residual_db = round(10 * np.log10(float(np.mean((mix - total).astype(np.float64) ** 2)) / e_mix + 1e-20), 1)

    if preset == "2" and set(stems) != {"vocals", "instrumental"}:
        if "vocals" not in stems:
            raise RuntimeError(f"model {model} has no vocals stem to build a 2-stem split from")
        inst = sum(y for k, y in stems.items() if k != "vocals")
        res.stems = {"vocals": stems["vocals"], "instrumental": inst}
    return res


def write_stems(res: StemResult, out_dir: Path, sr: int, bit_depth: str = "24") -> dict[str, Path]:
    d = out_dir / "stems"
    d.mkdir(parents=True, exist_ok=True)
    for old in d.glob("*.wav"):
        old.unlink()
    paths = {}
    for name, y in res.stems.items():
        p = d / f"{name}.wav"
        io.write(p, y, sr, bit_depth=bit_depth)
        paths[name] = p
    return paths


def read_stems(out_dir: Path) -> dict[str, tuple[np.ndarray, int]]:
    d = out_dir / "stems"
    return {p.stem: io.read(p) for p in sorted(d.glob("*.wav"))} if d.is_dir() else {}
