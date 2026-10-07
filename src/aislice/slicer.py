from __future__ import annotations

import re

import numpy as np

NAMED_UNITS = ["1/16", "1/8", "beat", "bar", "2bar", "4bar", "8bar"]
_NOTE = {"1/16": 0.25, "1/8": 0.5}


def unit_beats(unit: str, beats_per_bar: int) -> float:
    """Length of a slice unit in beats. Accepts 1/8, 1/16, beat, bar, or N + bar/beat (e.g. 8bar, 3beat)."""
    unit = unit.strip().lower()
    if unit in _NOTE:
        return _NOTE[unit]
    m = re.fullmatch(r"(\d*)\s*(bar|beat)s?", unit)
    if not m:
        raise ValueError(f"unknown slice unit: {unit!r} (try bar, 4bar, 8bar, beat, 1/8, 1/16)")
    n = int(m.group(1) or 1)
    if n < 1:
        raise ValueError("slice length must be at least 1")
    return n * (beats_per_bar if m.group(2) == "bar" else 1)


def slice_points(n_samples: int, sr: int, bpm: float, beats_per_unit: float, drop_partial: bool = False):
    """Exact grid boundaries computed cumulatively, so rounding never drifts."""
    unit = beats_per_unit * 60.0 / bpm * sr
    pts = []
    k = 0
    while True:
        s = int(round(k * unit))
        if s >= n_samples:
            break
        e = min(int(round((k + 1) * unit)), n_samples)
        if e - s < unit - 1 and drop_partial:
            break
        pts.append((s, e))
        k += 1
    return pts


def apply_fades(seg: np.ndarray, sr: int, fade_ms: float) -> np.ndarray:
    n = min(int(sr * fade_ms / 1000), len(seg) // 2)
    if n <= 0:
        return seg
    seg = seg.copy()
    ramp = np.linspace(0, 1, n, dtype=seg.dtype)[:, None]
    seg[:n] *= ramp
    seg[-n:] *= ramp[::-1]
    return seg


def folder_for(unit: str) -> str:
    """Output folder name for a slice unit: bar -> slices/, others -> slices_<unit>/."""
    return "slices" if unit == "bar" else "slices_" + unit.replace("/", "-").replace(" ", "")


def unit_for_folder(name: str) -> str | None:
    if name == "slices":
        return "bar"
    if name.startswith("slices_"):
        u = name[len("slices_"):].replace("-", "/")
        try:
            unit_beats(u, 4)
            return u
        except ValueError:
            return None
    return None


def write_slices(y, sr, bpm, unit, beats_per_bar, out_dir, stem, fade_ms=1.0, drop_partial=False,
                 fmt="wav", bit_depth="24"):
    """Cut `y` (grid starts at sample 0) into `unit` slices in out_dir/<folder_for(unit)>/. Returns count."""
    from . import io
    pts = slice_points(len(y), sr, bpm, unit_beats(unit, beats_per_bar), drop_partial)
    d = out_dir / folder_for(unit)
    d.mkdir(parents=True, exist_ok=True)
    for old in d.glob("*." + fmt):
        old.unlink()  # stale slices from a previous run would otherwise mix in
    label = unit.replace("/", "-").replace(" ", "")
    width = max(3, len(str(len(pts))))
    for i, (s, e) in enumerate(pts, 1):
        io.write(d / f"{stem}_{label}_{i:0{width}d}.{fmt}", apply_fades(y[s:e], sr, fade_ms), sr, fmt, bit_depth)
    return len(pts)
