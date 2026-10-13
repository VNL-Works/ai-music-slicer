from __future__ import annotations

import re

import numpy as np

NAMED_UNITS = ["1/16", "1/8", "beat", "bar", "2bar", "4bar", "8bar"]

# The standard layout used across aislice: melodic parts (and the full mix) as 8-bar loops, drums and bass as
# 4-bar loops. `--slice` can still override it, but nothing asks for a size any more.
STANDARD = "standard"
STEM_UNITS = {"drums": "4bar", "bass": "4bar"}
DEFAULT_UNIT = "8bar"


def standard_unit(name: str) -> str:
    """Slice unit for a row in the standard layout ('mix' for the full song)."""
    return STEM_UNITS.get(name, DEFAULT_UNIT)
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
                 fmt="wav", bit_depth="24", dest=None, normalize=True):
    """Cut `y` (grid starts at sample 0) into `unit` slices in out_dir/<folder_for(unit)>/ (or `dest`).

    Each slice is peak-normalized on its own (io.PEAK_DB), except near-silent ones (io.SILENT_DB below the
    loudest point of `y`); the gains go to <dest>/_gains.json. Returns the number of slices.
    Stem slices go in a subfolder per stem: dest=<unit folder>/<stem>.
    """
    import json

    from . import io
    pts = slice_points(len(y), sr, bpm, unit_beats(unit, beats_per_bar), drop_partial)
    d = dest if dest is not None else out_dir / folder_for(unit)
    d.mkdir(parents=True, exist_ok=True)
    for old in d.glob("*." + fmt):
        old.unlink()  # stale slices from a previous run would otherwise mix in
    label = unit.replace("/", "-").replace(" ", "")
    width = max(3, len(str(len(pts))))
    top = float(np.max(np.abs(y))) if len(y) else 0.0
    floor = top * 10 ** (io.SILENT_DB / 20)
    gains = {}
    for i, (s, e) in enumerate(pts, 1):
        name = f"{stem}_{label}_{i:0{width}d}.{fmt}"
        seg = apply_fades(y[s:e], sr, fade_ms)
        if normalize:
            seg, gains[name] = io.normalize(seg, floor=floor)
        io.write(d / name, seg, sr, fmt, bit_depth)
    gp = d / io.GAINS_FILE
    if normalize:
        gp.write_text(json.dumps(gains))
    elif gp.exists():
        gp.unlink()
    return len(pts)


def write_stem_slices(stems, sr, bpm, unit, beats_per_bar, out_dir, song_stem, **kw) -> dict[str, int]:
    """Slice every stem on the mix's grid, into <unit folder>/<stem name>/. unit="standard" uses each
    stem's own standard unit (drums/bass 4bar, the rest 8bar)."""
    out = {}
    for name, y in stems.items():
        u = standard_unit(name) if unit == STANDARD else unit
        out[name] = write_slices(y, sr, bpm, u, beats_per_bar, out_dir, f"{song_stem}_{name}",
                                 dest=out_dir / folder_for(u) / name, **kw)
    return out
