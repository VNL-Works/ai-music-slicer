"""Re-cut an existing aislice result: move the bar start by N beats and/or change slice units.

No re-analysis or re-stretching: the song is already on the grid, so this only moves where bar 1 is.
"""
from __future__ import annotations

import json
import math
from pathlib import Path

import numpy as np

from . import io, slicer
from . import stems as st


def _fit_len(a: np.ndarray, n: int) -> np.ndarray:
    return a[:n] if len(a) >= n else np.concatenate([a, np.zeros((n - len(a), a.shape[1]), a.dtype)])


def reslice(out_dir: Path, shift_beats: int = 0, units: list[str] | None = None, fade_ms: float = 1.0,
            drop_partial: bool = False, log=print) -> dict:
    out_dir = Path(out_dir)
    rpath = out_dir / "report.json"
    r = json.loads(rpath.read_text())
    bpm = float(r["target_bpm"])
    bpb = int(str(r.get("meter", "4/4")).split("/")[0])
    T = 60.0 / bpm
    bar = bpb * T
    full = next(out_dir.glob(f"*_{str(r['target_bpm']).replace('.', '_')}bpm.wav"), None) or \
        next(p for p in out_dir.glob("*.wav"))
    y, sr = io.read(full)
    stem = full.stem.rsplit("_", 1)[0]

    # bar lines (after the shift) sit at Dm + k*bar
    D = float(r["grid"]["first_downbeat_output_s"]) + shift_beats * T
    Dm = D - math.floor(D / bar + 1e-6) * bar
    # first audible moment (10 ms windows above -45 dBFS)
    w = max(1, int(0.010 * sr))
    mono = y.mean(1)
    n = len(mono) // w
    db = 20 * np.log10(np.sqrt((mono[:n * w].reshape(n, w) ** 2).mean(1)) + 1e-12)
    loud = np.flatnonzero(db > -45)
    t_a = loud[0] * w / sr if len(loud) else 0.0
    if t_a + 0.010 < Dm:
        # audible pickup before the first bar line: pad silence so it becomes a full bar
        offset = -int(round((bar - Dm) * sr))
        new_db = bar
    else:
        # start at the last bar line at/just before the music begins; drop silent bars before it
        offset = int(round((Dm + math.floor((t_a + 0.010 - Dm) / bar) * bar) * sr))
        new_db = 0.0

    def move(a: np.ndarray) -> np.ndarray:  # same trim/pad for the mix and every stem
        if offset >= 0:
            return a[offset:]
        return np.concatenate([np.zeros((-offset, a.shape[1]), a.dtype), a])

    y = move(y)
    io.write(full, y, sr)
    stems = {}
    for name, (sy, ssr) in st.read_stems(out_dir).items():
        stems[name] = _fit_len(move(sy), len(y))
        io.write(out_dir / "stems" / f"{name}.wav", stems[name], ssr)

    if units is None:  # re-cut whatever slice folders already exist
        units = [u for p in sorted(out_dir.iterdir()) if p.is_dir() and (u := slicer.unit_for_folder(p.name))]
    counts = {u: slicer.write_slices(y, sr, bpm, u, bpb, out_dir, stem, fade_ms, drop_partial) for u in units}
    for u in units:
        if stems:
            slicer.write_stem_slices(stems, sr, bpm, u, bpb, out_dir, stem, fade_ms=fade_ms,
                                     drop_partial=drop_partial)

    g = r["grid"]
    g["first_downbeat_output_s"] = round(new_db, 4)
    g["manual_shift_beats"] = int(g.get("manual_shift_beats", 0)) + shift_beats
    g["downbeat_confident"] = True if shift_beats else g.get("downbeat_confident")
    r["slices"] = {u: {"count": c, "samples_per_slice": round(slicer.unit_beats(u, bpb) * T * sr, 3)}
                   for u, c in counts.items()}
    r["duration"]["output_s"] = round(len(y) / sr, 3)
    rpath.write_text(json.dumps(r, indent=2))
    log(f"{out_dir.name}: bar 1 moved {shift_beats:+d} beat(s); " +
        ", ".join(f"{c} x {u}" for u, c in counts.items()) + (f" (+ stems: {', '.join(stems)})" if stems else ""))
    return counts
