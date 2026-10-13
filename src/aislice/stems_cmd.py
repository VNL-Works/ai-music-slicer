"""`aislice stems`: add stems to an existing result folder without re-processing the tempo."""
from __future__ import annotations

import json
from pathlib import Path

from . import io, slicer
from .pipeline import add_stems


def add_stems_to_result(out_dir: Path, preset: str = "4", model: str | None = None, model_dir: Path | None = None,
                        units: list[str] | None = None, fade_ms: float = 1.0, drop_partial: bool = False,
                        log=print, backend=None, fast: bool = False, exact: bool = False) -> dict:
    out_dir = Path(out_dir)
    rpath = out_dir / "report.json"
    r = json.loads(rpath.read_text())
    full = next(out_dir.glob(f"*_{str(r['target_bpm']).replace('.', '_')}bpm.wav"), None) or \
        next(p for p in out_dir.glob("*.wav"))
    y, sr = io.read(full)
    norm = r.get("normalize") or {}
    y = io.undo_gain(y, (norm.get("gains_db") or {}).get("mix", 0.0))   # separate from the original level
    song = full.stem.rsplit("_", 1)[0]
    bpb = int(str(r.get("meter", "4/4")).split("/")[0])
    if units is None:
        units = [slicer.STANDARD]
    log(f"{out_dir.name}:")
    info = add_stems(out_dir, y, sr, float(r["target_bpm"]), bpb, song, units, preset, model, model_dir,
                     fade_ms, drop_partial, log=log, backend=backend, fast=fast, exact=exact)
    r["stems"] = info
    g = {"mix": (norm.get("gains_db") or {}).get("mix", 0.0), **info.get("gains_db", {})}
    r["normalize"] = {"peak_db": io.PEAK_DB, "gains_db": g}
    rpath.write_text(json.dumps(r, indent=2))
    return info
