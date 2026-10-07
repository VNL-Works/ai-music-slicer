"""Lock the song to a fixed grid with a single variable-ratio Rubber Band pass."""
from __future__ import annotations

import math
import subprocess
import tempfile
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from . import io
from .tempomap import TempoMap


@dataclass
class WarpPlan:
    target_bpm: float
    beat_s: float
    pad_s: float                 # silence added before the stretched audio
    preroll_out_s: float         # stretched length of audio before the first grid beat
    first_downbeat_out_s: float
    src_points: np.ndarray       # source seconds
    dst_points: np.ndarray       # output seconds (before padding)
    out_duration_s: float        # stretched duration (before padding)
    ratios: np.ndarray           # per-beat stretch ratio (output/input)


def plan(tm: TempoMap, src_duration: float, target_bpm: float, preroll_silent: bool = False) -> WarpPlan:
    T = 60.0 / target_bpm
    g = tm.beats
    bpb = tm.beats_per_bar
    bar = bpb * T
    ratios = T / np.diff(g)
    r_start = float(np.median(ratios[:4]))
    r_end = float(np.median(ratios[-4:]))
    preroll = g[0] * r_start
    # P = output time of grid beat 0 (after padding) such that the first downbeat is on a bar line
    d = tm.downbeat_index
    k = math.ceil((preroll + d * T) / bar - 1e-9)
    P = k * bar - d * T
    pad = P - preroll
    if preroll_silent and d == 0 and pad > 0:
        # nothing audible before the first downbeat: trim instead of padding a bar of silence
        P, pad = 0.0, -preroll
    dst = preroll + np.arange(len(g)) * T
    out_dur = dst[-1] + (src_duration - g[-1]) * r_end
    src_pts = np.concatenate([[0.0], g, [src_duration]])
    dst_pts = np.concatenate([[0.0], dst, [out_dur]])
    return WarpPlan(target_bpm, T, pad, preroll, P + d * T, src_pts, dst_pts, out_dur, ratios)


def _has_r3() -> bool:
    """Rubber Band >= 3 has the higher-quality R3 engine (--fine); older builds don't."""
    try:
        out = subprocess.run(["rubberband", "--help"], capture_output=True, text=True)
        return "--fine" in (out.stdout + out.stderr)
    except FileNotFoundError as e:
        raise RuntimeError("the 'rubberband' command-line tool is not installed "
                           "(macOS: brew install rubberband; Debian/Ubuntu: apt install rubberband-cli)") from e


def render(src_wav: Path, wp: WarpPlan, sr: int, crisp: int = 5) -> np.ndarray:
    """Run rubberband with a time map; returns padded output (samples, channels)."""
    with tempfile.TemporaryDirectory() as td:
        td = Path(td)
        mp = td / "map.txt"
        with open(mp, "w") as f:
            for s, d in zip(wp.src_points[1:-1], wp.dst_points[1:-1]):
                f.write(f"{int(round(s * sr))} {int(round(d * sr))}\n")
        out = td / "out.wav"
        engine = ["--fine"] if _has_r3() else []
        subprocess.run(["rubberband", "-q", *engine, "--crisp", str(crisp),
                        "-M", str(mp), "-D", f"{wp.out_duration_s:.6f}",
                        str(src_wav), str(out)], check=True)
        y, osr = io.read(out)
    assert osr == sr
    n = int(round(wp.pad_s * sr))
    if n < 0:
        return y[-n:]
    return np.concatenate([np.zeros((n, y.shape[1]), dtype=y.dtype), y])
