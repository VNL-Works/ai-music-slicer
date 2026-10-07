"""Re-analyze the conformed output and measure how far beats are from the ideal grid."""
from __future__ import annotations

import librosa
import numpy as np

from .analysis import analyze


def verify(y_out: np.ndarray, sr: int, bpm: float, min_strength: float = 0.3) -> dict:
    mono = librosa.resample(y_out.mean(1), orig_sr=sr, target_sr=22050)
    a = analyze(mono, 22050, bpm_hint=bpm)
    T = 60.0 / bpm
    b = a.beats[a.beat_strength >= min_strength]
    err = (b - np.round(b / T) * T) * 1000
    # global offset (constant latency) is reported separately
    offset = float(np.median(err))
    e = np.abs(err)
    # tempo of output per 20 s window
    win = []
    for s in np.arange(0, a.duration, 20):
        m = (a.beats >= s) & (a.beats < s + 20) & (a.beat_strength >= min_strength)
        if m.sum() > 16:
            idx = np.where(m)[0]
            p = np.polyfit(idx, a.beats[idx], 1)
            win.append(round(60 / p[0], 2))
    return {
        "beats_checked": int(len(b)),
        "mean_err_ms": round(float(e.mean()), 2),
        "p95_err_ms": round(float(np.percentile(e, 95)), 2),
        "max_err_ms": round(float(e.max()), 2),
        "median_offset_ms": round(offset, 2),
        "output_bpm_per_20s": win,
        "pass": bool(np.percentile(e, 95) < 10.0),
        "_beats": a.beats, "_err": err, "_strong": a.beat_strength >= min_strength,
    }
