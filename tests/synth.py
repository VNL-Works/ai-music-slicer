"""Synthetic test tracks with a known tempo curve, beat times and bar structure.

Like real music, beats 1 and 3 are equally accented. Only the harmony (a chord that
changes on every downbeat) reveals where bar 1 is, which is the case that once fooled
the downbeat detector.
"""
from __future__ import annotations

import numpy as np

CHORDS = [(261.63, 329.63, 392.00), (349.23, 440.00, 523.25),   # C, F
          (392.00, 493.88, 587.33), (220.00, 261.63, 329.63)]   # G, Am


def beat_times(bpm_start: float, bpm_end: float, duration: float, start: float = 0.0) -> np.ndarray:
    """Beat times for a tempo that changes linearly from bpm_start to bpm_end over `duration`."""
    t, out = start, []
    while t < duration:
        out.append(t)
        bpm = bpm_start + (bpm_end - bpm_start) * (t / duration)
        t += 60.0 / bpm
    return np.array(out)


def render(beats: np.ndarray, duration: float, sr: int = 22050, beats_per_bar: int = 4,
           first_downbeat: int = 0, seed: int = 0) -> np.ndarray:
    """Mono float32: clicks on every beat (accent on beats 1 & 3) + a chord per bar."""
    rng = np.random.default_rng(seed)
    y = np.zeros(int(duration * sr) + sr, dtype=np.float64)
    n_click = int(0.03 * sr)
    env = np.exp(-np.arange(n_click) / (0.006 * sr))
    for i, b in enumerate(beats):
        pos = (i - first_downbeat) % beats_per_bar
        amp = 0.6 if pos in (0, 2) else 0.35          # 1 and 3 equally strong
        click = amp * env * (rng.standard_normal(n_click) * 0.5 + np.sin(2 * np.pi * 1800 * np.arange(n_click) / sr))
        s = int(round(b * sr))
        y[s:s + n_click] += click[: len(y) - s]
    # sustained chord per bar, changing exactly on downbeats
    db_idx = list(range(first_downbeat, len(beats), beats_per_bar))
    for k, i in enumerate(db_idx):
        s = int(round(beats[i] * sr))
        e = int(round(beats[i + beats_per_bar] * sr)) if i + beats_per_bar < len(beats) else int(duration * sr)
        t = np.arange(e - s) / sr
        tone = sum(np.sin(2 * np.pi * f * t) for f in CHORDS[k % len(CHORDS)]) / 3
        fade = np.minimum(1, np.minimum(t / 0.005, (t[-1] - t + 1e-9) / 0.01))
        y[s:e] += 0.12 * tone * fade
    # pickup beats before the first downbeat get the last chord
    return (y[: int(duration * sr)] * 0.8).astype(np.float32)
