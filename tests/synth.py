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


# ---------------------------------------------------------------- Picker test song
def _tone(f, n, sr, amp=0.2):
    t = np.arange(n) / sr
    env = np.minimum(1, np.arange(n) / (0.005 * sr)) * np.exp(-t * 3)
    return (amp * env * (np.sin(2 * np.pi * f * t) + 0.3 * np.sin(4 * np.pi * f * t))).astype(np.float32)


def _hit(n, sr, kind, rng):
    t = np.arange(n) / sr
    if kind == "kick":
        return (0.8 * np.sin(2 * np.pi * (50 + 80 * np.exp(-t * 30)) * t) * np.exp(-t * 18)).astype(np.float32)
    return (0.25 * rng.standard_normal(n) * np.exp(-t * 60)).astype(np.float32)


def pick_song(sr=22050, bpm=240.0, seed=0):
    """A grid-locked 6 x 8-bar song for the Picker, with stems and exact ground truth.

    Sections (8 bars each): A A B A B C.
      drums  : same pattern in sections 0-4, silent in 5            -> one group of 5, slice 6 empty
      melody : A = C E G B per beat, B = the same notes reversed     -> A {1,2,4} and B {3,5} must NOT merge
               (same pitch set, so slice-averaged features can't tell them apart), C = other notes
      vocals : sings in sections 0-1, one stray blip in section 3 -> slice 4 sparse, slices 3, 5, 6 silent
    Each section gets its own faint noise so repeats are close, not bit-identical (like AI audio).
    Returns (stems dict name -> (n, 2) float32, sr, bpm).
    """
    rng = np.random.default_rng(seed)
    spb = int(round(60 / bpm * sr))
    bar = 4 * spb
    sec = 8 * bar
    n = 6 * sec
    drums, melody, vocals = (np.zeros(n, np.float32) for _ in range(3))
    notes = {"A": [261.6, 329.6, 392.0, 493.9], "B": [493.9, 392.0, 329.6, 261.6], "C": [220.0, 293.7, 349.2, 440.0]}
    for s, name in enumerate("AABABC"):
        o = s * sec
        for b in range(32):
            p = o + b * spb
            if s < 5:
                drums[p:p + spb] += _hit(spb, sr, "kick" if b % 2 == 0 else "hat", rng)
            melody[p:p + spb] += _tone(notes[name][b % 4], spb, sr)
            if s < 2:
                vocals[p:p + spb] += _tone(notes[name][(b + 2) % 4] * 2, spb, sr, 0.15)
        noise = 0.002 * rng.standard_normal(sec).astype(np.float32)
        melody[o:o + sec] += noise
        if s < 5:
            drums[o:o + sec] += noise
    vocals[3 * sec + 5 * bar: 3 * sec + 5 * bar + spb // 4] += 0.3       # one stray blip
    st = {k: np.stack([v, v], 1) for k, v in (("drums", drums), ("melody", melody), ("vocals", vocals))}
    return st, sr, bpm


def write_pick_result(root, name="Song (1)", sr=22050, bpm=240.0, units=("standard",)):
    """Write a fake aislice result folder for pick_song(), the way the pipeline does: full mix + stems
    normalized on their own (gains in report.json), slices in the standard layout (drums 4bar, rest 8bar)
    unless `units` says otherwise, each slice normalized with a _gains.json."""
    import json
    from pathlib import Path

    from aislice import io, slicer
    stems, sr, bpm = pick_song(sr, bpm)
    n = int(round(48 * 4 * 60 / bpm * sr))       # pad to exactly 48 bars (the beat is rounded to whole samples)
    stems = {k: np.concatenate([y, np.zeros((max(0, n - len(y)), 2), np.float32)])[:n] for k, y in stems.items()}
    mix = sum(stems.values())
    gains = {}
    mix, gains["mix"] = io.normalize(mix)
    for k in list(stems):
        stems[k], gains[k] = io.normalize(stems[k])
    d = Path(root) / f"{name}_{bpm:g}bpm"
    (d / "stems").mkdir(parents=True)
    io.write(d / f"{name}_{bpm:g}bpm.wav", mix, sr)
    for k, y in stems.items():
        io.write(d / "stems" / f"{k}.wav", y, sr)
    for u in units:
        slicer.write_slices(mix, sr, bpm, slicer.standard_unit("mix") if u == "standard" else u, 4, d, name)
        slicer.write_stem_slices(stems, sr, bpm, u, 4, d, name)
    report = {"source": f"{name}.mp3", "sample_rate": sr, "target_bpm": bpm, "meter": "4/4",
              "slices": {"unit": units[0]}, "stems": {"model": "fake", "names": list(stems)},
              "normalize": {"peak_db": io.PEAK_DB, "gains_db": gains}}
    (d / "report.json").write_text(json.dumps(report))
    return d
