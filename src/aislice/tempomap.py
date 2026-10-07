"""Turn raw tracked beats into a clean, reliable beat grid (+ downbeat phase)."""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from .analysis import Analysis

DOWNBEAT_MIN_MARGIN = 0.2  # below this, warn: bar start may be wrong (fix with `aislice reslice --shift`)


@dataclass
class TempoMap:
    beats: np.ndarray            # cleaned + smoothed beat times (source seconds)
    confidence: np.ndarray       # 0..1 per beat
    beats_per_bar: int
    downbeat_index: int          # index into beats of the first downbeat
    raw_beats: np.ndarray
    low_conf_regions: list = field(default_factory=list)
    notes: list = field(default_factory=list)

    @property
    def local_bpm(self) -> np.ndarray:
        return 60.0 / np.diff(self.beats)


def _local_median(x: np.ndarray, half: int) -> np.ndarray:
    return np.array([np.median(x[max(0, i - half):i + half + 1]) for i in range(len(x))])


def _fix_missing_extra(beats: np.ndarray, notes: list) -> np.ndarray:
    """Split intervals ~2x/3x the local period, merge intervals ~0.5x."""
    for _ in range(3):
        ibi = np.diff(beats)
        med = _local_median(ibi, 8)
        ratio = ibi / med
        out = [beats[0]]
        i = 0
        changed = False
        while i < len(ibi):
            r = ratio[i]
            if r > 1.6:
                n = int(round(r))
                out.extend(beats[i] + ibi[i] * np.arange(1, n) / n)
                changed = True
                notes.append(f"inserted {n - 1} missing beat(s) near {beats[i]:.2f}s")
            if r < 0.6 and i + 1 < len(ibi):
                # drop the extra beat (merge with next interval)
                notes.append(f"removed extra beat near {beats[i + 1]:.2f}s")
                i += 1
                changed = True
                out.append(beats[i + 1])
                i += 1
                continue
            out.append(beats[i + 1])
            i += 1
        beats = np.array(out)
        if not changed:
            break
    return beats


def _weighted_local_linear(idx: np.ndarray, t: np.ndarray, w: np.ndarray, sigma: float) -> np.ndarray:
    """Smooth beat positions: weighted local linear regression of time vs. beat index."""
    out = np.empty_like(t)
    half = int(4 * sigma)
    for i in range(len(t)):
        lo, hi = max(0, i - half), min(len(t), i + half + 1)
        x = idx[lo:hi] - idx[i]
        ww = w[lo:hi] * np.exp(-0.5 * (x / sigma) ** 2)
        X = np.stack([np.ones_like(x), x], 1)
        A = X.T @ (X * ww[:, None])
        b = X.T @ (ww * t[lo:hi])
        out[i] = np.linalg.solve(A + 1e-9 * np.eye(2), b)[0]
    return out


def _confidence(a: Analysis, beats: np.ndarray) -> np.ndarray:
    fps = a.sr / a.hop
    fr = np.clip(np.round((beats + a.latency) * fps).astype(int), 0, len(a.onset_env) - 1)
    s = np.array([a.onset_env[max(0, f - 2):f + 3].max() for f in fr])
    return np.clip(s / (np.median(s) + 1e-9), 0, 1)


def _trim_unreliable_edges(beats, conf, notes):
    """Drop leading/trailing beats that are weak AND off-tempo (fade-ins/outs, noise)."""
    ibi = np.diff(beats)
    med = np.median(ibi)
    good = conf > 0.25
    # a beat is unreliable at the edges if weak; walk inwards until 4 consecutive good beats
    def first_run(mask):
        run = 0
        for i, m in enumerate(mask):
            run = run + 1 if m else 0
            if run == 4:
                return i - 3
        return 0
    start = first_run(good)
    end = len(beats) - 1 - first_run(good[::-1])
    if start > 0:
        notes.append(f"ignored {start} weak beat(s) at the start (before {beats[start]:.2f}s)")
    if end < len(beats) - 1:
        notes.append(f"ignored {len(beats) - 1 - end} weak beat(s) at the end (after {beats[end]:.2f}s); "
                     "grid extrapolated at the last reliable tempo")
    return start, end


def _downbeat_phase(a: Analysis, beats: np.ndarray, bpb: int) -> tuple[int, np.ndarray]:
    """Pick which beat is the bar's "1".

    Accents (bass hits, onsets) are strong on beats 1 AND 3 in 4/4, so on their own they can't tell
    1 from 3. Chord/harmony changes can: harmony mostly changes on the downbeat. Score =
    harmonic change at each phase (primary) + accent strength of its half-bar pair (secondary).
    """
    fps = a.sr / a.hop
    fr = np.clip(np.round((beats + a.latency) * fps).astype(int), 0, len(a.onset_env) - 1)
    n = len(fr)
    period = int(np.median(np.diff(fr))) if n > 1 else 1
    bass = np.array([a.bass_env[max(0, f - 3):f + 4].max() for f in fr])
    onset = np.array([a.onset_env[max(0, f - 3):f + 4].max() for f in fr])
    ch = a.chroma

    def change(f, w):
        pre = ch[:, max(0, f - w):f].mean(1)
        post = ch[:, f:f + w].mean(1)
        return 1 - pre @ post / (np.linalg.norm(pre) * np.linalg.norm(post) + 1e-9)

    harm = np.zeros(n)
    edge = bpb
    for i in range(edge, n - edge):
        harm[i] = change(fr[i], 2 * period) + change(fr[i], bpb * period)
    idx = np.arange(edge, n - edge)
    if len(idx) < 4 * bpb:
        return 0, np.zeros(bpb)
    z = lambda v: (v[idx] - v[idx].mean()) / (v[idx].std() + 1e-9)
    acc = z(bass) + 0.5 * z(onset)
    hz = z(harm)
    A = np.array([acc[idx % bpb == p].mean() for p in range(bpb)])
    H = np.array([hz[idx % bpb == p].mean() for p in range(bpb)])
    if bpb % 2 == 0:
        half = bpb // 2
        pair = np.array([A[p] + A[(p + half) % bpb] for p in range(bpb)])
        scores = H + 0.5 * pair
    else:
        scores = H + 0.5 * A
    # phase is relative to beat index 0 of `beats`
    return int(np.argmax(scores)), scores


def build_tempo_map(a: Analysis, beats_per_bar: int = 4, smooth_sigma_beats: float = 3.0) -> TempoMap:
    notes: list[str] = []
    raw = a.beats.copy()
    beats = _fix_missing_extra(raw, notes)
    conf = _confidence(a, beats)
    s, e = _trim_unreliable_edges(beats, conf, notes)
    core = beats[s:e + 1]
    cconf = conf[s:e + 1]

    idx = np.arange(len(core), dtype=float)
    w = 0.05 + cconf  # weak beats are pulled toward their neighbours
    smooth = _weighted_local_linear(idx, core, w, smooth_sigma_beats)

    # extrapolate grid to cover the whole song at the edge tempi
    p0 = np.median(np.diff(smooth[:9]))
    p1 = np.median(np.diff(smooth[-9:]))
    pre = []
    t = smooth[0] - p0
    while t > -1e-9:
        pre.append(t); t -= p0
    post = []
    t = smooth[-1] + p1
    while t < a.duration - 0.25 * p1:
        post.append(t); t += p1
    grid = np.concatenate([pre[::-1], smooth, post])
    gconf = np.concatenate([np.zeros(len(pre)), cconf, np.zeros(len(post))])

    # low-confidence regions (for the report)
    regions = []
    weak = gconf < 0.25
    i = 0
    while i < len(grid):
        if weak[i]:
            j = i
            while j + 1 < len(grid) and weak[j + 1]:
                j += 1
            if j - i >= 3:
                regions.append({"start_s": round(float(grid[i]), 2), "end_s": round(float(grid[j]), 2),
                                "beats": int(j - i + 1)})
            i = j + 1
        else:
            i += 1

    phase, scores = _downbeat_phase(a, grid, beats_per_bar)
    srt = np.sort(scores)[::-1]
    margin = float(srt[0] - srt[1]) if len(srt) > 1 else 0.0
    notes.append("downbeat phase scores " + ", ".join(f"{x:+.2f}" for x in scores))
    tm = TempoMap(grid, gconf, beats_per_bar, phase, raw, regions, notes)
    tm.downbeat_margin = margin
    tm.downbeat_confident = margin >= DOWNBEAT_MIN_MARGIN
    return tm
