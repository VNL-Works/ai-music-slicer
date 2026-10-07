"""Beat analysis: onset envelope, local tempo curve, time-varying DP beat tracker."""
from __future__ import annotations

from dataclasses import dataclass

import librosa
import numpy as np

ANALYSIS_SR = 22050
HOP = 64  # ~2.9 ms frames for timing precision
ATTACK_MARGIN = 0.002  # place grid lines this far before the attack so slices keep the transient


@dataclass
class Analysis:
    sr: int
    hop: int
    onset_env: np.ndarray        # onset strength (normalized)
    bass_env: np.ndarray         # low-frequency onset strength (for downbeats)
    chroma: np.ndarray           # chroma at analysis frames (for downbeats)
    local_bpm: np.ndarray        # per-frame local tempo estimate
    latency: float               # removed constant offset (s) between onset peaks and attacks
    beats: np.ndarray            # beat times (seconds), raw tracked
    beat_strength: np.ndarray    # onset strength at each beat (0..1)
    duration: float

    def frames_to_time(self, f):
        return np.asarray(f) * self.hop / self.sr


def _onset_envelopes(y: np.ndarray, sr: int):
    S = np.abs(librosa.stft(y, n_fft=1024, hop_length=HOP))
    mel = librosa.feature.melspectrogram(S=S**2, sr=sr, n_mels=96, fmax=11000)
    logmel = librosa.power_to_db(mel, ref=np.max)
    # SuperFlux-style: max-filter across frequency to suppress vibrato
    # S is already from a centered STFT, so center=False (otherwise librosa delays it ~46 ms)
    env = librosa.onset.onset_strength(S=logmel, sr=sr, hop_length=HOP, lag=2, max_size=3, center=False)
    bass = librosa.onset.onset_strength(S=logmel[:18], sr=sr, hop_length=HOP, lag=2, center=False)
    chroma = librosa.feature.chroma_stft(S=S, sr=sr, hop_length=HOP)
    norm = lambda e: e / (np.percentile(e, 99.5) + 1e-9)
    return norm(env), norm(bass), chroma


def attack_latency(y: np.ndarray, sr: int, beats: np.ndarray, strength: np.ndarray) -> float:
    """Median (beat marker - steepest attack) in seconds, measured on strong beats.

    The onset envelope peaks slightly after a note's attack; slicing needs the attack itself.
    """
    yh = librosa.effects.preemphasis(y, coef=0.97)
    h, w = 22, 88  # ~1 ms hop, ~4 ms window at 22.05 kHz
    edb = 20 * np.log10(librosa.feature.rms(y=yh, frame_length=w, hop_length=h, center=True)[0] + 1e-9)
    lat = []
    for b, s in zip(beats, strength):
        if s < 0.8:
            continue
        c = int(b * sr / h)
        lo, hi = c - int(0.06 * sr / h), c + int(0.03 * sr / h)
        if lo < 5 or hi >= len(edb):
            continue
        k = int(np.argmax(np.diff(edb[lo:hi])))
        lat.append(b - (lo + k + 1) * h / sr)
    return float(np.median(lat)) if len(lat) >= 16 else 0.0


def local_tempo_curve(env: np.ndarray, sr: int, hop: int, bpm_center: float,
                      win_s: float = 8.0, rel_range: float = 0.15) -> np.ndarray:
    """Per-frame tempo inside [center*(1-r), center*(1+r)] from a windowed autocorrelation."""
    fps = sr / hop
    win = int(win_s * fps)
    step = int(0.5 * fps)
    lo = 60 / (bpm_center * (1 + rel_range)) * fps
    hi = 60 / (bpm_center * (1 - rel_range)) * fps
    lags = np.arange(int(np.floor(lo)), int(np.ceil(hi)) + 1)
    w = np.hanning(win)
    centers, bpms = [], []
    x = np.pad(env - env.mean(), (win // 2, win // 2))
    for c in range(0, len(env), step):
        seg = x[c:c + win] * w
        if len(seg) < win:
            seg = np.pad(seg, (0, win - len(seg)))
        # autocorrelation at candidate lags, plus 2x lag for robustness (comb)
        ac = np.array([np.dot(seg[:-L], seg[L:]) + 0.5 * np.dot(seg[:-2 * L], seg[2 * L:]) for L in lags])
        k = int(np.argmax(ac))
        # parabolic refinement
        if 0 < k < len(ac) - 1:
            a, b, cc = ac[k - 1], ac[k], ac[k + 1]
            den = a - 2 * b + cc
            off = 0.5 * (a - cc) / den if den != 0 else 0.0
        else:
            off = 0.0
        L = lags[k] + off
        centers.append(c)
        bpms.append(60 * fps / L)
    bpms = np.array(bpms)
    # median smooth over ~5 s
    from scipy.ndimage import median_filter
    bpms = median_filter(bpms, size=11, mode="nearest")
    return np.interp(np.arange(len(env)), centers, bpms)


def track_beats(env: np.ndarray, local_bpm: np.ndarray, sr: int, hop: int,
                tightness: float = 300.0) -> np.ndarray:
    """Dynamic-programming beat tracker with a time-varying expected period."""
    fps = sr / hop
    n = len(env)
    period = 60 * fps / local_bpm
    # local score: onset env smoothed with a narrow gaussian (~10 ms)
    g = np.exp(-0.5 * (np.arange(-4, 5) / 1.5) ** 2)
    score_local = np.convolve(env, g / g.sum(), mode="same")
    cum = score_local.copy()
    back = -np.ones(n, dtype=int)
    for t in range(n):
        p = period[t]
        lo, hi = int(t - 1.35 * p), int(t - 0.7 * p)
        if hi <= 0:
            continue
        lo = max(lo, 0)
        cand = np.arange(lo, hi + 1)
        pen = -tightness * np.log((t - cand) / p) ** 2
        vals = cum[cand] + pen
        k = int(np.argmax(vals))
        if vals[k] > 0:
            cum[t] = score_local[t] + vals[k]
            back[t] = cand[k]
    # end: best cumulative score within last period, among local maxima
    p_end = period[-1]
    tail = np.arange(max(0, n - int(1.5 * p_end)), n)
    # only consider frames with some onset to avoid trailing silence
    thr = 0.5 * np.median(cum[librosa.util.localmax(cum)])
    tail_ok = tail[cum[tail] > thr] if np.any(cum[tail] > thr) else tail
    t = tail_ok[np.argmax(cum[tail_ok])]
    path = [t]
    while back[t] >= 0:
        t = back[t]
        path.append(t)
    beats = np.array(path[::-1], dtype=float)
    # trim leading/trailing beats over silence
    strength = score_local[beats.astype(int)]
    ok = strength > 0.15 * np.median(strength)
    first, last = np.argmax(ok), len(ok) - 1 - np.argmax(ok[::-1])
    beats = beats[first:last + 1]
    # sub-frame refinement: parabolic interpolation on local onset peak (+-2 frames)
    refined = []
    for b in beats.astype(int):
        lo, hi = max(b - 3, 1), min(b + 3, n - 2)
        k = lo + int(np.argmax(score_local[lo:hi + 1]))
        a, bb, c = score_local[k - 1], score_local[k], score_local[k + 1]
        den = a - 2 * bb + c
        off = 0.5 * (a - c) / den if den != 0 else 0.0
        refined.append(k + float(np.clip(off, -0.5, 0.5)))
    return np.array(refined) / fps


def analyze(y_mono: np.ndarray, sr: int, bpm_hint: float | None = None) -> Analysis:
    if sr != ANALYSIS_SR:
        y_mono = librosa.resample(y_mono, orig_sr=sr, target_sr=ANALYSIS_SR)
        sr = ANALYSIS_SR
    env, bass, chroma = _onset_envelopes(y_mono, sr)
    if bpm_hint is None:
        bpm_hint = float(librosa.feature.tempo(onset_envelope=env, sr=sr, hop_length=HOP)[0])
    local = local_tempo_curve(env, sr, HOP, bpm_hint)
    beats = track_beats(env, local, sr, HOP)
    fps = sr / HOP
    strength = np.clip(env[np.round(beats * fps).astype(int).clip(0, len(env) - 1)], 0, 1)
    lat = attack_latency(y_mono, sr, beats, strength)
    lat += ATTACK_MARGIN
    beats = beats - lat
    return Analysis(sr, HOP, env, bass, chroma, local, lat, beats, strength, len(y_mono) / sr)
