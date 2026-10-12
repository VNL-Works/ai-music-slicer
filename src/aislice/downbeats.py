"""Optional downbeat (bar 1) detection with beat_this (CPJKU, MIT; pip install "aislice[beats]").

beat_this is excellent at *where bars start* but its beat times are too coarse for warping
(20 ms frames, often a few ms late). So aislice keeps its own precise beat grid for timing, and
only asks beat_this which grid beat is the "1": every beat_this downbeat votes for the grid
phase (index mod beats_per_bar) of the grid beat nearest to it.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

INSTALL_HINT = 'beat_this downbeats need the optional extra: pip install "aislice[beats]"'
MIN_SHARE = 0.6        # vote share needed to trust beat_this over the harmony heuristic
_MODEL = {}


@dataclass
class PhaseVote:
    phase: int            # index (0..bpb-1) of the first downbeat in the grid
    share: float          # fraction of beat_this downbeats voting for it
    votes: list[int]
    n_downbeats: int


def available() -> bool:
    import importlib.util
    return importlib.util.find_spec("beat_this") is not None


def detect_downbeats(y_mono: np.ndarray, sr: int, device: str = "cpu") -> np.ndarray:
    """Downbeat times (s) from beat_this. Replaced by a fake in tests."""
    try:
        from beat_this.inference import Audio2Beats
    except ModuleNotFoundError as e:  # pragma: no cover
        raise RuntimeError(INSTALL_HINT) from e
    if device not in _MODEL:
        _MODEL[device] = Audio2Beats(checkpoint_path="final0", device=device, dbn=False)
    _, downbeats = _MODEL[device](y_mono, sr)
    return np.asarray(downbeats, dtype=float)


def vote_phase(grid: np.ndarray, downbeats: np.ndarray, beats_per_bar: int,
               max_dist: float | None = None) -> PhaseVote | None:
    """Map each detected downbeat to the nearest grid beat and vote on index mod beats_per_bar."""
    if len(grid) < 2 or len(downbeats) == 0:
        return None
    period = float(np.median(np.diff(grid)))
    max_dist = 0.35 * period if max_dist is None else max_dist
    idx = np.searchsorted(grid, downbeats).clip(1, len(grid) - 1)
    nearest = np.where(np.abs(grid[idx] - downbeats) < np.abs(grid[idx - 1] - downbeats), idx, idx - 1)
    close = np.abs(grid[nearest] - downbeats) <= max_dist
    if close.sum() == 0:
        return None
    votes = np.bincount(nearest[close] % beats_per_bar, minlength=beats_per_bar)
    phase = int(np.argmax(votes))
    return PhaseVote(phase, float(votes[phase] / votes.sum()), votes.tolist(), int(close.sum()))
