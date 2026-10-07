import numpy as np

from aislice.analysis import analyze
from aislice.tempomap import build_tempo_map
from synth import beat_times, render

SR = 22050


def _match(found, truth, tol=0.05):
    """For each true beat, the error to the nearest found beat (s)."""
    idx = np.searchsorted(found, truth).clip(1, len(found) - 1)
    near = np.where(np.abs(found[idx] - truth) < np.abs(found[idx - 1] - truth), found[idx], found[idx - 1])
    return near - truth


def test_tracks_drifting_tempo_precisely():
    truth = beat_times(120, 128, 40.0, start=0.5)
    y = render(truth, 40.0, SR)
    a = analyze(y, SR, bpm_hint=124)
    err = _match(a.beats, truth[2:-2])
    assert np.median(np.abs(err)) < 0.004            # < 4 ms
    assert np.percentile(np.abs(err), 95) < 0.010
    assert abs(len(a.beats) - len(truth)) <= 2


def test_beats_sit_on_attacks_not_after():
    """Regression: a double-centered onset envelope once put every beat ~42 ms late."""
    truth = beat_times(140, 140, 30.0, start=0.3)
    a = analyze(render(truth, 30.0, SR), SR, bpm_hint=140)
    err = _match(a.beats, truth[2:-2])
    assert -0.006 < np.median(err) < 0.002            # just before the attack (2 ms margin)


def test_downbeat_uses_harmony_not_accents():
    """Beats 1 and 3 are equally loud; only chord changes mark bar 1."""
    for first_db in (0, 1, 2, 3):
        truth = beat_times(118, 124, 40.0, start=0.4)
        y = render(truth, 40.0, SR, first_downbeat=first_db, seed=first_db)
        tm = build_tempo_map(analyze(y, SR, bpm_hint=121))
        # find the grid beat closest to the first true downbeat
        true_db = truth[first_db]
        k = int(np.argmin(np.abs(tm.beats - true_db)))
        assert (k - tm.downbeat_index) % 4 == 0, f"pickup={first_db}: picked {tm.downbeat_index}, want {k % 4}"
