import os

import numpy as np
import pytest

from aislice import downbeats as db
from aislice.pipeline import choose_downbeat


class TM:  # minimal stand-in for TempoMap
    def __init__(self, n=64, period=0.5, phase=0):
        self.beats = np.arange(n) * period + 0.2
        self.beats_per_bar = 4
        self.downbeat_index = phase
        self.downbeat_confident = False
        self.notes = []


def test_vote_phase_maps_downbeats_to_grid_phase():
    g = np.arange(40) * 0.5
    dbt = g[2::4] + 0.012                       # downbeats on grid beats 2, 6, 10, ... (slightly late)
    v = db.vote_phase(g, dbt, 4)
    assert v.phase == 2 and v.share == 1.0 and v.n_downbeats == 10


def test_vote_phase_ignores_far_off_downbeats():
    g = np.arange(40) * 0.5
    dbt = np.r_[g[1::4], g[3::4][:3] + 0.25]    # 3 strays between beats
    v = db.vote_phase(g, dbt, 4)
    assert v.phase == 1 and v.share == 1.0


def test_beat_this_overrides_harmony_when_confident(monkeypatch):
    tm = TM(phase=0)
    monkeypatch.setattr(db, "available", lambda: True)
    monkeypatch.setattr(db, "detect_downbeats", lambda y, sr: tm.beats[2::4] + 0.01)
    info = choose_downbeat(tm, np.zeros(10), 22050, "auto", log=lambda *a: None)
    assert tm.downbeat_index == 2 and tm.downbeat_confident
    assert info["method"] == "beat_this" and info["harmony_phase"] == 0
    assert any("moved bar 1 by 2" in n for n in tm.notes)


def test_inconclusive_beat_this_keeps_harmony(monkeypatch):
    tm = TM(phase=1)
    monkeypatch.setattr(db, "available", lambda: True)
    mixed = np.sort(np.r_[tm.beats[0::4], tm.beats[2::4]])   # 50/50 split
    monkeypatch.setattr(db, "detect_downbeats", lambda y, sr: mixed)
    info = choose_downbeat(tm, np.zeros(10), 22050, "auto", log=lambda *a: None)
    assert tm.downbeat_index == 1 and info["method"] == "harmony"


def test_harmony_mode_and_missing_model(monkeypatch):
    tm = TM(phase=3)
    monkeypatch.setattr(db, "available", lambda: False)
    assert choose_downbeat(tm, np.zeros(10), 22050, "auto", log=lambda *a: None)["method"] == "harmony"
    monkeypatch.setattr(db, "available", lambda: True)
    called = []
    monkeypatch.setattr(db, "detect_downbeats", lambda y, sr: called.append(1))
    choose_downbeat(tm, np.zeros(10), 22050, "harmony", log=lambda *a: None)
    assert not called and tm.downbeat_index == 3


def test_model_failure_falls_back_in_auto(monkeypatch):
    tm = TM(phase=1)
    monkeypatch.setattr(db, "available", lambda: True)

    def boom(y, sr):
        raise RuntimeError("download failed")
    monkeypatch.setattr(db, "detect_downbeats", boom)
    assert choose_downbeat(tm, np.zeros(10), 22050, "auto", log=lambda *a: None)["method"] == "harmony"
    with pytest.raises(RuntimeError):
        choose_downbeat(tm, np.zeros(10), 22050, "beat_this", log=lambda *a: None)


@pytest.mark.skipif(os.environ.get("AISLICE_REAL_BEATS") != "1",
                    reason="set AISLICE_REAL_BEATS=1 to run the real beat_this model (downloads ~80 MB)")
def test_real_beat_this_finds_bar_one():
    from aislice.analysis import analyze
    from aislice.tempomap import build_tempo_map
    from synth import beat_times, render
    sr = 22050
    for first_db in (0, 2):
        truth = beat_times(118, 124, 40.0, start=0.4)
        y = render(truth, 40.0, sr, first_downbeat=first_db, seed=first_db)
        tm = build_tempo_map(analyze(y, sr, bpm_hint=121))
        info = choose_downbeat(tm, y, sr, "beat_this", log=lambda *a: None)
        k = int(np.argmin(np.abs(tm.beats - truth[first_db])))
        assert (k - tm.downbeat_index) % 4 == 0, info
