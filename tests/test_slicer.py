import numpy as np
import pytest

from aislice import slicer


@pytest.mark.parametrize("unit,beats", [
    ("1/16", 0.25), ("1/8", 0.5), ("beat", 1), ("bar", 4), ("2bar", 8), ("4bar", 16),
    ("8bar", 32), ("16bar", 64), ("3beat", 3), ("8 bars", 32), ("BAR", 4),
])
def test_unit_beats(unit, beats):
    assert slicer.unit_beats(unit, 4) == beats


def test_unit_beats_respects_meter():
    assert slicer.unit_beats("bar", 3) == 3
    assert slicer.unit_beats("4bar", 3) == 12


@pytest.mark.parametrize("bad", ["0bar", "foo", "", "1/3", "-2bar"])
def test_unit_beats_rejects(bad):
    with pytest.raises(ValueError):
        slicer.unit_beats(bad, 4)


@pytest.mark.parametrize("bpm,sr", [(206, 48000), (120, 44100), (93.7, 48000), (174, 22050)])
def test_slice_points_exact_and_contiguous(bpm, sr):
    n = int(sr * 61.3)
    pts = slicer.slice_points(n, sr, bpm, 4)
    unit = 4 * 60 / bpm * sr
    # contiguous, covers everything, never drifts
    assert pts[0][0] == 0 and pts[-1][1] == n
    assert all(a[1] == b[0] for a, b in zip(pts, pts[1:]))
    lengths = {e - s for s, e in pts[:-1]}
    assert max(lengths) - min(lengths) <= 1
    assert all(abs(s - round(k * unit)) == 0 for k, (s, _) in enumerate(pts))


def test_drop_partial():
    sr, bpm = 48000, 120
    n = int(sr * 10.5)  # 5.25 bars of 2 s
    assert len(slicer.slice_points(n, sr, bpm, 4)) == 6
    assert len(slicer.slice_points(n, sr, bpm, 4, drop_partial=True)) == 5


def test_folder_names_roundtrip():
    for u in ["bar", "4bar", "8bar", "beat", "1/8", "1/16", "3beat"]:
        assert slicer.unit_for_folder(slicer.folder_for(u)) == u
    assert slicer.folder_for("bar") == "slices"
    assert slicer.unit_for_folder("not_slices") is None


def test_fades():
    seg = np.ones((4800, 2), dtype=np.float32)
    out = slicer.apply_fades(seg, 48000, 1.0)
    assert out[0, 0] == 0 and out[-1, 0] == 0 and out[2400, 0] == 1
    assert seg[0, 0] == 1  # input untouched
