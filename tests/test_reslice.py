import json

import numpy as np
import soundfile as sf

from aislice.reslice import reslice

SR, BPM = 48000, 120
T = 60 / BPM


def _fake_result(tmp_path, first_downbeat_s, silent_until_s=0.0):
    """An already-conformed song: a loud click on every downbeat, soft clicks on other beats."""
    d = tmp_path / "song_120bpm"
    d.mkdir()
    n = int(SR * 20)
    y = np.zeros((n, 2), np.float32)
    k = 0
    while True:
        t = first_downbeat_s + (k - 8) * T
        k += 1
        if t < silent_until_s:
            continue
        s = int(round(t * SR))
        if s >= n:
            break
        y[s:s + 200] = 0.8 if (k - 1) % 4 == 0 else 0.3
    sf.write(d / "song_120bpm.wav", y, SR, subtype="FLOAT")
    (d / "slices").mkdir()
    sf.write(d / "slices" / "stale_bar_999.wav", y[:100], SR)
    report = {"target_bpm": 120, "meter": "4/4", "duration": {"output_s": 20.0},
              "grid": {"first_downbeat_output_s": first_downbeat_s}}
    (d / "report.json").write_text(json.dumps(report))
    return d


def _loud_onsets(path):
    y, sr = sf.read(path)
    m = np.abs(y.mean(1)) > 0.5
    return np.flatnonzero(m & ~np.r_[False, m[:-1]]) / sr


def test_shift_moves_bar_one(tmp_path):
    # downbeats claimed at 0.0, but the loud clicks (true downbeats) are 2 beats later
    d = _fake_result(tmp_path, first_downbeat_s=0.0)
    y0, _ = sf.read(d / "song_120bpm.wav")
    # make the loud click land on beat 2: rewrite with an offset
    y = np.roll(y0, int(2 * T * SR), axis=0)
    sf.write(d / "song_120bpm.wav", y.astype(np.float32), SR, subtype="FLOAT")
    counts = reslice(d, shift_beats=2, log=lambda *a: None)
    onsets = _loud_onsets(d / "song_120bpm.wav")
    bar = 4 * T
    assert np.allclose((onsets + 1e-4) % bar, 1e-4, atol=1e-3), onsets[:4]  # loud clicks on bar lines
    assert counts == {"bar": len(list((d / "slices").glob("*.wav")))}
    assert not (d / "slices" / "stale_bar_999.wav").exists()
    r = json.loads((d / "report.json").read_text())
    assert r["grid"]["manual_shift_beats"] == 2


def test_new_units_and_equal_lengths(tmp_path):
    d = _fake_result(tmp_path, first_downbeat_s=0.0)
    counts = reslice(d, 0, units=["4bar", "8bar"], log=lambda *a: None)
    for u, folder in (("4bar", "slices_4bar"), ("8bar", "slices_8bar")):
        files = sorted((d / folder).glob("*.wav"))
        assert len(files) == counts[u]
        lens = {sf.info(f).frames for f in files[:-1]}
        assert max(lens) - min(lens) <= 1


def test_audible_pickup_is_padded_not_cut(tmp_path):
    d = _fake_result(tmp_path, first_downbeat_s=1.0)  # 2 audible beats (pickup) before bar 1
    before = sf.info(d / "song_120bpm.wav").frames
    reslice(d, 0, log=lambda *a: None)
    after = sf.info(d / "song_120bpm.wav").frames
    assert after > before  # padded with silence, nothing lost
    onsets = _loud_onsets(d / "song_120bpm.wav")
    assert abs(onsets[0] % (4 * T)) < 1e-3
