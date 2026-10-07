"""End-to-end: drifting synthetic song -> locked grid. Needs rubberband + ffmpeg."""
import json
from pathlib import Path

import numpy as np
import soundfile as sf

from aislice.pipeline import Options, process
from conftest import needs_rubberband
from synth import beat_times, render


@needs_rubberband
def test_end_to_end_locks_drifting_song(tmp_path):
    sr = 44100
    truth = beat_times(118, 126, 45.0, start=0.0)
    y = render(truth, 45.0, sr)
    src = tmp_path / "drifty song.wav"
    sf.write(src, np.stack([y, y], 1), sr)

    out = process(src, Options(bpm=120, slice_unit="bar", out_root=tmp_path / "out", plot=False),
                  log=lambda *a: None)
    r = json.loads((out / "report.json").read_text())

    v = r["verification"]
    assert v["pass"] and v["p95_err_ms"] < 10
    assert all(abs(b - 120) < 0.1 for b in v["output_bpm_per_20s"])
    assert r["detected"]["p5_bpm"] < 120 < r["detected"]["p95_bpm"]

    full = out / "drifty song_120bpm.wav"
    info = sf.info(full)
    assert info.samplerate == sr and info.channels == 2

    slices = sorted((out / "slices").glob("*.wav"))
    assert len(slices) == r["slices"]["count"]
    lens = {sf.info(f).frames for f in slices[:-1]}
    assert lens <= {round(2 * sr) - 1, round(2 * sr), round(2 * sr) + 1}  # 1 bar @120 = 2 s
