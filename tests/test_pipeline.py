"""End-to-end: drifting synthetic song -> locked grid. Needs rubberband + ffmpeg."""
import json
from pathlib import Path

import numpy as np
import pytest
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


@needs_rubberband
def test_main_command_songs_in_best_samples_out(tmp_path, monkeypatch):
    """The product: aislice <songs> --bpm N -> _best/<N>bpm/{Drums,Bass,Melody}, normalized, no prompts."""
    import aislice.cli as cli
    from aislice import export as ex
    from aislice import stems as st
    def fake_split(wav, model, out_dir, model_dir=None):           # every "stem" = a share of the mix
        out_dir.mkdir(parents=True, exist_ok=True)
        y, r = sf.read(wav, always_2d=True)
        files = {}
        for name, share in (("Drums", 0.3), ("Bass", 0.3), ("Other", 0.4)):
            f = out_dir / f"mix_({name})_fake.wav"
            sf.write(f, (share * y).astype(np.float32), r, subtype="FLOAT")
            files[name.lower()] = f
        return files
    monkeypatch.setattr(cli, "_require_stems_extra", lambda: None)
    monkeypatch.setattr(st, "run_backend", fake_split)
    monkeypatch.setattr("builtins.input", lambda *a: pytest.fail("the main command must not ask anything"))
    songs = tmp_path / "songs"
    songs.mkdir()
    sr = 44100
    for k, (a, b) in enumerate([(118, 124), (121, 117)]):
        y = render(beat_times(a, b, 40.0), 40.0, sr)
        sf.write(songs / f"song {k}.wav", np.stack([y, 0.8 * y], 1), sr)
    assert cli.main([str(songs), "--bpm", "120", "--out", str(tmp_path / "out"), "--best", "3"]) == 0
    out = tmp_path / "out"
    sessions = [p for p in out.iterdir()]
    assert len(sessions) == 1 and sessions[0].name.endswith("_120bpm")            # only the run's folder is left
    best = sessions[0]
    assert not any(p.is_dir() for p in best.iterdir())                             # flat
    for stem, bars in (("Drums", 4), ("Bass", 4), ("Melody", 8)):
        files = sorted(best.glob(f"{stem}_*.wav"))
        assert 1 <= len(files) <= 3 and files[0].name.startswith(f"{stem}_01_")
        for f in files:
            y, fsr = sf.read(f)
            assert abs(len(y) - bars * 2 * fsr) <= 1                    # exact 4 / 8 bars at 120 BPM
            assert abs(20 * np.log10(np.abs(y).max()) + 0.1) < 0.01     # normalized
            assert ex.read_acid(f)["bpm"] == 120
    assert {f.name.split("_")[2] for f in best.glob("*.wav")} <= {"song0", "song1"}
    # --keep-work keeps the per-song files (standard layout, normalized)
    assert cli.main([str(songs / "song 0.wav"), "--bpm", "120", "--out", str(tmp_path / "o2"), "--keep-work"]) == 0
    d = next((tmp_path / "o2").glob(".work_*")) / "song 0_120bpm"
    assert (d / "slices_4bar" / "drums").is_dir() and (d / "slices_8bar" / "melody").is_dir()
    assert not (d / "slices_8bar" / "drums").exists() and not (d / "slices").exists()
    r = json.loads((d / "report.json").read_text())
    assert set(r["normalize"]["gains_db"]) == {"mix", "drums", "bass", "melody"}
    y, _ = sf.read(d / "song 0_120bpm.wav")
    assert abs(20 * np.log10(np.abs(y).max()) + 0.1) < 0.01
