"""AI songs in, best samples out: one flat, timestamped folder per run."""
import json

import numpy as np
import pytest
import soundfile as sf
from synth import write_pick_result

from aislice import export as ex
from aislice.best import best_name, build_best, session_dir
from aislice.cli import main


def test_session_folder_names(tmp_path):
    d = session_dir(tmp_path, 206, when=0)
    assert d.parent == tmp_path and d.name.endswith("_206bpm") and len(d.name) == len("1970-01-01_0000_206bpm")
    d.mkdir()
    assert session_dir(tmp_path, 206, when=0).name == d.name + "_2"        # same minute: never mixed
    assert best_name("drums", 1, "Piano Core (1)", 206, "4bar", 25, "Am") == "Drums_01_PianoCore1_206bpm_Am_4bar25.wav"
    assert best_name("melody", 12, "x", 120.5, "8bar", 3, None) == "Melody_12_x_120_5bpm_8bar03.wav"


def test_best_samples_across_songs_flat(tmp_path):
    a = write_pick_result(tmp_path / "work", "One")
    b = write_pick_result(tmp_path / "work", "Two Song")
    out = tmp_path / "2026-10-13_1530_240bpm"
    files = build_best([a, b], out, n=2, log=lambda *x: None)
    assert all(f.parent == out for f in files)                                       # flat: no sub-folders
    assert not any(p.is_dir() for p in out.iterdir())
    names = sorted(f.name for f in files)
    assert {n.split("_")[0] for n in names} == {"Drums", "Melody", "Vocals"}          # no mix
    drums = [n for n in names if n.startswith("Drums_")]
    assert [n[:9] for n in drums] == ["Drums_01_", "Drums_02_"]
    assert {n.split("_")[2] for n in drums} == {"One", "TwoSong"}                    # ranked across songs
    assert all("_4bar" in n for n in drums) and all("_8bar" in n for n in names if n.startswith("Melody_"))
    for f in files:
        y, _ = sf.read(f)
        assert abs(20 * np.log10(np.abs(y).max()) + 0.1) < 0.01                      # normalized
        assert ex.read_acid(f)["bpm"] == 240
    info = json.loads((out / "best.json").read_text())
    assert info["per_instrument"] == 2 and info["songs"] == ["One", "Two Song"]


def test_bpm_is_required(tmp_path, capsys):
    f = tmp_path / "a.wav"
    sf.write(f, np.zeros((4800, 2), np.float32), 48000)
    with pytest.raises(SystemExit):
        main([str(f)])
    assert "--bpm" in capsys.readouterr().err
