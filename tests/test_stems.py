"""Stem separation plumbing, tested with a fake separator (no ML models needed)."""
import json
import os
from pathlib import Path

import librosa
import numpy as np
import pytest
import soundfile as sf

from aislice import stems as st
from aislice.reslice import reslice
from aislice.stems_cmd import add_stems_to_result


def fake_backend(names=("Vocals", "Drums", "Bass", "Other"), delay=0):
    """Mimics audio-separator: reads the file at 44.1 kHz, writes '<name>_(Stem)_model.wav' files.

    Splits the mix into bands so the stems sum back to the mix; `delay` (44.1 kHz samples)
    simulates a model that shifts its output.
    """
    def backend(wav, model, out_dir, model_dir=None):
        out_dir.mkdir(parents=True, exist_ok=True)
        y, sr = sf.read(wav, always_2d=True)
        y = librosa.resample(y.T, orig_sr=sr, target_sr=44100, res_type="soxr_vhq").T
        spec = np.fft.rfft(y, axis=0)
        edges = np.linspace(0, len(spec), len(names) + 1).astype(int)
        files = {}
        for i, n in enumerate(names):
            part = np.zeros_like(spec)
            part[edges[i]:edges[i + 1]] = spec[edges[i]:edges[i + 1]]
            s = np.fft.irfft(part, n=len(y), axis=0)
            if delay:
                s = np.concatenate([np.zeros((delay, s.shape[1])), s[:-delay]])
            f = out_dir / f"mix_({n})_{model.split('.')[0]}.wav"
            sf.write(f, s.astype(np.float32), 44100, subtype="FLOAT")
            files[st._stem_name(f.name)] = f
        return files
    return backend


def _song(sr=48000, seconds=8.0, seed=0):
    rng = np.random.default_rng(seed)
    t = np.arange(int(sr * seconds)) / sr
    # tones + band-limited noise (real music has ~no energy above the models' 22.05 kHz Nyquist)
    y = 0.2 * np.sin(2 * np.pi * 110 * t) + 0.1 * np.sin(2 * np.pi * 1760 * t)
    noise = np.fft.rfft(rng.standard_normal(len(t)))
    noise[int(len(noise) * 16000 / (sr / 2)):] = 0
    y = y + 0.02 * np.fft.irfft(noise, n=len(t)) / 0.6
    return np.stack([y, 0.9 * y], 1).astype(np.float32), sr


def test_stem_name_parsing():
    assert st._stem_name("song_(Vocals)_htdemucs_ft.wav") == "vocals"
    assert st._stem_name("a (b)_(No Drums)_model.wav") == "no_drums"
    assert st._stem_name("plain.wav") == "plain"


def test_separate_keeps_rate_length_and_sums_to_mix():
    mix, sr = _song()
    res = st.separate(mix, sr, "4", backend=fake_backend())
    assert set(res.stems) == {"vocals", "drums", "bass", "other"}
    for y in res.stems.values():
        assert y.shape == mix.shape
    assert res.residual_db < -30
    assert res.lag_samples == 0


def test_separate_corrects_model_delay():
    mix, sr = _song()
    res = st.separate(mix, sr, "4", backend=fake_backend(delay=441))   # 10 ms late at 44.1 kHz
    assert abs(res.lag_samples + 480) <= 1                               # = 10 ms at 48 kHz, shifted back
    assert res.residual_db < -25


def test_two_stem_preset_merges_non_vocals():
    mix, sr = _song()
    res = st.separate(mix, sr, "2", backend=fake_backend())
    assert list(res.stems) == ["vocals", "instrumental"]
    both = res.stems["vocals"] + res.stems["instrumental"]
    assert np.allclose(both[200:-200], mix[200:-200], atol=1e-4)   # edges: resampler ramp-in/out
    assert res.residual_db < -60


def test_unknown_preset():
    with pytest.raises(ValueError):
        st.separate(*_song(), preset="5", backend=fake_backend())


def _result(tmp_path, sr=48000, bpm=120):
    d = tmp_path / "song_120bpm"
    d.mkdir()
    mix, _ = _song(sr, seconds=20.0)
    sf.write(d / "song_120bpm.wav", mix, sr, subtype="FLOAT")
    (d / "slices").mkdir()
    (d / "slices_4bar").mkdir()
    r = {"target_bpm": bpm, "meter": "4/4", "duration": {"output_s": 20.0},
         "grid": {"first_downbeat_output_s": 0.0}}
    (d / "report.json").write_text(json.dumps(r))
    return d, mix


def test_add_stems_to_existing_result(tmp_path):
    d, mix = _result(tmp_path)
    info = add_stems_to_result(d, "4", backend=fake_backend(), log=lambda *a: None)
    assert sorted(p.name for p in (d / "stems").glob("*.wav")) == ["bass.wav", "drums.wav", "other.wav", "vocals.wav"]
    # every stem sliced on the same grid as the mix: 20 s @120 BPM = 10 bars / 2.5 x 4-bar
    for unit, n in (("slices", 10), ("slices_4bar", 3)):
        for name in ("vocals", "drums", "bass", "other"):
            files = sorted((d / unit / name).glob("*.wav"))
            assert len(files) == n
            assert files[0].name.startswith(f"song_{name}_")
    a = sf.read(d / "slices" / "vocals" / "song_vocals_bar_001.wav")[0]
    assert len(a) == 2 * 48000
    assert json.loads((d / "report.json").read_text())["stems"]["names"] == info["names"]


def test_reslice_moves_stems_with_the_mix(tmp_path):
    d, mix = _result(tmp_path)
    add_stems_to_result(d, "4", units=["bar"], backend=fake_backend(), log=lambda *a: None)
    reslice(d, shift_beats=1, log=lambda *a: None)   # music starts at 0 -> pickup padded to a full bar
    y, sr = sf.read(d / "song_120bpm.wav", always_2d=True)
    total = sum(sf.read(p, always_2d=True)[0] for p in (d / "stems").glob("*.wav"))
    assert total.shape == y.shape
    err = np.mean((y - total) ** 2) / np.mean(y ** 2)
    assert err < 1e-3                                  # stems still line up with the moved mix
    n_mix = len(list((d / "slices").glob("*.wav")))
    assert all(len(list((d / "slices" / s).glob("*.wav"))) == n_mix for s in ("vocals", "drums", "bass", "other"))


def test_cli_without_extra_fails_early(monkeypatch, tmp_path, capsys):
    import importlib.util

    from aislice.cli import main
    real = importlib.util.find_spec
    monkeypatch.setattr(importlib.util, "find_spec", lambda n, *a: None if n == "audio_separator" else real(n, *a))
    f = tmp_path / "a.wav"
    sf.write(f, np.zeros((4800, 2), np.float32), 48000)
    with pytest.raises(SystemExit) as e:
        main([str(f), "--bpm", "120", "--slice", "bar", "--stems", "4"])
    assert e.value.code == 2
    assert "aislice[stems]" in capsys.readouterr().err


@pytest.mark.skipif(os.environ.get("AISLICE_REAL_STEMS") != "1",
                    reason="set AISLICE_REAL_STEMS=1 to run a real separation model (downloads ~300 MB)")
def test_real_demucs_separation():
    from synth import beat_times, render
    sr = 44100
    y = render(beat_times(120, 120, 12.0), 12.0, sr)
    mix = np.stack([y, y], 1)
    md = os.environ.get("AISLICE_MODEL_DIR")
    res = st.separate(mix, sr, os.environ.get("AISLICE_STEMS_PRESET", "4"),
                      model=os.environ.get("AISLICE_STEM_MODEL") or None, model_dir=Path(md) if md else None)
    assert len(res.stems) >= 2
    for s in res.stems.values():
        assert s.shape == mix.shape
        assert np.sqrt(np.mean(s ** 2)) > 0                  # no empty/broken stem
    # htdemucs_ft uses one specialist model per source, so its stems don't sum exactly to the mix:
    # ~-19 dB on real music, ~-10 dB on these synthetic clicks. A broken separation is ~0 dB.
    assert res.residual_db < -6
    assert abs(res.lag_samples) < sr * 0.005


def test_three_stem_preset_drums_bass_melody():
    mix, sr = _song()
    res = st.separate(mix, sr, "3", backend=fake_backend())
    assert list(res.stems) == ["drums", "bass", "melody"]
    total = sum(res.stems.values())
    assert np.allclose(total[200:-200], mix[200:-200], atol=1e-4)


def test_rerun_with_other_split_removes_old_stem_folders(tmp_path):
    d, mix = _result(tmp_path)
    add_stems_to_result(d, "2", units=["bar"], backend=fake_backend(), log=lambda *a: None)
    assert (d / "slices" / "vocals").is_dir() and (d / "slices" / "instrumental").is_dir()
    add_stems_to_result(d, "3", units=["bar"], backend=fake_backend(), log=lambda *a: None)
    assert sorted(p.name for p in (d / "slices").iterdir() if p.is_dir()) == ["bass", "drums", "melody"]
    assert sorted(p.stem for p in (d / "stems").glob("*.wav")) == ["bass", "drums", "melody"]


def test_stems_cli_slice_none_means_no_stem_slices(tmp_path, monkeypatch):
    import aislice.stems_cmd as sc
    from aislice.cli import main
    d, mix = _result(tmp_path)
    seen = {}
    real = sc.add_stems_to_result
    monkeypatch.setattr(sc, "add_stems_to_result",
                        lambda *a, **k: seen.setdefault("units", a[4]) or real(*a, backend=fake_backend(), **k))
    monkeypatch.setattr("aislice.cli._require_stems_extra", lambda: None)
    assert main(["stems", str(d), "--stems", "3", "--slice", "none"]) == 0
    assert seen["units"] == []


def test_default_is_htdemucs_ft_and_fast_switch():
    seen = []

    def spy(wav, model, out_dir, model_dir=None):
        seen.append(model)
        return fake_backend()(wav, model, out_dir, model_dir)
    mix, sr = _song(seconds=2.0)
    st.separate(mix, sr, "3", backend=spy)
    st.separate(mix, sr, "3", backend=spy, fast=True)
    st.separate(mix, sr, "6", backend=spy, fast=True)          # no fine-tuned 6-stem: stays htdemucs_6s
    assert seen == ["htdemucs_ft.yaml", "htdemucs.yaml", "htdemucs_6s.yaml"]


def test_stems_exact_recombines_to_the_mix():
    def lossy(wav, model, out_dir, model_dir=None):
        files = fake_backend()(wav, model, out_dir, model_dir)
        for f in files.values():                      # model that "loses" 20% of every stem
            y, r = sf.read(f)
            sf.write(f, (0.8 * y).astype(np.float32), r, subtype="FLOAT")
        return files
    mix, sr = _song()
    plain = st.separate(mix, sr, "3", backend=lossy)
    assert plain.residual_db > -20
    exact = st.separate(mix, sr, "3", backend=lossy, exact=True)
    total = sum(exact.stems.values())
    assert np.allclose(total, mix, atol=1e-5)
    assert np.allclose(exact.stems["drums"], plain.stems["drums"])        # only melody gets the leftover
    assert any("leftover" in n for n in exact.notes)
