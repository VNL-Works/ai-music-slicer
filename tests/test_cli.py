import pytest

from aislice.cli import main


def test_version(capsys):
    with pytest.raises(SystemExit) as e:
        main(["--version"])
    assert e.value.code == 0


def test_invalid_slice_unit_is_rejected(capsys, tmp_path):
    f = tmp_path / "a.wav"
    f.write_bytes(b"")
    with pytest.raises(SystemExit) as e:
        main([str(f), "--bpm", "120", "--slice", "7/9"])
    assert e.value.code == 2
    assert "invalid slice unit" in capsys.readouterr().err


def test_no_audio_found(tmp_path, capsys):
    assert main([str(tmp_path), "--bpm", "120", "--slice", "bar"]) == 1


def test_reslice_help(capsys):
    with pytest.raises(SystemExit):
        main(["reslice", "--help"])
    assert "--shift" in capsys.readouterr().out
