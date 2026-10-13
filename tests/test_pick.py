"""The Picker: empty slices, repeat groups, key, export names + acid chunk, --auto, the local server."""
import json
import threading
import urllib.error
import urllib.request

import numpy as np
import pytest
import soundfile as sf
from synth import write_pick_result

from aislice import export as ex
from aislice import pick
from aislice.cli import main


@pytest.fixture(scope="module")
def song(tmp_path_factory):
    root = tmp_path_factory.mktemp("out")
    d = write_pick_result(root, units=("8bar", "bar"))
    return root, d


@pytest.fixture(scope="module")
def model(song):
    return pick.analyze_song(song[1], "8bar", use_cache=False)


def row(model, name):
    return next(r for r in model["rows"] if r["name"] == name)


def test_finds_songs_and_default_unit(song):
    root, d = song
    assert pick.find_songs(root) == (root.resolve(), [d.resolve()])
    assert pick.find_songs(d) == (root.resolve(), [d.resolve()])        # a song folder: _ready goes next to it
    info = pick.song_info(d)
    assert list(info["units"]) == ["bar", "8bar"] and pick.default_unit(info) == "8bar"
    assert list(info["units"]["8bar"]) == ["mix", "drums", "melody", "vocals"]
    assert all(len(fs) == 6 for fs in info["units"]["8bar"].values())


def test_empty_slices_are_hidden(model):
    v = row(model, "vocals")["cells"]
    assert [c["empty"] for c in v] == [False, False, True, True, True, True]
    assert [c["why"] for c in v] == ["", "", "silent", "sparse", "silent", "silent"]   # slice 4: one stray blip
    d = row(model, "drums")["cells"]
    assert [c["empty"] for c in d] == [False] * 5 + [True]
    assert not any(c["empty"] for c in row(model, "melody")["cells"])


def test_repeats_grouped_per_stem_in_beat_order(model):
    groups = lambda name: {frozenset(c["i"] for c in row(model, name)["cells"] if c["group"] == g)
                           for g in {c["group"] for c in row(model, name)["cells"] if not c["empty"]}}
    assert groups("drums") == {frozenset({1, 2, 3, 4, 5})}               # same beat every section
    # A and B use the same four notes in a different order: averaged features would merge them
    assert groups("melody") == {frozenset({1, 2, 4}), frozenset({3, 5}), frozenset({6})}
    for r in model["rows"]:
        for g in {c["group"] for c in r["cells"] if not c["empty"]}:
            members = [c for c in r["cells"] if c["group"] == g]
            assert sum(c["rep"] for c in members) == 1 and all(c["count"] == len(members) for c in members)


def test_order_matters_but_averages_dont():
    """Averaged over the slice, A and B look the same; beat by beat they don't."""
    from synth import pick_song
    import librosa
    stems, sr, bpm = pick_song()
    y = librosa.resample(stems["melody"][:, 0], orig_sr=sr, target_sr=pick.AN_SR)
    sec = len(y) // 6
    fa, fb, fa2 = (pick.slice_features(y[k * sec:(k + 1) * sec], bpm, 32, "tonal") for k in (0, 2, 1))
    avg = np.corrcoef(fa.mean(0), fb.mean(0))[0, 1]
    sim = pick.similarity_matrix([fa, fb, fa2], "tonal")
    assert avg > 0.95                       # the old way: "the same"
    assert sim[0, 1] < 0.3 < 0.9 < sim[0, 2]


def test_group_sim_zero_turns_grouping_off(song):
    m = pick.analyze_song(song[1], "8bar", {"group_sim": 0}, use_cache=False)
    assert all(c["count"] == 1 for r in m["rows"] for c in r["cells"])


def test_cache_makes_reopening_instant(song, monkeypatch):
    pick.analyze_song(song[1], "8bar")                                   # fills pick_cache.json
    monkeypatch.setattr(pick, "slice_features", lambda *a, **k: pytest.fail("cache was not used"))
    m = pick.analyze_song(song[1], "8bar")
    assert m["layout"]["drums"] == "8bar" and (song[1] / pick.CACHE_NAME).is_file()


def test_key_estimate():
    sr = 22050
    t = np.arange(int(sr * 1.0)) / sr
    chord = lambda fs: sum(np.sin(2 * np.pi * f * t) for f in fs)
    y = np.concatenate([chord(fs) for fs in ([220, 261.6, 329.6], [293.7, 349.2, 440], [329.6, 415.3, 493.9],
                                             [220, 261.6, 329.6])] * 3)
    k = pick.estimate_key(y.astype(np.float32), sr)
    assert k["key"] == "Am" and k["confident"]


def test_names():
    assert ex.clean_song("Piano Core (1)") == "PianoCore1"
    assert ex.ready_name("Piano Core", "drums", 206, "8bar", 3, "Am") == "PianoCore_Drums_206bpm_Am_8bar03.wav"
    assert ex.ready_name("x", "mix", 120.5, "bar", 12, None) == "x_Mix_120_5bpm_bar12.wav"
    assert ex.unit_label("1/8", 4) == "1-8_04"


def test_acid_chunk_roundtrip(tmp_path):
    src = tmp_path / "a.wav"
    y = np.random.default_rng(0).standard_normal((1000, 2)).astype(np.float32) * 0.1
    sf.write(src, y, 44100, subtype="PCM_24")
    dst = tmp_path / "b.wav"
    ex.write_with_acid(src, dst, ex.acid_chunk(206, 32, "4/4", 9))
    assert ex.read_acid(dst) == {"flags": 6, "root": 69, "beats": 32, "meter": "4/4", "bpm": 206.0}
    a, _ = sf.read(src)
    b, sr = sf.read(dst)
    assert sr == 44100 and np.array_equal(a, b)                          # audio untouched
    ex.write_with_acid(dst, dst.with_name("c.wav"), ex.acid_chunk(120, 4))  # replacing, not stacking
    assert ex.read_acid(dst.with_name("c.wav"))["bpm"] == 120 and dst.with_name("c.wav").stat().st_size == dst.stat().st_size


def test_export_names_copies_and_reexport(song, model, tmp_path):
    root, d = song
    picks = {"version": 1, "songs": {}, "exported": []}
    marks = ex.song_marks(picks, model["id"])["marks"]
    marks[ex.mark_key("8bar", "drums", 2)] = "keep"
    marks[ex.mark_key("8bar", "melody", 3)] = "keep"
    marks[ex.mark_key("8bar", "melody", 4)] = "skip"
    before = {p: p.stat().st_mtime_ns for p in d.rglob("*.wav")}
    files = ex.export(root, [model], picks, "song", log=lambda *a: None)
    key = "_" + model["key"]["key"] if model["key"].get("confident") else ""
    assert sorted(f.name for f in files) == sorted([f"Song1_Drums_240bpm{key}_8bar02.wav", f"Song1_Melody_240bpm{key}_8bar03.wav"])
    assert all(f.parent == root / "_ready" / "Song1" for f in files)
    acid = ex.read_acid(files[0])
    assert acid["bpm"] == 240 and acid["beats"] == 32
    a, _ = sf.read(files[0])
    b, _ = sf.read(d / row(model, "drums")["cells"][1]["file"])
    assert np.array_equal(a, b)
    assert {p: p.stat().st_mtime_ns for p in d.rglob("*.wav")} == before   # sources never touched
    saved = json.loads((root / "_ready" / "picks.json").read_text())
    assert saved["songs"][model["id"]]["marks"][ex.mark_key("8bar", "melody", 4)] == "skip"
    # re-export flat with one pick removed: old files go, new layout appears
    del marks[ex.mark_key("8bar", "melody", 3)]
    files2 = ex.export(root, [model], picks, "flat", log=lambda *a: None)
    assert [f.parent for f in files2] == [root / "_ready"] and not (root / "_ready" / "Song1").exists()


def test_name_collisions_get_suffix(song, model):
    root, _ = song
    m2 = dict(model, id=model["id"], song="Song 1")                      # cleans to the same name
    picks = {"version": 1, "songs": {}, "exported": []}
    ex.song_marks(picks, model["id"])["marks"][ex.mark_key("8bar", "drums", 1)] = "keep"
    files = ex.export(root, [model, m2], picks, "flat", log=lambda *a: None)
    assert len(files) == 2 and files[1].stem == files[0].stem + "_2"


def test_auto_pick_cli(tmp_path):
    write_pick_result(tmp_path, "Auto Song")
    assert main(["pick", str(tmp_path), "--auto", "--group-by", "stem"]) == 0
    got = sorted(str(p.relative_to(tmp_path / "_ready")) for p in (tmp_path / "_ready").rglob("*.wav"))
    stems = {p.split("/")[0] for p in got}
    assert stems == {"Drums", "Melody", "Vocals"}                       # the mix is hidden by default: not exported
    assert sum(p.startswith("Drums/") for p in got) == 1                 # one drum loop for the whole song
    assert sum(p.startswith("Melody/") for p in got) == 3                # A, B and C
    picks = json.loads((tmp_path / "_ready" / "picks.json").read_text())
    assert len(picks["exported"]) == len(got)


# ---------------------------------------------------------------- server
@pytest.fixture()
def server(song):
    from aislice.picker_server import serve
    got = {}
    ev = threading.Event()
    th = threading.Thread(target=serve, args=(song[0],), kwargs=dict(open_browser=False, log=lambda *a: None,
                          ready=lambda h, u: (got.update(h=h, url=u), ev.set())), daemon=True)
    th.start()
    assert ev.wait(10)
    yield got["h"], got["url"].split("?")[0].rstrip("/"), got["url"].split("t=")[1]
    got["h"].shutdown()


def _req(url, token=None, body=None, host=None):
    r = urllib.request.Request(url, data=None if body is None else json.dumps(body).encode(),
                               headers={k: v for k, v in (("X-Token", token), ("Host", host)) if v})
    try:
        with urllib.request.urlopen(r, timeout=30) as resp:
            return resp.status, resp.read()
    except urllib.error.HTTPError as e:
        return e.code, e.read()


def test_server_api_and_safety(server, song):
    h, base, token = server
    assert h.server_address[0] == "127.0.0.1"
    assert _req(base + "/api/songs")[0] == 403                           # no token
    assert _req(base + "/api/songs", token, host="evil.example")[0] == 403  # DNS rebinding
    code, body = _req(base + "/", token)
    assert code == 200 and b"aislice Picker" in body and token.encode() in body
    songs = json.loads(_req(base + "/api/songs", token)[1])["songs"]
    sid = songs[0]["id"]
    st = json.loads(_req(base + f"/api/song?id={urllib.parse.quote(sid)}&unit=8bar", token)[1])
    f = st["model"]["rows"][1]["cells"][0]["file"]
    code, wav = _req(base + f"/audio?id={urllib.parse.quote(sid)}&f={urllib.parse.quote(f)}", token)
    assert code == 200 and wav[:4] == b"RIFF"
    for bad in ("../../etc/passwd", "report.json", "/etc/hosts"):
        assert _req(base + f"/audio?id={urllib.parse.quote(sid)}&f={urllib.parse.quote(bad)}", token)[0] == 404
    k = ex.mark_key("8bar", "drums", 1)
    assert _req(base + "/api/mark", token, {"id": sid, "keys": [k], "value": "keep"})[0] == 200
    assert json.loads((song[0] / "_ready" / "picks.json").read_text())["songs"][sid]["marks"][k] == "keep"
    assert _req(base + "/api/mark", token, {"id": sid, "keys": [k], "value": "delete-all"})[0] == 400
    code, body = _req(base + "/api/export", token, {"group_by": "song"})
    assert code == 200 and len(json.loads(body)["files"]) >= 1


def test_best_ranking(model):
    for r in model["rows"]:
        ranked = sorted((c for c in r["cells"] if c["best"] is not None), key=lambda c: c["best"])
        assert [c["best"] for c in ranked] == list(range(len(ranked)))
        assert all(c["rep"] and not c["empty"] for c in ranked)
        keys = [pick.best_key(c) for c in ranked]
        assert keys == sorted(keys)
    mel = sorted((c for c in row(model, "melody")["cells"] if c["best"] is not None), key=lambda c: c["best"])
    assert mel[0]["count"] == 3                    # the part that repeats most (A, 3x) ranks first


def test_auto_best_n(tmp_path):
    write_pick_result(tmp_path, "Best Song")
    assert main(["pick", str(tmp_path), "--auto", "--best", "1", "--group-by", "flat"]) == 0
    names = sorted(p.name for p in (tmp_path / "_ready").glob("*.wav"))
    assert len(names) == 3 and {n.split("_")[1] for n in names} == {"Drums", "Melody", "Vocals"}


def test_server_starts_without_dns_lookup(monkeypatch, song):
    """macOS CI: socket.getfqdn() in HTTPServer.server_bind hung >10 s. The Picker must not call it."""
    import socket

    from aislice.picker_server import LocalServer, Picker, make_handler
    monkeypatch.setattr(socket, "getfqdn", lambda *a: pytest.fail("reverse DNS lookup"))
    h = LocalServer(("127.0.0.1", 0), make_handler(Picker(song[0]), "t"))
    assert h.server_name == "127.0.0.1" and h.server_port > 0
    h.server_close()
