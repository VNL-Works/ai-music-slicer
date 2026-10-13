"""Jam pad: sample-exact rendering of recorded pad launches, pad list, saving, and the API."""
import json
import threading
import urllib.parse

import numpy as np
import pytest
import soundfile as sf
from synth import write_pick_result

from aislice import export as ex
from aislice import jam, pick

SR, BPM, BPB = 1000, 120.0, 4          # 1 bar = 2 s = 2000 samples: easy to reason about
BAR = 2000


def clip(v, bars=2):
    """A recognisable clip: sample k holds v + k/1e6 (both channels)."""
    y = v + np.arange(bars * BAR, dtype=np.float64) / 1e6
    return np.stack([y, y], 1).astype(np.float32)


CLIPS = {"A": clip(0.1), "B": clip(0.3)}
load = CLIPS.__getitem__
mid = lambda b: slice(b * BAR + 50, (b + 1) * BAR - 50)          # away from the 3 ms fades


def test_clip_plays_in_phase_from_its_launch_bar():
    mix, st = jam.render([{"bar": 1, "track": "drums", "clip": "A"}], 4, BPM, BPB, 2, load, SR)
    y = st["drums"][:, 0]
    assert np.all(y[mid(0)] == 0)                                  # nothing before the launch
    a = CLIPS["A"][:, 0]
    assert np.allclose(y[mid(1)], a[mid(1)])                       # launched at bar 1 -> plays the clip's bar 2
    assert np.allclose(y[mid(2)], a[mid(0)])                       # then loops
    assert np.allclose(y[mid(3)], a[mid(1)])
    assert np.array_equal(mix, st["drums"])


def test_switch_stop_gain_and_mix_is_sum():
    ev = [{"bar": 0, "track": "drums", "clip": "A"}, {"bar": 2, "track": "drums", "clip": "B"},
          {"bar": 3, "track": "drums", "clip": None},
          {"bar": 0, "track": "bass", "clip": "B"}, {"bar": 1, "track": "bass", "gain": 0.5}]
    mix, st = jam.render(ev, 4, BPM, BPB, 2, load, SR)
    d, b = st["drums"][:, 0], st["bass"][:, 0]
    assert np.allclose(d[mid(2)], CLIPS["B"][mid(0), 0]) and np.all(d[mid(3)] == 0)
    assert np.allclose(b[mid(0)], CLIPS["B"][mid(0), 0]) and np.allclose(b[mid(1)], 0.5 * CLIPS["B"][mid(1), 0])
    assert np.allclose(mix, st["drums"] + st["bass"])
    assert abs(d[2 * BAR]) < 0.01                                  # fades in where the clip changes: no click


def test_render_rejects_silly_lengths():
    with pytest.raises(ValueError):
        jam.render([], 0, BPM, BPB, 2, load, SR)


def test_save_jam_files_each_normalized(tmp_path):
    loud = {"A": clip(0.9)}
    mix, st = jam.render([{"bar": 0, "track": "drums", "clip": "A"}, {"bar": 0, "track": "bass", "clip": "A"}],
                         2, BPM, BPB, 2, loud.__getitem__, SR)
    info = jam.save_jam(tmp_path, mix, st, SR, BPM, "4/4", 2, {"drums": "Drums", "bass": "Bass"})
    files = [tmp_path / f for f in info["files"]]
    assert [f.name.split("_")[-1] for f in files] == ["Mix.wav", "Bass.wav", "Drums.wav"]
    assert all(f.parent.parent == tmp_path / "_ready" / "jams" for f in files)
    y, sr = sf.read(files[0])
    assert sr == SR and len(y) == 2 * BAR
    for f in files:                                                # every file on its own peaks at -0.1 dB
        assert abs(20 * np.log10(np.abs(sf.read(f)[0]).max()) + 0.1) < 0.01
    assert info["gains_db"]["Mix"] < 0
    assert ex.read_acid(files[0]) == {"flags": 4, "root": 60, "beats": 8, "meter": "4/4", "bpm": 120.0}


@pytest.fixture(scope="module")
def songs(tmp_path_factory):
    root = tmp_path_factory.mktemp("jam")
    write_pick_result(root, "One")
    write_pick_result(root, "Two")
    return root


def test_pads_are_distinct_loops_without_empties(songs):
    _, dirs = pick.find_songs(songs)
    models = [pick.analyze_song(d) for d in dirs]
    picks = {"songs": {models[1]["id"]: {"marks": {ex.mark_key("8bar", "melody", 2): "keep"}}}}
    tracks = jam.build_pads(models, picks)
    assert {c["bars"] for t in tracks for c in t["clips"] if t["name"] == "drums"} == {4}   # drums: 4-bar loops
    assert {c["bars"] for t in tracks for c in t["clips"] if t["name"] == "melody"} == {8}
    assert [t["name"] for t in tracks] == ["drums", "melody", "vocals", "mix"]
    drums = next(t for t in tracks if t["name"] == "drums")
    assert len(drums["clips"]) == 2                                # one drum loop per song (all repeats)
    mel = next(t for t in tracks if t["name"] == "melody")
    assert mel["clips"][0]["kept"] and mel["clips"][0]["i"] == 2   # kept first, even though not the group's rep
    assert not any(c["i"] > 10 and c["row"] == "drums" for c in drums["clips"])   # silent section is not a pad


@pytest.fixture()
def server(songs):
    from aislice.picker_server import serve
    got, ev = {}, threading.Event()
    th = threading.Thread(target=serve, args=(songs,), kwargs=dict(open_browser=False, log=lambda *a: None, page="/jam",
                          ready=lambda h, u: (got.update(h=h, url=u), ev.set())), daemon=True)
    th.start()
    assert ev.wait(10)
    yield got["url"].split("/jam")[0], got["url"].split("t=")[1]
    got["h"].shutdown()


def test_jam_api_renders_a_recording(server, songs):
    from test_pick import _req
    base, token = server
    code, page = _req(base + "/jam", token)
    assert code == 200 and b"aislice Jam" in page
    st = json.loads(_req(base + "/api/jam", token)[1])
    assert st["bpm"] == 240 and [t["name"] for t in st["tracks"]][:2] == ["drums", "melody"]
    d = st["tracks"][0]["clips"][0]
    m = st["tracks"][1]["clips"][0]
    events = [{"bar": 0, "track": "drums", "clip": d["id"]}, {"bar": 2, "track": "melody", "clip": m["id"]}]
    code, body = _req(base + "/api/jam/render", token, {"bpm": 240, "bars": 6, "events": events})
    assert code == 200, body
    info = json.loads(body)
    y, sr = sf.read(songs / info["files"][0])
    assert len(y) == round(6 * 4 * 60 / 240 * sr) and len(info["files"]) == 3
    bad = [{"bar": 0, "track": "drums", "clip": "../../etc|drums|1"}]
    assert _req(base + "/api/jam/render", token, {"bars": 2, "events": bad})[0] == 400
    assert _req(base + "/api/jam/render", token, {"bars": 2, "events": []})[0] == 400
