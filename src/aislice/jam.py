"""Jam pad: build the pad list from Picker analysis, and render a recorded jam sample-exactly.

A jam is a list of events on a bar grid: {"bar": b, "track": t, "clip": <clip id> | None} launches or stops a
clip on a track from bar b; {"bar": b, "track": t, "gain": g} changes the track volume from bar b.
Clips loop and stay phrase-aligned: a clip launched at bar b plays from its bar (b mod clip_bars), the same
way the browser plays it, so what you heard is what gets rendered.
"""
from __future__ import annotations

import time
from pathlib import Path

import numpy as np

from . import export as ex
from . import pick

FADE_MS = 3.0   # tiny fade where a track switches clips mid-stream, so cuts never click
MAX_BARS = 2000


def build_pads(models: list[dict], picks: dict) -> list[dict]:
    """Tracks (one per stem name, full mix last) with one pad per distinct loop across all songs.

    Pads = each group's representative plus anything the user kept; empty slices never appear.
    Each clip gets a `rank` within its song and track (0 = best: kept, then most repeated, then loudest),
    and clips are ordered by rank so the songs are interleaved best-first."""
    tracks: dict[str, dict] = {}
    for si, m in enumerate(models):
        marks = picks.get("songs", {}).get(m["id"], {}).get("marks", {})
        for r in m["rows"]:
            t = tracks.setdefault(r["name"], {"name": r["name"], "label": "Full mix" if r["name"] == "mix"
                                              else r["label"], "clips": []})
            for c in r["cells"]:
                kept = marks.get(ex.mark_key(r["unit"], r["name"], c["i"])) == "keep"
                if c["empty"] or not (c["rep"] or kept) or c.get("partial"):
                    continue
                t["clips"].append({"id": f"{m['id']}|{r['name']}|{c['i']}", "sid": m["id"], "song": m["song"],
                                   "song_no": si, "row": r["name"], "i": c["i"], "file": c["file"],
                                   "unit": r["unit"], "bars": r["bars"],
                                   "kept": kept, "count": c["count"], "quiet": c["quiet"], "rms_db": c["rms_db"],
                                   "peaks": c["peaks"]})
    for t in tracks.values():
        # best first: kept, then not quiet, then most repeated (the hook), then loudest
        t["clips"].sort(key=lambda c: (not c["kept"], pick.best_key(c), c["song_no"], c["i"]))
        seen: dict[int, int] = {}
        for c in t["clips"]:                     # rank within (track, song): the page shows the top few per song
            c["rank"] = seen.get(c["song_no"], 0)
            seen[c["song_no"]] = c["rank"] + 1
        t["clips"].sort(key=lambda c: (c["rank"], not c["kept"], c["song_no"]))
    order = sorted(tracks, key=lambda n: (n == "mix", next((k for k, s in enumerate(
        ["drums", "bass", "melody", "piano", "guitar", "other", "vocals", "instrumental"]) if s == n), 50), n))
    return [tracks[n] for n in order if tracks[n]["clips"]]


def _segments(events: list[dict], track: str, bars: int):
    """Per bar: (clip id | None, gain) for one track."""
    evs = sorted((e for e in events if e.get("track") == track), key=lambda e: e["bar"])
    clip, gain, k, out = None, 1.0, 0, []
    for b in range(bars):
        while k < len(evs) and evs[k]["bar"] <= b:
            e = evs[k]
            if "clip" in e:
                clip = e["clip"]
            if "gain" in e:
                gain = float(min(max(e["gain"], 0.0), 2.0))
            k += 1
        out.append((clip, gain))
    return out


def render(events: list[dict], bars: int, bpm: float, bpb: int, clip_bars, load, sr: int):
    """Render a jam. `load(clip_id) -> (n, ch) float32 at sr`; `clip_bars` is a clip's length in bars (a number,
    or a function of the clip id, since drums/bass are 4-bar and melody 8-bar loops). Returns (mix, {track: audio})."""
    bars_of = clip_bars if callable(clip_bars) else (lambda _cid: clip_bars)
    if not 0 < bars <= MAX_BARS:
        raise ValueError(f"a jam must be 1-{MAX_BARS} bars")
    bar_len = bpb * 60.0 / bpm * sr
    n = int(round(bars * bar_len))
    tracks = sorted({e["track"] for e in events if "track" in e})
    cache, out = {}, {}
    fade = max(1, int(sr * FADE_MS / 1000))
    for t in tracks:
        y = None
        prev = None
        segs = _segments(events, t, bars)
        for b, (clip, gain) in enumerate(segs):
            s0, s1 = int(round(b * bar_len)), int(round((b + 1) * bar_len))
            if clip is None or gain == 0:
                prev = None
                continue
            if clip not in cache:
                cache[clip] = load(clip)
            a = cache[clip]
            if y is None:
                y = np.zeros((n, a.shape[1]), np.float32)
            if a.shape[1] != y.shape[1]:
                a = np.repeat(a.mean(1, keepdims=True), y.shape[1], axis=1)
            L = len(a)
            p0 = int(round((b % max(1, int(round(bars_of(clip))))) * bar_len))
            idx = (p0 + np.arange(s1 - s0)) % L
            seg = a[idx] * gain
            if prev != clip:                                   # clip starts here: fade in
                seg[:fade] *= np.linspace(0, 1, min(fade, len(seg)), dtype=np.float32)[:, None]
            nxt = segs[b + 1][0] if b + 1 < bars and segs[b + 1][1] > 0 else None
            if nxt != clip:                                    # clip ends after this bar: fade out
                seg[-fade:] *= np.linspace(1, 0, min(fade, len(seg)), dtype=np.float32)[:, None]
            y[s0:s1] += seg
            prev = clip
        if y is not None:
            out[t] = y
    ch = max((y.shape[1] for y in out.values()), default=2)
    mix = np.zeros((n, ch), np.float32)
    for y in out.values():
        mix += y if y.shape[1] == ch else np.repeat(y.mean(1, keepdims=True), ch, axis=1)
    return mix, out


def save_jam(root: Path, mix: np.ndarray, stems: dict, sr: int, bpm: float, meter: str, bars: int,
             labels: dict | None = None) -> dict:
    """Write <root>/_ready/jams/Jam_<time>_<bpm>/ with the mix + one WAV per track, all from bar 1, each with
    an ACID chunk and peak-normalized on its own to -0.1 dB. Returns paths relative to root."""
    from . import io
    name = f"Jam_{time.strftime('%Y%m%d-%H%M%S')}_{ex.bpm_label(bpm)}"
    d = Path(root) / ex.READY / "jams" / name
    k = 2
    while d.exists():
        d = d.with_name(f"{name}_{k}")
        k += 1
    d.mkdir(parents=True)
    beats = bars * int(str(meter).split("/")[0])
    files, gains = [], {}
    acid = ex.acid_chunk(bpm, beats, meter)
    for label, y in [("Mix", mix)] + [((labels or {}).get(t, ex.stem_label(t)), s) for t, s in stems.items()]:
        y, gains[label] = io.normalize(np.asarray(y, np.float32))   # each file on its own, peak at -0.1 dB
        tmp = d / f".{label}.wav"
        io.write(tmp, y, sr, bit_depth="24")
        dst = d / f"{d.name}_{label.replace(' ', '')}.wav"
        ex.write_with_acid(tmp, dst, acid)
        tmp.unlink()
        files.append(str(dst.relative_to(root)))
    return {"folder": str(d.relative_to(root)), "files": files, "bars": bars,
            "seconds": round(len(mix) / sr, 2), "gains_db": gains}
