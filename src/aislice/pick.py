"""`aislice pick` analysis: find empty slices, group repeats per stem, estimate the key.

Works on a finished result folder (report.json + slices_*/), never changes it. Results are cached in
<song folder>/pick_cache.json so re-opening the Picker is instant.
"""
from __future__ import annotations

import hashlib
import json
import re
from pathlib import Path

import numpy as np

from . import slicer

CACHE_NAME = "pick_cache.json"
CACHE_VERSION = 5
AN_SR = 22050          # analysis rate (mono)
HOP = 512
PEAK_BINS = 48         # waveform points per cell

# Levels are relative to the song's mix peak (= dBFS on a 0 dBFS master), so loud and quiet masters behave alike.
DEFAULTS = {
    "empty_db": -45.0,      # slice peak below this -> empty
    "active_db": -40.0,     # RMS frames above this count as "active"
    "empty_active": 0.15,   # active for less than this share of the slice -> empty
    "quiet_db": -20.0,      # RMS this far below the row's loudest slice -> "quiet" label (often splitter bleed)
    "group_sim": 0.60,      # beat-by-beat similarity (z-scored cosine) to join a group; 0 = no grouping
    "group_rep": "medoid",  # representative of a group: medoid (most typical) or loudest
}

STEM_ORDER = ["vocals", "drums", "bass", "melody", "guitar", "piano", "other", "instrumental"]
DRUMS = {"drums"}

# Krumhansl–Kessler key profiles
_MAJOR = np.array([6.35, 2.23, 3.48, 2.33, 4.38, 4.09, 2.52, 5.19, 2.39, 3.66, 2.29, 2.88])
_MINOR = np.array([6.33, 2.68, 3.52, 5.38, 2.60, 3.53, 2.54, 4.75, 3.98, 2.69, 3.34, 3.17])
NOTES = ["C", "C#", "D", "D#", "E", "F", "F#", "G", "G#", "A", "A#", "B"]
KEY_MIN_CORR, KEY_MIN_MARGIN = 0.6, 0.05


# ---------------------------------------------------------------- discovery
def find_songs(path: Path) -> tuple[Path, list[Path]]:
    """(output root, result folders). `path` is an output folder, or one song's result folder
    (then its parent is the root, so _ready/ always sits next to the song folders)."""
    path = Path(path).resolve()
    if (path / "report.json").is_file():
        return path.parent, [path]
    return path, sorted(p for p in path.iterdir() if p.is_dir() and (p / "report.json").is_file())


def _index(p: Path) -> int | None:
    m = re.search(r"_(\d+)$", p.stem)
    return int(m.group(1)) if m else None


def _wavs(d: Path) -> list[Path]:
    fs = [p for p in d.glob("*.wav") if _index(p) is not None]
    return sorted(fs, key=_index)


def song_info(d: Path) -> dict:
    """What a result folder holds: tempo, meter, song name, slice units and the files per row."""
    d = Path(d)
    r = json.loads((d / "report.json").read_text())
    bpm = float(r["target_bpm"])
    bpb = int(str(r.get("meter", "4/4")).split("/")[0])
    full = next((p for p in sorted(d.glob("*bpm.wav"))), None)
    song = full.stem.rsplit("_", 1)[0] if full else Path(r.get("source", d.name)).stem
    units = {}
    for p in d.iterdir():
        u = slicer.unit_for_folder(p.name) if p.is_dir() else None
        if not u:
            continue
        rows = {}
        mix = _wavs(p)
        if mix:
            rows["mix"] = mix
        stem_names = list((r.get("stems") or {}).get("names") or [])
        subs = [s for s in p.iterdir() if s.is_dir()]
        for s in sorted(subs, key=lambda s: (s.name not in stem_names,
                                             stem_names.index(s.name) if s.name in stem_names else 0, s.name)):
            fs = _wavs(s)
            if fs:
                rows[s.name] = fs
        if rows:
            units[u] = rows
    order = sorted(units, key=lambda u: slicer.unit_beats(u, bpb))
    return {"dir": d, "id": d.name, "song": song, "bpm": bpm, "bpb": bpb, "meter": r.get("meter", "4/4"),
            "full": full, "units": {u: units[u] for u in order}, "report": r}


ROW_ORDER = ["drums", "bass", "melody", "piano", "guitar", "other", "vocals", "instrumental"]


def layout(info: dict, unit: str | None = None) -> dict[str, tuple[str, list[Path]]]:
    """Which slices each row uses: {row: (unit, files)}. By default the standard layout (drums/bass 4bar,
    melody/mix 8bar); a row without its standard size falls back to the largest size it has. `unit` forces one
    size for every row (tests / old results). Rows: stems first (drums, bass, melody, ...), the mix last."""
    rows: dict[str, tuple[str, list[Path]]] = {}
    names = {r for rs in info["units"].values() for r in rs}
    for name in names:
        want = unit or slicer.standard_unit(name)
        have = [u for u, rs in info["units"].items() if name in rs]      # ascending size
        u = want if want in have else (None if unit else have[-1])
        if u:
            rows[name] = (u, info["units"][u][name])
    order = sorted(rows, key=lambda n: (n == "mix", ROW_ORDER.index(n) if n in ROW_ORDER else 50, n))
    return {n: rows[n] for n in order}


def default_unit(info: dict) -> str | None:
    """The largest slice unit that exists (8bar at fast tempos is the most useful loop length)."""
    return list(info["units"])[-1] if info["units"] else None


# ---------------------------------------------------------------- features
def _mono(y: np.ndarray) -> np.ndarray:
    return y.mean(1) if y.ndim == 2 else y


def _db(x: float) -> float:
    return float(20 * np.log10(max(x, 1e-10)))


def _per_beat(frames: np.ndarray, n_beats: int, spb_frames: float, agg=np.mean, sub: int = 1) -> np.ndarray:
    """Aggregate (dims, frames) into (n_beats*sub, dims) blocks on the beat grid (slice starts on a beat)."""
    out = np.zeros((n_beats * sub, frames.shape[0]), np.float32)
    step = spb_frames / sub
    for k in range(n_beats * sub):
        a, b = int(round(k * step)), int(round((k + 1) * step))
        if a < frames.shape[1] and b > a:
            out[k] = agg(frames[:, a:min(b, frames.shape[1])], axis=1)
    return out


def slice_features(y: np.ndarray, bpm: float, n_beats: int, kind: str) -> np.ndarray:
    """Beat-by-beat feature sequence (n_beats, dims) for one slice at AN_SR. kind: tonal | drums | mix."""
    import librosa
    spb = 60.0 / bpm * AN_SR / HOP
    parts = []
    if kind in ("tonal", "mix"):
        S = np.abs(librosa.stft(y, n_fft=2048, hop_length=HOP)) ** 2
        chroma = librosa.feature.chroma_stft(S=S, sr=AN_SR)
        mfcc = librosa.feature.mfcc(S=librosa.power_to_db(librosa.feature.melspectrogram(S=S, sr=AN_SR)), n_mfcc=13)[1:]
        c = _per_beat(chroma, n_beats, spb)
        c /= c.max(1, keepdims=True) + 1e-9
        parts += [c, _per_beat(mfcc, n_beats, spb)]
    if kind in ("drums", "mix"):
        onset = librosa.onset.onset_strength(y=y, sr=AN_SR, hop_length=HOP, center=False)[None, :]
        o = _per_beat(onset, n_beats, spb, agg=np.max, sub=4).reshape(n_beats, 4)
        mel = librosa.feature.melspectrogram(y=y, sr=AN_SR, hop_length=HOP, n_mels=8)
        parts += [o, _per_beat(np.log1p(mel * 100), n_beats, spb)]
    return np.concatenate(parts, axis=1)


def _block_sizes(kind: str) -> list[int]:
    return {"tonal": [12, 12], "drums": [4, 8], "mix": [12, 12, 4, 8]}[kind]


def similarity_matrix(feats: list[np.ndarray], kind: str) -> np.ndarray:
    """Cosine similarity of z-scored, beat-ordered feature sequences (blocks weighted equally)."""
    F = np.stack(feats).astype(np.float64)             # (n, beats, dims)
    mu = F.mean(axis=(0, 1), keepdims=True)
    sd = F.std(axis=(0, 1), keepdims=True) + 1e-6
    Z = (F - mu) / sd
    a = 0
    for size in _block_sizes(kind):                     # equal weight per feature block
        Z[..., a:a + size] /= np.sqrt(size)
        a += size
    V = Z.reshape(len(F), -1)
    V /= np.linalg.norm(V, axis=1, keepdims=True) + 1e-9
    return V @ V.T


def group_slices(sim: np.ndarray, threshold: float, loud: np.ndarray, rep: str = "medoid") -> list[list[int]]:
    """Leader clustering in time order: a slice joins a group only if it matches the group's LEADER,
    so groups can't snowball through chains of near matches. Returns groups (representative first)."""
    n = len(sim)
    if threshold <= 0:
        return [[i] for i in range(n)]
    leaders, groups = [], []
    for i in range(n):
        best = max(range(len(leaders)), key=lambda g: sim[i, leaders[g]], default=None)
        if best is not None and sim[i, leaders[best]] >= threshold:
            groups[best].append(i)
        else:
            leaders.append(i)
            groups.append([i])
    out = []
    for g in groups:
        if rep == "loudest":
            r = max(g, key=lambda i: loud[i])
        else:
            r = max(g, key=lambda i: sim[i, g].sum())
        out.append([r] + [i for i in g if i != r])
    return out


# ---------------------------------------------------------------- key
def estimate_key(y: np.ndarray, sr: int) -> dict:
    """Krumhansl key estimate. 'confident' is False when the best key barely beats the runner-up."""
    import librosa
    y = _mono(y).astype(np.float32)
    if sr != AN_SR:
        y = librosa.resample(y, orig_sr=sr, target_sr=AN_SR, res_type="soxr_hq")
    chroma = librosa.feature.chroma_stft(y=y, sr=AN_SR, n_fft=4096, hop_length=2048)
    prof = chroma.sum(1)
    if prof.sum() <= 0:
        return {"key": None, "name": None, "corr": 0.0, "margin": 0.0, "confident": False}
    scores = []
    for mode, p in (("", _MAJOR), ("m", _MINOR)):
        for pc in range(12):
            scores.append((float(np.corrcoef(prof, np.roll(p, pc))[0, 1]), pc, mode))
    scores.sort(reverse=True)
    (c, pc, mode), runner = scores[0], scores[1][0]
    return {"key": NOTES[pc] + mode, "name": f"{NOTES[pc]} {'minor' if mode else 'major'}", "pc": pc,
            "corr": round(c, 3), "margin": round(c - runner, 3),
            "confident": bool(c >= KEY_MIN_CORR and c - runner >= KEY_MIN_MARGIN)}


def _key_source(info: dict):
    """Audio for the key estimate: the non-drum stems if there are any, else the full mix."""
    from . import io
    sdir = info["dir"] / "stems"
    tonal = [p for p in sorted(sdir.glob("*.wav")) if p.stem not in DRUMS] if sdir.is_dir() else []
    if tonal:
        ys = [io.read(p) for p in tonal]
        n = min(len(y) for y, _ in ys)
        return sum(_mono(y[:n]) for y, _ in ys), ys[0][1], tonal
    if info["full"]:
        y, sr = io.read(info["full"])
        return y, sr, [info["full"]]
    return None, None, []


# ---------------------------------------------------------------- analysis
def _sig(files, extra) -> str:
    h = hashlib.sha1(json.dumps(extra, sort_keys=True).encode())
    for f in files:
        st = f.stat()
        h.update(f"{f}|{st.st_size}|{st.st_mtime_ns}".encode())
    return h.hexdigest()


def _load_cache(d: Path) -> dict:
    try:
        c = json.loads((d / CACHE_NAME).read_text())
        return c if c.get("version") == CACHE_VERSION else {"version": CACHE_VERSION}
    except (OSError, ValueError):
        return {"version": CACHE_VERSION}


def _save_cache(d: Path, c: dict) -> None:
    try:
        (d / CACHE_NAME).write_text(json.dumps(c))
    except OSError:
        pass   # a read-only folder just means no cache


def song_key(info: dict, cache: dict | None = None) -> dict:
    y, sr, files = _key_source(info)
    if y is None:
        return {"key": None, "name": None, "confident": False}
    sig = _sig(files, {"k": 1})
    if cache is not None and cache.get("key", {}).get("sig") == sig:
        return cache["key"]["value"]
    k = estimate_key(y, sr)
    if cache is not None:
        cache["key"] = {"sig": sig, "value": k}
    return k


def analyze_song(d: Path, unit: str | None = None, opts: dict | None = None, use_cache: bool = True) -> dict:
    """Analyse one result folder (standard layout, or one forced `unit`). Returns the Picker's JSON model."""
    import librosa

    from . import io
    o = {**DEFAULTS, **(opts or {})}
    info = song_info(d)
    if not info["units"]:
        raise ValueError(f"{d}: no slice folders found (run aislice first)")
    rows = layout(info, unit)
    if not rows:
        raise ValueError(f"{d}: no {unit} slices (have: {', '.join(info['units'])})")
    files = [f for _, fs in rows.values() for f in fs]
    lay = {n: u for n, (u, _) in rows.items()}
    lkey = unit or "standard"
    cache = _load_cache(info["dir"]) if use_cache else None
    gain_files = sorted({fs[0].parent / io.GAINS_FILE for _, fs in rows.values() if fs} | {info["dir"] / "report.json"})
    sig = _sig(files + [g for g in gain_files if g.exists()], {"layout": lay, "o": o, "v": CACHE_VERSION})
    key = song_key(info, cache)
    if cache is not None and cache.get("units", {}).get(lkey, {}).get("sig") == sig:
        model = cache["units"][lkey]["model"]
        model["key"] = key
        return model

    loaded = {}
    row_gain = ((info["report"].get("normalize") or {}).get("gains_db") or {})
    for name, (_, fs) in rows.items():
        file_gain = io.read_gains(fs[0].parent) if fs else {}
        for f in fs:
            y, sr = io.read(f)
            # files are peak-normalized on their own: undo it so empty/quiet/loudness compare real levels
            y = io.undo_gain(y, row_gain.get(name, 0.0) + file_gain.get(f.name, 0.0))
            m = _mono(y).astype(np.float32)
            loaded[(name, f)] = (float(np.abs(y).max()) if len(y) else 0.0, librosa.resample(m, orig_sr=sr, target_sr=AN_SR,
                                                                                    res_type="soxr_hq"), len(y) / sr)
    mix_peaks = [loaded[("mix", f)][0] for f in rows["mix"][1]] if "mix" in rows else [v[0] for v in loaded.values()]
    ref = max(max(mix_peaks), 1e-6)                     # song reference level = mix peak

    out_rows = []
    for name, (row_unit, fs) in rows.items():
        kind = "mix" if name == "mix" else ("drums" if name in DRUMS else "tonal")
        beats = slicer.unit_beats(row_unit, info["bpb"])
        n_beats = int(round(beats))
        full_len = beats * 60.0 / info["bpm"]
        cells = []
        for f in fs:
            peak, y, dur = loaded[(name, f)]
            rms = librosa.feature.rms(y=y, frame_length=2048, hop_length=HOP, center=False)[0] if len(y) >= 2048 \
                else np.array([np.sqrt(np.mean(y ** 2))]) if len(y) else np.zeros(1)
            rel = 20 * np.log10(np.maximum(rms, 1e-10) / ref)
            peak_db = _db(peak / ref)
            active = float(np.mean(rel > o["active_db"]))
            why = "silent" if peak_db < o["empty_db"] else ("sparse" if active < o["empty_active"] else "")
            bins = np.array_split(np.abs(y), PEAK_BINS) if len(y) >= PEAK_BINS else [np.abs(y)]
            cells.append({
                "i": _index(f), "file": str(f.relative_to(info["dir"])), "dur": round(dur, 3),
                "peak_db": round(peak_db, 1), "rms_db": round(_db(np.sqrt(np.mean(y ** 2)) / ref) if len(y) else -200, 1),
                "active": round(active, 2), "empty": bool(why), "why": why, "quiet": False,
                "partial": dur < full_len - 0.002,      # anything short of a full loop (last slice)
                "peaks": [round(float(b.max()) / ref, 3) if len(b) else 0 for b in bins],
                "group": None, "rep": True, "count": 1, "sim": None,
            })
        live = [k for k, c in enumerate(cells) if not c["empty"]]
        if live:
            top = max(cells[k]["rms_db"] for k in live)
            for k in live:
                cells[k]["quiet"] = cells[k]["rms_db"] < top + o["quiet_db"]
        full = [k for k in live if not cells[k]["partial"]]
        gid = 0
        if len(full) >= 2:
            feats = [slice_features(loaded[(name, fs[k])][1], info["bpm"], n_beats, kind) for k in full]
            sim = similarity_matrix(feats, kind)
            loud = np.array([cells[k]["rms_db"] for k in full])
            for g in group_slices(sim, float(o["group_sim"]), loud, o["group_rep"]):
                r = g[0]
                for j in g:
                    c = cells[full[j]]
                    c.update(group=gid, rep=(j == r), count=len(g), sim=round(float(sim[j, r]), 3))
                gid += 1
        for k, c in enumerate(cells):             # partial slices / singletons: their own group
            if c["group"] is None and not c["empty"]:
                c.update(group=gid, rep=True, count=1)
                gid += 1
        rank_best(cells)
        for c in cells:
            c["row"], c["unit"] = name, row_unit
        out_rows.append({"name": name, "label": "Mix" if name == "mix" else name.capitalize(), "kind": kind,
                         "unit": row_unit, "beats_per_slice": beats, "bars": beats / info["bpb"],
                         "cells": cells, "groups": gid})

    model = {"id": info["id"], "dir": str(info["dir"]), "song": info["song"], "bpm": info["bpm"],
             "meter": info["meter"], "bpb": info["bpb"],
             "layout": lay, "rows": out_rows, "options": o}
    if cache is not None:
        cache.setdefault("units", {})[lkey] = {"sig": sig, "model": model}
        _save_cache(info["dir"], cache)
    model = dict(model)
    model["key"] = key
    return model


def best_key(c: dict):
    """How good a loop probably is (lower = better): not quiet (likely splitter bleed), then most repeated
    (in AI songs the part that keeps coming back is usually the hook), then loudest (fullest)."""
    return (bool(c.get("quiet")), -int(c.get("count") or 1), -float(c.get("rms_db") or -200))


def rank_best(cells: list[dict]) -> None:
    """Set cell['best'] = 0, 1, 2 ... on each group's representative, best first. Others get None."""
    cand = [c for c in cells if c["rep"] and not c["empty"] and not c.get("partial")]
    for c in cells:
        c["best"] = None
    for k, c in enumerate(sorted(cand, key=lambda c: (best_key(c), c["i"]))):
        c["best"] = k


def summary(model: dict) -> str:
    parts = []
    for r in model["rows"]:
        live = [c for c in r["cells"] if not c["empty"]]
        parts.append(f"{r['label']} ({r['unit']}) {len(live)}/{len(r['cells'])} in {len({c['group'] for c in live})} groups")
    k = model.get("key") or {}
    key = f", key {k['key']}{'' if k.get('confident') else '?'}" if k.get("key") else ""
    return f"{model['song']}{key}: " + "; ".join(parts)


def auto_marks(model: dict, picks: dict, best: int | None = None, mix: bool = False) -> int:
    """--auto: keep each group's representative (skips empty and quiet slices), or only the `best` N per row.
    Never overrides the user's marks."""
    from .export import mark_key, song_marks
    marks = song_marks(picks, model["id"])["marks"]
    n = 0
    for r in model["rows"]:
        if r["name"] == "mix" and not mix and len(model["rows"]) > 1:    # the mix is hidden by default
            continue
        for c in r["cells"]:
            k = mark_key(r["unit"], r["name"], c["i"])
            if k in marks:
                continue
            if c["rep"] and not c["empty"] and not c["quiet"] and (best is None or (c.get("best") is not None
                                                                                   and c["best"] < best)):
                marks[k] = "keep"
                n += 1
    return n
