"""The end of the pipeline: the best loops of this run's songs, in one flat, timestamped folder.

<output>/<YYYY-MM-DD_HHMM>_<bpm>bpm/<Stem>_<NN>_<Song>_<bpm>bpm[_<Key>]_<unit><nn>.wav   (+ best.json)

Ranking (same as the Picker and Jam): not quiet (likely splitter bleed), then most repeated (in AI songs the
part that keeps coming back is usually the hook), then fullest. Ranked across all songs of the run, per
instrument; the full mix is not included.
"""
from __future__ import annotations

import json
import time
from pathlib import Path

from . import export as ex
from . import pick

DEFAULT_N = 8


def session_dir(out_root: Path, bpm: float, when: float | None = None) -> Path:
    """A new, unused <out_root>/<YYYY-MM-DD_HHMM>_<bpm>bpm folder name (suffix _2, _3 … if taken)."""
    base = Path(out_root) / f"{time.strftime('%Y-%m-%d_%H%M', time.localtime(when))}_{ex.bpm_label(bpm)}"
    d, k = base, 2
    while d.exists():
        d = base.with_name(f"{base.name}_{k}")
        k += 1
    return d


def best_name(stem: str, rank: int, song: str, bpm: float, unit: str, i: int, key: str | None) -> str:
    """'Drums_01_PianoCore_206bpm_Am_4bar25.wav': instrument first so a folder sorts by instrument, then rank."""
    rest = ex.ready_name(song, stem, bpm, unit, i, key)            # Song_Stem_bpm[_Key]_unit
    song_part, _, tail = rest.partition("_" + ex.stem_label(stem) + "_")
    return f"{ex.stem_label(stem)}_{rank:02d}_{song_part}_{tail}"


def build_best(song_dirs: list[Path], out: Path, n: int = DEFAULT_N, opts: dict | None = None,
               log=print) -> list[Path]:
    """Rank every loop of `song_dirs` per instrument and copy the top `n` of each into `out` (flat)."""
    models = []
    for d in song_dirs:
        try:
            models.append(pick.analyze_song(Path(d), None, opts, use_cache=False))
        except (ValueError, OSError, KeyError) as e:
            log(f"  {Path(d).name}: left out of the best samples ({e})")
    pools: dict[str, list] = {}
    for m in models:
        key = m.get("key") or {}
        for r in m["rows"]:
            if r["name"] == "mix":
                continue
            for c in r["cells"]:
                if c.get("best") is not None and not c["quiet"]:
                    pools.setdefault(r["name"], []).append((pick.best_key(c), m["song"], c["i"], m, r, c, key))
    out = Path(out)
    written, listing = [], []
    for stem, items in pools.items():
        items.sort(key=lambda t: t[:3])
        for rank, (_, _, _, m, r, c, key) in enumerate(items[:n], 1):
            name_key = key.get("key") if key.get("confident") else None
            dst = out / best_name(r["name"], rank, m["song"], m["bpm"], r["unit"], c["i"], name_key)
            out.mkdir(parents=True, exist_ok=True)
            ex.write_with_acid(Path(m["dir"]) / c["file"], dst,
                               ex.acid_chunk(m["bpm"], r["beats_per_slice"], m["meter"], key.get("pc") if name_key else None))
            written.append(dst)
            listing.append({"file": dst.name, "stem": r["name"], "rank": rank, "song": m["song"], "slice": c["i"],
                            "loop": r["unit"], "repeats": c["count"], "key": key.get("key")})
    if listing:
        (out / "best.json").write_text(json.dumps({"bpm": models[0]["bpm"], "songs": [m["song"] for m in models],
                                                   "per_instrument": n, "files": listing}, indent=1))
        by: dict[str, int] = {}
        for x in listing:
            by[x["stem"]] = by.get(x["stem"], 0) + 1
        log(f"Best samples from {len(models)} song(s): " + ", ".join(f"{v} {k}" for k, v in by.items()) + f" -> {out}")
    return written
