"""Picker picks (picks.json) and the ready-for-DAW export into <output>/_ready/."""
from __future__ import annotations

import json
import re
import shutil
import struct
from pathlib import Path

READY = "_ready"
PICKS = "picks.json"
GROUP_BY = ("song", "stem", "flat")


# ---------------------------------------------------------------- picks
def picks_path(root: Path) -> Path:
    return Path(root) / READY / PICKS


def load_picks(root: Path) -> dict:
    try:
        p = json.loads(picks_path(root).read_text())
        if isinstance(p, dict) and isinstance(p.get("songs"), dict):
            p.setdefault("version", 1)
            return p
    except (OSError, ValueError):
        pass
    return {"version": 1, "songs": {}, "group_by": "song", "exported": []}


def save_picks(root: Path, picks: dict) -> None:
    p = picks_path(root)
    p.parent.mkdir(parents=True, exist_ok=True)
    tmp = p.with_suffix(".tmp")
    tmp.write_text(json.dumps(picks, indent=1))
    tmp.replace(p)                      # atomic: a crash never leaves half a picks.json


def mark_key(unit: str, row: str, i: int) -> str:
    return f"{unit}|{row}|{i}"


def song_marks(picks: dict, song_id: str) -> dict:
    s = picks["songs"].setdefault(song_id, {})
    s.setdefault("marks", {})
    s.setdefault("heard", [])
    return s


# ---------------------------------------------------------------- names
def clean_song(name: str) -> str:
    """'Piano Core (1)' -> 'PianoCore1': no spaces, brackets or characters DAWs/file systems dislike."""
    s = re.sub(r"[\s()\[\]{}<>:\"/\\|?*'`,;!&$%^=+~]+", "", name)
    return s or "Song"


def stem_label(row: str) -> str:
    return "Mix" if row == "mix" else row[:1].upper() + row[1:]


def bpm_label(bpm: float) -> str:
    return f"{bpm:g}".replace(".", "_") + "bpm"


def unit_label(unit: str, i: int) -> str:
    lab = unit.replace("/", "-").replace(" ", "")
    return f"{lab}{'_' if lab[-1].isdigit() else ''}{i:02d}"


def ready_name(song: str, row: str, bpm: float, unit: str, i: int, key: str | None) -> str:
    parts = [clean_song(song), stem_label(row), bpm_label(bpm)]
    if key:
        parts.append(key)
    parts.append(unit_label(unit, i))
    return "_".join(parts) + ".wav"


# ---------------------------------------------------------------- acid chunk
def acid_chunk(bpm: float, beats: float, meter: str = "4/4", root_pc: int | None = None) -> bytes:
    """ACID loop info: tempo + beat count so FL Studio / ACID-aware DAWs loop the file at the right tempo."""
    num, den = (int(x) for x in str(meter).split("/"))
    flags = 0x04 | (0x02 if root_pc is not None else 0)     # stretch on, root note set; not a one-shot
    root = 60 + root_pc if root_pc is not None else 60
    body = struct.pack("<IHHfIHHf", flags, root, 0x8000, 0.0, int(round(beats)), den, num, float(bpm))
    return b"acid" + struct.pack("<I", len(body)) + body


def write_with_acid(src: Path, dst: Path, acid: bytes) -> None:
    """Copy a RIFF/WAVE file and add (or replace) its acid chunk. Audio data is copied byte for byte."""
    data = Path(src).read_bytes()
    if data[:4] != b"RIFF" or data[8:12] != b"WAVE":
        shutil.copyfile(src, dst)
        return
    chunks, pos = [], 12
    while pos + 8 <= len(data):
        cid, size = data[pos:pos + 4], struct.unpack("<I", data[pos + 4:pos + 8])[0]
        end = pos + 8 + size + (size & 1)
        if cid != b"acid":
            chunks.append(data[pos:min(end, len(data))])
        pos = end
    body = b"WAVE" + b"".join(chunks) + acid
    Path(dst).write_bytes(b"RIFF" + struct.pack("<I", len(body)) + body)


def read_acid(path: Path) -> dict | None:
    data = Path(path).read_bytes()
    pos = 12
    while pos + 8 <= len(data):
        cid, size = data[pos:pos + 4], struct.unpack("<I", data[pos + 4:pos + 8])[0]
        if cid == b"acid":
            f, root, _, _, beats, den, num, bpm = struct.unpack("<IHHfIHHf", data[pos + 8:pos + 32])
            return {"flags": f, "root": root, "beats": beats, "meter": f"{num}/{den}", "bpm": round(bpm, 3)}
        pos += 8 + size + (size & 1)
    return None


# ---------------------------------------------------------------- export
def kept(model: dict, picks: dict) -> list[tuple[dict, dict]]:
    """(row, cell) pairs marked keep for this song (each row at its own slice size)."""
    marks = picks["songs"].get(model["id"], {}).get("marks", {})
    out = []
    for r in model["rows"]:
        for c in r["cells"]:
            if marks.get(mark_key(r["unit"], r["name"], c["i"])) == "keep":
                out.append((r, c))
    return out


def export(root: Path, models: list[dict], picks: dict, group_by: str = "song", log=print) -> list[Path]:
    """Copy every kept slice into <root>/_ready/ with DAW-friendly names + an acid chunk.

    Re-exporting first removes only the files the previous export wrote; the slices_* folders are never touched."""
    if group_by not in GROUP_BY:
        raise ValueError(f"group_by must be one of {', '.join(GROUP_BY)}")
    root = Path(root)
    ready = root / READY
    ready.mkdir(parents=True, exist_ok=True)
    for rel in picks.get("exported", []):
        p = (ready / rel).resolve()
        if p.is_file() and ready.resolve() in p.parents:
            p.unlink()
    for d in sorted((p for p in ready.rglob("*") if p.is_dir()), key=lambda p: -len(p.parts)):
        if not any(d.iterdir()):
            d.rmdir()
    written, records, used = [], [], set()
    for m in models:
        key = m.get("key") or {}
        name_key = key.get("key") if key.get("confident") else None
        for r, c in kept(m, picks):
            fname = ready_name(m["song"], r["name"], m["bpm"], r["unit"], c["i"], name_key)
            sub = {"song": clean_song(m["song"]), "stem": stem_label(r["name"]), "flat": ""}[group_by]
            dst = ready / sub / fname if sub else ready / fname
            stem_, n = dst.stem, 2
            while str(dst).lower() in used or dst.exists():
                dst = dst.with_name(f"{stem_}_{n}.wav")
                n += 1
            used.add(str(dst).lower())
            dst.parent.mkdir(parents=True, exist_ok=True)
            beats = r["beats_per_slice"] if not c.get("partial") else m["bpm"] * c["dur"] / 60
            write_with_acid(root / m["id"] / c["file"], dst,
                            acid_chunk(m["bpm"], beats, m["meter"], key.get("pc") if name_key else None))
            written.append(dst)
            records.append({"file": str(dst.relative_to(ready)), "song": m["id"], "row": r["name"],
                            "unit": r["unit"], "i": c["i"], "source": str(Path(m["id"]) / c["file"]),
                            "key": key.get("key"), "key_confident": bool(key.get("confident"))})
    picks["exported"] = [rec["file"] for rec in records]
    picks["last_export"] = records
    picks["group_by"] = group_by
    save_picks(root, picks)
    log(f"exported {len(written)} file(s) to {ready}")
    return written
