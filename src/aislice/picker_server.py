"""Local web server for the Picker: one HTML page + a small JSON API.

Safety: binds to 127.0.0.1 only, every request needs a random per-run token, the Host header must be
localhost (blocks DNS-rebinding), and audio is only served from inside the output folder.
"""
from __future__ import annotations

import json
import secrets
import threading
import webbrowser
from http import HTTPStatus
import socketserver
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from importlib import resources
from pathlib import Path
from urllib.parse import parse_qs, urlparse

from . import export as ex
from . import pick


class Picker:
    """State shared by all requests: the output root, songs, analysis cache and picks.json."""

    def __init__(self, path: Path, opts: dict | None = None):
        self.root, dirs = pick.find_songs(path)
        if not dirs:
            raise ValueError(f"no aislice results in {path} (folders with a report.json)")
        self.dirs = {d.name: d for d in dirs}
        self.opts = opts or {}
        self.picks = ex.load_picks(self.root)
        self.models: dict[str, dict] = {}
        self.lock = threading.RLock()

    def songs(self) -> list[dict]:
        out = []
        for k, (sid, d) in enumerate(self.dirs.items()):
            info = pick.song_info(d)
            marks = self.picks["songs"].get(sid, {}).get("marks", {})
            out.append({"id": sid, "song": info["song"], "bpm": info["bpm"], "no": k,
                        "kept": sum(1 for v in marks.values() if v == "keep")})
        return out

    def model(self, sid: str) -> dict:
        if sid not in self.dirs:
            raise KeyError(sid)
        with self.lock:
            if sid not in self.models:
                m = pick.analyze_song(self.dirs[sid], None, self.opts)
                for r in m["rows"]:
                    for c in r["cells"]:
                        c["sid"] = sid
                self.models[sid] = m
            return self.models[sid]

    def song_state(self, sid: str) -> dict:
        m = self.model(sid)
        with self.lock:
            s = ex.song_marks(self.picks, sid)
            return {"model": m, "marks": {sid: s["marks"]}, "heard": {sid: s["heard"]}}

    def best_state(self, n: int = 6) -> dict:
        """One list per stem with the N best loops across ALL songs (same ranking as everywhere: not quiet,
        most repeated, fullest). Songs at another BPM are left out (they wouldn't fit together)."""
        songs = self.songs()
        bpms = [s["bpm"] for s in songs]
        bpm = max(set(bpms), key=bpms.count)
        models = [self.model(s["id"]) for s in songs if s["bpm"] == bpm]
        rows: dict[str, dict] = {}
        for m in models:
            no = next(s["no"] for s in songs if s["id"] == m["id"])
            for r in m["rows"]:
                row = rows.setdefault(r["name"], {"name": r["name"], "label": r["label"], "kind": r["kind"],
                                                  "unit": r["unit"], "bars": r["bars"], "cells": []})
                for c in r["cells"]:
                    if c.get("best") is not None:
                        members = [x["i"] for x in r["cells"] if x["group"] == c["group"]]
                        row["cells"].append({**c, "song": m["song"], "song_no": no, "members": members})
        for row in rows.values():
            row["cells"].sort(key=lambda c: (pick.best_key(c), c["song_no"], c["i"]))
            for k, c in enumerate(row["cells"]):
                c["overall"] = k
        order = sorted(rows, key=lambda nm: (nm == "mix", pick.ROW_ORDER.index(nm) if nm in pick.ROW_ORDER else 50, nm))
        with self.lock:
            marks = {m["id"]: dict(ex.song_marks(self.picks, m["id"])["marks"]) for m in models}
            heard = {m["id"]: list(ex.song_marks(self.picks, m["id"])["heard"]) for m in models}
        return {"model": {"id": "*", "song": "Best of all songs", "bpm": bpm, "meter": models[0]["meter"],
                          "bpb": models[0]["bpb"], "rows": [rows[k] for k in order], "best_of_all": True,
                          "songs": [{"id": m["id"], "song": m["song"]} for m in models], "key": None},
                "marks": marks, "heard": heard}

    def mark(self, sid: str, keys: list[str], value: str | None) -> dict:
        if value not in ("keep", "skip", None):
            raise ValueError("value must be keep, skip or null")
        with self.lock:
            marks = ex.song_marks(self.picks, sid)["marks"]
            for k in keys:
                if value is None:
                    marks.pop(k, None)
                else:
                    marks[k] = value
            ex.save_picks(self.root, self.picks)
            return marks

    def heard(self, sid: str, key: str) -> None:
        with self.lock:
            h = ex.song_marks(self.picks, sid)["heard"]
            if key not in h:
                h.append(key)
                ex.save_picks(self.root, self.picks)

    def export(self, group_by: str) -> list[str]:
        with self.lock:
            models = [self.model(s["id"]) for s in self.songs()]
            files = ex.export(self.root, models, self.picks, group_by, log=lambda *a: None)
            return [str(f.relative_to(self.root)) for f in files]

    # ------------------------------------------------------------ jam pad
    def jam_state(self, bpm: float | None = None) -> dict:
        from . import jam
        songs = self.songs()
        bpms = sorted({s["bpm"] for s in songs})
        bpm = bpm if bpm in bpms else max(bpms, key=lambda b: sum(s["bpm"] == b for s in songs))
        models = [self.model(s["id"]) for s in songs if s["bpm"] == bpm]
        with self.lock:
            pads = jam.build_pads(models, self.picks)
        m0 = models[0]
        return {"bpm": bpm, "bpms": bpms, "bpb": m0["bpb"], "meter": m0["meter"], "tracks": pads,
                "songs": [{"id": m["id"], "song": m["song"]} for m in models]}

    def jam_render(self, body: dict) -> dict:
        from . import io, jam
        import numpy as np
        st = self.jam_state(body.get("bpm"))
        clips = {c["id"]: c for t in st["tracks"] for c in t["clips"]}
        events = []
        for e in body.get("events", []):
            ev = {"bar": int(e["bar"]), "track": str(e["track"])}
            if "clip" in e:
                if e["clip"] is not None and e["clip"] not in clips:
                    raise ValueError(f"unknown clip {e['clip']}")
                ev["clip"] = e["clip"]
            if "gain" in e:
                ev["gain"] = float(e["gain"])
            events.append(ev)
        used = [c for c in clips if any(e.get("clip") == c for e in events)]
        if not used:
            raise ValueError("nothing was playing while recording")
        sr = io.read(self.audio_path(clips[used[0]]["sid"], clips[used[0]]["file"]))[1]

        def load(cid):
            c = clips[cid]
            y, r = io.read(self.audio_path(c["sid"], c["file"]))
            if r != sr:
                import librosa
                y = librosa.resample(y.T, orig_sr=r, target_sr=sr, res_type="soxr_hq").T
            return np.ascontiguousarray(y, dtype=np.float32)
        mix, stems = jam.render(events, int(body["bars"]), st["bpm"], st["bpb"], lambda cid: clips[cid]["bars"],
                                load, sr)
        labels = {t["name"]: t["label"] for t in st["tracks"]}
        with self.lock:
            info = jam.save_jam(self.root, mix, stems, sr, st["bpm"], st["meter"], int(body["bars"]), labels)
        info["path"] = str(self.root / info["folder"])
        return info

    def audio_path(self, sid: str, rel: str) -> Path:
        base = self.dirs[sid].resolve()
        p = (base / rel).resolve()
        if base not in p.parents or p.suffix.lower() != ".wav" or not p.is_file():
            raise PermissionError(rel)
        return p


def make_handler(app: Picker, token: str):
    pages = {p: resources.files("aislice").joinpath(f).read_text(encoding="utf-8")
             for p, f in (("/", "picker.html"), ("/jam", "jam.html"))}

    class Handler(BaseHTTPRequestHandler):
        server_version = "aislice-picker"

        def log_message(self, *a):          # quiet console
            pass

        def _ok_origin(self) -> bool:
            host = (self.headers.get("Host") or "").split(":")[0]
            return host in ("127.0.0.1", "localhost")

        def _send(self, code, body: bytes, ctype="application/json", extra=None):
            self.send_response(code)
            self.send_header("Content-Type", ctype)
            self.send_header("Content-Length", str(len(body)))
            self.send_header("Cache-Control", "no-store")
            self.send_header("X-Content-Type-Options", "nosniff")
            for k, v in (extra or {}).items():
                self.send_header(k, v)
            self.end_headers()
            self.wfile.write(body)

        def _json(self, obj, code=200):
            self._send(code, json.dumps(obj).encode())

        def _err(self, code, msg):
            self._json({"error": msg}, code)

        def _auth(self, q) -> bool:
            t = self.headers.get("X-Token") or (q.get("t") or [""])[0]
            return self._ok_origin() and secrets.compare_digest(t, token)

        def do_GET(self):
            u = urlparse(self.path)
            q = parse_qs(u.query)
            if not self._auth(q):
                return self._err(HTTPStatus.FORBIDDEN, "forbidden")
            try:
                if u.path in pages:
                    return self._send(200, pages[u.path].replace("__TOKEN__", token).encode(), "text/html; charset=utf-8",
                                      {"Content-Security-Policy": "default-src 'self' 'unsafe-inline' blob:; "
                                                                  "media-src 'self' blob:; connect-src 'self'"})
                if u.path == "/api/songs":
                    return self._json({"root": str(self.server.app.root), "songs": app.songs(),
                                       "group_by": app.picks.get("group_by", "song")})
                if u.path == "/api/song":
                    sid = q["id"][0]
                    if sid == "*":
                        return self._json(app.best_state(int((q.get("n") or ["6"])[0])))
                    return self._json(app.song_state(sid))
                if u.path == "/api/jam":
                    b = (q.get("bpm") or [None])[0]
                    return self._json(app.jam_state(float(b) if b else None))
                if u.path == "/audio":
                    p = app.audio_path(q["id"][0], q["f"][0])
                    return self._send(200, p.read_bytes(), "audio/wav")
                return self._err(404, "not found")
            except (KeyError, PermissionError) as e:
                return self._err(404, f"not found: {e}")
            except ValueError as e:
                return self._err(400, str(e))

        def do_POST(self):
            u = urlparse(self.path)
            if not self._auth(parse_qs(u.query)):
                return self._err(HTTPStatus.FORBIDDEN, "forbidden")
            try:
                n = int(self.headers.get("Content-Length") or 0)
                if n > 1_000_000:
                    return self._err(413, "too large")
                body = json.loads(self.rfile.read(n) or b"{}")
                if u.path == "/api/mark":
                    return self._json({"marks": app.mark(body["id"], list(body["keys"]), body.get("value"))})
                if u.path == "/api/heard":
                    app.heard(body["id"], body["key"])
                    return self._json({"ok": True})
                if u.path == "/api/jam/render":
                    return self._json(app.jam_render(body))
                if u.path == "/api/export":
                    files = app.export(body.get("group_by", "song"))
                    return self._json({"files": files, "folder": str(app.root / ex.READY)})
                return self._err(404, "not found")
            except (KeyError, ValueError, TypeError, PermissionError) as e:
                return self._err(400, str(e))

    return Handler


class LocalServer(ThreadingHTTPServer):
    """ThreadingHTTPServer without the reverse-DNS lookup of the bind address in server_bind(), which can hang
    for many seconds on macOS. The server only ever listens on 127.0.0.1 anyway."""
    daemon_threads = True

    def server_bind(self):
        socketserver.TCPServer.server_bind(self)
        host, port = self.server_address[:2]
        self.server_name, self.server_port = host, port


def serve(path: Path, opts: dict | None = None, port: int = 0, open_browser: bool = True,
          log=print, ready=None, page: str = "/"):
    app = Picker(path, opts)
    token = secrets.token_urlsafe(16)
    httpd = LocalServer(("127.0.0.1", port), make_handler(app, token))
    httpd.app = app
    url = f"http://127.0.0.1:{httpd.server_address[1]}{page}?t={token}"
    log(f"Picker running at {url}\n  {len(app.dirs)} song(s) in {app.root}. Press Ctrl+C to stop.")
    if ready:
        ready(httpd, url)
    if open_browser:
        threading.Timer(0.3, webbrowser.open, (url,)).start()
    try:
        httpd.serve_forever()
    except KeyboardInterrupt:
        log("\nPicker stopped. Picks are saved in " + str(ex.picks_path(app.root)))
    finally:
        httpd.server_close()
