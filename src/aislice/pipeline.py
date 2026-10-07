from __future__ import annotations

import json
import re
import tempfile
from dataclasses import dataclass
from pathlib import Path

import librosa
import numpy as np

from . import io, slicer, warp
from .analysis import analyze
from .tempomap import build_tempo_map
from .verify import verify


@dataclass
class Options:
    bpm: float
    slice_unit: str = "bar"          # bar, beat, 2bar, 4bar, 1/8, 1/16, none
    beats_per_bar: int = 4
    out_root: Path = Path("aislice_output")
    fmt: str = "wav"
    bit_depth: str = "24"
    fade_ms: float = 1.0
    drop_partial: bool = False
    plot: bool = True


def _safe(name: str) -> str:
    return re.sub(r"[^\w\-. ()]+", "_", name).strip()


def detect(path: Path, bpm_hint: float | None = None):
    with tempfile.TemporaryDirectory() as td:
        wav = Path(td) / "src.wav"
        io.decode_to_wav(path, wav)
        y, sr = io.read(wav)
    mono = librosa.resample(y.mean(1), orig_sr=sr, target_sr=22050)
    return analyze(mono, 22050, bpm_hint=bpm_hint)


def relate_tempo(detected: float, target: float) -> float:
    """Pick the octave of the detected tempo closest to the target (half/double-time)."""
    cands = [detected * f for f in (0.5, 1, 2)]
    return min(cands, key=lambda c: abs(np.log(c / target)))


def process(path: Path, opt: Options, log=print) -> Path:
    stem = _safe(path.stem)
    bpm_tag = f"{opt.bpm:g}".replace(".", "_")
    out_dir = opt.out_root / f"{stem}_{bpm_tag}bpm"
    out_dir.mkdir(parents=True, exist_ok=True)

    with tempfile.TemporaryDirectory() as td:
        wav = Path(td) / "src.wav"
        io.decode_to_wav(path, wav)
        y, sr = io.read(wav)
        src_dur = len(y) / sr

        # 1. analysis (hint: the target; the tracker searches +-15% around the matching octave)
        mono = librosa.resample(y.mean(1), orig_sr=sr, target_sr=22050)
        rough = float(librosa.feature.tempo(y=mono, sr=22050)[0])
        hint = relate_tempo(rough, opt.bpm)
        a = analyze(mono, 22050, bpm_hint=hint)
        log(f"  tracked {len(a.beats)} beats (attack calibration {a.latency*1000:+.1f} ms)")

        # 2. tempo map
        tm = build_tempo_map(a, beats_per_bar=opt.beats_per_bar)
        if not tm.downbeat_confident:
            log("  WARNING: not sure where bar 1 starts. Listen to slice 1; if it's off, run:")
            log(f"    aislice reslice \"{opt.out_root}/{stem}_{bpm_tag}bpm\" --shift 1|2|3")
        lb = tm.local_bpm[tm.confidence[:-1] > 0.25]

        # 3. warp
        # anything audible before the first downbeat? (keep 10 ms before it as attack margin)
        cut = int(max(0.0, tm.beats[tm.downbeat_index] - 0.010) * sr)
        pre_db = 20 * np.log10(np.sqrt(np.mean(y[:cut] ** 2)) + 1e-12) if cut > 0 else -120.0
        wp = warp.plan(tm, src_dur, opt.bpm, preroll_silent=pre_db < -45)
        y_out = warp.render(wav, wp, sr)

    full = out_dir / f"{stem}_{bpm_tag}bpm.{opt.fmt}"
    io.write(full, y_out, sr, opt.fmt, opt.bit_depth)

    # 4. slices
    slice_info = None
    if opt.slice_unit != "none":
        upb = slicer.unit_beats(opt.slice_unit, opt.beats_per_bar)
        pts = slicer.slice_points(len(y_out), sr, opt.bpm, upb, opt.drop_partial)
        sdir = out_dir / "slices"
        sdir.mkdir(exist_ok=True)
        label = opt.slice_unit.replace("/", "-")
        width = max(3, len(str(len(pts))))
        for i, (s, e) in enumerate(pts, 1):
            seg = slicer.apply_fades(y_out[s:e], sr, opt.fade_ms)
            io.write(sdir / f"{stem}_{label}_{i:0{width}d}.{opt.fmt}", seg, sr, opt.fmt, opt.bit_depth)
        first_db = wp.first_downbeat_out_s
        slice_info = {"unit": opt.slice_unit, "count": len(pts),
                      "samples_per_slice": round(upb * 60 / opt.bpm * sr, 3),
                      "first_downbeat_slice": int(round(first_db / (upb * 60 / opt.bpm))) + 1}

    # 5. verify
    v = verify(y_out, sr, opt.bpm)
    log(f"  verify: mean {v['mean_err_ms']} ms, p95 {v['p95_err_ms']} ms -> {'PASS' if v['pass'] else 'CHECK'}")

    report = {
        "source": path.name,
        "sample_rate": sr,
        "target_bpm": opt.bpm,
        "meter": f"{opt.beats_per_bar}/4",
        "detected": {
            "median_bpm": round(float(np.median(lb)), 2),
            "p5_bpm": round(float(np.percentile(lb, 5)), 2),
            "p95_bpm": round(float(np.percentile(lb, 95)), 2),
            "bpm_per_20s": _bpm_windows(tm),
        },
        "stretch": {"min_ratio": round(float(wp.ratios.min()), 4), "max_ratio": round(float(wp.ratios.max()), 4)},
        "grid": {"beats": int(len(tm.beats)), "first_downbeat_source_s": round(float(tm.beats[tm.downbeat_index]), 3),
                 "downbeat_margin": round(tm.downbeat_margin, 3),
                 "downbeat_confident": tm.downbeat_confident,
                 "first_downbeat_output_s": round(wp.first_downbeat_out_s, 4),
                 "silence_padded_s": round(wp.pad_s, 4)},
        "low_confidence_regions": tm.low_conf_regions,
        "notes": tm.notes,
        "verification": {k: val for k, val in v.items() if not k.startswith("_")},
        "slices": slice_info,
        "duration": {"source_s": round(src_dur, 3), "output_s": round(len(y_out) / sr, 3)},
    }
    (out_dir / "report.json").write_text(json.dumps(report, indent=2))
    if opt.plot:
        from .report import plot
        plot(out_dir / "tempo.png", path.name, tm, opt.bpm, v)
    return out_dir


def _bpm_windows(tm):
    b = tm.beats
    out = []
    for s in np.arange(0, b[-1], 20):
        m = (b >= s) & (b < s + 20) & (tm.confidence > 0.25)
        idx = np.where(m)[0]
        if len(idx) > 8:
            p = np.polyfit(idx, b[idx], 1)
            out.append({"from_s": int(s), "bpm": round(60 / p[0], 2)})
    return out
