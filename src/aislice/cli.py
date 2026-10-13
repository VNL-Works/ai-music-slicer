"""aislice — lock AI-generated music to a fixed BPM grid and slice it."""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

from . import __version__
from .io import collect_inputs

SLICE_CHOICES = ["bar", "2bar", "4bar", "8bar", "beat", "1/8", "1/16", "none"]


def _valid_unit(u: str) -> bool:
    from .slicer import unit_beats
    try:
        unit_beats(u, 4)
        return True
    except ValueError:
        return False


def _slice_arg(u: str) -> str:
    if u.lower() == "standard":
        return "standard"
    if u != "none" and not _valid_unit(u):
        raise argparse.ArgumentTypeError(f"invalid slice unit {u!r}: use bar, 2bar, 4bar, 8bar, Nbar, beat, Nbeat, 1/8, 1/16 or none")
    return u.lower()


def reslice_main(argv) -> int:
    ap = argparse.ArgumentParser(prog="aislice reslice",
                                 description="Re-cut existing results: move bar 1 by N beats and/or change slice units.")
    ap.add_argument("folders", nargs="+", help="output folder(s) made by aislice, e.g. aislice_output/song_206bpm")
    ap.add_argument("--shift", type=int, default=0, help="move bar 1 later by N beats (negative = earlier)")
    ap.add_argument("--slice", help="comma-separated units, e.g. bar,4bar,8bar (default: re-cut existing folders)")
    ap.add_argument("--fade-ms", type=float, default=1.0)
    ap.add_argument("--drop-partial", action="store_true")
    args = ap.parse_args(argv)
    units = [_slice_arg(u.strip()) for u in args.slice.split(",")] if args.slice else None
    from .reslice import reslice
    for f in args.folders:
        reslice(Path(f), args.shift, units, args.fade_ms, args.drop_partial)
    return 0


STEM_CHOICES = ["none", "2", "3", "4", "6"]


def _require_stems_extra() -> None:
    import importlib.util
    if importlib.util.find_spec("audio_separator") is None:
        print('Stem separation needs the optional extra:  pip install "aislice[stems]"', file=sys.stderr)
        raise SystemExit(2)


def _add_stem_args(ap, hide: bool = False):
    h = (lambda t: argparse.SUPPRESS) if hide else (lambda t: t)
    ap.add_argument("--stems", choices=STEM_CHOICES,
                    help=h("separate stems after locking the tempo: 2 (vocals/instrumental), 3 (drums/bass/melody), 4 or 6 "
                         '(needs: pip install "aislice[stems]")'))
    ap.add_argument("--stem-model", help=h("any audio-separator model filename (overrides the --stems preset), "
                                         "e.g. model_bs_roformer_ep_317_sdr_12.9755.ckpt"))
    ap.add_argument("--stems-fast", action="store_true",
                    help=h("use Demucs (htdemucs) instead of the default BS-Roformer-SW: much faster on CPU, lower quality"))
    ap.add_argument("--stems-exact", action="store_true",
                    help=h("add what the model leaves out into the melody/other stem, so the stems sum exactly "
                         "to the mix"))
    ap.add_argument("--model-dir", help=h("folder to cache separation models in"))


def stems_main(argv) -> int:
    ap = argparse.ArgumentParser(prog="aislice stems",
                                 description="Add stems to existing aislice results (no re-processing of the tempo).")
    ap.add_argument("folders", nargs="+", help="output folder(s) made by aislice, e.g. aislice_output/song_206bpm")
    _add_stem_args(ap)
    ap.add_argument("--slice", help="advanced: comma-separated units to slice stems by (default: the standard layout, "
                                     "drums/bass 4bar, the rest 8bar)")
    ap.add_argument("--fade-ms", type=float, default=1.0)
    ap.add_argument("--drop-partial", action="store_true")
    args = ap.parse_args(argv)
    units = [u for u in (_slice_arg(x.strip()) for x in args.slice.split(",")) if u != "none"] if args.slice else None
    _require_stems_extra()
    from .stems_cmd import add_stems_to_result
    failed = 0
    for f in args.folders:
        try:
            add_stems_to_result(Path(f), args.stems if args.stems not in (None, "none") else "4", args.stem_model,
                                Path(args.model_dir) if args.model_dir else None, units, args.fade_ms,
                                args.drop_partial, fast=args.stems_fast, exact=args.stems_exact)
        except Exception as e:
            failed += 1
            print(f"{f}: FAILED: {e}", file=sys.stderr)
    return 1 if failed else 0


def _add_pick_args(ap):
    ap.add_argument("--empty-db", type=float, default=-45.0,
                    help="hide slices whose peak is below this, in dB relative to the song's mix peak (default -45)")
    ap.add_argument("--empty-active", type=float, default=0.15,
                    help="hide slices that have sound (RMS above -40 dB) for less than this share of the slice (default 0.15)")
    ap.add_argument("--group-sim", type=float, default=0.60,
                    help="how alike slices must be (beat by beat) to be grouped as repeats, 0-1 (default 0.6; 0 = off)")
    ap.add_argument("--group-rep", choices=["medoid", "loudest"], default="medoid",
                    help="which slice represents a group: the most typical one (default) or the loudest")


def _pick_opts(args) -> dict:
    return {"empty_db": args.empty_db, "empty_active": args.empty_active, "group_sim": args.group_sim,
            "group_rep": args.group_rep}


def run_auto_pick(path: Path, opts: dict, group_by: str, log=print, best: int | None = None) -> int:
    from . import export as ex
    from . import pick
    root, dirs = pick.find_songs(path)
    if not dirs:
        print(f"No aislice results in {path}", file=sys.stderr)
        return 1
    picks = ex.load_picks(root)
    models = []
    for d in dirs:
        try:
            m = pick.analyze_song(d, None, opts)
        except ValueError as e:
            log(f"  {d.name}: skipped ({e})")
            continue
        n = pick.auto_marks(m, picks, best)
        log(f"  {pick.summary(m)} -> keeping {n} new")
        models.append(m)
    ex.export(root, models, picks, group_by, log=log)
    return 0


def pick_main(argv) -> int:
    ap = argparse.ArgumentParser(prog="aislice pick",
                                 description="Audition slices in the browser, keep the good ones and export them "
                                             "DAW-ready to <output>/_ready/. Empty slices are hidden and repeats grouped.")
    ap.add_argument("folder", nargs="?", default="aislice_output",
                    help="output folder (or one song's result folder); default: aislice_output")
    ap.add_argument("--auto", action="store_true",
                    help="no browser: keep one slice per group (skipping empty and quiet ones) and export")
    ap.add_argument("--best", type=int, metavar="N",
                    help="with --auto: keep only the N best loops per stem (most repeated, not quiet, fullest)")
    ap.add_argument("--group-by", choices=["song", "stem", "flat"], default="song", help="_ready/ folder layout")
    ap.add_argument("--port", type=int, default=0, help="local port (default: any free port)")
    ap.add_argument("--no-open", action="store_true", help="don't open the browser")
    _add_pick_args(ap)
    args = ap.parse_args(argv)
    path = Path(args.folder)
    if not path.is_dir():
        print(f"{path}: not a folder", file=sys.stderr)
        return 1
    if args.auto:
        return run_auto_pick(path, _pick_opts(args), args.group_by, best=args.best)
    from .picker_server import serve
    try:
        serve(path, _pick_opts(args), args.port, not args.no_open)
    except ValueError as e:
        print(e, file=sys.stderr)
        return 1
    return 0


def jam_main(argv) -> int:
    ap = argparse.ArgumentParser(prog="aislice jam",
                                 description="Launchpad in the browser: play loops from all your songs and stems in sync, "
                                             "record a jam and get a mix + stems in <output>/_ready/jams/.")
    ap.add_argument("folder", nargs="?", default="aislice_output",
                    help="output folder (or one song's result folder); default: aislice_output")
    ap.add_argument("--port", type=int, default=0, help="local port (default: any free port)")
    ap.add_argument("--no-open", action="store_true", help="don't open the browser")
    _add_pick_args(ap)
    args = ap.parse_args(argv)
    path = Path(args.folder)
    if not path.is_dir():
        print(f"{path}: not a folder", file=sys.stderr)
        return 1
    from .picker_server import serve
    try:
        serve(path, _pick_opts(args), args.port, not args.no_open, page="/jam")
    except ValueError as e:
        print(e, file=sys.stderr)
        return 1
    return 0


def main(argv=None) -> int:
    argv = sys.argv[1:] if argv is None else argv
    if argv and argv[0] == "reslice":
        return reslice_main(argv[1:])
    if argv and argv[0] == "stems":
        return stems_main(argv[1:])
    if argv and argv[0] == "pick":
        return pick_main(argv[1:])
    if argv and argv[0] == "jam":
        return jam_main(argv[1:])
    ap = argparse.ArgumentParser(prog="aislice", description=__doc__ + " AI songs in, best samples out: "
                                 "aislice songs/ --bpm 206 locks every song to the BPM, splits drums, bass and melody, "
                                 "cuts loops (drums/bass 4 bars, melody 8 bars, each normalized to -0.1 dB) and puts the "
                                 "best ones in one folder per run: <out>/<date>_<time>_<bpm>bpm/.",
                                 epilog="Optional tools: aislice pick <out> (audition and hand-pick), aislice jam <out> "
                                        "(launchpad), aislice reslice <song folder> --shift N (fix the bar start).")
    ap.add_argument("inputs", nargs="+", help="audio files and/or folders")
    ap.add_argument("--bpm", type=float, required=True, help="target BPM")
    ap.add_argument("--best", type=int, default=None, metavar="N",
                    help="how many best loops per instrument (default 8)")
    ap.add_argument("--out", default="aislice_output", help="output root folder")
    ap.add_argument("--meter", default="4/4", help="time signature, e.g. 4/4, 3/4")
    ap.add_argument("--slice", type=_slice_arg, help=argparse.SUPPRESS)
    ap.add_argument("--format", default="wav", choices=["wav", "flac"], help=argparse.SUPPRESS)
    ap.add_argument("--bit-depth", default="24", choices=["16", "24", "32f"], help=argparse.SUPPRESS)
    ap.add_argument("--fade-ms", type=float, default=1.0, help=argparse.SUPPRESS)
    ap.add_argument("--drop-partial", action="store_true", help=argparse.SUPPRESS)
    ap.add_argument("--no-plot", action="store_true", help=argparse.SUPPRESS)
    _add_stem_args(ap, hide=True)
    ap.add_argument("--downbeats", choices=["auto", "beat_this", "harmony"], default="auto", help=argparse.SUPPRESS)
    ap.add_argument("--pick", action="store_true", help=argparse.SUPPRESS)
    ap.add_argument("--keep-work", action="store_true", help=argparse.SUPPRESS)   # keep songs/stems/all loops
    ap.add_argument("-y", "--yes", action="store_true", help=argparse.SUPPRESS)
    ap.add_argument("--version", action="version", version=__version__)
    args = ap.parse_args(argv)

    files = collect_inputs(args.inputs)
    if not files:
        print("No audio files found.", file=sys.stderr)
        return 1
    if not 20 <= args.bpm <= 400:
        print("--bpm must be between 20 and 400", file=sys.stderr)
        return 2
    print(f"Found {len(files)} audio file(s).")

    stems = args.stems if args.stems is not None else "3"       # drums, bass, melody: always split
    stems = None if stems == "none" else stems
    if stems or args.stem_model:
        _require_stems_extra()
    if args.downbeats == "beat_this":
        from .downbeats import INSTALL_HINT, available
        if not available():
            print(INSTALL_HINT, file=sys.stderr)
            raise SystemExit(2)

    from .pipeline import Options, process  # heavy imports after arg parsing

    import shutil

    from .best import DEFAULT_N, build_best, session_dir

    slice_unit = args.slice or "standard"
    out_root = Path(args.out)
    session = session_dir(out_root, args.bpm)
    simple = slice_unit != "none" and args.format == "wav"      # the normal case: songs in -> best samples out
    # per-song working files (locked song, stems, every loop) go to a hidden folder that is removed at the end
    work = out_root / f".work_{session.name}" if simple else out_root
    opt = Options(bpm=args.bpm, slice_unit=slice_unit, beats_per_bar=int(args.meter.split("/")[0]),
                  out_root=work, fmt=args.format, bit_depth=args.bit_depth,
                  fade_ms=args.fade_ms, drop_partial=args.drop_partial, plot=not args.no_plot,
                  stems=stems, stem_model=args.stem_model, downbeats=args.downbeats, stems_fast=args.stems_fast,
                  stems_exact=args.stems_exact,
                  model_dir=Path(args.model_dir) if args.model_dir else None)
    failed, done = 0, []
    for i, f in enumerate(files, 1):
        print(f"[{i}/{len(files)}] {f.name}")
        try:
            done.append(process(f, opt))
        except Exception as e:  # keep going with the rest of the batch
            failed += 1
            print(f"  FAILED: {e}", file=sys.stderr)
    if not simple:
        print(f"Done. Output in {out_root}/" + (f" ({failed} failed)" if failed else ""))
        return 1 if failed else 0
    written = build_best(done, session, args.best or DEFAULT_N) if done else []
    if args.pick and done:
        from .picker_server import serve
        serve(work)
    if not args.keep_work and not args.pick:
        shutil.rmtree(work, ignore_errors=True)
    elif done:
        print(f"Working files kept in {work}/")
    if written:
        print(f"Done. {len(written)} best samples in {session}/" + (f" ({failed} song(s) failed)" if failed else ""))
    else:
        print("Done, but no samples came out" + (f" ({failed} song(s) failed)" if failed else "."), file=sys.stderr)
    return 1 if failed or not written else 0


if __name__ == "__main__":
    sys.exit(main())
