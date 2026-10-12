"""aislice — lock AI-generated music to a fixed BPM grid and slice it."""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

from . import __version__
from .io import collect_inputs

SLICE_CHOICES = ["bar", "2bar", "4bar", "8bar", "beat", "1/8", "1/16", "none"]


def _ask(prompt: str, default: str) -> str:
    try:
        ans = input(f"? {prompt} [{default}]: ").strip()
    except EOFError:
        ans = ""
    return ans or default


def _ask_bpm(default: float | None) -> float:
    while True:
        ans = _ask("Target BPM", f"{default:g}" if default else "120")
        try:
            v = float(ans)
            if 20 <= v <= 400:
                return v
        except ValueError:
            pass
        print("  Please enter a number between 20 and 400.")


def _valid_unit(u: str) -> bool:
    from .slicer import unit_beats
    try:
        unit_beats(u, 4)
        return True
    except ValueError:
        return False


def _slice_arg(u: str) -> str:
    if u != "none" and not _valid_unit(u):
        raise argparse.ArgumentTypeError(f"invalid slice unit {u!r}: use bar, 2bar, 4bar, 8bar, Nbar, beat, Nbeat, 1/8, 1/16 or none")
    return u.lower()


def _ask_slice() -> str:
    print("? Slice by:")
    for i, c in enumerate(SLICE_CHOICES, 1):
        print(f"    {i}) {c}" + ("  (full song only)" if c == "none" else ""))
    while True:
        ans = _ask(f"Choose 1-{len(SLICE_CHOICES)}, or type e.g. 16bar", "1")
        if ans.isdigit() and 1 <= int(ans) <= len(SLICE_CHOICES):
            return SLICE_CHOICES[int(ans) - 1]
        if ans == "none" or _valid_unit(ans):
            return ans
        print("  Not a valid choice.")


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


def _ask_stems() -> str | None:
    print("? Separate stems? (needs the [stems] extra; slow on CPU)")
    print("    1) none\n    2) 2 stems: vocals + instrumental\n    3) 3 stems: drums, bass, melody\n"
          "    4) 4 stems: vocals, drums, bass, other\n    5) 6 stems: vocals, drums, bass, guitar, piano, other")
    while True:
        ans = _ask("Choose 1-5", "1")
        m = {"1": None, "2": "2", "3": "3", "4": "4", "5": "6", "none": None}
        if ans in m:
            return m[ans]
        print("  Not a valid choice.")


def _require_stems_extra() -> None:
    import importlib.util
    if importlib.util.find_spec("audio_separator") is None:
        print('Stem separation needs the optional extra:  pip install "aislice[stems]"', file=sys.stderr)
        raise SystemExit(2)


def _add_stem_args(ap):
    ap.add_argument("--stems", choices=STEM_CHOICES,
                    help="separate stems after locking the tempo: 2 (vocals/instrumental), 3 (drums/bass/melody), 4 or 6 "
                         '(needs: pip install "aislice[stems]")')
    ap.add_argument("--stem-model", help="any audio-separator model filename (overrides the --stems preset), "
                                         "e.g. model_bs_roformer_ep_317_sdr_12.9755.ckpt")
    ap.add_argument("--stems-fast", action="store_true",
                    help="use htdemucs instead of the default htdemucs_ft: ~4x faster, slightly lower quality")
    ap.add_argument("--stems-exact", action="store_true",
                    help="add what the model leaves out into the melody/other stem, so the stems sum exactly "
                         "to the mix (htdemucs_ft leaves ~1%% out)")
    ap.add_argument("--model-dir", help="folder to cache separation models in")


def stems_main(argv) -> int:
    ap = argparse.ArgumentParser(prog="aislice stems",
                                 description="Add stems to existing aislice results (no re-processing of the tempo).")
    ap.add_argument("folders", nargs="+", help="output folder(s) made by aislice, e.g. aislice_output/song_206bpm")
    _add_stem_args(ap)
    ap.add_argument("--slice", help="comma-separated units to slice stems by (default: the slice folders that exist)")
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


def main(argv=None) -> int:
    argv = sys.argv[1:] if argv is None else argv
    if argv and argv[0] == "reslice":
        return reslice_main(argv[1:])
    if argv and argv[0] == "stems":
        return stems_main(argv[1:])
    ap = argparse.ArgumentParser(prog="aislice", description=__doc__,
                                 epilog="Fix a wrong bar start afterwards: aislice reslice <output folder> --shift N. "
                                        "Add stems to existing results: aislice stems <output folder> --stems 4")
    ap.add_argument("inputs", nargs="+", help="audio files and/or folders")
    ap.add_argument("--bpm", type=float, help="target BPM")
    ap.add_argument("--slice", type=_slice_arg, help="slice unit: bar, 2bar, 4bar, 8bar (any Nbar), beat (any Nbeat), 1/8, 1/16, none")
    ap.add_argument("--meter", default="4/4", help="time signature, e.g. 4/4, 3/4")
    ap.add_argument("--out", default="aislice_output", help="output root folder")
    ap.add_argument("--format", default="wav", choices=["wav", "flac"])
    ap.add_argument("--bit-depth", default="24", choices=["16", "24", "32f"])
    ap.add_argument("--fade-ms", type=float, default=1.0)
    ap.add_argument("--drop-partial", action="store_true")
    ap.add_argument("--no-plot", action="store_true")
    _add_stem_args(ap)
    ap.add_argument("--downbeats", choices=["auto", "beat_this", "harmony"], default="auto",
                    help="how to find bar 1: beat_this model (pip install \"aislice[beats]\"), the built-in "
                         "harmony heuristic, or auto (beat_this when installed)")
    ap.add_argument("-y", "--yes", action="store_true", help="accept defaults, no prompts")
    ap.add_argument("--version", action="version", version=__version__)
    args = ap.parse_args(argv)

    files = collect_inputs(args.inputs)
    if not files:
        print("No audio files found.", file=sys.stderr)
        return 1
    print(f"Found {len(files)} audio file(s).")

    from .pipeline import Options, detect, process  # heavy imports after arg parsing

    bpm = args.bpm
    if bpm is None:
        print("Analyzing tempo ...")
        est = []
        for f in files:
            a = detect(f)
            import numpy as np
            med = float(60 / np.median(np.diff(a.beats)))
            est.append(med)
            print(f"  {f.name}: ~{med:.1f} BPM")
        default = round(sorted(est)[len(est) // 2])
        bpm = default if args.yes else _ask_bpm(default)
    slice_unit = args.slice or ("bar" if args.yes else _ask_slice())
    if args.stems is not None:
        stems = None if args.stems == "none" else args.stems
    else:
        stems = None if (args.yes or args.stem_model) else _ask_stems()
    if stems or args.stem_model:
        _require_stems_extra()
    if args.downbeats == "beat_this":
        from .downbeats import INSTALL_HINT, available
        if not available():
            print(INSTALL_HINT, file=sys.stderr)
            raise SystemExit(2)

    opt = Options(bpm=bpm, slice_unit=slice_unit, beats_per_bar=int(args.meter.split("/")[0]),
                  out_root=Path(args.out), fmt=args.format, bit_depth=args.bit_depth,
                  fade_ms=args.fade_ms, drop_partial=args.drop_partial, plot=not args.no_plot,
                  stems=stems, stem_model=args.stem_model, downbeats=args.downbeats, stems_fast=args.stems_fast,
                  stems_exact=args.stems_exact,
                  model_dir=Path(args.model_dir) if args.model_dir else None)
    failed = 0
    for i, f in enumerate(files, 1):
        print(f"[{i}/{len(files)}] {f.name}")
        try:
            d = process(f, opt)
            print(f"  -> {d}")
        except Exception as e:  # keep going with the rest of the batch
            failed += 1
            print(f"  FAILED: {e}", file=sys.stderr)
    print(f"Done. Output in {opt.out_root}/" + (f" ({failed} failed)" if failed else ""))
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
