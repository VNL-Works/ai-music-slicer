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


def main(argv=None) -> int:
    argv = sys.argv[1:] if argv is None else argv
    if argv and argv[0] == "reslice":
        return reslice_main(argv[1:])
    ap = argparse.ArgumentParser(prog="aislice", description=__doc__,
                                 epilog="Fix a wrong bar start afterwards: aislice reslice <output folder> --shift N")
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

    opt = Options(bpm=bpm, slice_unit=slice_unit, beats_per_bar=int(args.meter.split("/")[0]),
                  out_root=Path(args.out), fmt=args.format, bit_depth=args.bit_depth,
                  fade_ms=args.fade_ms, drop_partial=args.drop_partial, plot=not args.no_plot)
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
