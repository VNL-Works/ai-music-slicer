# AGENTS.md

Guidance for AI coding agents (and humans) working on this repo.

## What this is
`aislice` is a Python CLI. It locks AI-generated music, which drifts in tempo, to a fixed BPM grid and slices it by bar, beat or N bars. The README is the user-facing doc.

## Setup & run
```bash
brew install ffmpeg rubberband        # system deps (Linux: apt install ffmpeg rubberband-cli)
pip install -e .
aislice ./temp --bpm 206 --slice 8bar # or run with no flags for interactive prompts
python -m aislice --help
```

## Layout
```
src/aislice/
  cli.py        argparse + interactive prompts (entry point: aislice.cli:main)
  pipeline.py   per-file orchestration -> output folder, report.json, tempo.png
  analysis.py   onset envelope (~2.9 ms frames), local tempo curve, DP beat tracker, attack calibration
  tempomap.py   clean beats: missed/extra beats, weak edge beats, smoothing, downbeat phase
  warp.py       source-beat -> target-grid time map; single Rubber Band pass (R3 if available)
  slicer.py     unit parsing (bar, Nbar, beat, Nbeat, 1/8, 1/16), exact cumulative slice points, write_slices
  reslice.py    `aislice reslice`: move bar 1 by N beats / change units on an existing result (no re-stretch)
  verify.py     re-track the output and measure beat error vs. the ideal grid
  report.py     before/after tempo plot
  io.py         ffmpeg decode, soundfile read/write
```

## Invariants: don't break these
- **Warp in one pass** with a time map. Never stretch slice-by-slice (it causes seams and drift).
- **Slice points are computed cumulatively** (`round(k * samples_per_unit)`), so rounding never accumulates. Full slices may differ by at most 1 sample.
- **Pitch is preserved**; only time changes.
- **Beat times mark note attacks.** The `onset_strength` calls use `center=False` because the STFT is already centered. With `center=True`, librosa delays the envelope ~46 ms and every slice cuts off its attack. `attack_latency()` calibrates any remaining offset, and `ATTACK_MARGIN` (2 ms) puts grid lines just before the transient.
- **The first downbeat lands on a bar line** of the output. Silent lead-in is trimmed. An audible pickup is padded with silence up to a bar boundary.
- **Analysis uses a mono 22.05 kHz copy; rendering always uses the original audio** at its native sample rate and channel count.
- **Batch runs keep going when one file fails.**

## Verifying changes
- Run `pytest` (install with `pip install -e ".[dev]"`). `tests/synth.py` builds drifting click+chord tracks with exact ground truth. `test_beats_sit_on_attacks_not_after` and `test_downbeat_uses_harmony_not_accents` are regression guards for real bugs: keep them passing, and don't loosen their tolerances to make a change pass.
- Check `report.json` → `verification`: target p95 < 10 ms (current tracks give 3–4 ms), and `output_bpm_per_20s` should equal the target BPM.
- `verify.py` uses the same tracker as the analysis, so it can't catch a constant timing offset. When you touch analysis or warp, also measure the raw note attacks in the output against the grid (steepest rise of a ~4 ms RMS envelope), independently of the tracker.
- Test audio goes in `temp/` (gitignored). Output goes to `aislice_output/` (gitignored). Never commit audio.

## Releasing
1. Bump `__version__` in `src/aislice/__init__.py` and add a `CHANGELOG.md` entry.
2. Merge to `main` and wait for CI to pass.
3. Create a GitHub Release with tag `vX.Y.Z` (it must match `__version__`). `.github/workflows/release.yml` builds the package and publishes it to PyPI through Trusted Publishing.

## Conventions
- Python ≥ 3.10, type hints, small pure functions per stage; keep heavy imports out of CLI arg parsing.
- Dependencies stay light: numpy, scipy, librosa, soundfile, matplotlib, plus ffmpeg and rubberband binaries. Discuss before adding ML frameworks (e.g. PyTorch for beat_this).
- Keep README's option list in sync with `slicer.unit_beats` / `cli.SLICE_CHOICES`.

## Known gaps / next work
- Downbeat detection is heuristic. Accents can't separate beat 1 from beat 3, so harmonic change picks the "1" and the accents of each half-bar pair are only a tie-breaker. Low margin → CLI warning + `downbeat_confident: false` in report.json. Phrase alignment for 4-/8-bar slices isn't detected. Ideas: repetition/self-similarity, section boundaries, optional beat_this, per-section re-alignment when an AI track drops or adds a beat.
- Expressive outros (ritardando) are flattened onto the grid by design.
- License: MIT. Rubber Band (GPL) is only called as an external binary. Never vendor or link it.
