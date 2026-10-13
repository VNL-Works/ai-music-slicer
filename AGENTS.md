# AGENTS.md

Guidance for AI coding agents (and humans) working on this repo.

## What this is
`aislice` is a Python CLI. It locks AI-generated music, which drifts in tempo, to a fixed BPM grid and slices it by bar, beat or N bars. The README is the user-facing doc.

## Setup & run
```bash
brew install ffmpeg rubberband        # system deps (Linux: apt install ffmpeg rubberband-cli)
pip install -e .
aislice ./temp --bpm 206              # songs in, best samples out (aislice_output/<date>_<time>_206bpm/)
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
  reslice.py    `aislice reslice`: move bar 1 by N beats / change units on an existing result (no re-stretch); moves stems too
  stems.py      optional stem separation via audio-separator (lazy import; [stems] extra), resample/align/merge
  stems_cmd.py  `aislice stems`: add stems to an existing result
  downbeats.py  optional beat_this downbeats ([beats] extra): votes which grid beat is bar 1
  pick.py       Picker analysis: empty slices, beat-by-beat repeat groups per stem, key estimate, pick_cache.json
  export.py     picks.json, DAW-ready names, ACID chunk, export to <output>/_ready/
  picker_server.py  `aislice pick`: stdlib http.server on 127.0.0.1 (token + Host check) serving picker.html + JSON API
  picker.html   the Picker page (vanilla JS, Web Audio loops); shipped in the wheel
  jam.py        Jam pad: pad list from Picker analysis, sample-exact render of recorded bar events, save to _ready/jams/
  jam.html      the Jam page (`aislice jam`, /jam): bar-quantised clip launching with a Web Audio look-ahead scheduler
  verify.py     re-track the output and measure beat error vs. the ideal grid
  report.py     before/after tempo plot
  io.py         ffmpeg decode, soundfile read/write
```

## The product (the user's MUST)
AI songs in (BPM given with --bpm) → best samples out: `aislice_output/<YYYY-MM-DD_HHMM>_<bpm>bpm/<Stem>_<NN>_….wav`,
flat, one folder per run (keep folder depth minimal). Working files go to `.work_<session>/` and are deleted afterwards
unless `--keep-work`. **No interactive prompts and no
required human step in between.** The Picker and Jam are optional extras the user runs explicitly. Don't add menus,
questions or size options to the main flow.

## Invariants: don't break these
- **Standard loop layout:** drums and bass 4-bar, melody/other stems and the full mix 8-bar (`slicer.standard_unit`).
- **Every audio file written is peak-normalized on its own to -0.1 dB** (`io.normalize`), except near-silent slices (`io.SILENT_DB` below their track). Record every gain (report.json `normalize.gains_db`, `_gains.json` per slice folder) and undo it wherever real levels matter (Picker analysis, stems separation, reslice).
- **Stems are always split** (preset 3: drums / bass / melody) in the main command; separation runs on the un-normalized song.
- **Warp in one pass** with a time map. Never stretch slice-by-slice (it causes seams and drift).
- **Slice points are computed cumulatively** (`round(k * samples_per_unit)`), so rounding never accumulates. Full slices may differ by at most 1 sample.
- **Pitch is preserved**; only time changes.
- **Beat times mark note attacks.** The `onset_strength` calls use `center=False` because the STFT is already centered. With `center=True`, librosa delays the envelope ~46 ms and every slice cuts off its attack. `attack_latency()` calibrates any remaining offset, and `ATTACK_MARGIN` (2 ms) puts grid lines just before the transient.
- **The first downbeat lands on a bar line** of the output. Silent lead-in is trimmed. An audible pickup is padded with silence up to a bar boundary.
- **Analysis uses a mono 22.05 kHz copy; rendering always uses the original audio** at its native sample rate and channel count.
- **Batch runs keep going when one file fails.**
- **Stems are separated from the grid-locked song, never from the source**, so they share its grid. Each stem is resampled back to the song's rate, padded or trimmed to its exact length, and lag-corrected against the mix. Anything that moves the mix (e.g. `reslice`) must move the stems identically.
- **beat_this only picks bar 1, never the beat times.** Its beats are 20 ms-quantised and ~10 ms late on our tracks. The warp needs aislice's ~1 ms grid.
- **The Picker never changes results.** It only reads report.json and slices; it writes pick_cache.json in song folders and everything else under `<output>/_ready/`. Export removes only files the previous export wrote.
- **Picker groups compare slices beat by beat in order, and only against a group's leader** (no chaining). Don't switch to slice-averaged features: `test_order_matters_but_averages_dont` guards this.
- **The Picker server binds to 127.0.0.1 only**, checks a per-run token and the Host header, and serves only `.wav` files inside a song folder.
- **Jam playback and render must agree.** A clip launched at bar b plays from its bar (b mod clip_bars) in both jam.html and `jam.render`, so the saved file matches what was heard.
- **ML imports stay lazy.** Import `audio_separator`/`beat_this`/torch only inside `stems.run_backend` / `downbeats.detect_downbeats`, so `--help`, the tests and the Picker work without them. The main command needs `[stems]` (it says so and exits if missing); install with `.[all]`.

## Verifying changes
- Run `pytest` (install with `pip install -e ".[dev]"`). Stem tests use a fake separator. The real-model tests run only with `AISLICE_REAL_STEMS=1` / `AISLICE_REAL_BEATS=1` (needs `.[all]` and network for the model downloads). `tests/synth.py` builds drifting click+chord tracks with exact ground truth. `test_beats_sit_on_attacks_not_after` and `test_downbeat_uses_harmony_not_accents` are regression guards for real bugs: keep them passing, and don't loosen their tolerances to make a change pass.
- Check `report.json` → `verification`: target p95 < 10 ms (current tracks give 3–4 ms), and `output_bpm_per_20s` should equal the target BPM.
- `verify.py` uses the same tracker as the analysis, so it can't catch a constant timing offset. When you touch analysis or warp, also measure the raw note attacks in the output against the grid (steepest rise of a ~4 ms RMS envelope), independently of the tracker.
- Test audio goes in `temp/` (gitignored). Output goes to `aislice_output/` (gitignored). Never commit audio.

## Releasing
1. Bump `__version__` in `src/aislice/__init__.py` and add a `CHANGELOG.md` entry.
2. Merge to `main` and wait for CI to pass.
3. Create a GitHub Release with tag `vX.Y.Z` (it must match `__version__`). `.github/workflows/release.yml` builds the package and publishes it to PyPI through Trusted Publishing.

## Conventions
- Python ≥ 3.10, type hints, small pure functions per stage; keep heavy imports out of CLI arg parsing.
- Core dependencies stay light: numpy, scipy, librosa, soundfile, matplotlib, plus ffmpeg and rubberband binaries. PyTorch only comes in through the optional `[stems]` / `[beats]` extras. Never make ML a core dependency.
- Keep README's option list in sync with `slicer.unit_beats` / `cli.SLICE_CHOICES`.

## Known gaps / next work
- Without the [beats] extra, downbeat detection is heuristic. Accents can't separate beat 1 from beat 3, so harmonic change picks the "1" and the accents of each half-bar pair are only a tie-breaker. With beat_this (≥60% vote share) it was right on all 4 test tracks. Low margin → CLI warning + `downbeat_confident: false` in report.json. Phrase alignment for 4-/8-bar slices isn't detected. Ideas: repetition/self-similarity, section boundaries, per-section re-alignment when an AI track drops or adds a beat, and using the drum/bass stems (when present) to improve beat and downbeat tracking.
- Expressive outros (ritardando) are flattened onto the grid by design.
- License: MIT. Rubber Band (GPL) is only called as an external binary. Never vendor or link it.
