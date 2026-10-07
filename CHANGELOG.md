# Changelog

## 0.1.0 (2026-10-07)

First public release.

- Lock a drifting (AI-generated) song to a fixed BPM grid in one Rubber Band R3 pass, with pitch preserved.
- Drift-following beat tracker; beat markers are calibrated to note attacks, so slices keep their transients.
- Tempo-map cleaning: missed and extra beats, weak fade-in/out beats, and smoothing of tracking jitter.
- Downbeat detection from harmonic change, with a warning when it isn't confident.
- Slicing by `1/16`, `1/8`, `beat`, `bar`, `2bar`, `4bar`, `8bar` or any `Nbar`/`Nbeat`. Slices are sample-accurate and never drift.
- `aislice reslice` moves bar 1 by N beats, or re-cuts in other units, without re-processing.
- Per-song `report.json` (tempo map, corrections, verification) and `tempo.png`.
- Test suite with synthetic drifting tracks; CI on Python 3.10–3.13.
