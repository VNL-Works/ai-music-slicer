# Changelog

## 0.3.0 (2026-10-12)

- Stems now default to **BS-Roformer-SW** (6-stem BS-RoFormer), which won a second listening test against `htdemucs_ft`. The 2/3/4 presets merge its six stems. `--stems-fast` still uses Demucs `htdemucs` (`htdemucs_6s` for 6 stems). The old default is still available with `--stem-model htdemucs_ft.yaml`.

## 0.2.0 (2026-10-12)

- **Stem separation (optional).** `--stems 2|3|4|6` splits the grid-locked song into stems (vocals/instrumental; drums/bass/melody; vocals/drums/bass/other; plus guitar/piano) and slices each stem with the same cut points as the mix. Install with `pip install "aislice[stems]"`.
- `aislice stems <folder>` adds stems to an existing result without re-processing the tempo.
- `--stem-model` accepts any audio-separator model (Demucs, RoFormer, MDX, VR). Models are cached in `~/.cache/aislice/models`.
- Stems are checked against the mix: any model delay is corrected automatically, and the leftover difference is reported in `report.json`.
- `aislice reslice` now moves stems together with the mix.
- **beat_this downbeats (optional, `pip install "aislice[beats]"`).** The state-of-the-art beat_this model decides where bar 1 is. aislice's own beat grid still does the precise timing. Falls back to the harmony rule when beat_this is inconclusive or not installed. `--downbeats auto|beat_this|harmony`. The decision and votes are in `report.json`.
- `--stems 3`: drums, bass and melody (everything else).
- Stems default to `htdemucs_ft` (picked in a listening test against `htdemucs` and BS-Roformer-SW). `--stems-fast` uses `htdemucs`. `--stems-exact` makes the stems sum exactly to the mix.
- `aislice stems --slice none` no longer crashes.
- CI: macOS job and a real-model job (Demucs + beat_this).

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
