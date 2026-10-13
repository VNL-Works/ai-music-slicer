# Changelog

## 0.4.0 (2026-10-14)

- **AI songs in, best samples out.** `aislice songs/ --bpm 206` now goes all the way: tempo lock, stems (drums / bass / melody, always), loops, and the best loops of all songs of the run, ranked, in one flat folder per run: `aislice_output/<date>_<time>_<bpm>bpm/Drums_01_….wav` (`--best N`, default 8). No prompts: `--bpm` is required. The working files are deleted at the end (`--keep-work` keeps them for the Picker / Jam / reslice).
- **One loop layout everywhere:** drums and bass in 4-bar loops, melody (and the full mix) in 8-bar loops. The slice-size question and menus are gone (`--slice` still exists, hidden, for special cases).
- **Every audio file is normalized** on its own to a −0.1 dB peak: the song, stems, slices, exports and jams. Near-silent slices are not boosted. Gains are stored (`report.json` → `normalize`, `_gains.json` next to slices) so the Picker still sees the real levels.
- Picker: opens on **★ Best of all songs**, rows drawn to time scale, the full mix row hidden by default (`M`). Jam: fixed loop sizes, fixed a bug where stopping a recording in the last moment of a bar never finished it.
- `aislice reslice` keeps each stem in the folder it was in, and re-normalizes.
- **Picker (`aislice pick`).** A local browser page to audition every slice in a grid (mix + stems), keep or skip with the keyboard and export the keepers DAW-ready to `_ready/`. `--pick` opens it after processing.
  - Empty and near-empty slices are hidden; quiet (likely bleed) slices are labelled.
  - Repeats are grouped per stem by comparing slices beat by beat, so a verse and a chorus on the same chords stay apart. Groups don't chain; the most typical slice represents each group.
  - Export names like `Song_Drums_206bpm_Am_8bar03.wav`, an ACID tempo chunk in every file, key estimate (only in the name when confident), `--group-by song|stem|flat`, `picks.json` saved as you go.
  - **Best** view (default): the top N loops per stem, ranked by not-quiet, most repeated, fullest. `B` toggles Best/All.
  - `aislice pick --auto` keeps one slice per group and exports without a browser; `--best N` keeps only the N best per stem.
  - The server only listens on 127.0.0.1, needs a per-run token and only serves files inside the output folder.
- **Jam pad (`aislice jam`).** A launchpad in the browser: one column per stem, pads from all songs at the same BPM, bar-quantised launch, loops stay phrase-aligned. Record a jam and get a sample-exact mix + one WAV per column in `_ready/jams/`.

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
