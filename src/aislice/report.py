from __future__ import annotations

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


def plot(path, title, tm, bpm, v):
    fig, ax = plt.subplots(2, 1, figsize=(11, 6), sharex=False)
    b = tm.beats
    raw = tm.raw_beats
    ax[0].plot(raw[1:], 60 / np.diff(raw), ".", ms=2, color="#bbbbbb", label="raw beat-to-beat")
    ok = tm.confidence[:-1] > 0.25
    ax[0].plot(b[1:][ok], tm.local_bpm[ok], "-", lw=1.6, color="#2b6cb0", label="cleaned tempo map")
    ax[0].axhline(bpm, color="#c53030", lw=1.2, ls="--", label=f"target {bpm:g} BPM")
    lb = tm.local_bpm[ok]
    ax[0].set_ylim(min(lb.min(), bpm) - 6, max(lb.max(), bpm) + 6)
    ax[0].set_ylabel("BPM"); ax[0].set_xlabel("source time (s)")
    ax[0].set_title(f"{title} — before")
    ax[0].legend(loc="upper left", fontsize=8)
    ob, err, strong = v["_beats"], v["_err"], v["_strong"]
    ax[1].plot(ob[strong], err, ".", ms=3, color="#2f855a")
    ax[1].axhline(0, color="k", lw=0.6)
    ax[1].set_ylim(-30, 30)
    ax[1].set_ylabel("beat error vs grid (ms)"); ax[1].set_xlabel("output time (s)")
    ax[1].set_title(f"after — p95 error {v['p95_err_ms']} ms, output tempo {np.median(v['output_bpm_per_20s']):.2f} BPM")
    fig.tight_layout()
    fig.savefig(path, dpi=110)
    plt.close(fig)
