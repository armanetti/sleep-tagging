#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
export_timeline_sliding.py

Sliding-window coherence for the time scrubber of the Nocturnal Atlas.
numpy + pandas only, so it runs anywhere. Output: web/data_coh/timeline.json

For every 120 s window, stepped every 30 s across the night:

    s_n(f) = (1/L) sum_t x_n(t) e^{-2i pi f t}      per 4 s chunk
    C(f)   = < s(f) s(f)^H >_chunks
    Coh_ij = |C_ij| / sqrt(C_ii C_jj)

Two choices worth knowing about.

1. The backbone is the set of electrode pairs whose coherence VARIES most across
   the night, not the pairs whose coherence is highest. Volume conduction keeps
   neighbouring electrodes above 0.95 all night long: ranking by strength just
   redraws the montage. Ranking by standard deviation finds the links that sleep
   actually switches.

2. Stored bytes are coherence rescaled per frequency over the 1st to 99th
   percentile of the backbone across the whole night. Raw coherence sits in a
   narrow band near the top of [0, 1], so absolute bytes would waste 95% of the
   range. The real bounds are kept in meta.range so the page can print them.

The phase view is the mean of every window scored as that stage: identical
estimator, identical chunk count, identical bias, so the plate does not jump
when you move between the time view and the phase view.
"""

from __future__ import annotations
import base64, json, time
from pathlib import Path
import numpy as np
import pandas as pd

FS, EPOCH_SEC = 128, 30
SAMP_EPOCH = FS * EPOCH_SEC
FREQS = np.array([1.0, 6.5, 10.0, 12.5])          # slow, theta, alpha, sigma
BAND_NAMES = ["slow", "theta", "alpha", "sigma"]
WIN_S, STEP_S, CHUNK_S = 120.0, 30.0, 4.0
N_BACKBONE = 220
SCALE = 1e6                                        # V -> uV
STAGES = ("W", "N1", "N2", "N3", "R")

ROOT = Path(__file__).resolve().parent.parent
DATA = ROOT / "Data_for_analysis"
OUT = ROOT / "web" / "data_coh"


def b64(x):
    return base64.b64encode(
        np.clip(np.asarray(x) * 255, 0, 255).round().astype(np.uint8).tobytes()).decode()


def distance_to_criticality(C, n_chunks):
    """g(omega) from Calvo et al. 2024, vectorised over frequency."""
    N, _, K = C.shape
    off = ~np.eye(N, dtype=bool)
    tr = np.real(np.einsum("nnk->k", C)) / N
    a = np.abs(C)
    M2 = np.sqrt((a[off, :] ** 2).mean(axis=0))
    cij = a.reshape(N * N, K).mean(axis=0)
    T = max(n_chunks, 2)
    d = np.sqrt(np.abs(M2 ** 2 - (tr ** 2 - cij ** 2) / (T - 1))) / tr
    return np.sqrt(np.clip(1.0 - np.sqrt(1.0 / (1.0 + N * d ** 2)), 0, None))


def main():
    t0 = time.time()
    npy = next(DATA.glob("EPCTL06*.npy"))
    X = np.load(npy, mmap_mode="r")
    N, T = X.shape

    labels = pd.read_csv(DATA / "EPCTL06.csv", header=None,
                         names=["L", "onset", "dur"])["L"].astype(str).values
    n_epochs = min(len(labels), T // SAMP_EPOCH)
    labels = labels[:n_epochs]

    win, step, chunk = int(WIN_S * FS), int(STEP_S * FS), int(CHUNK_S * FS)
    n_chunks = win // chunk
    starts = list(range(0, T - win + 1, step))
    nW, K = len(starts), len(FREQS)

    iu, ju = np.triu_indices(N, 1)
    E = np.exp(-2j * np.pi * np.outer(np.arange(chunk) / FS, FREQS))
    off = ~np.eye(N, dtype=bool)

    print(f"{N} channels, {T/FS/3600:.2f} h, {nW} windows of {WIN_S:.0f}s in "
          f"{n_chunks} chunks of {CHUNK_S:.0f}s (df = {1/CHUNK_S:.2f} Hz)", flush=True)

    pair = np.empty((nW, K, len(iu)), dtype=np.float32)
    node = np.empty((nW, K, N), dtype=np.float32)
    gcrit = np.empty((nW, K), dtype=np.float32)
    stage_at = []

    for w, s in enumerate(starts):
        seg = np.array(X[:, s:s + win], dtype=np.float64) * SCALE
        ch = np.ascontiguousarray(
            seg[:, :n_chunks * chunk].reshape(N, n_chunks, chunk).transpose(1, 0, 2))
        S = (ch @ E) / chunk
        C = np.einsum("cnk,cmk->nmk", S, S.conj()) / n_chunks
        d = np.real(np.einsum("nnk->nk", C)).copy()
        d[d <= 0] = 1.0
        coh = np.abs(C) / np.sqrt(d[:, None, :] * d[None, :, :])

        pair[w] = coh[iu, ju, :].T
        node[w] = coh[off, :].reshape(N, N - 1, K).mean(axis=1).T
        gcrit[w] = distance_to_criticality(C, n_chunks)
        stage_at.append(str(labels[min(int((s + win / 2) // SAMP_EPOCH), n_epochs - 1)]))
        if w % 200 == 0:
            print(f"  window {w}/{nW}  {time.time()-t0:.0f}s", flush=True)

    # backbone: the pairs sleep actually switches, not the ones anatomy fixes
    spread = pair.std(axis=0).max(axis=0)                       # (n_pairs,)
    keep = np.argsort(spread)[::-1][:N_BACKBONE]
    keep = keep[np.argsort(spread[keep])[::-1]]
    bi, bj = iu[keep], ju[keep]
    B = pair[:, :, keep]                                        # (nW, K, nB)

    lo = np.percentile(B, 1, axis=(0, 2))
    hi = np.percentile(B, 99, axis=(0, 2))
    Bn = np.clip((B - lo[None, :, None]) / (hi - lo)[None, :, None], 0, 1)
    nlo = node.min(axis=(0, 2)); nhi = node.max(axis=(0, 2))
    Nn = np.clip((node - nlo[None, :, None]) / (nhi - nlo)[None, :, None], 0, 1)

    stage_arr = np.array(stage_at)
    phases = {}
    for st in STAGES:
        m = stage_arr == st
        if not m.any():
            continue
        phases[st] = {
            "n_windows": int(m.sum()),
            "minutes": round(float(m.sum() * STEP_S / 60), 1),
            "edges": [b64(Bn[m, k, :].mean(axis=0)) for k in range(K)],
            "nodes": [b64(Nn[m, k, :].mean(axis=0)) for k in range(K)],
            "g": [round(float(gcrit[m, k].mean()), 4) for k in range(K)],
            "coherence": [round(float(B[m, k, :].mean()), 4) for k in range(K)],
        }

    out = {
        "meta": {
            "subject": "EPCTL06", "fs": FS, "epoch_sec": EPOCH_SEC,
            "n_epochs": int(n_epochs), "n_channels": int(N),
            "estimator": "magnitude coherence",
            "win_s": WIN_S, "step_s": STEP_S, "chunk_s": CHUNK_S,
            "n_chunks_per_window": int(n_chunks),
            "freq_resolution_hz": round(1 / CHUNK_S, 3),
            "range": {"edges_lo": [round(float(v), 4) for v in lo],
                      "edges_hi": [round(float(v), 4) for v in hi],
                      "nodes_lo": [round(float(v), 4) for v in nlo],
                      "nodes_hi": [round(float(v), 4) for v in nhi]},
            "generated": time.strftime("%Y-%m-%dT%H:%M:%S"),
        },
        "stages": [str(x) for x in labels],
        "phases": phases,
        "sliding": {
            "freqs": [float(f) for f in FREQS],
            "bands": BAND_NAMES,
            "t0_s": float(win / 2 / FS), "dt_s": float(STEP_S), "n_windows": nW,
            "stage_at_center": stage_at,
            "g": [[round(float(v), 4) for v in row] for row in gcrit],
            "backbone": {"i": bi.tolist(), "j": bj.tolist()},
            "edges": [[b64(Bn[w, k, :]) for k in range(K)] for w in range(nW)],
            "nodes": [[b64(Nn[w, k, :]) for k in range(K)] for w in range(nW)],
        },
    }
    OUT.mkdir(parents=True, exist_ok=True)
    p = OUT / "timeline.json"
    p.write_text(json.dumps(out, separators=(",", ":")))
    print(f"coherence range per band: "
          + ", ".join(f"{BAND_NAMES[k]} {lo[k]:.3f}-{hi[k]:.3f}" for k in range(K)))
    print(f"wrote {p} ({p.stat().st_size/1e6:.2f} MB) in {time.time()-t0:.0f}s")


if __name__ == "__main__":
    main()
