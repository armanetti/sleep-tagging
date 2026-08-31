#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
export_atlas.py

Data for the Nocturnal Atlas page. One file: web/data_coh/atlas_plates.json
Needs numpy, pandas, fooof (or specparam). Nothing else. Runs in ~15 s.

    python export_atlas.py


The network, following fdc_paper_figures.ipynb
----------------------------------------------
Same recipe as the notebook, which is the one whose plates read well:

    C(f)    complex frequency-dependent covariance, corr_type="covariance"
    w_ij    = Re C_ij(f)                       signed, not the modulus
    keep    the strongest N pairs with w_ij > 0

Taking the real part and keeping only the positive side is what makes the plate
sparse: Re C > 0 means the two electrodes oscillate IN PHASE at that frequency,
and roughly half of the strong pairs are in antiphase and get dropped. The
notebook did this by comparing the signed value against a threshold computed on
the modulus, which quietly halved the 5% it looked like it was keeping.

Covariance rather than coherence, also as in the notebook. Coherence divides the
power out, so its top pairs spread evenly over every neighbouring electrode and
the plate turns into a uniform lattice. Covariance is power weighted, so the
strong pairs cluster around the electrodes where the rhythm actually lives.

EDGE_PCT is the knob: the threshold is the EDGE_PCT-th percentile of |Re C| over
the upper triangle, exactly as in the notebook. It is a quantile rather than an
absolute value because covariance scales with power, which goes as 1/f, so the
magnitudes at 1 Hz and at 27 Hz differ by orders of magnitude and no fixed cut
could serve both. Raise it to draw fewer links.

The number of links that survive is NOT the same on every plate, and that is the
point: it depends on how many of the strong pairs are in phase. A plate with more
links is more coupled at that rhythm. Fixing the count would throw that away.


The frequencies
---------------
Each stage is drawn at its own FOOOF peaks, fitted on its own mean spectrum.
aperiodic_mode='knee' over 0.75 to 30 Hz: a single-slope fit cannot follow the
bend at low frequency and its misfit invents a peak at the bottom edge, and the
upper limit stays clear of the 45 Hz low-pass and the 50 Hz mains, whose filter
shoulder is otherwise read as a row of gamma peaks. Peaks wider than 4 Hz are
rejected as fit artefacts.

Every stage is estimated from the same number of chunks: coherence and
covariance bias both go as 1/sqrt(n_chunks), so unequal sample sizes would make
the stages with less data look more coupled. 200 chunks also puts the rank of C
above the 83 channels, which 50 did not.

Dots are the topography of the rhythm: power inside the peak over power in two
flanking sidebands, in dB, per electrode. Slow and fast spindles are told apart
by that topography, not by a frequency line, because the conventional 13 Hz
boundary is a population average and this subject sits below it.
"""

from __future__ import annotations
import base64, json, time
from pathlib import Path

import numpy as np
import pandas as pd

FS, EPOCH_SEC = 128, 30
SAMP_EPOCH = FS * EPOCH_SEC
CHUNK_S, N_CHUNKS, CHUNKS_PER_EPOCH = 4.0, 200, 7
SCALE = 1e6
STAGES = ("W", "N1", "N2", "N3", "R")

EDGE_PCT = 95.0                    # the threshold, as in the notebook. The knob.
SPEC_LO, SPEC_HI, SPEC_N = 0.5, 45.0, 300
FIT_LO, FIT_HI = 0.75, 30.0
MIN_PEAK_HZ, MAX_PEAK_BW, MAX_PEAKS = 1.0, 4.0, 3

NREM = ("N2", "N3")
SPINDLE_LO, SPINDLE_HI = 11.0, 16.0

ROOT = Path(__file__).resolve().parent.parent
DATA = ROOT / "Data_for_analysis"
OUT = ROOT / "web" / "data_coh"


def b64(x, scale=255):
    return base64.b64encode(
        np.clip(np.asarray(x) * scale, 0, 255).round().astype(np.uint8).tobytes()).decode()


def b64i(x):
    return base64.b64encode(np.asarray(x).astype(np.uint8).tobytes()).decode()


def band_of(stage, f, anterior=None):
    if f < 1.5:  return {"rhythm": "slow oscillation", "glyph": "δ"}
    if f < 4.0:  return {"rhythm": "delta waves", "glyph": "δ"}
    if f < 8.0:  return {"rhythm": "theta waves", "glyph": "θ"}
    if stage in NREM and SPINDLE_LO <= f < SPINDLE_HI:
        return {"rhythm": "slow spindles" if anterior else "fast spindles", "glyph": "σ"}
    if f < 12.0: return {"rhythm": "alpha waves", "glyph": "α"}
    if f < 16.0: return {"rhythm": "sigma waves", "glyph": "σ"}
    if f < 30.0: return {"rhythm": "beta waves", "glyph": "β"}
    return {"rhythm": "gamma waves", "glyph": "γ"}


def region_of(topo, xs):
    """One word for where the rhythm sits: weighted centroid of the topography
    along the anterior-posterior axis (ch_pos x is forward)."""
    w = np.clip(np.asarray(topo, float), 0, 1) ** 3
    if w.sum() <= 0:
        return "diffuse", 0.0
    r = (float((w * xs).sum() / w.sum()) - float(np.median(xs))) / \
        max(float(xs.max() - xs.min()), 1e-9)
    name = ("frontal" if r > 0.16 else "fronto-central" if r > 0.05 else
            "central" if r > -0.05 else "centro-parietal" if r > -0.16 else
            "parieto-occipital")
    return name, round(r, 3)


def load_fooof():
    try:
        from fooof import FOOOF
        return FOOOF, "fooof"
    except ImportError:
        from specparam import SpectralModel
        return SpectralModel, "specparam"


def fourier(chunks, freqs):
    """Same convention as fdc.correlation_freq, vectorised over frequency."""
    L = chunks.shape[2]
    E = np.exp(-2j * np.pi * np.outer(np.arange(L) / FS, freqs))
    return (chunks @ E) / L


def covariance(chunks, freqs):
    S = fourier(chunks, freqs)
    return np.einsum("cnk,cmk->nmk", S, S.conj()) / S.shape[0]


def distance_to_criticality(C, n_chunks):
    """g(omega) from Calvo et al. 2024."""
    N, _, K = C.shape
    off = ~np.eye(N, dtype=bool)
    tr = np.real(np.einsum("nnk->k", C)) / N
    a = np.abs(C)
    M2 = np.sqrt((a[off, :] ** 2).mean(axis=0))
    cij = a.reshape(N * N, K).mean(axis=0)
    T = max(n_chunks, 2)
    d = np.sqrt(np.abs(M2 ** 2 - (tr ** 2 - cij ** 2) / (T - 1))) / tr
    return np.sqrt(np.clip(1.0 - np.sqrt(1.0 / (1.0 + N * d ** 2)), 0, None))


def stage_chunks(X, labels, stage, chunk):
    """Chunks spread over as many of the stage's interior epochs as it has, so
    the estimate is not taken from a single stretch of the night."""
    idx = np.where(labels == stage)[0]
    inner = np.array([i for i in idx if 0 < i < len(labels) - 1
                      and labels[i - 1] == stage and labels[i + 1] == stage])
    if len(inner) >= max(6, len(idx) // 3):
        idx = inner
    if len(idx) == 0:
        return None
    per = max(1, min(CHUNKS_PER_EPOCH, int(np.ceil(N_CHUNKS / len(idx)))))
    need = int(np.ceil(N_CHUNKS / per))
    if len(idx) > need:
        idx = idx[np.unique(np.linspace(0, len(idx) - 1, need).round().astype(int))]
    out = []
    for e in idx:
        seg = np.array(X[:, e * SAMP_EPOCH:(e + 1) * SAMP_EPOCH], float) * SCALE
        for c in range(per):
            out.append(seg[:, c * chunk:(c + 1) * chunk])
            if len(out) >= N_CHUNKS:
                break
        if len(out) >= N_CHUNKS:
            break
    return np.ascontiguousarray(np.stack(out)), len(idx)


def main():
    t0 = time.time()
    FooofCls, backend = load_fooof()
    chunk = int(CHUNK_S * FS)

    X = np.load(next(DATA.glob("EPCTL06*.npy")), mmap_mode="r")
    N, T = X.shape
    pos = pd.read_csv(DATA / "ch_pos.csv", index_col=0)
    if len(pos) != N:
        raise SystemExit(f"ch_pos has {len(pos)} rows, the array has {N} channels")
    names = [str(n) for n in pos.index]
    xs = pos["X"].to_numpy(float)

    # the hypnogram has NO header: the first line is already an epoch
    labels = pd.read_csv(DATA / "EPCTL06.csv", header=None,
                         names=["L", "onset", "dur"])["L"].astype(str).values
    n_epochs = min(len(labels), T // SAMP_EPOCH)
    labels = labels[:n_epochs]

    spec_f = np.linspace(SPEC_LO, SPEC_HI, SPEC_N)
    iu, ju = np.triu_indices(N, 1)
    n_pairs = len(iu)

    print(f"{N} channels, {n_epochs*EPOCH_SEC/3600:.2f} h, {N_CHUNKS} chunks of "
          f"{CHUNK_S:.0f} s per stage ({1/CHUNK_S:.2f} Hz resolution), "
          f"threshold at the {EDGE_PCT:.0f}th percentile of |Re C|, "
          f"backend {backend}", flush=True)

    out_stages, report = {}, []

    for st in STAGES:
        got = stage_chunks(X, labels, st, chunk)
        if got is None:
            print(f"  {st}: no epochs, skipped"); continue
        ch, n_ep = got

        P = (np.abs(fourier(ch, spec_f)) ** 2).mean(axis=0)          # (N, K)
        fm = FooofCls(peak_width_limits=[0.5, 6.0], max_n_peaks=6,
                      min_peak_height=0.05, peak_threshold=2.0,
                      aperiodic_mode="knee")
        fm.fit(spec_f, P.mean(axis=0), freq_range=[FIT_LO, FIT_HI])
        pk = np.atleast_2d(np.asarray(fm.peak_params_, float))
        if pk.size == 0:
            pk = np.zeros((0, 3))
        n_raw = len(pk)
        if len(pk):
            pk = pk[(pk[:, 0] >= MIN_PEAK_HZ) & (pk[:, 2] <= MAX_PEAK_BW)]
        if len(pk):
            pk = pk[np.argsort(pk[:, 1])[::-1][:MAX_PEAKS]]
            pk = pk[np.argsort(pk[:, 0])]
        print(f"  {st}: {n_ep} epochs, {ch.shape[0]} chunks, r2 {fm.r_squared_:.3f}, "
              f"{len(pk)} peaks of {n_raw}", flush=True)
        if not len(pk):
            continue

        C = covariance(ch, pk[:, 0])                                  # at the peaks
        g = distance_to_criticality(C, ch.shape[0])

        entries = []
        for k, (cf, pw, bw) in enumerate(pk):
            half = max(bw, 0.5) / 2
            core = (spec_f >= cf - half) & (spec_f <= cf + half)
            flank = (((spec_f >= cf - 3*half) & (spec_f <= cf - 1.5*half)) |
                     ((spec_f >= cf + 1.5*half) & (spec_f <= cf + 3*half)))
            r = 10 * np.log10(P[:, core].mean(axis=1) / P[:, flank].mean(axis=1))
            a, b = np.percentile(r, 5), np.percentile(r, 95)
            topo = np.clip((r - a) / max(b - a, 1e-9), 0, 1)
            region, anter = region_of(topo, xs)

            # the notebook's rule: threshold on the modulus, test on the signed
            # value, so only the in-phase half of the strong pairs survives
            w = np.real(C[:, :, k])[iu, ju]
            thr = float(np.percentile(np.abs(w), EDGE_PCT))
            order = np.where(w > thr)[0]
            order = order[np.argsort(w[order])[::-1]]
            wk = w[order]
            norm = (wk - thr) / max(wk.max() - thr, 1e-30)

            entries.append({
                "freq": round(float(cf), 2),
                "bandwidth": round(float(bw), 2),
                "power": round(float(pw), 3),
                "g": round(float(g[k]), 4),
                "band": band_of(st, float(cf), anterior=anter > 0.0),
                "region": region,
                "n_edges": int(len(order)),
                "top_pct": round(100 * len(order) / n_pairs, 2),
                "in_phase_pct": round(100 * float((w > 0).mean()), 1),
                "edges": {"i": b64i(iu[order]), "j": b64i(ju[order]), "w": b64(norm)},
                "nodes": b64(topo),
            })
            report.append((st, entries[-1], topo))

        out_stages[st] = {
            "minutes": round(float((labels == st).sum() * EPOCH_SEC / 60), 1),
            "n_epochs": int(n_ep), "n_chunks": int(ch.shape[0]),
            "peaks": entries,
        }

    out = {
        "meta": {
            "subject": "EPCTL06", "fs": FS, "epoch_sec": EPOCH_SEC,
            "n_channels": int(N), "hours": round(n_epochs * EPOCH_SEC / 3600, 2),
            "chunk_s": CHUNK_S, "n_chunks_per_stage": N_CHUNKS,
            "freq_resolution_hz": round(1 / CHUNK_S, 3),
            "edge_percentile": EDGE_PCT,
            "edges": "pairs whose Re C(f) exceeds the "
                     f"{EDGE_PCT:.0f}th percentile of |Re C|, so in phase and strong",
            "nodes": "peak power over flanking sidebands, in dB, per electrode",
            "fooof": {"backend": backend, "aperiodic_mode": "knee",
                      "fit_range": [FIT_LO, FIT_HI]},
            "generated": time.strftime("%Y-%m-%dT%H:%M:%S"),
        },
        "electrodes": [{"name": str(n), "x": round(float(r.X), 3),
                        "y": round(float(r.Y), 3), "z": round(float(r.Z), 3)}
                       for n, r in pos.iterrows()],
        "stages": out_stages,
    }

    OUT.mkdir(parents=True, exist_ok=True)
    p = OUT / "atlas_plates.json"
    p.write_text(json.dumps(out, separators=(",", ":")))

    print(f"\n  {'st':>2}  {'freq':>8}  {'rhythm':<16} {'region':<18} "
          f"{'pw':>5} {'g':>6} {'links':>6} {'top%':>6} {'in-phase%':>10}   top electrodes")
    for st, e, topo in report:
        top = ", ".join(names[i] for i in np.argsort(topo)[::-1][:5])
        print(f"  {st:>2}  {e['freq']:6.2f}Hz  {e['band']['rhythm']:<16} "
              f"{e['region']:<18} {e['power']:5.2f} {e['g']:6.3f} {e['n_edges']:6d} "
              f"{e['top_pct']:6.2f} {e['in_phase_pct']:10.1f}   {top}")
    print(f"\nwrote {p} ({p.stat().st_size/1e3:.0f} kB) in {time.time()-t0:.0f}s")


if __name__ == "__main__":
    main()
