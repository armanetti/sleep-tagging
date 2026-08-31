# sleep-tagging

Characterising sleep stages from the frequency-dependent covariance of EEG, and
drawing the result as an interactive atlas.

**[Open the atlas](https://armanetti.github.io/sleep-tagging/)**

One night of polysomnography, 83 scalp electrodes at 128 Hz. For every sleep
stage the page shows the frequencies at which that stage actually oscillates,
where each rhythm sits on the scalp, and how those areas couple.

- **Dots** are the topography of the rhythm: power inside the spectral peak over
  power in two flanking sidebands, in dB, per electrode.
- **Lines** are the electrode pairs whose complex covariance at that rhythm is
  both strong and in phase, above the 95th percentile of |Re C|.

Waking alpha comes out parieto-occipital, slow spindles frontal near 11.2 Hz,
fast spindles centro-parietal near 12.8 Hz, REM theta fronto-central. Slow and
fast spindles are separated by their topography rather than by a fixed frequency
boundary, because the conventional 13 Hz line is a population average.

## Method

Frequency-dependent covariance, following Calvo, Martorell, Morales, Di Santo and
Muñoz, *Frequency-dependent covariance reveals critical spatio-temporal patterns
of synchronized activity in the human brain*, [arXiv:2403.15092](https://arxiv.org/abs/2403.15092).

    s_n(f) = (1/L) Σ_t x_n(t) e^(−2πi f t)        per 4 s chunk
    C(f)   = ⟨ s(f) s(f)^H ⟩ over chunks

Spectral peaks are found with [FOOOF](https://fooof-tools.github.io/) in
`aperiodic_mode='knee'` over 0.75 to 30 Hz. Every stage is estimated from the
same number of chunks, because covariance bias goes as 1/√n and unequal sample
sizes would make the stages with less data look more coupled.

## Layout

    src/fdc.py                  frequency-dependent covariance, criticality estimator
    src/sleep_stage_function.py epoch extraction from the hypnogram
    src/export_atlas.py         builds web/data_coh/atlas_plates.json
    web/index.html              the page, no build step
    web/vendor/                 three.js, vendored so the page has no CDN dependency
    Data_for_analysis/          recordings, not in the repository

## Reproducing the data

Put the preprocessed `.npy`, the hypnogram and `ch_pos.csv` in
`Data_for_analysis/`, then:

    cd src
    python export_atlas.py

Takes about 15 seconds and writes a 16 kB JSON. `EDGE_PCT` near the top of the
file sets how many links each plate draws.

## Running the page locally

    cd web
    python -m http.server 8000

The page fetches its data, so opening `index.html` directly from the filesystem
will not work. `python build_standalone.py` writes `dist/index.html` with the
library and the data inlined, which does work from a plain file.

Built at BrainHack 2025.
