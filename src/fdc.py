# -*- coding: utf-8 -*-
"""
fdc.py

Core tools for Frequency-Dependent Covariance (FDC/FDPCA) analyses:

- 3D visualization of eigenmodes and node activity
- Reconstruction of spatiotemporal patterns from FDC eigenvectors
- Frequency-dependent covariance estimation
- Distance-to-criticality estimator (Beyond mean field-style)
- Simple surrogate (reshuffling) and PCA spectrum

Intended usage (after installing / cloning the repo)::

    from fdc.fdc import (
        plot_leading_pattern,
        animate_nodes,
        spatio_temporal,
        reshuffling,
        PCA,
        correlation_freq,
        POWER_SPECTRUM,
        Distance_criticality,
        fourier_freq,
    )

All arrays use the convention:
- N: number of nodes / channels
- T: number of time points
"""

from __future__ import annotations

from pathlib import Path
from typing import Optional, Sequence, Tuple

import numpy as np
import matplotlib.pyplot as plt
from matplotlib import animation
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401  # needed for 3D projection


# ============================================================================
# 3D visualization of eigenmodes and node activity
# ============================================================================

def plot_leading_pattern(
    U_F: np.ndarray,
    eigs: Sequence[float],
    X3D: Sequence[float],
    Y3D: Sequence[float],
    Z3D: Sequence[float],
    sign: float = +1.0,
    leading: int = 0,
    satura: float = 1.0,
    size: float = 360.0,
) -> plt.Figure:
    """
    Plot 3D scatter projections of one eigenvector pattern in three views.

    Parameters
    ----------
    U_F
        Matrix of eigenvectors, shape (N_nodes, N_modes).
        Columns should be eigenvectors, e.g. from `np.linalg.eigh`.
    eigs
        Corresponding eigenvalues, shape (N_modes,).
    X3D, Y3D, Z3D
        3D coordinates of each node, length N_nodes.
    sign
        Optional sign flip (+1 or -1) to choose the orientation of the vector.
    leading
        Index of the eigenvector to visualise (default: 0).
    satura
        Saturation factor for color normalization (larger = less saturated).
    size
        Marker size for scatter plots.

    Returns
    -------
    fig
        Matplotlib figure with three 3D scatter views.

    Notes
    -----
    Colors:
        - Positive components → red
        - Negative components → blue
        - Magnitude → intensity
    """
    U_F = np.asarray(U_F)
    eigs = np.asarray(eigs)
    X3D = np.asarray(X3D)
    Y3D = np.asarray(Y3D)
    Z3D = np.asarray(Z3D)

    n_nodes = U_F.shape[0]
    assert X3D.shape[0] == Y3D.shape[0] == Z3D.shape[0] == n_nodes, "Coordinate lengths must match U_F.shape[0]."

    # Select the chosen eigenvector and print its eigenvalue (importance)
    pattern = sign * U_F[:, leading]
    importance = eigs[leading]
    print(f"Eigenvalue of selected mode: {importance}")

    # Normalize values for color mapping
    norm = np.max(np.abs(pattern)) / float(satura) if np.max(np.abs(pattern)) > 0 else 1.0

    # Build RGB colors
    colors = np.zeros((n_nodes, 3))
    for i in range(n_nodes):
        val = pattern[i] / norm
        if val > 0:
            colors[i, 0] = min(val, 1.0)  # red channel
        else:
            colors[i, 2] = min(-val, 1.0)  # blue channel

    # Figure with three views
    fig = plt.figure(figsize=(9, 6))

    ax1 = fig.add_subplot(1, 3, 1, projection="3d")
    ax1.scatter(X3D, Y3D, Z3D, c=colors, s=size)
    ax1.view_init(elev=0, azim=90)

    ax2 = fig.add_subplot(1, 3, 2, projection="3d")
    ax2.scatter(Y3D, -X3D, Z3D, c=colors, s=size)
    ax2.view_init(elev=90, azim=-90)

    ax3 = fig.add_subplot(1, 3, 3, projection="3d")
    ax3.scatter(X3D, Y3D, Z3D, c=colors, s=size)
    ax3.view_init(elev=0, azim=-90)

    for ax in (ax1, ax2, ax3):
        _style_ax(ax)

    fig.subplots_adjust(wspace=-0.2)
    return fig


def paint_colors(
    x: Sequence[float],
    maximo: float,
    minimo: float,
    satura: float = 1.0,
) -> np.ndarray:
    """
    Map a 1D array to RGB colors (red for positive, blue for negative).

    Parameters
    ----------
    x
        1D array of length N_nodes with values to color-code.
    maximo, minimo
        Global max/min used to normalize the color scale.
    satura
        Saturation factor. Values are divided by (maximo / satura) or
        (minimo / satura) when mapping to color intensities.

    Returns
    -------
    colors : (N_nodes, 3) array
        RGB colors in [0, 1].
    """
    x = np.asarray(x)
    colors = np.zeros((len(x), 3))

    norm_max = float(maximo) / float(satura) if maximo != 0 else 1.0
    norm_min = float(minimo) / float(satura) if minimo != 0 else -1.0

    for i, val in enumerate(x):
        if val > 0:
            ratio = val / norm_max
            colors[i, :] = [min(ratio, 1.0), 0.0, 0.0]
        else:
            ratio = abs(val / norm_min)
            colors[i, :] = [0.0, 0.0, min(ratio, 1.0)]

    return colors


def _style_ax(ax: Axes3D) -> None:
    """
    Apply minimalist styling to a 3D axis: no ticks, no panes, white background.
    """
    ax.grid(False)
    ax.set_xticklabels([])
    ax.set_yticklabels([])
    ax.set_zticklabels([])

    ax.xaxis.line.set_color((1, 1, 1, 0))
    ax.yaxis.line.set_color((1, 1, 1, 0))
    ax.zaxis.line.set_color((1, 1, 1, 0))

    ax.xaxis.set_pane_color((1, 1, 1, 0.0))
    ax.yaxis.set_pane_color((1, 1, 1, 0.0))
    ax.zaxis.set_pane_color((1, 1, 1, 0.0))

    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_zticks([])

    ax.set_facecolor("white")
    ax.patch.set_alpha(0.0)


def animate_nodes(
    x: np.ndarray,
    X3D: Sequence[float],
    Y3D: Sequence[float],
    Z3D: Sequence[float],
    satura: float = 1.5,
    size: float = 200.0,
    interval_ms: int = 50,
    fps: int = 20,
    writer: str = "ffmpeg",
    output_path: Optional[str | Path] = None,
    dpi: int = 150,
) -> Optional[Path]:
    """
    Create a three-panel 3D animation of node colors over time and save to MP4.

    Parameters
    ----------
    x
        Array of shape (N_nodes, T). Node activity over time.
    X3D, Y3D, Z3D
        Node coordinates, length N_nodes.
    satura
        Saturation factor passed to :func:`paint_colors`.
    size
        Marker size for scatter plots.
    interval_ms
        Delay between frames in milliseconds for interactive preview.
    fps
        Frames per second for saved video.
    writer
        Matplotlib writer to use (e.g. "ffmpeg").
    output_path
        Path to save the video (str or Path). If None, a file dialog
        is opened (if possible) or the movie is saved as "animation.mp4"
        in the current directory.
    dpi
        Resolution of the saved video.

    Returns
    -------
    output_path
        Path where the video was saved, or None if saving was cancelled.
    """
    x = np.asarray(x)
    X3D = np.asarray(X3D)
    Y3D = np.asarray(Y3D)
    Z3D = np.asarray(Z3D)

    n_nodes, T = x.shape
    assert (
        X3D.shape[0] == Y3D.shape[0] == Z3D.shape[0] == n_nodes
    ), "Coordinate lengths must match N_nodes."

    # Ask user for a path if not provided
    if output_path is None:
        try:
            import tkinter as tk
            from tkinter import filedialog

            root = tk.Tk()
            root.withdraw()
            fname = filedialog.asksaveasfilename(
                title="Save animation as...",
                defaultextension=".mp4",
                filetypes=[("MP4 video", "*.mp4")],
                initialfile="animation.mp4",
            )
            root.update()
            root.destroy()
            if not fname:
                print("Save cancelled.")
                return None
            output_path = Path(fname)
        except Exception:
            # Headless or tkinter not available: fall back to current dir
            output_path = Path.cwd() / "animation.mp4"
            print(f"tkinter not available; saving to default: {output_path}")

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    # Precompute extremes across the whole movie
    maximo = float(np.max(x))
    minimo = float(np.min(x))

    fig = plt.figure(figsize=(9, 6))
    ax3 = fig.add_subplot(1, 3, 3, projection="3d")
    ax2 = fig.add_subplot(1, 3, 2, projection="3d")
    ax1 = fig.add_subplot(1, 3, 1, projection="3d")

    # Initial colors from first frame
    c0 = paint_colors(x[:, 0], maximo, minimo, satura=satura)

    sc3 = ax3.scatter3D(X3D, Y3D, Z3D, color=c0, s=size)
    ax3.view_init(0, -90)
    _style_ax(ax3)

    sc2 = ax2.scatter3D(Y3D, -X3D, Z3D, color=c0, s=size)
    ax2.view_init(90, -90)
    _style_ax(ax2)

    sc1 = ax1.scatter3D(X3D, Y3D, Z3D, color=c0, s=size)
    ax1.view_init(0, 90)
    _style_ax(ax1)

    fig.subplots_adjust(wspace=-0.2, hspace=-0.4)

    def update(frame: int):
        colors = paint_colors(x[:, frame], maximo, minimo, satura=satura)
        sc1.set_color(colors)
        sc2.set_color(colors)
        sc3.set_color(colors)
        return sc1, sc2, sc3

    ani = animation.FuncAnimation(
        fig, update, frames=T, interval=interval_ms, blit=False
    )

    try:
        ani.save(str(output_path), writer=writer, fps=fps, dpi=dpi)
        plt.close(fig)
        print(f"Saved animation to: {output_path}")
        return output_path
    except Exception as e:  # pragma: no cover
        plt.close(fig)
        raise RuntimeError(
            f"Failed to save video. Make sure '{writer}' is installed (e.g., FFmpeg). "
            f"Original error: {e}"
        )


# ============================================================================
# Spatiotemporal pattern from frequency-dependent covariance
# ============================================================================

def spatio_temporal(
    C_freq: np.ndarray,
    freqs: Sequence[float],
    ts: Sequence[float],
    freq: int = 0,
    leading: int = -1,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Reconstruct the spatiotemporal pattern of a given eigenmode at a given frequency.

    Parameters
    ----------
    C_freq
        Frequency-dependent covariance, shape (N, N, n_freqs).
    freqs
        Frequencies in Hz, length n_freqs.
    ts
        Times (in seconds) at which to evaluate the pattern, length T_time.
    freq
        Index of the frequency in ``freqs`` to analyse.
    leading
        Index of the eigenvector (mode) at that frequency. Use -1 for the
        mode with the largest eigenvalue.

    Returns
    -------
    sp : ndarray, shape (N, T_time)
        Spatiotemporal pattern: activity of each node as a function of time.
    sorting : ndarray, shape (N,)
        Indices that sort nodes by how close their phase is to 0 (in radians).

    Notes
    -----
    The pattern is:

        sp_i(t) = |v_i| * cos(2π f t + arg(v_i))

    where v is the chosen eigenvector of C_freq[:, :, freq].
    """
    C_freq = np.asarray(C_freq)
    freqs = np.asarray(freqs)
    ts = np.asarray(ts)

    print(f"Frequency analysed: {freqs[freq]} Hz")

    # Eigen-decomposition at selected frequency
    eigs, U = np.linalg.eigh(C_freq[:, :, freq])

    # Select eigenvector (e.g. leading = -1 → largest eigenvalue)
    v = U[:, leading]
    v = v / v[0]  # fix arbitrary global phase

    phases = np.angle(v)
    modules = np.abs(v)

    # Distance of phases to 0 (mod 2π)
    distance = np.minimum(np.abs(phases), 2 * np.pi - np.abs(phases))
    sorting = np.argsort(distance)

    n_nodes = len(v)
    sp = np.zeros((n_nodes, len(ts)))

    omega = 2 * np.pi * freqs[freq]
    for i, t in enumerate(ts):
        sp[:, i] = modules * np.cos(omega * t + phases)

    return sp, sorting


# ============================================================================
# Surrogate / PCA utilities
# ============================================================================

def reshuffling(X: np.ndarray) -> np.ndarray:
    """
    Circularly reshuffle each node's time series independently.

    Parameters
    ----------
    X
        Input time series, array of shape (N, T).

    Returns
    -------
    Y
        Reshuffled time series, same shape (N, T). For each node i,
        X[i] is rotated by a random shift, preserving its temporal
        structure but destroying synchrony across nodes.
    """
    X = np.asarray(X)
    N, T = X.shape

    shifts = np.random.randint(T, size=N)
    Y = np.zeros_like(X)

    for i in range(N):
        indices = (np.arange(T) + shifts[i]) % T
        Y[i, :] = X[i, indices]

    return Y

# NEW FUNCTION 
def phase_randomization(X: np.ndarray) -> np.ndarray:
    """Phase randomization in Fourier space."""
    X = np.asarray(X)
    N, T = X.shape
    
    Y = np.zeros_like(X)
    
    for i in range(N):
        fft_x = np.fft.rfft(X[i])
        random_phases = np.exp(2j * np.pi * np.random.rand(len(fft_x)))
        random_phases[0] = 1  # Keep DC component real
        if T % 2 == 0:
            random_phases[-1] = 1  # Keep Nyquist real for even length
        fft_randomized = fft_x * random_phases
        Y[i] = np.fft.irfft(fft_randomized, n=T)
    
    return Y

# NEW FUNCTION
def block_permutation(X: np.ndarray, block_length: int = 2400) -> np.ndarray:
    """Block permutation with fixed block length."""
    X = np.asarray(X)
    N, T = X.shape
    
    n_blocks = T // block_length
    remainder = T % block_length
    
    Y = np.zeros_like(X)
    
    for i in range(N):
        blocks = X[i, :n_blocks * block_length].reshape(n_blocks, block_length)
        permuted_indices = np.random.permutation(n_blocks)
        Y[i, :n_blocks * block_length] = blocks[permuted_indices].ravel()
        if remainder > 0:
            Y[i, n_blocks * block_length:] = X[i, n_blocks * block_length:]
    
    return Y


def PCA(X: np.ndarray) -> np.ndarray:
    """
    Compute eigenvalues of the correlation matrix (principal components spectrum).

    Parameters
    ----------
    X
        Time series, shape (N, T).

    Returns
    -------
    eigs
        Eigenvalues of the correlation matrix, sorted ascending as in
        :func:`numpy.linalg.eigh`.

    Notes
    -----
    The time series for each node are first z-scored, then the correlation
    matrix is computed as (X X^T / T).
    """
    X = np.asarray(X)
    N, T = X.shape

    # Standardize each time series (z-score)
    x_norm = np.zeros_like(X, dtype=float)
    for n in range(N):
        mu = np.mean(X[n, :])
        sigma = np.std(X[n, :])
        if sigma == 0:
            x_norm[n, :] = 0.0
        else:
            x_norm[n, :] = (X[n, :] - mu) / sigma

    # Correlation (here actually covariance of standardized variables)
    C = (x_norm @ x_norm.T) / float(T)

    eigs, _ = np.linalg.eigh(C)
    return eigs


# ============================================================================
# Frequency-dependent covariance and power spectrum
# ============================================================================

def correlation_freq(
    X: np.ndarray,
    time_step: float,
    frequency: float,
    n_chunks: int = 1,
    corr_type: str = "covariance",
) -> np.ndarray:
    """
    Compute the complex covariance/correlation matrix at a given frequency.

    Parameters
    ----------
    X
        Time series, array of shape (N, T).
    time_step
        Sampling interval (dt) in seconds.
    frequency
        Frequency of interest in Hz.
    n_chunks
        Number of chunks to split the time series into for averaging.
        If n_chunks > 1, the time series is split along the time axis
        and a covariance estimate is computed for each chunk, then
        averaged.
    corr_type
        Either "covariance" (default) or "correlation".
        If "correlation", the result is normalized by the diagonal
        (i.e., unit diagonal).

    Returns
    -------
    C
        Complex covariance/correlation matrix at the given frequency,
        shape (N, N).
    """
    X = np.asarray(X)
    N, T = X.shape
    omega = 2 * np.pi * frequency

    C = np.zeros((N, N), dtype=complex)

    for n in range(n_chunks):
        n_samples = int(T / n_chunks)
        start = n * n_samples
        stop = (n + 1) * n_samples
        X_chunk = X[:, start:stop]
        Sw = fourier_freq(X_chunk, time_step, omega)  # shape (N,)
        C_sample = Sw.reshape(N, 1) @ np.conjugate(Sw.reshape(1, N))
        C += C_sample

    C /= float(n_chunks)

    if corr_type.lower() == "correlation":
        diag = np.real(np.diag(C))
        # Avoid division by zero
        diag[diag == 0] = 1.0
        norm = np.sqrt(diag.reshape(N, 1) * diag.reshape(1, N))
        C = C / norm

    return C


def POWER_SPECTRUM(
    X: np.ndarray,
    dt: float,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Compute the (single-sided) power spectrum of one or many time series.

    Parameters
    ----------
    X
        Time series, shape (N, T) or (T,). If 1D, treated as a single
        time series.
    dt
        Sampling interval (seconds).

    Returns
    -------
    f
        Frequencies in Hz, shape (T/2 + 1,).
    P
        Amplitude spectrum (magnitude of FFT) with shape:
        - (N, T/2 + 1) if X is 2D
        - (T/2 + 1,) if X is 1D

    Notes
    -----
    This function follows the usual single-sided FFT convention where
    all frequencies except DC and Nyquist are multiplied by 2.
    """
    X = np.asarray(X)

    if X.ndim == 1:
        X = X[np.newaxis, :]

    N, T = X.shape
    freqs = np.linspace(0, 1, T // 2 + 1) / (2 * dt)

    P2 = np.abs(np.fft.fft(X, axis=1))
    P1 = P2[:, : T // 2 + 1]
    if T > 3:
        P1[:, 1:-2] *= 2

    if N == 1:
        return freqs, P1[0]
    return freqs, P1


# ============================================================================
# Distance to criticality (Beyond mean field estimator)
# ============================================================================

def Distance_criticality(
    C: np.ndarray,
    N: int,
    T: int,
    omegas: Sequence[float],
) -> np.ndarray:
    """
    Estimate distance to criticality g(ω) from frequency-dependent covariance.

    Parameters
    ----------
    C
        Covariance matrix (or matrices), either:
        - shape (N, N, n_omegas) for multiple frequencies, or
        - shape (N, N) for a single frequency.
    N
        Number of units (nodes).
    T
        Effective number of independent samples / chunks used to estimate
        C (appears in the unbiased estimator for the second moment).
    omegas
        Angular frequencies (rad/s). Only the length of this array is used
        when C has shape (N, N, n_omegas).

    Returns
    -------
    g_inferred
        If len(omegas) > 1:
            array of shape (2, n_omegas), where:
                g_inferred[0, :] is the biased estimator,
                g_inferred[1, :] is the unbiased estimator.
        If len(omegas) == 1:
            array of shape (2,), same meaning.

    Notes
    -----
    This follows the moment-based estimator of the effective coupling g
    in nearly critical random networks (Beyond mean field), using:

        Delta^2 = M2 / trace^2

    and then:

        g = sqrt(1 - sqrt(1 / (1 + N * Delta^2)))

    where M2 is the second moment of off-diagonal covariances and
    trace is the mean diagonal covariance.
    """
    C = np.asarray(C)
    omegas = np.asarray(omegas)

    off_mask = ~np.eye(N, dtype=bool)

    if len(omegas) > 1:
        g_inferred = np.zeros((2, len(omegas)))

        for w in range(len(omegas)):
            Cw = C[:, :, w]

            trace0 = np.real(np.trace(Cw)) / float(N)
            M2 = np.sqrt(np.real(np.mean(np.abs(Cw[off_mask]) ** 2)))

            Delta0 = M2 / trace0

            cij_mean = np.mean(np.abs(Cw))
            Delta_unbiased = np.sqrt(
                np.abs(M2**2 - (trace0**2 - cij_mean**2) / (T - 1))
            ) / trace0

            g_inferred[0, w] = np.sqrt(1.0 - np.sqrt(1.0 / (1.0 + N * Delta0**2)))
            g_inferred[1, w] = np.sqrt(
                1.0 - np.sqrt(1.0 / (1.0 + N * Delta_unbiased**2))
            )

    else:
        Cw = C
        g_inferred = np.zeros(2)

        trace0 = np.real(np.trace(Cw)) / float(N)
        M2 = np.sqrt(np.real(np.mean(np.abs(Cw[off_mask]) ** 2)))

        Delta0 = M2 / trace0

        cij_mean = np.mean(np.abs(Cw))
        Delta_unbiased = np.sqrt(
            np.abs(M2**2 - (trace0**2 - cij_mean**2) / (T - 1))
        ) / trace0

        g_inferred[0] = np.sqrt(1.0 - np.sqrt(1.0 / (1.0 + N * Delta0**2)))
        g_inferred[1] = np.sqrt(
            1.0 - np.sqrt(1.0 / (1.0 + N * Delta_unbiased**2))
        )

    return g_inferred



# ============================================================================
# Fourier transform at a single frequency
# ============================================================================

def fourier_freq(
    X: np.ndarray,
    time_step: float,
    omega: float,
) -> np.ndarray:
    """
    Compute Fourier component of time series at a single angular frequency.

    Parameters
    ----------
    X
        Time series, shape (N, T).
    time_step
        Sampling interval (dt) in seconds.
    omega
        Angular frequency in rad/s (omega = 2π f).

    Returns
    -------
    ps
        Complex Fourier amplitudes of shape (N,), normalized so that
        multiplying by (T * dt) recovers the sum in time.
    """
    X = np.asarray(X)
    N, T = X.shape
    t = np.arange(T) * time_step

    ps = np.zeros(N, dtype=complex)
    for n in range(N):
        ps[n] = np.sum(X[n, :] * np.exp(-1j * omega * t)) * time_step

    ps /= float(T * time_step)
    return ps