"""Relaxation-timescale spectrum of a memory trajectory.

Given an aggregate autocorrelation ``C(tau)`` we invert

    C(tau) ~= sum_k  w_k * exp(-tau / T_k),   w_k >= 0

over a fixed log-spaced grid of candidate timescales ``T_k`` using
non-negative least squares (the standard approach for relaxation spectra in
dynamic light scattering / rheology).  Peaks in ``w_k`` reveal the timescales
actually present in the memory -- so a model that *claims* multiple memory
speeds can be checked against what its trajectory actually contains.

A small first-difference (Tikhonov) penalty keeps the spectrum from splitting a
single true peak across many adjacent grid points without smearing distinct
peaks together.

This module also offers a wavelet octave-band energy view
(:func:`octave_band_energy`) as an independent, assumption-light cross-check.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
from scipy.optimize import nnls

__all__ = ["relaxation_spectrum", "dominant_timescales", "octave_band_energy", "Peak"]


def relaxation_spectrum(
    C: np.ndarray,
    *,
    n_grid: int = 48,
    t_min: float = 0.5,
    t_max: float | None = None,
    smooth: float = 1e-3,
) -> tuple[np.ndarray, np.ndarray]:
    """Recover ``(timescales, weights)`` from an autocorrelation ``C``.

    ``timescales`` is a log-spaced grid; ``weights`` are the non-negative
    amplitudes.  ``smooth`` is the strength of a first-difference penalty on the
    (log-grid) weights.
    """
    C = np.asarray(C, dtype=float)
    L = len(C)
    taus = np.arange(L, dtype=float)
    if t_max is None:
        t_max = max(4.0 * L, 8.0)
    timescales = np.geomspace(t_min, t_max, n_grid)

    A = np.exp(-taus[:, None] / timescales[None, :])  # (L, n_grid)
    b = C.copy()

    if smooth > 0:
        # penalize curvature of the spectrum (second difference), scaled to A
        D = np.zeros((n_grid - 2, n_grid))
        for i in range(n_grid - 2):
            D[i, i] = 1.0
            D[i, i + 1] = -2.0
            D[i, i + 2] = 1.0
        scale = (
            smooth
            * np.linalg.norm(A, ord="fro")
            / max(np.linalg.norm(D, ord="fro"), 1e-12)
        )
        A = np.vstack([A, scale * D])
        b = np.concatenate([b, np.zeros(n_grid - 2)])

    w, _ = nnls(A, b, maxiter=10 * n_grid)
    return timescales, w


@dataclass
class Peak:
    """A merged spectral peak."""

    timescale: float
    weight: float
    members: list[int] = field(default_factory=list)


def dominant_timescales(
    timescales: np.ndarray,
    weights: np.ndarray,
    *,
    rel_thresh: float = 0.05,
) -> list[Peak]:
    """Merge contiguous non-negligible grid bins into peaks.

    A peak's timescale is the weight-weighted geometric mean of its members;
    only peaks carrying at least ``rel_thresh`` of the total weight survive.
    Returned sorted by timescale (ascending).
    """
    timescales = np.asarray(timescales, dtype=float)
    weights = np.asarray(weights, dtype=float)
    total = weights.sum()
    if total <= 0:
        return []
    active = weights > rel_thresh * weights.max()

    peaks: list[Peak] = []
    i = 0
    n = len(weights)
    while i < n:
        if not active[i]:
            i += 1
            continue
        j = i
        while j < n and active[j]:
            j += 1
        members = list(range(i, j))
        wseg = weights[members]
        wsum = wseg.sum()
        # weighted geometric mean of the timescale
        logT = np.average(np.log(timescales[members]), weights=wseg)
        peaks.append(
            Peak(timescale=float(np.exp(logT)), weight=float(wsum), members=members)
        )
        i = j

    peaks = [p for p in peaks if p.weight >= rel_thresh * total]
    peaks.sort(key=lambda p: p.timescale)
    return peaks


def effective_n_timescales(weights: np.ndarray) -> float:
    """Participation ratio of the spectral weights: ``(sum w)^2 / sum w^2``.

    ~1 for a single-speed memory, grows toward the number of distinct modes.
    """
    w = np.asarray(weights, dtype=float)
    s1 = w.sum()
    s2 = (w * w).sum()
    if s2 <= 0:
        return 0.0
    return float(s1 * s1 / s2)


def octave_band_energy(
    X: np.ndarray,
    *,
    wavelet: str = "db4",
    level: int | None = None,
) -> np.ndarray:
    """Wavelet octave-band energy of a trajectory (assumption-light cross-check).

    Returns the fraction of detail energy in each octave (finest -> coarsest).
    A single-timescale signal concentrates energy in one band; genuinely
    multi-timescale memory spreads it across bands.
    """
    import pywt

    X = np.asarray(X, dtype=float)
    if X.ndim == 1:
        X = X[:, None]
    if X.ndim == 3:  # ensemble -> stack runs along the dimension axis
        n, T, d = X.shape
        X = X.transpose(1, 0, 2).reshape(T, n * d)
    X = X - X.mean(axis=0, keepdims=True)
    T = X.shape[0]
    max_level = pywt.dwt_max_level(T, pywt.Wavelet(wavelet).dec_len)
    if level is None:
        level = max(1, max_level)
    level = min(level, max_level)
    if level < 1:
        raise ValueError("trajectory too short for a wavelet decomposition")

    band_energy = np.zeros(level)
    for j in range(X.shape[1]):
        coeffs = pywt.wavedec(X[:, j], wavelet, level=level)
        details = coeffs[1:]  # cA, cD_level, ..., cD_1  -> drop approximation
        for k, cd in enumerate(details):
            band_energy[k] += float(np.sum(np.asarray(cd) ** 2))
    total = band_energy.sum()
    if total <= 0:
        return band_energy
    return band_energy / total
