"""Synthetic memory trajectories with *known* ground-truth timescales.

These generators exist so the instrument can be validated before it is ever
pointed at a real model: if ``chronospect`` cannot recover timescales it
*injected itself*, it must not be trusted on a real network.  See
:mod:`chronospect.sensitivity`.

All generators are deterministic given ``seed`` and return arrays shaped
``(n_traj, T, d)``.  Randomness lives only in the data; every estimator that
consumes these arrays is a deterministic function of its input.
"""

from __future__ import annotations

import numpy as np

__all__ = ["single_timescale", "multi_timescale", "aging_process"]


def _ar1(rng: np.random.Generator, T: int, a: float, sigma: float = 1.0) -> np.ndarray:
    """One unit-variance AR(1)/OU path of length ``T`` with retention ``a``.

    The innovation scale is chosen as ``sqrt(1 - a**2)`` so the stationary
    variance is ``sigma**2`` regardless of the timescale -- otherwise slow
    channels would carry far more energy than fast ones and dominate the
    variance-weighted autocorrelation, hiding the fast timescale.
    """
    innov = sigma * np.sqrt(max(1.0 - a * a, 1e-6))
    x = np.empty(T)
    x[0] = rng.normal(0.0, sigma)
    for t in range(1, T):
        x[t] = a * x[t - 1] + rng.normal(0.0, innov)
    return x


def _a_from_timescale(timescale: float) -> float:
    """AR(1) retention ``a = exp(-1/T)`` so the autocorrelation is ``exp(-tau/T)``."""
    return float(np.exp(-1.0 / float(timescale)))


def single_timescale(
    T: int = 1024,
    d: int = 16,
    *,
    timescale: float = 20.0,
    n_traj: int = 8,
    sigma: float = 1.0,
    seed: int = 0,
) -> np.ndarray:
    """Stationary memory with exactly one timescale (an AR(1) per dimension)."""
    rng = np.random.default_rng(seed)
    a = _a_from_timescale(timescale)
    out = np.empty((n_traj, T, d))
    for i in range(n_traj):
        for j in range(d):
            out[i, :, j] = _ar1(rng, T, a, sigma)
    return out


def multi_timescale(
    T: int = 1024,
    d: int = 24,
    *,
    timescales: tuple[float, ...] = (5.0, 100.0),
    weights: tuple[float, ...] | None = None,
    n_traj: int = 8,
    sigma: float = 1.0,
    seed: int = 0,
) -> np.ndarray:
    """Stationary memory mixing several timescales.

    Dimensions are partitioned across the requested timescales (sizes given by
    ``weights``); each dimension is an independent AR(1).  The aggregate
    autocorrelation is the weighted sum of the corresponding exponentials, so a
    correct spectrum estimator recovers every ``timescale``.
    """
    rng = np.random.default_rng(seed)
    ts = np.asarray(timescales, dtype=float)
    k = len(ts)
    if k > d:
        raise ValueError(f"need d >= number of timescales ({k}); got d={d}")
    if weights is None:
        weights = tuple(1.0 / k for _ in range(k))
    w = np.asarray(weights, dtype=float)
    w = w / w.sum()
    # allocate dimensions to timescales: one each, then distribute the rest by
    # largest remainder so the counts provably sum to exactly d.
    counts = np.ones(k, dtype=int)
    remaining = d - k
    if remaining > 0:
        frac = w * remaining
        add = np.floor(frac).astype(int)
        counts += add
        deficit = remaining - int(add.sum())
        order = np.argsort(-(frac - np.floor(frac)))
        for i in range(deficit):
            counts[order[i % k]] += 1
    assert counts.sum() == d, (counts, d)

    out = np.empty((n_traj, T, int(counts.sum())))
    for i in range(n_traj):
        col = 0
        for ts_k, c_k in zip(ts, counts, strict=True):
            a = _a_from_timescale(ts_k)
            for _ in range(c_k):
                out[i, :, col] = _ar1(rng, T, a, sigma)
                col += 1
    return out


def aging_process(
    T: int = 1024,
    d: int = 16,
    *,
    base_timescale: float = 8.0,
    aging_rate: float = 200.0,
    n_traj: int = 8,
    sigma: float = 1.0,
    seed: int = 0,
) -> np.ndarray:
    """Non-stationary (aging) memory: the local timescale grows with time.

    The instantaneous retention is ``a_t = exp(-1 / (base * (1 + t/aging_rate)))``
    so early states relax fast and late states relax slowly -- a glassy,
    coarsening-like memory whose two-time correlation depends on the waiting
    time.  The innovation is scaled by ``sqrt(1 - a_t**2)`` so the *variance*
    stays ~constant and the aging signal is a pure timescale ramp rather than a
    confounded amplitude ramp.  Used to validate
    :func:`chronospect.estimators.twotime.aging_index`.
    """
    rng = np.random.default_rng(seed)
    t = np.arange(T, dtype=float)
    a_t = np.exp(-1.0 / (base_timescale * (1.0 + t / aging_rate)))
    innov = sigma * np.sqrt(np.clip(1.0 - a_t**2, 1e-6, None))
    out = np.empty((n_traj, T, d))
    for i in range(n_traj):
        for j in range(d):
            x = np.empty(T)
            x[0] = rng.normal(0.0, sigma)
            for tt in range(1, T):
                x[tt] = a_t[tt] * x[tt - 1] + rng.normal(0.0, innov[tt])
            out[i, :, j] = x
    return out
