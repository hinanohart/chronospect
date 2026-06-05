"""Data-driven observability / memory-capacity estimators.

We never assume access to the driving inputs, so we use *empirical* observability
proxies built from the state trajectory itself:

* :func:`effective_rank` -- the Roy-Vetterli effective rank (entropy of the
  normalized singular spectrum), i.e. how many independent directions a matrix
  carries.
* :func:`windowed_effective_rank` -- effective rank of the state covariance in a
  sliding window: how many memory directions are simultaneously active.
* :func:`capacity_vs_horizon` -- effective rank of the lag-``tau``
  cross-covariance ``E[x_t x_{t+tau}^T]`` as a function of ``tau``.  This is the
  memory-capacity-versus-horizon curve: how many independent past directions
  survive ``tau`` steps.  A single-mode memory collapses to rank ~1 quickly; a
  genuinely multi-timescale memory keeps a higher rank out to longer horizons.
"""

from __future__ import annotations

import numpy as np

__all__ = [
    "effective_rank",
    "windowed_effective_rank",
    "capacity_vs_horizon",
]


def _as_ensemble(X: np.ndarray) -> np.ndarray:
    X = np.asarray(X, dtype=float)
    if X.ndim == 2:
        X = X[None, ...]
    if X.ndim != 3:
        raise ValueError(f"expected (T, d) or (n, T, d), got shape {X.shape}")
    return X


def effective_rank(singular_values: np.ndarray, eps: float = 1e-12) -> float:
    """Effective rank ``exp(H(p))`` of a singular spectrum (Roy & Vetterli, 2007)."""
    s = np.asarray(singular_values, dtype=float)
    s = s[s > eps]
    if s.size == 0:
        return 0.0
    p = s / s.sum()
    H = -np.sum(p * np.log(p))
    return float(np.exp(H))


def windowed_effective_rank(
    X: np.ndarray,
    window: int,
    *,
    step: int = 1,
) -> tuple[np.ndarray, np.ndarray]:
    """Effective rank of the state covariance over sliding windows.

    Returns ``(centers, ranks)`` where ``centers`` are the window centre times.
    """
    Xe = _as_ensemble(X)
    _n, T, _d = Xe.shape
    if window < 2 or window > T:
        raise ValueError("window must be in [2, T]")
    centers, ranks = [], []
    for start in range(0, T - window + 1, step):
        seg = Xe[:, start : start + window, :]  # (n, window, d)
        seg = seg - seg.mean(axis=1, keepdims=True)
        flat = seg.reshape(-1, seg.shape[-1])  # (n*window, d)
        s = np.linalg.svd(flat, compute_uv=False)
        centers.append(start + window / 2.0)
        # covariance eigenvalues are s**2; effective rank of the covariance
        ranks.append(effective_rank(s**2))
    return np.asarray(centers), np.asarray(ranks)


def _inv_sqrt(Sigma: np.ndarray, ridge: float) -> np.ndarray:
    """Symmetric ``Sigma^{-1/2}`` with a relative ridge for stability."""
    vals, vecs = np.linalg.eigh(Sigma)
    vals = np.clip(vals, 0.0, None)
    floor = ridge * (vals.max() if vals.size and vals.max() > 0 else 1.0)
    inv_sqrt = 1.0 / np.sqrt(vals + floor)
    return (vecs * inv_sqrt) @ vecs.T


def capacity_vs_horizon(
    X: np.ndarray,
    max_lag: int | None = None,
    *,
    demean: bool = True,
    ridge: float = 1e-3,
) -> tuple[np.ndarray, np.ndarray]:
    """Memory capacity (summed squared canonical correlations) vs horizon.

    For each lag ``tau`` we whiten the state and read the canonical correlations
    between ``x_t`` and ``x_{t+tau}`` (singular values of the whitened
    cross-covariance, clipped to ``[0, 1]``).  ``capacity[tau] = sum(rho_k**2)``
    is the effective number of past directions still linearly recoverable after
    ``tau`` steps.

    Unlike a raw cross-covariance rank, whitening removes the noise-floor
    inflation that makes a *faded* memory look high-rank: once the true
    correlation is gone the canonical correlations collapse to ~0, so
    ``capacity`` decays to 0 for every memory -- just more slowly for a genuinely
    long-timescale one.  ``capacity[0]`` is ~``d``.
    """
    Xe = _as_ensemble(X)
    _n, T, d = Xe.shape
    if max_lag is None:
        max_lag = T // 2
    max_lag = int(min(max_lag, T - 1))
    if demean:
        Xe = Xe - Xe.mean(axis=1, keepdims=True)

    flat = Xe.reshape(-1, d)
    Sigma = (flat.T @ flat) / flat.shape[0]
    W = _inv_sqrt(Sigma, ridge)
    Z = Xe @ W  # whitened trajectory (W symmetric)

    lags = np.arange(max_lag + 1)
    capacity = np.empty(max_lag + 1)
    for tau in lags:
        a = Z[:, : T - tau, :].reshape(-1, d)
        b = Z[:, tau:, :].reshape(-1, d)
        m = a.shape[0]
        M = (a.T @ b) / m  # whitened cross-covariance ~ canonical-correlation matrix
        s = np.linalg.svd(M, compute_uv=False)
        s = np.clip(s, 0.0, 1.0)
        capacity[tau] = float(np.sum(s * s))
    return lags, capacity
