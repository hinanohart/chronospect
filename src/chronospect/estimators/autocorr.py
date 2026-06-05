"""Aggregate autocorrelation of a memory trajectory.

A trajectory ``X`` is ``(T, d)`` (one run) or ``(n, T, d)`` (an ensemble of
``n`` runs).  We compute the variance-weighted average of the per-dimension
normalized autocorrelation ``C(tau)`` under a (locally) stationary assumption,
i.e. averaging the lag-``tau`` product over all valid start times ``t_w``.

For a single AR(1)/Ornstein-Uhlenbeck channel with retention ``a`` the result
is ``C(tau) = a**tau = exp(-tau / T)`` with ``T = -1 / ln a``.  For a mix of
independent channels the result is the variance-weighted sum of such
exponentials -- which is exactly what :mod:`chronospect.estimators.spectrum`
inverts to recover the individual timescales.
"""

from __future__ import annotations

import numpy as np

__all__ = ["aggregate_autocorr"]


def _as_ensemble(X: np.ndarray) -> np.ndarray:
    X = np.asarray(X, dtype=float)
    if X.ndim == 2:
        X = X[None, ...]
    if X.ndim != 3:
        raise ValueError(f"expected (T, d) or (n, T, d), got shape {X.shape}")
    return X


def aggregate_autocorr(
    X: np.ndarray,
    max_lag: int | None = None,
    *,
    demean: bool = True,
    weight_by_variance: bool = True,
    var_floor: float = 1e-12,
) -> np.ndarray:
    """Return ``C[tau]`` for ``tau = 0 .. max_lag`` (``C[0] == 1`` by construction).

    Parameters
    ----------
    X:
        Trajectory ``(T, d)`` or ensemble ``(n, T, d)``.
    max_lag:
        Largest lag; defaults to ``T // 2``.
    demean:
        Subtract each dimension's temporal mean before correlating.
    weight_by_variance:
        Weight dimensions by their variance fraction (so high-energy memory
        directions dominate); otherwise a plain mean over dimensions.
    """
    Xe = _as_ensemble(X)
    _n, T, _d = Xe.shape
    if max_lag is None:
        max_lag = T // 2
    max_lag = int(min(max_lag, T - 1))
    if max_lag < 1:
        raise ValueError("trajectory too short for autocorrelation")

    if demean:
        Xe = Xe - Xe.mean(axis=1, keepdims=True)

    # per (run, dim) variance, pooled across the ensemble
    var = Xe.var(axis=1).mean(axis=0)  # (d,)
    keep = var > var_floor
    if not np.any(keep):
        return np.concatenate([[1.0], np.zeros(max_lag)])
    Xk = Xe[:, :, keep]
    vk = var[keep]
    w = vk / vk.sum() if weight_by_variance else np.full(vk.shape, 1.0 / vk.size)

    C = np.empty(max_lag + 1)
    for tau in range(max_lag + 1):
        a = Xk[:, : T - tau, :]
        b = Xk[:, tau:, :]
        cov = (a * b).mean(axis=(0, 1))  # (dk,) lag-tau autocovariance
        ac = cov / vk  # normalized -> per-dim autocorrelation
        C[tau] = float(np.sum(w * ac))
    # guard tiny numerical drift at tau=0
    if C[0] != 0:
        C = C / C[0]
    return C
