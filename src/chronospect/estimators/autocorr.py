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
    bias_correct: bool = False,
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
    bias_correct:
        Correct the finite-sample bias introduced by subtracting the *sample*
        mean (``demean``). Demeaning lowers every estimated autocovariance by
        roughly ``Var(x_bar)``, which is largest *relative* to the signal at long
        lags and drags the tail toward (and through) zero, under-estimating long
        timescales. When ``True`` a Bartlett-weighted estimate of ``Var(x_bar)``
        (reusing the lags already computed) is added back to every autocovariance
        before re-normalizing, which lifts the long-lag tail and noticeably
        improves recovery of long timescales (see :func:`chronospect.calibrate`).

        **Default is** ``False``. This is a deliberate, measured choice: lifting
        the tail also injects a small amount of long-timescale weight into a
        genuinely *single*-timescale spectrum (the inversion cannot tell a
        bias-shrunk long timescale from a correctly-zero short-timescale tail --
        that ambiguity is the inherent ill-posedness). Empirically this inflates
        the effective number of timescales for a single-speed memory on a sizeable
        fraction of seeds, which would weaken the single-vs-multi discrimination
        the instrument is built on. So the correction is *opt-in*: enable it when
        you specifically want better long-timescale calibration, and consult
        :func:`chronospect.calibrate` for the recovered-vs-injected curve (and its
        cost) at your settings. The residual finite-window shrinkage that remains
        even with ``bias_correct=True`` is inherent and is disclosed, not removed.
        Has no effect when ``demean=False``.

        Note: adding ``x_bar**2`` directly is *not* equivalent -- it overshoots
        and injects spurious peaks -- so the Bartlett estimator is used instead.
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

    # raw (biased) lag-tau autocovariance per kept dimension: gamma[tau, dim]
    gamma = np.empty((max_lag + 1, vk.size))
    for tau in range(max_lag + 1):
        a = Xk[:, : T - tau, :]
        b = Xk[:, tau:, :]
        gamma[tau] = (a * b).mean(axis=(0, 1))

    if bias_correct and demean:
        # Var(x_bar) ~= (1/T)[gamma(0) + 2 sum_{k>=1}(1 - k/T) gamma(k)], per dim,
        # using Bartlett (triangular) lag weights over the lags we already have.
        k = np.arange(max_lag + 1)
        bartlett = 1.0 - k / T
        var_xbar = (gamma[0] + 2.0 * (bartlett[1:, None] * gamma[1:]).sum(axis=0)) / T
        var_xbar = np.clip(var_xbar, 0.0, None)  # a variance estimate is non-negative
        gamma = gamma + var_xbar[None, :]

    denom = np.where(gamma[0] > var_floor, gamma[0], np.inf)  # dead dims contribute 0
    ac = gamma / denom[None, :]  # (max_lag+1, dk) per-dim autocorrelation
    C = np.asarray(ac @ w, dtype=float)  # variance-weighted sum over dims
    if C[0] != 0:
        C = C / C[0]  # guard tiny numerical drift at tau=0
    return C
