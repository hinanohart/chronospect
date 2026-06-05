"""Two-time correlation and an aging index.

Borrowed from the statistical physics of glasses: the two-time correlation
``C(t_w, tau)`` measures how much the state at waiting time ``t_w`` still
overlaps with the state ``tau`` steps later.

* If the dynamics are **stationary / single-speed**, ``C`` is
  time-translation invariant: it depends on ``tau`` only, and the curves for
  different ``t_w`` collapse.
* If the memory **ages** (a hierarchy of slow modes building up during
  training), ``C(t_w, tau)`` depends on ``t_w`` too -- older states decorrelate
  more slowly.  This is exactly the non-stationarity a multi-timescale memory
  is supposed to exhibit, and it is invisible to end-task accuracy / BWT.

:func:`aging_index` summarizes the ``t_w`` dependence as a single number
(0 = stationary).
"""

from __future__ import annotations

import numpy as np

__all__ = ["two_time_correlation", "aging_index"]


def _as_ensemble(X: np.ndarray) -> np.ndarray:
    X = np.asarray(X, dtype=float)
    if X.ndim == 2:
        X = X[None, ...]
    if X.ndim != 3:
        raise ValueError(f"expected (T, d) or (n, T, d), got shape {X.shape}")
    return X


def two_time_correlation(
    X: np.ndarray,
    t_ws: np.ndarray,
    taus: np.ndarray,
    *,
    tw_window: int = 0,
    demean: bool = True,
) -> np.ndarray:
    """Return ``C[i, j] ~= corr(x_{t_ws[i]}, x_{t_ws[i] + taus[j]})``.

    Correlation is the cosine overlap across dimensions, averaged over the
    ensemble **and** over a window of start times ``[t_w - tw_window,
    t_w + tw_window]``.  Block-averaging over a window in which the dynamics are
    quasi-stationary sharply reduces the finite-ensemble noise that would
    otherwise fake a waiting-time trend; set ``tw_window=0`` for the raw
    single-time estimate.  Entries with no valid ``t_w' + tau`` are ``nan``.
    """
    Xe = _as_ensemble(X)
    _n, T, _d = Xe.shape
    if demean:
        Xe = Xe - Xe.mean(axis=1, keepdims=True)
    t_ws = np.asarray(t_ws, dtype=int)
    taus = np.asarray(taus, dtype=int)
    tw_window = max(0, int(tw_window))

    C = np.full((len(t_ws), len(taus)), np.nan)
    for i, tw in enumerate(t_ws):
        lo = max(0, tw - tw_window)
        hi = min(T - 1, tw + tw_window)
        if hi < lo:
            continue
        starts = np.arange(lo, hi + 1)
        for j, tau in enumerate(taus):
            s = starts[(starts + tau >= 0) & (starts + tau < T)]
            if s.size == 0:
                continue
            a = Xe[:, s, :]  # (n, |s|, d)
            b = Xe[:, s + tau, :]
            na = np.linalg.norm(a, axis=2)  # (n, |s|)
            nb = np.linalg.norm(b, axis=2)
            denom = na * nb
            dot = np.sum(a * b, axis=2)  # (n, |s|)
            ok = denom > 1e-12
            if not np.any(ok):
                continue
            C[i, j] = float(np.mean(dot[ok] / denom[ok]))
    return C


def _half_life(curve: np.ndarray, taus: np.ndarray, frac: float) -> float:
    """First lag where ``curve`` drops below ``frac`` (linear-interpolated)."""
    curve = np.asarray(curve, dtype=float)
    taus = np.asarray(taus, dtype=float)
    valid = ~np.isnan(curve)
    curve, taus = curve[valid], taus[valid]
    if curve.size < 2:
        return float("nan")
    if curve[0] <= frac:
        # already below the threshold at the first sampled lag; do not extrapolate
        return float(taus[0])
    for k in range(1, len(curve)):
        if curve[k] <= frac:
            c0, c1 = curve[k - 1], curve[k]
            t0, t1 = taus[k - 1], taus[k]
            if c0 == c1:
                return float(t1)
            return float(t0 + (frac - c0) * (t1 - t0) / (c1 - c0))
    return float("nan")  # never decayed below frac within the window


def aging_index(
    C: np.ndarray,
    t_ws: np.ndarray,
    taus: np.ndarray,
    *,
    frac: float = 1.0 / np.e,
    alpha: float = 0.05,
) -> float:
    """Quantify how strongly the decay timescale grows with waiting time.

    For each waiting time we read the decorrelation half-life (lag where ``C``
    falls to ``frac``), fit ``half_life ~ a + beta * t_w`` and report the
    *relative* increase the trend predicts across the observed window:

        index = beta * (t_w_max - t_w_min) / mean_half_life

    To avoid mistaking finite-ensemble noise for aging, the slope is reported
    only when it is **positive and statistically significant**: its one-sided
    Student-t statistic must exceed the critical value for the actual degrees of
    freedom at level ``alpha`` (so three waiting times demand a much larger
    t-stat than ten do); otherwise the index is exactly 0.  A stationary memory
    therefore reads ~0 regardless of how it is split across timescales, while a
    genuinely aging memory reads large.  Returns ``nan`` if fewer than three
    waiting times yield a finite half-life.
    """
    from scipy.stats import t as _student_t

    C = np.asarray(C, dtype=float)
    taus = np.asarray(taus, dtype=float)
    tw = np.asarray(t_ws, dtype=float)
    hls = np.array([_half_life(C[i], taus, frac) for i in range(C.shape[0])])
    ok = np.isfinite(hls)
    if ok.sum() < 3:
        return float("nan")
    tw_ok, hl_ok = tw[ok], hls[ok]
    mean = hl_ok.mean()
    span = tw_ok.max() - tw_ok.min()
    if mean <= 0 or span <= 0:
        return float("nan")

    # ordinary least-squares of half_life ~ a + beta * t_w, with slope t-stat
    A = np.vstack([np.ones_like(tw_ok), tw_ok]).T
    coef, *_ = np.linalg.lstsq(A, hl_ok, rcond=None)
    beta = coef[1]
    resid = hl_ok - A @ coef
    dof = tw_ok.size - 2
    sxx = float(np.sum((tw_ok - tw_ok.mean()) ** 2))
    if dof >= 1 and sxx > 0:
        s2 = float(np.sum(resid**2)) / dof
        se_beta = np.sqrt(s2 / sxx) if s2 > 0 else 0.0
        t_stat = beta / se_beta if se_beta > 0 else np.inf
        t_crit = float(_student_t.ppf(1.0 - alpha, dof))
    else:
        t_stat, t_crit = np.inf, 0.0  # cannot assess significance -> do not gate

    if beta <= 0 or t_stat < t_crit:
        return 0.0
    return float(beta * span / mean)
