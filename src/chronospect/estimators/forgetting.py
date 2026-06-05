"""Benna-Fusi forgetting yardstick (reference only, never a headline claim).

A cascade memory tuned for many timescales (Benna & Fusi, Nat. Neurosci. 2016)
forgets as a power law ``C(tau) ~ tau**-0.5`` rather than a single exponential.
We fit both an exponential and a power law to the decay of an autocorrelation
and report which describes it better, plus the fitted power-law exponent.

This is a *reference yardstick*: a single power-law fit is close to trivial, so
:func:`forgetting_fit` is meant to contextualize the spectrum / two-time
results, not to stand alone.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

__all__ = ["forgetting_fit", "ForgettingFit"]


@dataclass
class ForgettingFit:
    better: str  # "power_law" | "exponential" | "inconclusive"
    power_law_exponent: float
    power_law_r2: float
    exponential_timescale: float
    exponential_r2: float


def _r2(y: np.ndarray, yhat: np.ndarray) -> float:
    y = np.asarray(y, dtype=float)
    ss_res = float(np.sum((y - yhat) ** 2))
    ss_tot = float(np.sum((y - y.mean()) ** 2))
    if ss_tot <= 0:
        return 0.0
    return 1.0 - ss_res / ss_tot


def forgetting_fit(
    C: np.ndarray,
    *,
    tau_min: int = 1,
    floor: float = 1e-4,
) -> ForgettingFit:
    """Fit exponential vs power-law decay to ``C(tau)`` (for ``tau >= tau_min``).

    Fits are done in log space on the positive part of the curve.  ``better`` is
    decided by R^2 with a small margin; near-ties are ``"inconclusive"``.
    """
    C = np.asarray(C, dtype=float)
    taus = np.arange(len(C), dtype=float)
    mask = (taus >= tau_min) & (C > floor)
    t = taus[mask]
    c = C[mask]
    if t.size < 3:
        return ForgettingFit("inconclusive", float("nan"), 0.0, float("nan"), 0.0)

    logc = np.log(c)

    # exponential: log C = log A - tau / T   ->  linear in tau
    A1 = np.vstack([np.ones_like(t), -t]).T
    coef_e, *_ = np.linalg.lstsq(A1, logc, rcond=None)
    inv_T = coef_e[1]
    T = 1.0 / inv_T if inv_T > 0 else float("inf")
    exp_hat = A1 @ coef_e
    r2_exp = _r2(logc, exp_hat)

    # power law: log C = log A - p * log tau  ->  linear in log tau
    logt = np.log(t)
    A2 = np.vstack([np.ones_like(logt), -logt]).T
    coef_p, *_ = np.linalg.lstsq(A2, logc, rcond=None)
    p = coef_p[1]
    pl_hat = A2 @ coef_p
    r2_pl = _r2(logc, pl_hat)

    margin = 0.02
    if r2_pl > r2_exp + margin:
        better = "power_law"
    elif r2_exp > r2_pl + margin:
        better = "exponential"
    else:
        better = "inconclusive"

    return ForgettingFit(
        better=better,
        power_law_exponent=float(p),
        power_law_r2=float(r2_pl),
        exponential_timescale=float(T),
        exponential_r2=float(r2_exp),
    )
