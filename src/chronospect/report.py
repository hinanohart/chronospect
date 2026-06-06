"""High-level ``analyze`` entry point that assembles a TimescaleReport.

This is the one call most users need: hand it a memory trajectory and it
returns every diagnostic with a plain-language reading of whether the memory
looks genuinely multi-timescale, single-speed, or aging.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field

import numpy as np

from .estimators.autocorr import aggregate_autocorr
from .estimators.forgetting import ForgettingFit, forgetting_fit
from .estimators.observability import capacity_vs_horizon
from .estimators.spectrum import (
    dominant_timescales,
    effective_n_timescales,
    octave_band_energy,
    relaxation_spectrum,
)
from .estimators.twotime import aging_index, two_time_correlation

__all__ = ["analyze", "TimescaleReport"]


@dataclass
class TimescaleReport:
    n_dominant_timescales: int
    dominant_timescales: list[float]
    effective_n_timescales: float
    capacity_horizon_half: float  # lag at which memory capacity falls to half
    aging_index: float
    forgetting: ForgettingFit
    octave_band_energy: list[float]
    verdict: str
    # raw curves for plotting / auditing (not part of the headline)
    autocorr: list[float] = field(default_factory=list)
    spectrum_timescales: list[float] = field(default_factory=list)
    spectrum_weights: list[float] = field(default_factory=list)

    def to_dict(self) -> dict:
        d = asdict(self)
        d["forgetting"] = asdict(self.forgetting)
        return d


def _capacity_half(lags: np.ndarray, cap: np.ndarray) -> float:
    valid = np.isfinite(cap)
    lags, cap = np.asarray(lags)[valid], np.asarray(cap)[valid]
    if cap.size == 0 or cap[0] <= 0:
        return float("nan")
    target = cap[0] / 2.0
    for k in range(1, len(cap)):
        if cap[k] <= target:
            c0, c1 = cap[k - 1], cap[k]
            l0, l1 = float(lags[k - 1]), float(lags[k])
            if c0 == c1:
                return l1
            return l0 + (target - c0) * (l1 - l0) / (c1 - c0)
    return float("nan")


def analyze(
    X: np.ndarray,
    *,
    max_lag: int | None = None,
    rel_thresh: float = 0.08,
    aging_waiting_times: np.ndarray | None = None,
    bias_correct: bool = False,
) -> TimescaleReport:
    """Compute the full timescale report for a memory trajectory ``X``.

    ``X`` is ``(T, d)`` or an ensemble ``(n, T, d)``.

    ``bias_correct`` (default ``False``) toggles the demeaning-bias correction in
    :func:`chronospect.aggregate_autocorr`. It is off by default because, although
    it improves long-timescale recovery, it can inflate the apparent spectral width
    of a single-speed memory (see that function and :func:`chronospect.calibrate`);
    enable it when long-timescale calibration matters more than the single-vs-multi
    headline.
    """
    X = np.asarray(X, dtype=float)
    if X.ndim not in (2, 3):
        raise ValueError(f"X must be (T, d) or (n, T, d); got shape {X.shape}")
    T = X.shape[-2]
    if T < 32:
        raise ValueError(f"trajectory too short to analyze (T={T}); need at least ~32 steps")
    if max_lag is None:
        max_lag = min(T // 2, 600)

    C = aggregate_autocorr(X, max_lag=max_lag, bias_correct=bias_correct)
    grid, w = relaxation_spectrum(C)
    peaks = dominant_timescales(grid, w, rel_thresh=rel_thresh)
    neff = effective_n_timescales(w)

    lags, cap = capacity_vs_horizon(X, max_lag=max_lag)
    cap_half = _capacity_half(lags, cap)

    if aging_waiting_times is None:
        aging_waiting_times = np.linspace(0.08 * T, 0.78 * T, 6).astype(int)
    taus = np.arange(0, min(max_lag, T // 3), max(1, T // 600))
    tw_window = max(1, T // 40)
    Ctt = two_time_correlation(X, aging_waiting_times, taus, tw_window=tw_window)
    ai = aging_index(Ctt, aging_waiting_times, taus)

    fit = forgetting_fit(C)
    obe = octave_band_energy(X)

    verdict = _verdict(len(peaks), neff, ai)

    return TimescaleReport(
        n_dominant_timescales=len(peaks),
        dominant_timescales=[round(p.timescale, 3) for p in peaks],
        effective_n_timescales=round(neff, 3),
        capacity_horizon_half=round(float(cap_half), 3),
        aging_index=round(float(ai), 3) if np.isfinite(ai) else float("nan"),
        forgetting=fit,
        octave_band_energy=[round(float(x), 4) for x in obe],
        verdict=verdict,
        autocorr=[round(float(x), 5) for x in C],
        spectrum_timescales=[round(float(x), 3) for x in grid],
        spectrum_weights=[round(float(x), 5) for x in w],
    )


def _verdict(n_peaks: int, neff: float, ai: float) -> str:
    aging = np.isfinite(ai) and ai >= 0.3
    if n_peaks >= 2 and neff >= 1.8:
        base = "multi-timescale"
    else:
        base = "effectively single-speed"
    if aging:
        return f"{base}; non-stationary (aging)"
    return f"{base}; approximately stationary"
