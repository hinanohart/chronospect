"""Fix-B: disclose the instrument's timescale calibration as a curve.

The relaxation-spectrum inversion is, for a finite observation window, an
ill-posed Laplace inversion: long timescales are systematically under-recovered
(shrunk), and only a longer window genuinely reduces this. The opt-in demeaning-
bias correction (``bias_correct``; see :func:`chronospect.aggregate_autocorr`)
removes the *identifiable* finite-sample component of that shrinkage and lifts
long-timescale recovery, but it cannot remove the residual finite-window
shrinkage, and it carries a cost (it inflates the apparent spectral width of a
genuinely single-speed memory).

:func:`calibrate` measures both sides of that trade-off on synthetic data with
known timescales: the recovered-versus-injected ratio (with a bootstrap CI) and
the spectral-width cost on a single-speed memory. It is a *disclosure* tool -- it
tells you how to read a recovered long timescale (as a shrinkage-affected
estimate / lower bound, not an exact value), and what enabling the correction
buys and costs. It does not silently rescale anything.

All numbers it returns are functions of synthetic data with a fixed bootstrap
seed, so :func:`calibrate` is reproducible.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field

import numpy as np

from .estimators.autocorr import aggregate_autocorr
from .estimators.spectrum import (
    dominant_timescales,
    effective_n_timescales,
    relaxation_spectrum,
)
from .synthetic import single_timescale

__all__ = ["calibrate", "CalibrationCurve", "CalibrationPoint"]

DEFAULT_TAUS = (7.0, 20.0, 50.0, 100.0, 200.0, 300.0)


@dataclass
class CalibrationPoint:
    injected: float
    recovered_median: float  # median recovered timescale (NOT a claim of exact value)
    ratio_median: float  # recovered / injected
    ratio_ci_low: float  # bootstrap 95% CI on the ratio
    ratio_ci_high: float
    n_seeds: int


@dataclass
class CalibrationCurve:
    bias_correct: bool
    T: int
    n_traj: int
    n_boot: int
    boot_seed: int
    points: list[CalibrationPoint] = field(default_factory=list)
    # disclosed cost of the correction: effective # timescales the instrument
    # reports for a genuinely single-speed memory (ideal is ~1.0; larger means the
    # spectrum is artificially broadened).
    single_speed_neff_median: float = float("nan")

    def to_dict(self) -> dict:
        return asdict(self)


def _recovered_timescale(tau: float, T: int, n_traj: int, seed: int, bias_correct: bool) -> float:
    X = single_timescale(T=T, timescale=float(tau), n_traj=n_traj, seed=seed)
    C = aggregate_autocorr(X, max_lag=min(600, T // 2), bias_correct=bias_correct)
    grid, w = relaxation_spectrum(C)
    peaks = dominant_timescales(grid, w, rel_thresh=0.08)
    if not peaks:
        return float("nan")
    return float(max(peaks, key=lambda p: p.weight).timescale)


def _single_speed_neff(T: int, n_traj: int, seed: int, bias_correct: bool) -> float:
    X = single_timescale(T=T, timescale=20.0, n_traj=n_traj, seed=seed)
    C = aggregate_autocorr(X, max_lag=min(600, T // 2), bias_correct=bias_correct)
    _grid, w = relaxation_spectrum(C)
    return float(effective_n_timescales(w))


def calibrate(
    injected_taus: tuple[float, ...] = DEFAULT_TAUS,
    *,
    T: int = 2048,
    n_traj: int = 8,
    seeds: tuple[int, ...] = (0, 1, 2, 3, 4, 5, 6, 7),
    bias_correct: bool = True,
    n_boot: int = 400,
    boot_seed: int = 0,
) -> CalibrationCurve:
    """Measure recovered-vs-injected timescale calibration on synthetic data.

    For each injected timescale, a single-timescale ensemble is generated for each
    seed, analysed (optionally with ``bias_correct``), and the dominant recovered
    timescale recorded; the curve reports the median ratio and a bootstrap CI.
    ``single_speed_neff_median`` reports the spectral-width cost of the chosen
    ``bias_correct`` setting on a genuinely single-speed memory.

    This is synthetic by construction (the only place ground-truth timescales are
    known); treat the curve as how to *read* the instrument, not as a correction
    factor to apply.
    """
    rng = np.random.default_rng(boot_seed)
    curve = CalibrationCurve(
        bias_correct=bool(bias_correct),
        T=int(T),
        n_traj=int(n_traj),
        n_boot=int(n_boot),
        boot_seed=int(boot_seed),
    )
    for tau in injected_taus:
        ratios = np.array(
            [_recovered_timescale(tau, T, n_traj, 300 + s, bias_correct) / tau for s in seeds],
            dtype=float,
        )
        valid = ratios[np.isfinite(ratios)]
        if valid.size == 0:
            curve.points.append(
                CalibrationPoint(
                    float(tau), float("nan"), float("nan"), float("nan"), float("nan"), 0
                )
            )
            continue
        med = float(np.median(valid))
        boots = np.array(
            [
                float(np.median(rng.choice(valid, size=valid.size, replace=True)))
                for _ in range(n_boot)
            ]
        )
        lo, hi = (float(x) for x in np.percentile(boots, [2.5, 97.5]))
        curve.points.append(
            CalibrationPoint(
                injected=float(tau),
                recovered_median=float(med * tau),
                ratio_median=med,
                ratio_ci_low=lo,
                ratio_ci_high=hi,
                n_seeds=int(valid.size),
            )
        )
    neffs = [_single_speed_neff(T, n_traj, 400 + s, bias_correct) for s in seeds]
    curve.single_speed_neff_median = float(np.median(neffs))
    return curve
