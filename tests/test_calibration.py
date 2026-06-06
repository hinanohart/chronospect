"""S3 tests: the demeaning-bias correction (Fix-A) and the calibration
disclosure (Fix-B).

These pin down the three claims the v0.2 calibration work makes:
* enabling the correction lifts long-timescale recovery (without overshooting);
* it is OFF by default, so the v0.1 single-vs-multi headline is unchanged;
* :func:`chronospect.calibrate` reports both the benefit and the disclosed cost
  (single-speed spectral-width inflation), and is reproducible.
"""

from __future__ import annotations

import numpy as np

import chronospect as cs
from chronospect.estimators.autocorr import aggregate_autocorr
from chronospect.estimators.spectrum import dominant_timescales, relaxation_spectrum
from chronospect.synthetic import single_timescale

_LONG_TAU = 300.0
_SEEDS = (300, 301, 302)


def _recovered_ratio(tau: float, *, bias_correct: bool) -> float:
    """Median recovered/injected ratio over a few seeds (robust to one noisy seed)."""
    rs: list[float] = []
    for seed in _SEEDS:
        X = single_timescale(T=2048, timescale=tau, n_traj=8, seed=seed)
        C = aggregate_autocorr(X, max_lag=600, bias_correct=bias_correct)
        grid, w = relaxation_spectrum(C)
        peaks = dominant_timescales(grid, w, rel_thresh=0.08)
        assert peaks, f"no peak recovered for tau={tau}, bias_correct={bias_correct}"
        rs.append(max(peaks, key=lambda p: p.weight).timescale / tau)
    return float(np.median(rs))


def test_bias_correct_lifts_long_timescale_recovery():
    # Fix-A: the long timescale is under-recovered without correction; the
    # correction moves the estimate UP toward the truth, but must not overshoot.
    off = _recovered_ratio(_LONG_TAU, bias_correct=False)
    on = _recovered_ratio(_LONG_TAU, bias_correct=True)
    assert off < 0.55, f"uncorrected long-tau recovery unexpectedly high: {off:.3f}"
    assert on > off, f"correction did not lift recovery: off={off:.3f} on={on:.3f}"
    assert on <= 1.45, f"correction overshoots the injected timescale: {on:.3f}"


def test_analyze_default_is_uncorrected():
    # Fix-A is opt-in: analyze()'s default must reproduce the v0.1 (uncorrected)
    # autocovariance path exactly, so existing users' headline is unchanged.
    X = single_timescale(T=1024, timescale=40.0, n_traj=6, seed=7)
    default = cs.analyze(X)
    off = cs.analyze(X, bias_correct=False)
    assert default.dominant_timescales == off.dominant_timescales
    assert default.effective_n_timescales == off.effective_n_timescales
    assert default.autocorr == off.autocorr


def test_analyze_bias_correct_changes_autocorr():
    # Sanity: the flag actually does something (the corrected autocovariance differs).
    X = single_timescale(T=1024, timescale=40.0, n_traj=6, seed=7)
    off = cs.analyze(X, bias_correct=False)
    on = cs.analyze(X, bias_correct=True)
    assert off.autocorr != on.autocorr


def test_calibrate_discloses_benefit_and_cost():
    taus = (40.0, _LONG_TAU)
    common = dict(T=2048, n_traj=8, seeds=(0, 1, 2), n_boot=64, boot_seed=0)
    off = cs.calibrate(taus, bias_correct=False, **common)
    on = cs.calibrate(taus, bias_correct=True, **common)

    for curve in (off, on):
        assert len(curve.points) == len(taus)
        for p in curve.points:
            assert np.isfinite(p.ratio_median)
            assert p.ratio_ci_low <= p.ratio_median <= p.ratio_ci_high
        assert np.isfinite(curve.single_speed_neff_median)

    # disclosed benefit: long-tau recovery improves when the correction is on
    off_long = next(p for p in off.points if p.injected == _LONG_TAU)
    on_long = next(p for p in on.points if p.injected == _LONG_TAU)
    assert on_long.ratio_median > off_long.ratio_median

    # disclosed cost: a genuinely single-speed memory's spectrum is broadened by
    # the correction (this is exactly why Fix-A is opt-in, not default-on).
    assert on.single_speed_neff_median >= off.single_speed_neff_median


def test_calibrate_is_reproducible():
    common = dict(T=1024, n_traj=4, seeds=(0, 1), bias_correct=True, n_boot=32, boot_seed=0)
    a = cs.calibrate((50.0,), **common)
    b = cs.calibrate((50.0,), **common)
    assert a.to_dict() == b.to_dict()
