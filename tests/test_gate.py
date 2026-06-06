import numpy as np
import pytest

from chronospect.estimators.autocorr import aggregate_autocorr
from chronospect.estimators.spectrum import dominant_timescales, relaxation_spectrum
from chronospect.sensitivity import (
    CALIB_BANDS,
    CALIB_HOLDOUT_TAUS,
    CALIB_SEEDS,
    CALIB_T,
    run_gate,
)
from chronospect.synthetic import single_timescale

CORE = ("G1", "G2", "G3", "G4", "G5")


@pytest.mark.parametrize("seed", [0, 1, 2, 3])
def test_core_gate_passes(seed):
    # The v0.1 core checks (G1-G5) must hold across seeds; the v0.2 calibration
    # correction must not regress them. (Replaces the v0.1 full-gate parametrized
    # test: same checks, finer granularity.)
    result = run_gate(seed=seed)
    core = [c for c in result.checks if c.name[:2] in CORE]
    assert core, str(result)
    assert all(c.passed for c in core), str(result)


def test_full_gate_passes():
    # All seven pre-registered checks pass once the S3 demeaning-bias correction
    # lands (G6 enables it explicitly). The pre-registered G6 band was NOT changed
    # to make this green -- see test_g6_passes_and_has_teeth.
    result = run_gate(seed=0)
    assert result.passed, str(result)


def test_gate_has_all_checks():
    result = run_gate(seed=0)
    names = {c.name for c in result.checks}
    assert {
        "G1_recover_two_timescales",
        "G2_single_is_single",
        "G3_capacity_horizon",
        "G4_aging_detected",
        "G5_no_false_aging",
        "G6_calibration_holdout",
        "G7_real_model_smoke",
    } <= names


def test_g6_passes_and_has_teeth():
    # G6 enables the demeaning-bias correction and must now PASS...
    result = run_gate(seed=0)
    g6 = next(c for c in result.checks if c.name == "G6_calibration_holdout")
    assert g6.passed, f"G6 should pass once Fix-A lands: {g6.detail}"

    # ...but the gate must still be non-vacuous: WITHOUT the correction the longest
    # hold-out timescale recovers BELOW its pre-registered band, so passing G6
    # genuinely requires the correction (the band is not loose enough to pass either
    # way). This reproduces G6's recovery computation with bias_correct=False.
    tau = 300.0
    lo, _hi = CALIB_BANDS[tau]
    rs: list[float] = []
    for i in range(len(CALIB_SEEDS)):
        X = single_timescale(T=CALIB_T, timescale=tau, n_traj=8, seed=200 + i)
        C = aggregate_autocorr(X, max_lag=min(600, CALIB_T // 2), bias_correct=False)
        grid, w = relaxation_spectrum(C)
        pks = dominant_timescales(grid, w, rel_thresh=0.08)
        rs.append(max(pks, key=lambda p: p.weight).timescale / tau if pks else np.nan)
    uncorrected = float(np.nanmedian(rs))
    assert uncorrected < lo, (
        f"gate is vacuous: uncorrected tau={tau} recovery {uncorrected:.3f} "
        f"already inside band [{lo}, _]"
    )


def test_g7_smoke_passes_with_torch():
    pytest.importorskip("torch")
    result = run_gate(seed=0)
    g7 = next(c for c in result.checks if c.name == "G7_real_model_smoke")
    assert g7.passed, g7.detail


def test_calib_holdout_disjoint_from_g1():
    # Pre-registration guard: hold-out timescales must not overlap G1's (5, 100).
    assert set(CALIB_HOLDOUT_TAUS).isdisjoint({5.0, 100.0})
