import pytest

from chronospect.sensitivity import CALIB_HOLDOUT_TAUS, run_gate

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


@pytest.mark.xfail(
    reason="G6 calibration fails on the v0.1 estimator (documents the long-timescale "
    "bias); the demeaning-bias correction lands in the next commit (S3). The "
    "pre-registered G6 band is NOT changed when this turns green.",
    strict=False,
)
def test_full_gate_passes():
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


def test_g6_is_non_vacuous_pre_fix():
    # Before the demeaning-bias correction, G6 must FAIL -- otherwise the gate would
    # be vacuous (passing regardless of the bias it exists to catch). Replaced by a
    # green G6 assertion in S3 once Fix-A lands.
    result = run_gate(seed=0)
    g6 = next(c for c in result.checks if c.name == "G6_calibration_holdout")
    assert not g6.passed, f"G6 unexpectedly passes pre-Fix-A: {g6.detail}"


def test_g7_smoke_passes_with_torch():
    pytest.importorskip("torch")
    result = run_gate(seed=0)
    g7 = next(c for c in result.checks if c.name == "G7_real_model_smoke")
    assert g7.passed, g7.detail


def test_calib_holdout_disjoint_from_g1():
    # Pre-registration guard: hold-out timescales must not overlap G1's (5, 100).
    assert set(CALIB_HOLDOUT_TAUS).isdisjoint({5.0, 100.0})
