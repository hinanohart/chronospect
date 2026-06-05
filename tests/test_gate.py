import pytest

from chronospect.sensitivity import run_gate


@pytest.mark.parametrize("seed", [0, 1, 2, 3])
def test_gate_passes(seed):
    result = run_gate(seed=seed)
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
    } <= names
