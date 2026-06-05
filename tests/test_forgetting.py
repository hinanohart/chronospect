import numpy as np

from chronospect.estimators.forgetting import forgetting_fit


def test_exponential_recognized():
    taus = np.arange(0, 200, dtype=float)
    C = np.exp(-taus / 30.0)
    fit = forgetting_fit(C)
    assert fit.better == "exponential"
    assert abs(fit.exponential_timescale - 30.0) / 30.0 < 0.25


def test_power_law_recognized():
    taus = np.arange(1, 400, dtype=float)
    C = np.concatenate([[1.0], (taus[1:] ** -0.5)])
    fit = forgetting_fit(C)
    assert fit.better == "power_law"
    # Benna-Fusi optimal exponent ~0.5
    assert abs(fit.power_law_exponent - 0.5) < 0.2


def test_too_short_is_inconclusive():
    fit = forgetting_fit(np.array([1.0, 0.5]))
    assert fit.better == "inconclusive"
