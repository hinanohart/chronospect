import numpy as np

from chronospect.estimators.twotime import aging_index, two_time_correlation
from chronospect.synthetic import aging_process, multi_timescale, single_timescale

T_WS = np.array([100, 400, 800, 1200, 1600])
TAUS = np.arange(0, 250, 3)


def _ai(X):
    C = two_time_correlation(X, T_WS, TAUS, tw_window=64)
    return aging_index(C, T_WS, TAUS)


def test_two_time_shape_and_origin():
    X = single_timescale(T=2048, timescale=20.0, n_traj=8, seed=0)
    C = two_time_correlation(X, T_WS, TAUS, tw_window=64)
    assert C.shape == (len(T_WS), len(TAUS))
    # correlation at tau=0 is ~1
    assert np.all(np.abs(C[:, 0] - 1.0) < 1e-6)


def test_aging_detected():
    ai = _ai(aging_process(T=2048, n_traj=8, seed=3))
    assert np.isfinite(ai) and ai >= 0.3


def test_stationary_single_not_aging():
    ai = _ai(single_timescale(T=2048, timescale=20.0, n_traj=8, seed=4))
    assert ai == 0.0  # significance gate -> exactly 0 when no real trend


def test_stationary_multi_not_aging():
    # the key false-positive guard: multi-speed but stationary must NOT read as aging
    ai = _ai(multi_timescale(T=2048, timescales=(5.0, 100.0), n_traj=8, seed=5))
    assert ai < 0.25
