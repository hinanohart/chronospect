import numpy as np

from chronospect.estimators.observability import (
    capacity_vs_horizon,
    effective_rank,
    windowed_effective_rank,
)
from chronospect.synthetic import multi_timescale, single_timescale


def test_effective_rank_basics():
    assert abs(effective_rank(np.array([1.0, 0, 0, 0])) - 1.0) < 1e-9
    # k equal singular values -> effective rank k
    assert abs(effective_rank(np.ones(4)) - 4.0) < 1e-9
    assert effective_rank(np.array([])) == 0.0


def test_capacity_starts_high_decays():
    X = single_timescale(T=2048, timescale=10.0, n_traj=8, seed=0)
    lags, cap = capacity_vs_horizon(X, max_lag=200)
    assert cap[0] > cap[-1]  # capacity fades with horizon
    assert cap[0] > 5.0  # ~ d independent directions at lag 0


def test_multi_retains_more_than_fast_at_horizon():
    Xm = multi_timescale(T=2048, timescales=(5.0, 100.0), n_traj=8, seed=0)
    Xf = single_timescale(T=2048, timescale=5.0, n_traj=8, seed=1)
    _, cap_m = capacity_vs_horizon(Xm, max_lag=200)
    _, cap_f = capacity_vs_horizon(Xf, max_lag=200)
    assert cap_m[200] > cap_f[200]


def test_windowed_effective_rank_shape():
    X = single_timescale(T=512, timescale=8.0, n_traj=4, seed=2)
    centers, ranks = windowed_effective_rank(X, window=64, step=32)
    assert centers.shape == ranks.shape
    assert np.all(ranks > 0)
