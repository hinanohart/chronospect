import pytest

from chronospect.synthetic import aging_process, multi_timescale, single_timescale


@pytest.mark.parametrize(
    "d,timescales",
    [(24, (5.0, 100.0)), (4, (2.0, 5.0, 10.0)), (3, (1.0, 2.0, 3.0)), (10, (5.0,))],
)
def test_multi_timescale_dim_count_exact(d, timescales):
    X = multi_timescale(T=128, d=d, timescales=timescales, n_traj=2, seed=0)
    assert X.shape == (2, 128, d)


def test_multi_timescale_rejects_more_timescales_than_dims():
    with pytest.raises(ValueError):
        multi_timescale(T=128, d=2, timescales=(1.0, 2.0, 3.0, 4.0, 5.0))


def test_single_timescale_unit_variance():
    X = single_timescale(T=4096, timescale=20.0, n_traj=4, seed=0)
    # stationary variance is ~sigma**2 == 1 regardless of timescale
    assert abs(X.var() - 1.0) < 0.2


def test_aging_process_variance_is_controlled():
    # variance should stay roughly flat across time (aging is a timescale ramp,
    # not an amplitude ramp)
    X = aging_process(T=4096, d=8, n_traj=8, seed=0)
    early = X[:, :500, :].var()
    late = X[:, -500:, :].var()
    assert late / early < 2.5  # was ~9x before the fix
