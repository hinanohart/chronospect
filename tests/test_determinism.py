"""Honest-marketing guard: every metric is a deterministic function of its input.

If any estimator secretly injected randomness (e.g. ``np.random`` in a return
path), analysing the *same* trajectory twice would disagree.  Reported numbers
must come from the data, not from a random number generator.
"""

import numpy as np

from chronospect import analyze
from chronospect.synthetic import multi_timescale


def test_analyze_is_deterministic():
    X = multi_timescale(T=1500, timescales=(7.0, 80.0), n_traj=6, seed=11)
    a = analyze(X).to_dict()
    b = analyze(X).to_dict()
    assert a == b


def test_synthetic_seed_reproducible():
    x1 = multi_timescale(T=512, timescales=(5.0, 50.0), n_traj=3, seed=2)
    x2 = multi_timescale(T=512, timescales=(5.0, 50.0), n_traj=3, seed=2)
    assert np.array_equal(x1, x2)


def test_different_seed_differs():
    x1 = multi_timescale(T=512, timescales=(5.0, 50.0), n_traj=3, seed=2)
    x2 = multi_timescale(T=512, timescales=(5.0, 50.0), n_traj=3, seed=3)
    assert not np.array_equal(x1, x2)
