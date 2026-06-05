import numpy as np

from chronospect.estimators.autocorr import aggregate_autocorr
from chronospect.estimators.spectrum import (
    dominant_timescales,
    effective_n_timescales,
    octave_band_energy,
    relaxation_spectrum,
)
from chronospect.synthetic import multi_timescale, single_timescale


def _within_factor(a, b, factor):
    return (1.0 / factor) <= (a / b) <= factor


def test_single_timescale_one_peak():
    X = single_timescale(T=2048, timescale=20.0, n_traj=8, seed=0)
    C = aggregate_autocorr(X, max_lag=400)
    grid, w = relaxation_spectrum(C)
    peaks = dominant_timescales(grid, w, rel_thresh=0.08)
    assert len(peaks) == 1
    assert _within_factor(peaks[0].timescale, 20.0, 2.0)
    assert effective_n_timescales(w) < 1.8


def test_multi_timescale_recovered():
    X = multi_timescale(T=2048, timescales=(5.0, 100.0), n_traj=8, seed=0)
    C = aggregate_autocorr(X, max_lag=600)
    grid, w = relaxation_spectrum(C)
    peaks = dominant_timescales(grid, w, rel_thresh=0.08)
    recovered = sorted(p.timescale for p in peaks)
    assert len(peaks) >= 2
    for truth in (5.0, 100.0):
        assert any(_within_factor(r, truth, 2.0) for r in recovered), recovered


def test_autocorr_starts_at_one():
    X = single_timescale(T=512, timescale=10.0, n_traj=4, seed=1)
    C = aggregate_autocorr(X, max_lag=100)
    assert abs(C[0] - 1.0) < 1e-9
    assert np.all(C[1:] < 1.0 + 1e-6)


def test_octave_band_energy_sums_to_one():
    X = single_timescale(T=1024, timescale=15.0, n_traj=4, seed=2)
    obe = octave_band_energy(X)
    assert abs(obe.sum() - 1.0) < 1e-9
    assert np.all(obe >= 0)
