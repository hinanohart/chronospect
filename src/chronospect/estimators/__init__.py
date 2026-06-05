"""Estimators that turn a memory trajectory into timescale diagnostics."""

from .autocorr import aggregate_autocorr
from .forgetting import ForgettingFit, forgetting_fit
from .observability import capacity_vs_horizon, effective_rank, windowed_effective_rank
from .spectrum import (
    Peak,
    dominant_timescales,
    effective_n_timescales,
    octave_band_energy,
    relaxation_spectrum,
)
from .twotime import aging_index, two_time_correlation

__all__ = [
    "aggregate_autocorr",
    "relaxation_spectrum",
    "dominant_timescales",
    "effective_n_timescales",
    "octave_band_energy",
    "Peak",
    "capacity_vs_horizon",
    "effective_rank",
    "windowed_effective_rank",
    "two_time_correlation",
    "aging_index",
    "forgetting_fit",
    "ForgettingFit",
]
