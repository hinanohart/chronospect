"""chronospect -- measure the timescale spectrum and aging of a model's memory.

A model that *claims* multiple memory speeds (Titans, HOPE / Nested Learning,
Continuum Memory Systems, test-time learners) may secretly collapse to a single
effective timescale.  ``chronospect`` is a CPU-only measurement instrument that
reads this off a logged memory trajectory:

* a relaxation-timescale **spectrum** (which timescales are actually present),
* a memory-**capacity-versus-horizon** curve (how far back state stays useful),
* a two-time **aging** index (is the memory stationary or coarsening?), and
* a Benna-Fusi forgetting **yardstick** (power-law vs exponential decay).

Start with :func:`analyze`; validate the instrument with :func:`run_gate`.
"""

from __future__ import annotations

from .calibration import CalibrationCurve, CalibrationPoint, calibrate
from .estimators import (
    ForgettingFit,
    Peak,
    aggregate_autocorr,
    aging_index,
    capacity_vs_horizon,
    dominant_timescales,
    effective_n_timescales,
    effective_rank,
    forgetting_fit,
    octave_band_energy,
    relaxation_spectrum,
    two_time_correlation,
    windowed_effective_rank,
)
from .loggers import (
    TrajectoryRecorder,
    from_snapshots,
    record_rnn_states,
    record_titans_memory,
)
from .report import TimescaleReport, analyze
from .sensitivity import GateResult, run_gate

__version__ = "0.2.0a1"

__all__ = [
    "analyze",
    "TimescaleReport",
    "run_gate",
    "GateResult",
    "calibrate",
    "CalibrationCurve",
    "CalibrationPoint",
    "TrajectoryRecorder",
    "record_rnn_states",
    "record_titans_memory",
    "from_snapshots",
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
    "__version__",
]
