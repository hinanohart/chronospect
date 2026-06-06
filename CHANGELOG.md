# Changelog

All notable changes to chronospect are recorded here. Versions follow
[Semantic Versioning](https://semver.org/) (alpha series: `0.x.y`).

## [0.2.0a1]

Adds calibration disclosure and an exploratory real-model case study. **No
default behaviour changes**: the v0.1 readings are reproduced exactly by the
defaults.

### Added
- Pre-registered gate checks `G6` (calibration on **hold-out** timescales
  `(7, 40, 150, 300)`, disjoint from the `G1` timescales so calibration cannot be
  tuned to the gate grid) and `G7` (torch-gated real-model smoke, skipped and
  counted as passing when torch is absent) in `chronospect gate`. The
  pre-registered criteria docstring is corrected to cover G1–G7.
- An **opt-in** `bias_correct` flag on `aggregate_autocorr` (and on `analyze`):
  a Bartlett-weighted sample-mean bias correction that lifts long-timescale
  recovery (synthetic injected τ=300 improves from recovered/τ 0.45 to 0.61) at
  the cost of broadening a single-speed spectrum (effective number of timescales
  1.22 → 1.55). It is **off by default**.
- `calibrate()` (with `CalibrationCurve`, `CalibrationPoint`): reports the
  recovered-versus-injected timescale curve with bootstrap confidence intervals
  and the disclosed single-speed cost (`results/calibration_v0.2.json`).
- `record_titans_memory` logger (records a chunked test-time memory module's
  per-step readout) and top-level exports of the trajectory loggers
  (`record_rnn_states`, `TrajectoryRecorder`, `from_snapshots`).
- Exploratory real-model case study (`examples/run_case_study.py`,
  `examples/case_study.md`, `results/realmodel_v0.2.json`): trained-versus-
  untrained memory-trajectory spectra for a GRU and a Titans `NeuralMemory`. This
  is an *exploratory case study, not a validation*, and makes no cross-model
  comparison.
- CI guard against ranking / comparative-performance wording in user-facing docs.

### Changed
- No default behaviour change. The demeaning-bias correction is **opt-in**
  (`bias_correct=True`); by default `aggregate_autocorr` behaves exactly as in
  v0.1, so existing readings are reproduced. The residual finite-window shrinkage
  at long timescales is **disclosed as a calibration curve**, not corrected away:
  long timescales are reported as shrinkage-affected estimates / lower bounds,
  never as exact point recovery.

### Notes
- chronospect remains a measurement instrument: it characterises the timescale
  structure of a logged memory trajectory and does not rank or improve models.
