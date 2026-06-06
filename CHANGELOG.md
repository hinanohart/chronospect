# Changelog

All notable changes to chronospect are recorded here. Versions follow
[Semantic Versioning](https://semver.org/) (alpha series: `0.x.y`).

## [Unreleased]

### Added
- Pre-registered gate checks `G6` (calibration on **hold-out** timescales,
  disjoint from the `G1` timescales) and `G7` (real-model smoke test, skipped
  when torch is absent) in `chronospect gate`.
- `bias_correct` option on `aggregate_autocorr` (Bartlett-weighted sample-mean
  bias correction; default on) and a `calibrate()` routine that reports the
  recovered-versus-injected timescale curve with bootstrap confidence intervals.
- Exploratory real-model case study (`examples/run_case_study.py`,
  `examples/case_study.md`): trained-versus-untrained memory-trajectory spectra
  for a GRU and a Titans `NeuralMemory`. This is an *exploratory case study, not
  a validation*.
- CI guard against ranking / comparative-performance wording in user-facing docs.

### Changed
- `aggregate_autocorr` applies the demeaning-bias correction by default; the
  previous behaviour is available with `bias_correct=False`. Recovered
  timescales on the calibration grid change from <!--MEASURED@S3:before--> to
  <!--MEASURED@S3:after-->; the residual finite-window shrinkage at long
  timescales is *disclosed as a calibration curve*, not corrected away. The
  instrument reports long timescales as shrinkage-affected estimates / lower
  bounds and does not claim exact point recovery.

### Notes
- chronospect remains a measurement instrument: it characterises the timescale
  structure of a logged memory trajectory and does not rank or improve models.
