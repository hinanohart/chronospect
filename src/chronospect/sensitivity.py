"""The pre-registered sensitivity gate.

Before ``chronospect`` is pointed at any real model it must pass this gate: it
must recover timescales it injected into synthetic data, and must tell a
stationary memory apart from an aging one.  The pass/fail criteria below are
fixed *in advance* (pre-registered) precisely so the instrument cannot be
quietly tuned until the pictures look nice.

Run it with ``chronospect gate`` or :func:`run_gate`.

Pre-registered criteria
-----------------------
G1  multi_timescale(5, 100): the spectrum recovers two peaks, each within a
    factor of ``TIMESCALE_TOL`` of the injected value.
G2  single_timescale(20): the spectrum has exactly one dominant peak and the
    effective number of timescales is < ``SINGLE_NEFF_MAX``.
G3  capacity-vs-horizon: the multi-timescale memory retains more capacity at a
    long horizon than the single fast-timescale memory.
G4  aging: aging_index(aging_process) exceeds aging_index(stationary) by at
    least ``AGING_MARGIN`` and the stationary index is below ``AGING_FLAT_MAX``.
G5  no false aging: a stationary *multi*-timescale memory's aging index stays
    below ``AGING_FLAT_MAX`` (guards against small-ensemble false positives).
G6  calibration on HOLD-OUT timescales (7, 40, 150, 300 -- disjoint from the
    G1 timescales, so calibration cannot be tuned by fitting the G1 grid):
    single-timescale recovery (recovered/injected, median over seeds) lands in
    a pre-registered TWO-SIDED band per timescale, AND a single-timescale memory
    still resolves to exactly one peak (an over-correction is a failure too).
    The bands were fixed BEFORE the v0.2 demeaning-bias correction; with the
    v0.1 estimator the longest hold-out timescale fails the band, which is what
    the correction must lift -- the gate therefore has teeth and is not vacuous.
G7  real-model smoke (torch-gated): an ``nn.GRU`` memory trajectory flows
    through :func:`chronospect.analyze` end to end and yields a finite headline
    and a non-empty verdict. Skipped (and counted as passing) when torch is not
    installed, so the core test matrix stays torch-free.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from .estimators.autocorr import aggregate_autocorr
from .estimators.observability import capacity_vs_horizon
from .estimators.spectrum import (
    dominant_timescales,
    effective_n_timescales,
    relaxation_spectrum,
)
from .estimators.twotime import aging_index, two_time_correlation
from .synthetic import aging_process, multi_timescale, single_timescale

__all__ = ["run_gate", "GateResult"]

# --- pre-registered thresholds (do not tune to make figures pretty) ---
TIMESCALE_TOL = 2.0  # recovered timescale within this multiplicative factor
SINGLE_NEFF_MAX = 1.8  # single-speed memory: effective # timescales below this
AGING_MARGIN = 0.30  # aging index must exceed stationary by at least this
AGING_FLAT_MAX = 0.25  # a stationary memory's aging index must stay below this

# --- G6 calibration thresholds (v0.2; pre-registered at S2 BEFORE the estimator
#     change, informed by a READ-ONLY measurement of the v0.1 estimator, and never
#     loosened afterwards). Ratio = recovered_dominant_timescale / injected, median
#     over CALIB_SEEDS. Bands are TWO-SIDED: an over-correction that overshoots a
#     timescale (or injects a spurious peak) must also fail. The hold-out timescales
#     are disjoint from the G1 timescales (5, 100) so the calibration cannot be tuned
#     by fitting the G1 grid. With the v0.1 estimator the longest hold-out timescale
#     (300) recovers ~0.50 and FAILS its band below -- that gap is exactly what the
#     v0.2 demeaning-bias correction must close, which keeps G6 non-vacuous. ---
CALIB_HOLDOUT_TAUS = (7.0, 40.0, 150.0, 300.0)
CALIB_T = 2048
CALIB_SEEDS = (0, 1, 2)
CALIB_BANDS: dict[float, tuple[float, float]] = {
    7.0: (0.60, 1.50),  # short: should stay ~1.0; band guards against distortion
    40.0: (0.60, 1.50),
    150.0: (0.55, 1.50),
    300.0: (0.58, 1.45),  # binding: v0.1 ~0.50 (fails); correction must reach >= 0.58
}


@dataclass
class Check:
    name: str
    passed: bool
    detail: str


@dataclass
class GateResult:
    passed: bool
    checks: list[Check] = field(default_factory=list)

    def __str__(self) -> str:  # pragma: no cover - cosmetic
        lines: list[str] = [f"GATE {'PASS' if self.passed else 'FAIL'}"]
        for c in self.checks:
            lines.append(f"  [{'ok' if c.passed else 'XX'}] {c.name}: {c.detail}")
        return "\n".join(lines)


def _within(value: float, target: float, tol: float) -> bool:
    if value <= 0 or target <= 0:
        return False
    ratio = value / target
    return (1.0 / tol) <= ratio <= tol


def run_gate(seed: int = 0, *, T: int = 2048) -> GateResult:
    """Run all pre-registered checks; return a :class:`GateResult`.

    ``T`` is the synthetic trajectory length; horizons and waiting times scale
    with it.  The pre-registered thresholds were fixed at the default ``T=2048``;
    very small ``T`` is rejected because the long-horizon checks are meaningless
    when there is no long horizon.
    """
    if T < 512:
        raise ValueError("run_gate needs T >= 512 (long-horizon checks need a long horizon)")
    checks: list[Check] = []

    # ---- G1: recover two injected timescales ----
    true_ts = (5.0, 100.0)
    Xm = multi_timescale(T=T, timescales=true_ts, n_traj=8, seed=seed)
    Cm = aggregate_autocorr(Xm, max_lag=min(600, T // 2))
    grid, w = relaxation_spectrum(Cm)
    peaks = dominant_timescales(grid, w, rel_thresh=0.08)
    recovered = sorted(p.timescale for p in peaks)
    g1 = len(peaks) >= 2 and all(
        any(_within(r, t, TIMESCALE_TOL) for r in recovered) for t in true_ts
    )
    checks.append(
        Check(
            "G1_recover_two_timescales",
            bool(g1),
            f"injected={true_ts} recovered={[round(r, 1) for r in recovered]}",
        )
    )

    # ---- G2: single timescale -> one peak ----
    Xs = single_timescale(T=T, timescale=20.0, n_traj=8, seed=seed + 1)
    Cs = aggregate_autocorr(Xs, max_lag=min(600, T // 2))
    grid_s, w_s = relaxation_spectrum(Cs)
    peaks_s = dominant_timescales(grid_s, w_s, rel_thresh=0.08)
    neff_s = effective_n_timescales(w_s)
    g2 = len(peaks_s) == 1 and neff_s < SINGLE_NEFF_MAX
    checks.append(
        Check(
            "G2_single_is_single",
            bool(g2),
            f"n_peaks={len(peaks_s)} n_eff={neff_s:.2f} (<{SINGLE_NEFF_MAX})",
        )
    )

    # ---- G3: multi-timescale retains capacity at a long horizon ----
    long_h = min(200, T // 4)
    _, cap_m = capacity_vs_horizon(Xm, max_lag=long_h)
    Xs_fast = single_timescale(T=T, timescale=5.0, n_traj=8, seed=seed + 2)
    _, cap_s = capacity_vs_horizon(Xs_fast, max_lag=long_h)
    g3 = cap_m[long_h] > cap_s[long_h]
    checks.append(
        Check(
            "G3_capacity_horizon",
            bool(g3),
            f"cap_multi[{long_h}]={cap_m[long_h]:.3f} > cap_fast[{long_h}]={cap_s[long_h]:.3f}",
        )
    )

    # ---- G4: aging detected on a genuinely aging memory (realistic small ensemble) ----
    t_ws = (np.linspace(0.05, 0.78, 5) * T).astype(int)
    taus = np.arange(0, min(250, T // 8), 3)
    tw_window = min(64, T // 32)
    Xa = aging_process(T=T, n_traj=8, seed=seed + 3)
    Ca = two_time_correlation(Xa, t_ws, taus, tw_window=tw_window)
    ai_age = aging_index(Ca, t_ws, taus)
    Xstat = single_timescale(T=T, timescale=20.0, n_traj=8, seed=seed + 4)
    Cstat = two_time_correlation(Xstat, t_ws, taus, tw_window=tw_window)
    ai_stat = aging_index(Cstat, t_ws, taus)
    g4 = (
        np.isfinite(ai_age)
        and np.isfinite(ai_stat)
        and ai_stat < AGING_FLAT_MAX
        and (ai_age - ai_stat) >= AGING_MARGIN
    )
    checks.append(
        Check(
            "G4_aging_detected",
            bool(g4),
            f"aging={ai_age:.3f} stationary={ai_stat:.3f} (margin>={AGING_MARGIN})",
        )
    )

    # ---- G5: no false aging on a stationary *multi*-timescale memory (small ensemble) ----
    # This is the guard against calling a legitimately multi-speed but stationary
    # memory "aging" just because the ensemble is small.
    Cmt = two_time_correlation(Xm, t_ws, taus, tw_window=tw_window)
    ai_multi = aging_index(Cmt, t_ws, taus)
    g5 = np.isfinite(ai_multi) and ai_multi < AGING_FLAT_MAX
    checks.append(
        Check(
            "G5_no_false_aging",
            bool(g5),
            f"stationary multi-timescale aging={ai_multi:.3f} (<{AGING_FLAT_MAX})",
        )
    )

    # ---- G6: calibration accuracy on HOLD-OUT timescales (v0.2) ----
    # Single-timescale recovery on timescales disjoint from G1's. Uses the same
    # aggregate_autocorr the instrument uses, so the v0.2 demeaning-bias correction
    # (default-on) flows through here automatically once it lands.
    g6_ratios: dict[float, float] = {}
    g6_parts: list[bool] = []
    for tau in CALIB_HOLDOUT_TAUS:
        rs: list[float] = []
        for i in range(len(CALIB_SEEDS)):
            Xt = single_timescale(T=CALIB_T, timescale=tau, n_traj=8, seed=200 + i)
            Ct = aggregate_autocorr(Xt, max_lag=min(600, CALIB_T // 2))
            gt, wt = relaxation_spectrum(Ct)
            pks = dominant_timescales(gt, wt, rel_thresh=0.08)
            rs.append(max(pks, key=lambda p: p.weight).timescale / tau if pks else np.nan)
        med = float(np.nanmedian(rs))
        g6_ratios[tau] = med
        lo, hi = CALIB_BANDS[tau]
        g6_parts.append(bool(np.isfinite(med) and lo <= med <= hi))
    # over-correction guard: a single-timescale memory must still resolve to one peak
    Xsg = single_timescale(T=CALIB_T, timescale=20.0, n_traj=8, seed=250)
    gsg, wsg = relaxation_spectrum(aggregate_autocorr(Xsg, max_lag=min(600, CALIB_T // 2)))
    g6_one_peak = len(dominant_timescales(gsg, wsg, rel_thresh=0.08)) == 1
    g6 = all(g6_parts) and g6_one_peak
    checks.append(
        Check(
            "G6_calibration_holdout",
            bool(g6),
            "recovered/injected "
            + ", ".join(
                f"{int(t)}:{g6_ratios[t]:.2f}[{CALIB_BANDS[t][0]},{CALIB_BANDS[t][1]}]"
                for t in CALIB_HOLDOUT_TAUS
            )
            + f" single20_one_peak={g6_one_peak}",
        )
    )

    # ---- G7: real-model smoke (torch-gated) ----
    # Skipped (counted as passing) without torch so the core matrix stays torch-free.
    try:
        import torch

        from .loggers import record_rnn_states
        from .report import analyze as _analyze

        torch.manual_seed(seed)
        gru = torch.nn.GRU(8, 16)
        states = record_rnn_states(gru, torch.randn(512, 8))
        rep = _analyze(states)
        g7 = (
            bool(np.isfinite(rep.effective_n_timescales))
            and isinstance(rep.verdict, str)
            and len(rep.verdict) > 0
        )
        g7_detail = (
            f"GRU(8,16) smoke: n_ts={rep.n_dominant_timescales} "
            f"neff={rep.effective_n_timescales} verdict={rep.verdict!r}"
        )
    except ImportError:
        g7, g7_detail = True, "skipped (torch not installed)"
    except Exception as exc:  # a real pipeline failure is reported, not hidden
        g7, g7_detail = False, f"FAILED: {type(exc).__name__}: {exc}"
    checks.append(Check("G7_real_model_smoke", bool(g7), g7_detail))

    passed = all(c.passed for c in checks)
    return GateResult(passed=passed, checks=checks)
