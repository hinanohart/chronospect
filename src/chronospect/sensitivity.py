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
        lines = [f"GATE {'PASS' if self.passed else 'FAIL'}"]
        for c in self.checks:
            lines.append(f"  [{'ok' if c.passed else 'XX'}] {c.name}: {c.detail}")
        return "\n".join(lines)


def _within(value: float, target: float, tol: float) -> bool:
    if value <= 0 or target <= 0:
        return False
    ratio = value / target
    return (1.0 / tol) <= ratio <= tol


def run_gate(seed: int = 0, *, T: int = 2048) -> GateResult:
    """Run all pre-registered checks; return a :class:`GateResult`."""
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
    long_h = 200
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
    t_ws = np.array([100, 400, 800, 1200, 1600])
    taus = np.arange(0, 250, 3)
    tw_window = 64
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

    passed = all(c.passed for c in checks)
    return GateResult(passed=passed, checks=checks)
