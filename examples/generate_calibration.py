"""Regenerate ``results/calibration_v0.2.json`` -- the provenance for the
calibration numbers quoted in the README and CHANGELOG.

This is fully deterministic (fixed data seeds and a fixed bootstrap seed), so the
recovered-vs-injected numbers reproduce exactly on a clean clone; only the ``env``
timestamp/versions differ. Run from the repo root:

    python examples/generate_calibration.py
"""

from __future__ import annotations

import datetime as _dt
import json
import platform
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path

import numpy
import scipy

import chronospect as cs

INJECTED = (7.0, 20.0, 50.0, 100.0, 200.0, 300.0)
T = 2048
N_TRAJ = 8
SEEDS = (0, 1, 2, 3, 4, 5, 6, 7)
N_BOOT = 400
BOOT_SEED = 0


def _pkg(name: str) -> str:
    try:
        return version(name)
    except PackageNotFoundError:
        return "unknown"


def main() -> None:
    off = cs.calibrate(
        INJECTED,
        T=T,
        n_traj=N_TRAJ,
        seeds=SEEDS,
        bias_correct=False,
        n_boot=N_BOOT,
        boot_seed=BOOT_SEED,
    )
    on = cs.calibrate(
        INJECTED,
        T=T,
        n_traj=N_TRAJ,
        seeds=SEEDS,
        bias_correct=True,
        n_boot=N_BOOT,
        boot_seed=BOOT_SEED,
    )

    longest = INJECTED[-1]
    off_long = next(p for p in off.points if p.injected == longest)
    on_long = next(p for p in on.points if p.injected == longest)

    payload = {
        "schema": "chronospect-calibration-v0.2",
        "generated_by": "examples/generate_calibration.py",
        "note": (
            "Synthetic single-timescale recovery. Numbers are reproducible; treat the "
            "recovered/injected ratio as how to READ the instrument (long timescales are "
            "shrinkage-affected estimates / lower bounds), not as a rescaling factor."
        ),
        "env": {
            "utc": _dt.datetime.now(_dt.timezone.utc).isoformat(timespec="seconds"),
            "python": platform.python_version(),
            "numpy": numpy.__version__,
            "scipy": scipy.__version__,
            "pywavelets": _pkg("PyWavelets"),
            "chronospect": cs.__version__,
        },
        "settings": {
            "injected_taus": list(INJECTED),
            "T": T,
            "n_traj": N_TRAJ,
            "seeds": list(SEEDS),
            "n_boot": N_BOOT,
            "boot_seed": BOOT_SEED,
        },
        "curves": {"bias_correct_off": off.to_dict(), "bias_correct_on": on.to_dict()},
        "summary": {
            "single_speed_neff_median": {
                "off": round(off.single_speed_neff_median, 4),
                "on": round(on.single_speed_neff_median, 4),
            },
            "longest_injected_tau": longest,
            "longest_tau_ratio_median": {
                "off": round(off_long.ratio_median, 4),
                "on": round(on_long.ratio_median, 4),
            },
        },
    }

    out = Path(__file__).resolve().parent.parent / "results" / "calibration_v0.2.json"
    out.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(f"wrote {out}")
    s = payload["summary"]
    print(
        f"longest tau ({longest}) recovered ratio: off={s['longest_tau_ratio_median']['off']} "
        f"on={s['longest_tau_ratio_median']['on']}"
    )
    print(
        f"single-speed n_eff: off={s['single_speed_neff_median']['off']} "
        f"on={s['single_speed_neff_median']['on']}"
    )


if __name__ == "__main__":
    main()
