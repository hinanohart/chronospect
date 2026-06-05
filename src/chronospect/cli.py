"""``chronospect`` command-line interface.

    chronospect gate                 # run the pre-registered sensitivity gate
    chronospect demo                 # analyze a synthetic multi-timescale memory
    chronospect analyze traj.npz     # analyze your own trajectory

``analyze`` reads a ``.npz``/``.npy`` file holding an array shaped ``(T, d)`` or
``(n, T, d)`` (for ``.npz``, key ``X`` if present, else the first array).
"""

from __future__ import annotations

import argparse
import json
import sys

import numpy as np


def _load_array(path: str) -> np.ndarray:
    # dispatch on what np.load actually returns, not on the file extension,
    # so a renamed archive or a plain .npy named .npz still loads correctly.
    obj = np.load(path, allow_pickle=False)
    if isinstance(obj, np.lib.npyio.NpzFile):
        key = "X" if "X" in obj.files else obj.files[0]
        return np.asarray(obj[key], dtype=float)
    return np.asarray(obj, dtype=float)


def _cmd_gate(args: argparse.Namespace) -> int:
    from .sensitivity import run_gate

    result = run_gate(seed=args.seed)
    print(result)
    return 0 if result.passed else 1


def _cmd_demo(args: argparse.Namespace) -> int:
    from .report import analyze
    from .synthetic import multi_timescale

    X = multi_timescale(
        T=args.length, timescales=(5.0, 100.0), n_traj=args.ensemble, seed=args.seed
    )
    report = analyze(X)
    print("# synthetic memory with injected timescales 5 and 100")
    print(json.dumps(_summary(report), indent=2))
    return 0


def _cmd_analyze(args: argparse.Namespace) -> int:
    from .report import analyze

    X = _load_array(args.path)
    report = analyze(X, max_lag=args.max_lag)
    out = report.to_dict() if args.full else _summary(report)
    print(json.dumps(out, indent=2))
    return 0


def _summary(report) -> dict:
    return {
        "verdict": report.verdict,
        "dominant_timescales": report.dominant_timescales,
        "n_dominant_timescales": report.n_dominant_timescales,
        "effective_n_timescales": report.effective_n_timescales,
        "capacity_horizon_half": report.capacity_horizon_half,
        "aging_index": report.aging_index,
        "forgetting_better": report.forgetting.better,
        "power_law_exponent": round(report.forgetting.power_law_exponent, 3),
        "octave_band_energy": report.octave_band_energy,
    }


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="chronospect",
        description="Measure the timescale spectrum and aging of a model's memory.",
    )
    sub = p.add_subparsers(dest="command", required=True)

    g = sub.add_parser("gate", help="run the pre-registered sensitivity gate")
    g.add_argument("--seed", type=int, default=0)
    g.set_defaults(func=_cmd_gate)

    d = sub.add_parser("demo", help="analyze a synthetic multi-timescale memory")
    d.add_argument("--seed", type=int, default=0)
    d.add_argument("--length", type=int, default=2048)
    d.add_argument("--ensemble", type=int, default=8)
    d.set_defaults(func=_cmd_demo)

    a = sub.add_parser("analyze", help="analyze a trajectory file (.npz/.npy)")
    a.add_argument("path")
    a.add_argument("--max-lag", type=int, default=None, dest="max_lag")
    a.add_argument("--full", action="store_true", help="emit raw curves too")
    a.set_defaults(func=_cmd_analyze)
    return p


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    return int(args.func(args))


if __name__ == "__main__":  # pragma: no cover
    sys.exit(main())
