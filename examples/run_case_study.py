"""Exploratory real-model case study (NOT validation).

This script trains two small memory models on a toy next-step-prediction task
whose target mixes a fast and a slow timescale, then uses ``chronospect`` to read
the timescale spectrum of each model's memory trajectory *before* and *after*
training. The point is narrow and honest:

  * it shows the instrument **responds to learning** -- the recovered spectrum
    changes once the model has been trained on structured data (a forward pass on
    an untrained model carries no task structure, so a before/after change is the
    evidence that the readings are not an artefact of the probe);
  * it is **exploratory**: a single tiny config on CPU, synthetic toy data, two
    architectures looked at independently. It is NOT a benchmark, NOT a validation
    of either model, and makes NO quality comparison between the two models.

Each model is reported on its own terms (observed timescale structure only).
There is deliberately no cross-model comparison or ranking.

Reproducibility: fixed seeds make the numbers deterministic on a given
torch/titans build; the exact versions are stamped into the output JSON. Run from
the repo root:

    pip install -e ".[torch]" && pip install "titans-pytorch==0.5.3"
    python examples/run_case_study.py

If ``titans-pytorch`` is not installed the script degrades to the GRU-only study
(mode ``real-subset``) and records why -- it never silently skips a model.
"""

from __future__ import annotations

import datetime as _dt
import json
import platform
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path

import numpy as np

import chronospect as cs
from chronospect.synthetic import multi_timescale

# --- toy task / probe configuration (fixed; deterministic) ---
TIMESCALES = (5.0, 50.0)  # the toy target mixes a fast and a slow timescale
D_SIG = 8  # signal channels (>= number of timescales)
T_TRAIN = 512
BATCH = 16
T_PROBE = 2048  # long enough for the slow timescale; divisible by the chunk size
N_PROBE = 8  # probe ensemble size (better-conditioned spectrum, like the gate)
GRU_HIDDEN = 16
GRU_STEPS = 300
GRU_LR = 1e-2
TITANS_DIM = 16
TITANS_CHUNK = 16
TITANS_STEPS = 30
TITANS_LR = 1e-2

_SEED_TRAIN = 1000
_SEED_PROBE = 2000


def _pkg(name: str) -> str:
    try:
        return version(name)
    except PackageNotFoundError:
        return "not installed"


def _spectrum_summary(rep: cs.TimescaleReport) -> dict:
    return {
        "n_dominant_timescales": rep.n_dominant_timescales,
        "dominant_timescales": rep.dominant_timescales,
        "effective_n_timescales": rep.effective_n_timescales,
        "verdict": rep.verdict,
        "spectrum_weights": rep.spectrum_weights,
    }


def _changed(before: dict, after: dict) -> bool:
    # "the instrument responded to training": the recovered spectrum is not
    # bit-identical before vs after. Compared on the (rounded) weight vector.
    return before["spectrum_weights"] != after["spectrum_weights"]


def _probe_signals(d: int) -> np.ndarray:
    """Held-out probe ensemble (n, T_PROBE, d), disjoint seeds from training."""
    return multi_timescale(T=T_PROBE, d=d, timescales=TIMESCALES, n_traj=N_PROBE, seed=_SEED_PROBE)


def gru_case_study() -> dict:
    import torch
    from torch import nn

    torch.manual_seed(_SEED_TRAIN)

    class GRUNextStep(nn.Module):
        def __init__(self, d_in: int, hidden: int) -> None:
            super().__init__()
            self.gru = nn.GRU(d_in, hidden)
            self.head = nn.Linear(hidden, d_in)

        def forward(self, x: torch.Tensor) -> torch.Tensor:  # x: (T, batch, d_in)
            out, _ = self.gru(x)
            return self.head(out)

    model = GRUNextStep(D_SIG, GRU_HIDDEN)

    # training data: (batch, T, d) -> (T, batch, d)
    train_np = multi_timescale(
        T=T_TRAIN, d=D_SIG, timescales=TIMESCALES, n_traj=BATCH, seed=_SEED_TRAIN
    )
    x = torch.tensor(np.transpose(train_np, (1, 0, 2)), dtype=torch.float32)

    probe_np = _probe_signals(D_SIG)
    probe = [torch.tensor(probe_np[i], dtype=torch.float32) for i in range(N_PROBE)]

    def record_probe() -> np.ndarray:
        return np.stack([cs.record_rnn_states(model.gru, p) for p in probe], axis=0)

    before = cs.analyze(record_probe())

    opt = torch.optim.Adam(model.parameters(), lr=GRU_LR)
    loss_fn = nn.MSELoss()
    losses: list[float] = []
    model.train()
    for _ in range(GRU_STEPS):
        opt.zero_grad()
        pred = model(x)
        loss = loss_fn(pred[:-1], x[1:])
        loss.backward()
        opt.step()
        losses.append(float(loss.item()))

    after = cs.analyze(record_probe())

    b, a = _spectrum_summary(before), _spectrum_summary(after)
    return {
        "model": "vanilla GRU (nn.GRU, next-step prediction toy task)",
        "memory_trajectory": "top-layer GRU hidden state per step (record_rnn_states)",
        "training_steps": GRU_STEPS,
        "train_loss_first": round(losses[0], 6),
        "train_loss_last": round(losses[-1], 6),
        "train_loss_decreased": losses[-1] < losses[0],
        "spectrum_before": b,
        "spectrum_after": a,
        "spectrum_changed_after_training": _changed(b, a),
    }


def titans_case_study() -> dict | None:
    try:
        import torch
        from titans_pytorch import NeuralMemory
    except ImportError:
        return None

    torch.manual_seed(_SEED_TRAIN)
    mem = NeuralMemory(dim=TITANS_DIM, chunk_size=TITANS_CHUNK)

    train_np = multi_timescale(
        T=T_TRAIN, d=TITANS_DIM, timescales=TIMESCALES, n_traj=BATCH, seed=_SEED_TRAIN
    )
    seq = torch.tensor(train_np, dtype=torch.float32)  # (batch, T, dim)

    probe_np = _probe_signals(TITANS_DIM)
    probe = torch.tensor(probe_np, dtype=torch.float32)  # (n, T_PROBE, dim)

    before = cs.analyze(cs.record_titans_memory(mem, probe))

    opt = torch.optim.Adam(mem.parameters(), lr=TITANS_LR)
    losses: list[float] = []
    mem.train()
    for _ in range(TITANS_STEPS):
        opt.zero_grad()
        retrieved, _ = mem(seq)
        loss = torch.nn.functional.mse_loss(retrieved[:, :-1], seq[:, 1:])
        loss.backward()
        opt.step()
        losses.append(float(loss.item()))

    after = cs.analyze(cs.record_titans_memory(mem, probe))

    b, a = _spectrum_summary(before), _spectrum_summary(after)
    return {
        "model": "Titans NeuralMemory (titans-pytorch, next-step prediction toy task)",
        "memory_trajectory": "per-step memory readout (record_titans_memory)",
        "training_steps": TITANS_STEPS,
        "train_loss_first": round(losses[0], 6),
        "train_loss_last": round(losses[-1], 6),
        "train_loss_decreased": losses[-1] < losses[0],
        "spectrum_before": b,
        "spectrum_after": a,
        "spectrum_changed_after_training": _changed(b, a),
    }


def main() -> None:
    results: list[dict] = []
    skipped: dict[str, str] = {}

    results.append(gru_case_study())

    titans = titans_case_study()
    if titans is not None:
        results.append(titans)
    else:
        skipped["lucidrains/titans-pytorch"] = (
            "titans-pytorch not importable in this environment; install with "
            "`pip install titans-pytorch` to include it. GRU study still ran."
        )
    skipped["obekt/HOPE-nested-learning"] = (
        "deferred to v0.3 (unofficial repo, API-drift risk; GRU + Titans suffice "
        "for an exploratory study)."
    )

    mode = "real" if len(results) >= 2 else "real-subset"

    # --- mechanical anti-hollow guards (forward-only studies are rejected) ---
    for r in results:
        assert r["training_steps"] > 0, f"{r['model']}: no training happened"
        assert r["spectrum_changed_after_training"], (
            f"{r['model']}: spectrum identical before/after training -- the study "
            "would be hollow (forward-only)."
        )
    training_steps_min = min(r["training_steps"] for r in results)
    before_after_changed = all(r["spectrum_changed_after_training"] for r in results)

    payload = {
        "schema": "chronospect-realmodel-casestudy-v0.2",
        "study_type": "exploratory case study (NOT validation)",
        "disclaimer": (
            "Exploratory only: tiny CPU configs, synthetic toy data, single seed per "
            "model, no hyperparameter search. Each model is described on its own terms "
            "(observed timescale structure); there is no cross-model comparison or "
            "ranking and no claim that the recovered timescales match the injected "
            "ones. Real-model validation is deferred to a future release."
        ),
        "mode": mode,
        "toy_task": {
            "task": "next-step prediction",
            "injected_timescales": list(TIMESCALES),
            "signal_channels": D_SIG,
            "T_train": T_TRAIN,
            "T_probe": T_PROBE,
            "probe_ensemble": N_PROBE,
            "note": (
                "injected timescales describe the TARGET signal, not a ground truth "
                "the model's memory is expected to reproduce."
            ),
        },
        "env": {
            "utc": _dt.datetime.now(_dt.timezone.utc).isoformat(timespec="seconds"),
            "python": platform.python_version(),
            "numpy": np.__version__,
            "torch": _pkg("torch"),
            "titans_pytorch": _pkg("titans-pytorch"),
            "chronospect": cs.__version__,
        },
        "training_steps_min": training_steps_min,
        "before_after_changed": before_after_changed,
        "models_run": [r["model"] for r in results],
        "models_skipped": list(skipped.keys()),
        "skip_reasons": skipped,
        "results": results,
    }

    out = Path(__file__).resolve().parent.parent / "results" / "realmodel_v0.2.json"
    out.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(f"wrote {out}  (mode={mode})")
    for r in results:
        print(
            f"  {r['model'].split('(')[0].strip()}: "
            f"loss {r['train_loss_first']:.4f} -> {r['train_loss_last']:.4f}; "
            f"n_dominant {r['spectrum_before']['n_dominant_timescales']} -> "
            f"{r['spectrum_after']['n_dominant_timescales']}; "
            f"verdict before={r['spectrum_before']['verdict']!r} "
            f"after={r['spectrum_after']['verdict']!r}"
        )


if __name__ == "__main__":
    main()
