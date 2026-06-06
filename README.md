# chronospect

**A CPU-only measurement instrument for the *timescale spectrum* and *aging dynamics* of a model's memory.**

Modern architectures *claim* to hold memory at multiple timescales — Titans, HOPE / Nested
Learning's Continuum Memory System, test-time learners, deep state-space models. The claim is
architectural. What has been missing is an **instrument that checks whether a model's memory is
actually multi-timescale, or whether it has quietly collapsed to a single effective speed** — and
whether that memory is *stationary* or *aging* during training.

`chronospect` reads a logged memory trajectory and answers:

- **Spectrum** — *which* relaxation timescales are actually present (not just how many)?
- **Capacity-vs-horizon** — how many independent past directions survive `τ` steps?
- **Aging** — is the memory time-translation invariant, or does it coarsen (older memory relaxing more slowly)?
- **Forgetting yardstick** — does decay follow a single exponential or the Benna–Fusi power law `τ^-1/2`?

It needs only a trajectory array and runs on a laptop CPU. No GPU, no model weights bundled, MIT licensed.

> **Status: v0.1.0a1 (alpha).** The validation numbers below come from **synthetic** ground-truth data
> with known injected timescales (see [Validation](#validation-the-sensitivity-gate)). A small
> [exploratory case study](examples/case_study.md) looks at real GRU and Titans memory before/after
> training, but `chronospect` has **not** been used to publish a *validated* finding about a real model.
> It is a measurement tool whose calibration is **characterised, not hidden** — including a
> [residual long-timescale bias](#calibration-how-to-read-a-recovered-timescale) it discloses rather
> than corrects away.

---

## Why these methods (the cross-disciplinary part)

Each diagnostic imports a mature idea from another field that, as far as we can find, has **not** been
applied to model memory as a shipped instrument:

| Diagnostic | Borrowed from | What it measures |
|---|---|---|
| `relaxation_spectrum` | dynamic-light-scattering / rheology relaxation spectra (non-negative least squares over a log-grid of exponentials) | the timescales present in the autocorrelation |
| `two_time_correlation` + `aging_index` | the **two-time correlation `C(t_w, t_w+τ)` of glassy / spin-glass physics** | whether memory dynamics are stationary or *aging* |
| `capacity_vs_horizon` | the **observability Gramian** of control theory, made data-driven via whitened canonical correlations | how many past directions remain recoverable at horizon `τ` |
| `octave_band_energy` | **wavelet multi-resolution analysis** | update energy per frequency octave (an assumption-light cross-check) |
| `forgetting_fit` | the **Benna–Fusi cascade memory** optimal forgetting law (`τ^-1/2`) | exponential vs power-law decay (a *reference yardstick*, never a headline) |

## Install

```bash
pip install chronospect            # core (numpy, scipy, PyWavelets)
pip install "chronospect[torch]"   # + the optional torch trajectory loggers
```

Or from source:

```bash
git clone https://github.com/hinanohart/chronospect
cd chronospect && pip install -e ".[dev]"
```

## Quickstart

```bash
chronospect gate     # validate the instrument on synthetic ground truth
chronospect demo     # analyze a synthetic memory with injected timescales 5 and 100
```

```python
import chronospect as cs

# X is your memory trajectory: (T, d) for one run, or (n, T, d) for an ensemble.
report = cs.analyze(X)
print(report.verdict)               # e.g. "multi-timescale; approximately stationary"
print(report.dominant_timescales)   # e.g. [3.6, 68.7]
print(report.aging_index)           # 0.0 == stationary; larger == aging
print(report.capacity_horizon_half) # lag at which memory capacity halves
```

### Getting a trajectory out of a model

```python
from chronospect.loggers import TrajectoryRecorder, record_rnn_states, from_snapshots

# 1) generic: record any per-step state you can name
rec = TrajectoryRecorder()
for step in rollout:
    rec.record(model.memory_state())     # array-like or torch tensor
X = rec.trajectory()                      # (T, d)

# 2) torch forward hook (records a module's output each call)
with TrajectoryRecorder().attach(model.memory_module) as rec:
    model(batch)
X = rec.trajectory()

# 3) RNN hidden states, step by step
X = record_rnn_states(torch.nn.GRU(8, 16), inputs)   # (T, 16)

# 4) weight-space memory across continual-learning tasks
X = from_snapshots([snapshot_after_task_i for i in range(n_tasks)])  # (n_tasks, d)
```

## Validation: the sensitivity gate

Before pointing the instrument at any real model it must recover structure it *injected itself*.
The pass/fail criteria are **pre-registered** in `chronospect/sensitivity.py` (fixed in advance, not
tuned until the pictures look nice). If `chronospect` can't recover timescales it planted in a toy, it
must not be trusted on a network.

```text
$ chronospect gate
GATE PASS
  [ok] G1_recover_two_timescales: injected=(5.0, 100.0) recovered=[3.6, 65.2]
  [ok] G2_single_is_single: n_peaks=1 n_eff=1.20 (<1.8)
  [ok] G3_capacity_horizon: cap_multi[200]=1.004 > cap_fast[200]=0.093
  [ok] G4_aging_detected: aging=0.990 stationary=0.000 (margin>=0.3)
  [ok] G5_no_false_aging: stationary multi-timescale aging=0.000 (<0.25)
  [ok] G6_calibration_holdout: recovered/injected 7:1.00[0.6,1.5], 40:0.90[0.6,1.5], 150:0.78[0.55,1.5], 300:0.64[0.58,1.45] single20_one_peak=True
  [ok] G7_real_model_smoke: skipped (torch not installed)
```

- **G1/G2** the spectrum recovers two well-separated injected timescales (to within a factor of ~2,
  the log-grid resolution) and reports a single-speed memory as single.
- **G3** a genuinely multi-timescale memory retains more capacity at a long horizon than a fast one.
- **G4/G5** an *aging* process is flagged, while a stationary memory — **even one split across several
  timescales** — is **not** (the aging index uses a significance-gated slope, so finite-ensemble noise
  does not fake aging).
- **G6** calibration on **hold-out** timescales `(7, 40, 150, 300)` — disjoint from the G1 timescales,
  so the calibration cannot be tuned by fitting the gate grid — checks that recovery lands in a
  pre-registered two-sided band (and that a single-speed memory still resolves to one peak). The band
  was fixed *before* the v0.2 calibration code; see [Calibration](#calibration-how-to-read-a-recovered-timescale).
- **G7** a torch-gated end-to-end smoke (an `nn.GRU` trajectory flows through `analyze`). With
  `chronospect[torch]` installed it runs the GRU smoke; on the torch-free core it is skipped and counted
  as passing, so the core test matrix stays GPU- and torch-free.

The gate is also the test suite (`pytest`), run across multiple seeds.

## Calibration: how to read a recovered timescale

Recovering a relaxation timescale from a finite window is an ill-posed inverse problem: **long
timescales are systematically under-recovered** (the larger the true timescale relative to the
observation window, the more it is shrunk). `chronospect` **discloses** this instead of hiding it. The
table below is generated by `cs.calibrate(...)` on synthetic single-timescale data with *known* answers
(`results/calibration_v0.2.json`, reproducible):

| injected τ | recovered/τ (default) | recovered/τ (`bias_correct=True`) |
|---:|:---:|:---:|
| 7   | 0.96 | 0.97 |
| 20  | 0.93 | 0.97 |
| 50  | 0.83 | 0.91 |
| 100 | 0.71 | 0.80 |
| 200 | 0.56 | 0.68 |
| 300 | 0.45 | 0.61 |

Read a recovered long timescale as a **shrinkage-affected estimate / lower bound**, not an exact value.

An **opt-in** demeaning-bias correction (`aggregate_autocorr(..., bias_correct=True)`, also reachable as
`analyze(..., bias_correct=True)`) removes the *identifiable* finite-sample part of the shrinkage and
lifts long-timescale recovery (e.g. τ=300 from 0.45 to 0.61) — but it **cannot** remove the residual
finite-window shrinkage, and it has a cost: it broadens the apparent spectrum of a genuinely
single-speed memory (effective number of timescales 1.22 → 1.55 in the same run). It is therefore **off
by default** — the default preserves the single-vs-multi discrimination the gate checks — and you enable
it when long-timescale calibration matters more than the single-vs-multi headline. `chronospect` never
silently rescales a reading and never claims exact point recovery (it does not claim a recovered 100
from an injected 100).

## Exploratory case study (not validation)

[`examples/case_study.md`](examples/case_study.md) trains a small `nn.GRU` and a Titans `NeuralMemory`
on a toy next-step-prediction task and reads each model's memory-trajectory spectrum **before and after
training** on a held-out probe (`results/realmodel_v0.2.json`; regenerate with `python
examples/run_case_study.py`). It is deliberately narrow: it shows the instrument **responds to learning**
— the recovered spectrum changes once the model has been trained — and nothing more. It is
**exploratory** (tiny CPU configs, synthetic toy data, one seed per model) and makes **no cross-model
comparison or ranking** and no claim that a recovered timescale equals an injected one. For instance the
Titans readout's verdict moves from "effectively single-speed" before training to "multi-timescale"
after, and the GRU's recovered fast peak shifts with training; see the case study for the full tables
and caveats.

## Honest limitations

- **Synthetic validation; one exploratory real-model case study.** The validated numbers come from toy
  AR(1)/aging processes with known answers; the real-model [case study](#exploratory-case-study-not-validation)
  is exploratory, not a validation. Treat readings on real models as exploratory until corroborated.
- **Long timescales are under-recovered** (an ill-posed finite-window inversion); resolution is ~a
  factor of 2. Read the spectrum as bands present / lower bounds, not exact constants — see the
  [calibration table](#calibration-how-to-read-a-recovered-timescale) and the opt-in `bias_correct`
  correction, which lifts but cannot eliminate the long-timescale shrinkage.
- **Aging detection needs an ensemble.** With a single trajectory the two-time correlation is noisy;
  pass several runs (`(n, T, d)`) for a reliable `aging_index`.
- **The autocorrelation assumes local stationarity** for the spectrum; on strongly non-stationary
  memory read the two-time / aging output first.
- The Benna–Fusi power-law fit is a **reference yardstick only** — a single power-law fit is close to
  trivial and is never reported as a standalone claim.

## Prior work and relation to it

There are excellent implementations of multi-timescale *architectures* and continual-learning
*benchmarks*, but we could not find a shipped instrument that *measures* a memory's timescale spectrum
and aging. In particular, the Nested Learning / HOPE line explicitly **describes** "gradient-memory
dashboards" and frequency-band diagnosis of forgetting as a direction; the public HOPE/Nested-Learning
repositories ship *training* dashboards (loss / throughput), not a memory-timescale measurement tool.
`chronospect` operationalizes that measurement, model-agnostically, from a logged trajectory.

- Benna & Fusi, *Computational principles of synaptic memory consolidation*, Nat. Neurosci. 2016.
- Roy & Vetterli, *The effective rank of a matrix*, EUSIPCO 2007.
- Two-time correlation & aging: standard tools in the statistical physics of glasses.
- Behrouz et al., *Nested Learning / Titans* (multi-timescale memory architectures).
- Continual-learning frameworks (e.g. Avalanche, LibContinual) report accuracy / BWT-FWT — a different,
  complementary axis from timescale structure.

This is a measurement instrument, not a method that improves a model. If you find a place where this
measurement is already shipped, please open an issue — we would rather cite it than duplicate it.

## License

MIT © 2026 hinanohart. See [LICENSE](LICENSE).
