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

> **Status: v0.1.0a1 (alpha).** Every number below is from **synthetic** ground-truth validation — see
> [Validation](#validation-the-sensitivity-gate). `chronospect` has **not** yet been used to publish a
> finding about a real trained model; it is a measurement tool whose calibration is checked against
> data with *known* injected timescales.

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
```

- **G1/G2** the spectrum recovers two well-separated injected timescales (to within a factor of ~2,
  the log-grid resolution) and reports a single-speed memory as single.
- **G3** a genuinely multi-timescale memory retains more capacity at a long horizon than a fast one.
- **G4/G5** an *aging* process is flagged, while a stationary memory — **even one split across several
  timescales** — is **not** (the aging index uses a significance-gated slope, so finite-ensemble noise
  does not fake aging).

The gate is also the test suite (`pytest`), run across multiple seeds.

## Honest limitations

- **Synthetic validation only.** Numbers here come from toy AR(1)/aging processes with known answers.
  Treat readings on real models as exploratory until corroborated.
- **Timescale resolution is ~a factor of 2**; long timescales near the observation window are biased
  low. Read the spectrum as "bands present", not exact constants.
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
