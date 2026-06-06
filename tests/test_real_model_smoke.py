"""Real-model smoke tests (torch-gated, tiny + fast).

These keep the real-model path honest in CI without running the full case study
(see examples/run_case_study.py for that). The core matrix stays torch-free:
each test skips cleanly when its optional dependency is absent.
"""

from __future__ import annotations

import numpy as np
import pytest

import chronospect as cs
from chronospect.synthetic import multi_timescale


def test_record_titans_memory_shapes_no_torch():
    # record_titans_memory is duck-typed: it works on any callable returning a
    # readout (or (readout, state)) without importing torch. This pins its shape
    # contract and runs on the torch-free core matrix.
    class FakeReadout:
        def __init__(self, arr):
            self.arr = arr

        def __call__(self, _seq):
            return self.arr, "state"  # (readout, state) tuple form

    single = FakeReadout(np.zeros((40, 4)))
    assert cs.record_titans_memory(single, None).shape == (40, 4)

    batch1 = FakeReadout(np.zeros((1, 40, 4)))
    assert cs.record_titans_memory(batch1, None).shape == (40, 4)  # singleton collapsed

    ens = FakeReadout(np.zeros((3, 40, 4)))
    assert cs.record_titans_memory(ens, None).shape == (3, 40, 4)

    bad = FakeReadout(np.zeros((4,)))
    with pytest.raises(ValueError):
        cs.record_titans_memory(bad, None)


def test_gru_memory_pipeline_smoke():
    torch = pytest.importorskip("torch")
    torch.manual_seed(0)
    gru = torch.nn.GRU(4, 8)
    sig = multi_timescale(T=256, d=4, timescales=(5.0, 40.0), n_traj=1, seed=0)[0]
    states = cs.record_rnn_states(gru, torch.tensor(sig, dtype=torch.float32))
    rep = cs.analyze(states)
    assert states.shape == (256, 8)
    assert np.isfinite(rep.effective_n_timescales)
    assert isinstance(rep.verdict, str) and rep.verdict


def test_gru_training_changes_spectrum():
    # The load-bearing claim of the case study, in miniature: training changes
    # the recovered spectrum (a forward-only study would be hollow).
    torch = pytest.importorskip("torch")
    from torch import nn

    torch.manual_seed(0)

    class GRUNextStep(nn.Module):
        def __init__(self):
            super().__init__()
            self.gru = nn.GRU(4, 8)
            self.head = nn.Linear(8, 4)

        def forward(self, x):
            out, _ = self.gru(x)
            return self.head(out)

    model = GRUNextStep()
    train = multi_timescale(T=128, d=4, timescales=(5.0, 40.0), n_traj=4, seed=1)
    x = torch.tensor(np.transpose(train, (1, 0, 2)), dtype=torch.float32)
    probe = torch.tensor(
        multi_timescale(T=256, d=4, timescales=(5.0, 40.0), n_traj=1, seed=2)[0],
        dtype=torch.float32,
    )

    before = cs.analyze(cs.record_rnn_states(model.gru, probe)).to_dict()["spectrum_weights"]
    opt = torch.optim.Adam(model.parameters(), lr=1e-2)
    loss_fn = nn.MSELoss()
    losses: list[float] = []
    for _ in range(25):
        opt.zero_grad()
        pred = model(x)
        loss = loss_fn(pred[:-1], x[1:])
        loss.backward()
        opt.step()
        losses.append(loss.item())
    after = cs.analyze(cs.record_rnn_states(model.gru, probe)).to_dict()["spectrum_weights"]

    assert losses[-1] < losses[0]  # training actually reduced the loss
    assert before != after  # the instrument responded to training


def test_titans_readout_pipeline_smoke():
    pytest.importorskip("torch")
    NeuralMemory = pytest.importorskip("titans_pytorch").NeuralMemory
    import torch

    torch.manual_seed(0)
    mem = NeuralMemory(dim=8, chunk_size=8)
    probe = torch.tensor(
        multi_timescale(T=256, d=8, timescales=(5.0, 40.0), n_traj=2, seed=0),
        dtype=torch.float32,
    )
    traj = cs.record_titans_memory(mem, probe)
    rep = cs.analyze(traj)
    assert traj.shape == (2, 256, 8)
    assert np.isfinite(rep.effective_n_timescales)
    assert isinstance(rep.verdict, str) and rep.verdict
