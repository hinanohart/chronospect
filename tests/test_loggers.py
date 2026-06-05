import numpy as np
import pytest

from chronospect.loggers import TrajectoryRecorder, from_snapshots

torch = pytest.importorskip("torch", reason="torch optional")


def test_recorder_collects_frames():
    rec = TrajectoryRecorder()
    for t in range(20):
        rec.record(np.arange(8) * 0.1 + t)
    X = rec.trajectory()
    assert X.shape == (20, 8)


def test_from_snapshots():
    snaps = [np.ones(10) * i for i in range(5)]
    X = from_snapshots(snaps)
    assert X.shape == (5, 10)


def test_recorder_forward_hook_on_gru():
    rnn = torch.nn.GRU(input_size=4, hidden_size=6, batch_first=False)
    rec = TrajectoryRecorder()
    with rec.attach(rnn):
        for _ in range(15):
            x = torch.randn(1, 1, 4)
            rnn(x)
    X = rec.trajectory()
    assert X.shape[0] == 15
    assert X.shape[1] > 0


def test_record_rnn_states():
    from chronospect.loggers import record_rnn_states

    rnn = torch.nn.LSTM(input_size=3, hidden_size=5)
    inputs = torch.randn(25, 3)
    X = record_rnn_states(rnn, inputs)
    assert X.shape == (25, 5)
