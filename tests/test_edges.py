import numpy as np
import pytest

from chronospect import analyze
from chronospect.loggers import TrajectoryRecorder
from chronospect.synthetic import single_timescale


def test_trajectory_inconsistent_widths_raises():
    rec = TrajectoryRecorder()
    rec.record(np.zeros(8))
    rec.record(np.zeros(16))  # different width -> must not be silently truncated
    with pytest.raises(ValueError):
        rec.trajectory()


def test_trajectory_empty_raises():
    with pytest.raises(ValueError):
        TrajectoryRecorder().trajectory()


def test_analyze_rejects_too_short():
    with pytest.raises(ValueError):
        analyze(single_timescale(T=16, n_traj=2, seed=0))


def test_analyze_rejects_bad_ndim():
    with pytest.raises(ValueError):
        analyze(np.zeros(100))  # 1-D is neither (T,d) nor (n,T,d)


def test_record_rnn_states_rejects_batch_gt_1():
    torch = pytest.importorskip("torch")
    rnn = torch.nn.GRU(input_size=3, hidden_size=5)
    from chronospect.loggers import record_rnn_states

    with pytest.raises(ValueError):
        record_rnn_states(rnn, torch.randn(20, 4, 3))  # batch=4 not allowed
