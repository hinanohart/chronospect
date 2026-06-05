"""Helpers for turning a model's memory into a trajectory array.

``chronospect`` analyses a trajectory shaped ``(T, d)`` (one run) or
``(n, T, d)`` (an ensemble).  How you obtain that trajectory depends on what you
call "memory":

* **recurrent / SSM state** -- the hidden state at each step (GRU/LSTM,
  Mamba-style scans, Titans test-time memory).  Use :class:`TrajectoryRecorder`
  with a forward hook, or :func:`record_rnn_states` for ``nn.GRU`` / ``nn.LSTM``.
* **fast-weights / memory bank** -- flatten the memory tensor each step and feed
  it to the recorder.
* **weight-space memory across tasks** (continual learning) -- snapshot a
  parameter tensor after each task with :func:`from_snapshots`.

Torch is an optional dependency; importing this module does not require it.
"""

from __future__ import annotations

from typing import Any

import numpy as np

__all__ = ["TrajectoryRecorder", "record_rnn_states", "from_snapshots"]


def _to_row(x: Any) -> np.ndarray:
    """Flatten one frame to a 1-D float vector (detaching torch tensors)."""
    if hasattr(x, "detach"):
        x = x.detach().to("cpu").numpy()
    arr = np.asarray(x, dtype=float)
    return arr.reshape(-1)


class TrajectoryRecorder:
    """Collect per-step state vectors and stack them into a trajectory.

    Example
    -------
    >>> rec = TrajectoryRecorder()
    >>> for step in rollout:                 # doctest: +SKIP
    ...     rec.record(model.memory_state())
    >>> X = rec.trajectory()                  # (T, d)            # doctest: +SKIP
    """

    def __init__(self) -> None:
        self._frames: list[np.ndarray] = []
        self._handle = None

    def record(self, state: Any) -> None:
        """Append one frame (any array-like or torch tensor)."""
        self._frames.append(_to_row(state))

    def forward_hook(self, _module: Any, _inputs: Any, output: Any) -> None:
        """A ``torch`` forward hook: records the module output each call.

        For modules returning a tuple (e.g. ``nn.GRU`` -> ``(out, h)``) the first
        element is recorded; override :meth:`record` calls for finer control.
        """
        out = output[0] if isinstance(output, tuple) else output
        self.record(out)

    def attach(self, module: Any) -> TrajectoryRecorder:
        """Register :meth:`forward_hook` on a torch ``module``; returns self."""
        self._handle = module.register_forward_hook(self.forward_hook)
        return self

    def detach(self) -> None:
        """Remove a previously attached hook."""
        if self._handle is not None:
            self._handle.remove()
            self._handle = None

    def __enter__(self) -> TrajectoryRecorder:
        return self

    def __exit__(self, *exc: object) -> None:
        self.detach()

    def trajectory(self) -> np.ndarray:
        """Return the recorded ``(T, d)`` trajectory (frames padded/truncated to a common width)."""
        if not self._frames:
            raise ValueError("no frames recorded")
        width = min(f.size for f in self._frames)
        return np.stack([f[:width] for f in self._frames], axis=0)


def record_rnn_states(rnn: Any, inputs: Any) -> np.ndarray:
    """Run an ``nn.GRU``/``nn.LSTM`` step by step and return hidden states ``(T, d)``.

    ``inputs`` is a ``(T, input_size)`` or ``(T, 1, input_size)`` tensor.  Only
    the top-layer hidden state is recorded.  Requires torch.
    """
    import torch

    rnn.eval()
    x = inputs
    if x.dim() == 2:
        x = x.unsqueeze(1)  # (T, 1, input_size)
    T = x.shape[0]
    h = None
    frames = []
    with torch.no_grad():
        for t in range(T):
            _, h = rnn(x[t : t + 1], h)
            hid = h[0] if isinstance(h, tuple) else h  # LSTM -> (h, c)
            frames.append(_to_row(hid[-1]))  # top layer
    return np.stack(frames, axis=0)


def from_snapshots(snapshots: list[Any]) -> np.ndarray:
    """Build a trajectory from a list of weight/state snapshots (e.g. per task).

    Each snapshot is flattened to a row; the result is ``(n_snapshots, d)``,
    a weight-space memory trajectory suitable for :func:`chronospect.analyze`.
    """
    if len(snapshots) < 2:
        raise ValueError("need at least two snapshots")
    rows = [_to_row(s) for s in snapshots]
    width = min(r.size for r in rows)
    return np.stack([r[:width] for r in rows], axis=0)
