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

__all__ = [
    "TrajectoryRecorder",
    "record_rnn_states",
    "record_titans_memory",
    "from_snapshots",
]


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
        """A ``torch`` forward hook: records the module's output tensor each call.

        The output is flattened as-is, so this is best for a module that emits a
        single state vector per call (batch size 1).  For an ``nn.GRU``/``nn.LSTM``
        the first tuple element is the *output sequence*, not the hidden state --
        use :func:`record_rnn_states` if you want hidden states, or call
        :meth:`record` manually with the exact tensor you want.
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
        """Return the recorded ``(T, d)`` trajectory.

        Every recorded frame must have the same width; a mismatch raises rather
        than silently truncating, since dropping columns would discard exactly
        the memory content the instrument is meant to measure.
        """
        if not self._frames:
            raise ValueError("no frames recorded")
        widths = {f.size for f in self._frames}
        if len(widths) > 1:
            raise ValueError(
                f"recorded frames have inconsistent widths {sorted(widths)}; "
                "record a fixed-size state each step (reduce/flatten consistently)"
            )
        return np.stack(self._frames, axis=0)


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
    if x.dim() != 3 or x.shape[1] != 1:
        raise ValueError(
            "record_rnn_states expects inputs shaped (T, input) or (T, 1, input) "
            f"(batch size 1); got {tuple(inputs.shape)}"
        )
    T = x.shape[0]
    h = None
    frames = []
    with torch.no_grad():
        for t in range(T):
            _, h = rnn(x[t : t + 1], h)
            hid = h[0] if isinstance(h, tuple) else h  # LSTM -> (h, c)
            frames.append(_to_row(hid[-1]))  # top layer
    return np.stack(frames, axis=0)


def record_titans_memory(memory: Any, seq: Any) -> np.ndarray:
    """Record a chunked test-time memory module's per-step *readout* trajectory.

    ``memory`` is any module whose ``forward(seq)`` returns either the read-out
    tensor or a ``(readout, state)`` tuple -- Titans' ``NeuralMemory`` is the
    canonical example.  ``seq`` is a ``(T, dim)``, ``(1, T, dim)`` or
    ``(n, T, dim)`` torch tensor.  Returns the readout as ``(T, d)`` (single
    sequence) or ``(n, T, d)`` (ensemble), ready for :func:`chronospect.analyze`.

    Why the readout, not the raw fast-weights?  ``analyze`` consumes a trajectory
    shaped ``(T, d)``; a flattened fast-weight snapshot per chunk has ``d`` far
    larger than the number of chunks ``T``, which makes the capacity estimator's
    whitening singular (and slow).  The readout -- what the memory returns when
    queried at each step -- is the low-dimensional *observable* memory signal and
    keeps ``T >> d``.  It still reflects the test-time-updated memory state, so
    training-induced changes show up in its timescale spectrum.

    Duck-typed (no dependency on any specific memory library); torch is only used
    to detach the tensor.
    """
    out = memory(seq)
    readout = out[0] if isinstance(out, tuple) else out
    if hasattr(readout, "detach"):
        readout = readout.detach().to("cpu").numpy()
    arr = np.asarray(readout, dtype=float)
    if arr.ndim == 2:  # (T, d)
        return arr
    if arr.ndim == 3:  # (n, T, d); collapse a singleton batch to (T, d)
        return arr[0] if arr.shape[0] == 1 else arr
    raise ValueError(f"memory readout must be (T, d) or (n, T, d); got shape {arr.shape}")


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
