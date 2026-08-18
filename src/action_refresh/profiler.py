"""Timing utilities — three layers.

Layer 1 (always on): wall-clock via ``time.perf_counter_ns()``.
Layer 2 (GPU stages): ``torch.cuda.Event`` at controlled boundaries. We do
  NOT synchronize the whole device on every event — only around the block
  we're measuring. Overuse of cuda.synchronize() destroys throughput.
Layer 3 (deep dive, opt-in): PyTorch Profiler / Nsight — hooked around a
  handful of requests, not every request.

Torch is imported lazily so this module can be used in the client without
depending on CUDA when only wall timing is needed.
"""
from __future__ import annotations

import contextlib
import time
from dataclasses import dataclass, field
from typing import Any, Iterator


def _now_ns() -> int:
    return time.perf_counter_ns()


@dataclass
class StageTimer:
    """Accumulates named stage durations in ms. Call `stage(name)` as a context."""

    _ms: dict[str, float] = field(default_factory=dict)
    _stack: list[tuple[str, int]] = field(default_factory=list)

    @contextlib.contextmanager
    def stage(self, name: str) -> Iterator[None]:
        t0 = _now_ns()
        try:
            yield
        finally:
            dt_ms = (_now_ns() - t0) / 1e6
            self._ms[name] = self._ms.get(name, 0.0) + dt_ms

    def as_dict(self) -> dict[str, float]:
        return dict(self._ms)


@dataclass
class CudaStageTimer:
    """CUDA-event stage timer. Synchronize only inside `finalize()`.

    ``device`` is a **CUDA-visible** index (i.e. after ``CUDA_VISIBLE_DEVICES``
    remapping), which is what ``torch.cuda.device()`` expects — so the default
    of 0 means "the first GPU this process can see", not "physical GPU 0".
    That is correct here, but do NOT reuse this number for NVML power
    sampling: NVML ignores ``CUDA_VISIBLE_DEVICES``. See
    ``action_refresh.energy.nvml_index_for_uuid``.
    """

    device: int = 0
    _events: list[tuple[str, Any, Any]] = field(default_factory=list)

    def _torch(self):  # type: ignore[no-untyped-def]
        import torch  # local import; keeps this module CPU-only when torch is absent

        if not torch.cuda.is_available():
            raise RuntimeError("CudaStageTimer: no CUDA device available")
        return torch

    @contextlib.contextmanager
    def stage(self, name: str) -> Iterator[None]:
        torch = self._torch()
        with torch.cuda.device(self.device):
            start = torch.cuda.Event(enable_timing=True)
            end = torch.cuda.Event(enable_timing=True)
            start.record()
            try:
                yield
            finally:
                end.record()
                self._events.append((name, start, end))

    def finalize(self) -> dict[str, float]:
        """Synchronize once, then read all elapsed times. Returns ms per name."""
        if not self._events:
            return {}
        torch = self._torch()
        with torch.cuda.device(self.device):
            torch.cuda.synchronize()
            out: dict[str, float] = {}
            for name, s, e in self._events:
                out[name] = out.get(name, 0.0) + s.elapsed_time(e)
        return out


@contextlib.contextmanager
def torch_profile(
    trace_path: str,
    with_flops: bool = True,
    activities: tuple[str, ...] = ("cpu", "cuda"),
) -> Iterator[Any]:
    """Wrap around a code region to produce a chrome trace at `trace_path`.

    Cheap when unused: the import is lazy and the context does nothing if
    called without an active `with` block.
    """
    import torch
    from torch.profiler import ProfilerActivity, profile

    act = []
    if "cpu" in activities:
        act.append(ProfilerActivity.CPU)
    if "cuda" in activities and torch.cuda.is_available():
        act.append(ProfilerActivity.CUDA)

    with profile(activities=act, with_flops=with_flops, record_shapes=True) as prof:
        yield prof
    prof.export_chrome_trace(trace_path)


@contextlib.contextmanager
def flop_counter() -> Iterator[Any]:
    """Wrap around a code region to count FLOPs of supported ops.

    Reports only what ``FlopCounterMode`` supports — custom fused kernels are
    NOT counted. Report coverage separately when using this.
    """
    from torch.utils.flop_counter import FlopCounterMode

    with FlopCounterMode() as ctr:
        yield ctr
