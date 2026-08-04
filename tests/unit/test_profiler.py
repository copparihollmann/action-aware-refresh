"""Unit tests for the CPU-side profiler (CUDA layer is smoke-tested separately)."""
from __future__ import annotations

import time

from action_refresh.profiler import StageTimer


def test_stage_timer_accumulates() -> None:
    t = StageTimer()
    with t.stage("a"):
        time.sleep(0.01)
    with t.stage("b"):
        time.sleep(0.005)
    with t.stage("a"):
        time.sleep(0.01)
    d = t.as_dict()
    assert "a" in d and "b" in d
    # Loose bounds so a busy CI doesn't flake.
    assert d["a"] >= 15.0  # two 10ms stages, minus jitter margin
    assert d["b"] >= 3.0
