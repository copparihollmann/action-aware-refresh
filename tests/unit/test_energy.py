"""Unit test for energy integration math (no GPU required)."""
from __future__ import annotations

from action_refresh.energy import EnergyMeter, EnergySample


def test_integrate_trapezoidal() -> None:
    m = EnergyMeter()
    # Manually seed samples: constant 100 W for 2 s → 200 J.
    m._samples = [
        EnergySample(0.0, 100.0),
        EnergySample(1.0, 100.0),
        EnergySample(2.0, 100.0),
    ]
    assert abs(m.integrate_j() - 200.0) < 1e-6
    # Linear ramp 0 → 200 W over 2 s → integral = 200 J.
    m._samples = [
        EnergySample(0.0, 0.0),
        EnergySample(1.0, 100.0),
        EnergySample(2.0, 200.0),
    ]
    assert abs(m.integrate_j() - 200.0) < 1e-6


def test_empty_returns_zero() -> None:
    assert EnergyMeter().integrate_j() == 0.0
