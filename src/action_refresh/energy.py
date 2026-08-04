"""NVML-based energy sampler.

The `nvidia-smi --query-gpu=total_energy_consumption` counter is unavailable
on this driver/GPU, so we sample instantaneous power and integrate over
time. Method:

  1. Idle baseline: record mean power over N seconds with no workload.
  2. During a run: sample power at fixed cadence (default 10 Hz) in a
     background thread and integrate via trapezoidal rule.
  3. Report both gross energy and idle-adjusted energy. Always label as
     ESTIMATED (documented sampling cadence in the record).

Use ``EnergyMeter`` as a context manager around the region of interest.
"""
from __future__ import annotations

import threading
import time
from dataclasses import dataclass, field
from typing import Optional

try:
    import pynvml  # type: ignore[import-untyped]
except Exception:  # noqa: BLE001
    pynvml = None  # type: ignore[assignment]


@dataclass
class EnergySample:
    t_s: float
    power_w: float


@dataclass
class EnergyMeter:
    device_index: int = 0
    sample_hz: float = 10.0
    _samples: list[EnergySample] = field(default_factory=list)
    _thread: Optional[threading.Thread] = None
    _stop: threading.Event = field(default_factory=threading.Event)
    _handle: object = None
    _t0: float = 0.0

    def _require_nvml(self) -> None:
        if pynvml is None:
            raise RuntimeError("pynvml not installed — cannot sample GPU power")

    def _init_handle(self) -> None:
        self._require_nvml()
        pynvml.nvmlInit()
        self._handle = pynvml.nvmlDeviceGetHandleByIndex(self.device_index)

    def _sample_loop(self) -> None:
        dt = 1.0 / self.sample_hz
        while not self._stop.is_set():
            try:
                mw = pynvml.nvmlDeviceGetPowerUsage(self._handle)  # milliwatts
                self._samples.append(
                    EnergySample(t_s=time.perf_counter() - self._t0, power_w=mw / 1000.0)
                )
            except Exception:  # noqa: BLE001
                pass
            self._stop.wait(dt)

    def __enter__(self) -> "EnergyMeter":
        self._init_handle()
        self._samples.clear()
        self._stop.clear()
        self._t0 = time.perf_counter()
        self._thread = threading.Thread(target=self._sample_loop, daemon=True)
        self._thread.start()
        return self

    def __exit__(self, *exc: object) -> None:
        self._stop.set()
        if self._thread is not None:
            self._thread.join(timeout=2.0)
        try:
            pynvml.nvmlShutdown()
        except Exception:  # noqa: BLE001
            pass

    def integrate_j(self) -> float:
        """Trapezoidal integration of power(t) → energy in joules."""
        s = self._samples
        if len(s) < 2:
            return 0.0
        total = 0.0
        for a, b in zip(s[:-1], s[1:]):
            total += 0.5 * (a.power_w + b.power_w) * (b.t_s - a.t_s)
        return total

    def mean_power_w(self) -> float:
        if not self._samples:
            return 0.0
        return sum(s.power_w for s in self._samples) / len(self._samples)

    def sample_count(self) -> int:
        return len(self._samples)


def measure_idle_power(device_index: int, duration_s: float = 5.0, sample_hz: float = 10.0) -> float:
    """Return mean idle power in watts. Call before your workload starts."""
    with EnergyMeter(device_index=device_index, sample_hz=sample_hz) as m:
        time.sleep(duration_s)
    return m.mean_power_w()
