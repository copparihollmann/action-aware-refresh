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


def nvml_index_for_uuid(uuid: str) -> int:
    """Resolve a GPU UUID to its physical NVML index.

    NVML indices ignore ``CUDA_VISIBLE_DEVICES``, so a process pinned to CUDA
    device 0 may well be running on physical GPU 2. Sampling power by the
    CUDA-visible index would then silently report a *different* GPU's power.
    Always resolve through the UUID recorded in configs/topology.yaml.
    """
    if pynvml is None:
        raise RuntimeError("pynvml not installed — cannot resolve GPU UUID")
    pynvml.nvmlInit()
    try:
        want = uuid.strip()
        for i in range(pynvml.nvmlDeviceGetCount()):
            got = pynvml.nvmlDeviceGetUUID(pynvml.nvmlDeviceGetHandleByIndex(i))
            if isinstance(got, bytes):
                got = got.decode()
            if got.strip() == want:
                return i
        raise RuntimeError(f"no NVML device with uuid={want!r}")
    finally:
        pynvml.nvmlShutdown()


@dataclass
class EnergyMeter:
    """Samples power on ONE physical GPU.

    Pass ``uuid`` whenever it is known; ``device_index`` alone is a physical
    NVML index and is only correct when it happens to match the GPU the
    workload is actually on.
    """

    device_index: int = 0
    uuid: Optional[str] = None
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
        if self.uuid is not None:
            # Verify the recorded index still points at the recorded GPU, and
            # correct it if the driver renumbered. Never guess silently.
            got = pynvml.nvmlDeviceGetUUID(pynvml.nvmlDeviceGetHandleByIndex(self.device_index))
            if isinstance(got, bytes):
                got = got.decode()
            if got.strip() != self.uuid.strip():
                for i in range(pynvml.nvmlDeviceGetCount()):
                    other = pynvml.nvmlDeviceGetUUID(pynvml.nvmlDeviceGetHandleByIndex(i))
                    if isinstance(other, bytes):
                        other = other.decode()
                    if other.strip() == self.uuid.strip():
                        self.device_index = i
                        break
                else:
                    raise RuntimeError(
                        f"no NVML device with uuid={self.uuid!r} "
                        f"(index {self.device_index} reports {got!r})"
                    )
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


    def duration_s(self) -> float:
        """Wall-clock span actually covered by samples."""
        if len(self._samples) < 2:
            return 0.0
        return self._samples[-1].t_s - self._samples[0].t_s

    def idle_adjusted_j(self, idle_power_w: float) -> float:
        """Energy above the idle floor, in joules.

        Reported alongside — never instead of — ``integrate_j()``. Both are
        ESTIMATED: this driver does not expose a total-energy counter, so
        everything here is a power integral at ``sample_hz``.
        """
        adjusted = self.integrate_j() - idle_power_w * self.duration_s()
        return max(0.0, adjusted)


def measure_idle_power(
    device_index: int,
    duration_s: float = 5.0,
    sample_hz: float = 10.0,
    uuid: Optional[str] = None,
) -> float:
    """Return mean idle power in watts. Call before your workload starts."""
    with EnergyMeter(device_index=device_index, uuid=uuid, sample_hz=sample_hz) as m:
        time.sleep(duration_s)
    return m.mean_power_w()
