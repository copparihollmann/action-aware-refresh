# Decision log

Append-only. Newest at bottom. Every non-obvious choice made during the
project — with the reasoning — lives here.

---

## 2026-08-03 — Session 1 kickoff

**Deployment topology: single-host multi-GPU, native `uv`, no containers.**

- Reason: host has no `nvidia-container-toolkit`, no CDI, no docker, no
  sudo. Rootless podman 5.6 alone cannot expose GPUs to containers on this
  system. Trying to work around that would eat all the session time.
- Alternative rejected: request sysadmin install `nvidia-container-toolkit`.
  Kept as a future ask if the native uv path proves painful for Cosmos or
  RoboLab (e.g. Isaac Sim system deps).
- Impact: the spec's preferred `docker build` path for Cosmos Framework is
  replaced with a native `uv sync` inside `third_party/cosmos-framework/`.
  Same dependency group (`cu130-train` + `policy-server`) applies.

**GPU assignment: GPU 0 → Cosmos server, GPU 1 → RoboLab client.**

- Both are 98 GB Blackwell — plenty of headroom for either workload alone.
- The machine is currently shared with user `ken_ho` (small memory, ~90–95%
  util spikes). Correctness runs are fine. Timing measurements will wait
  for a quiet window and record `nvidia-smi` snapshots at run start.

**Python: 3.11 via uv.**

- Cosmos + RoboLab both officially target 3.11. Available via uv (system
  Python is 3.9 which is too old).

**Energy: NVML power-integration, not total-energy counter.**

- `nvidia-smi --query-gpu=total_energy_consumption` is unsupported on this
  GPU/driver. `energy.py` samples power at 10 Hz and trapezoid-integrates.
  Every energy field will be labeled ESTIMATED and record the sampling
  cadence in the run metadata.
