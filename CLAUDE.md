# CLAUDE.md — repo-local operating rules

## Environment invariants
- Host: `firesim2`, **Ubuntu 24.04.4**, glibc 2.39. No sudo.
  Docker daemon and `nvidia-container-toolkit` exist on the host, but **our uid is
  not in the `docker` group** and rootless prerequisites (`newuidmap`, `/etc/subuid`)
  are absent → container path unavailable to us. It is also unnecessary: this host
  already *is* the upstream image's base (`ubuntu24.04`), and nothing in either
  install needs sudo. All installs are native, user-space, via `uv`.
- **4× NVIDIA L40S** (SM 8.9 Ada, 46068 MiB = 45.0 GiB each), driver 580.126.18,
  CUDA 13.0. GPU 0 → Cosmos server, GPU 2 → RoboLab/Isaac (separate PCIe switches);
  GPUs 1, 3 spare. Pin via `configs/topology.yaml`; assert the recorded UUID.
- **VRAM is the tight constraint**: the checkpoint is 32.9 GB of bf16 weights
  (~30.4 GB transformer ≈ 15.2B params) against 45.0 GiB → ~14 GiB headroom.
  Batched multi-env requests may not fit. If it OOMs: reduce concurrency, or shard
  across GPUs 0+1 and record that as *the* baseline topology. **Never quantize to
  make the baseline fit** — quantization is a separate experiment (spec §5).
- Host CUDA toolkit is 12.6 (`/usr/local/cuda`), but the driver reports CUDA 13.0,
  so the documented wheel group is **`cu130-train`** (`cu128-train` is the fallback).
  Do *not* set `TRITON_PTXAS_PATH` to the 12.6 `ptxas`: that Dockerfile line exists
  because triton's bundled ptxas lagged Blackwell, and on SM 8.9 the bundled one is
  correct — pointing cu130 torch at a 12.6 ptxas invites a mismatch.
- **SM 8.9 changes the attention backend.** `get_backend_list` in
  `cosmos_framework/model/attention/backends.py` gives `arch_tag>=80` →
  `["flash2","cudnn","natten"]`; `flash3` is Hopper-only. So absolute latencies are
  not comparable to NVIDIA's published FA3 numbers. All our comparisons are
  within-machine — record the selected backend in every result.
- `nsys`/`ncu` absent → PyTorch profiler only. `total_energy_consumption` is not a
  valid NVML field on these GPUs → energy is power-integrated at 10 Hz and must
  always be labelled ESTIMATED. GPU 0 idles at ~67 W vs ~34 W for GPUs 1–3
  (P0 vs P8), so report idle-adjusted energy alongside gross.
- HF token via `HF_TOKEN` env var only. Never write to disk. `hf auth login` OK.
  Note: `nvidia/Cosmos3-Nano-Policy-DROID` is **not gated** — no token needed.

## Research constraints (from spec)
- Do NOT retrain Cosmos3 from scratch. LoRA / adapters / small gates only.
- Do NOT optimize primarily for image quality.
- Task success vs total compute (with all overhead counted) is the main metric.
- Every claimed speedup requires: normalized FLOPs, measured GPU time, end-to-end latency, task success.
- Never accept licenses on the user's behalf (Cosmos, HF gated, Isaac EULA). Stop and ask.
- No run > 4 GPU-hours without explicit approval (report estimate first).
- Preserve upstream repos — modifications land as local branches under
  `third_party/*/research/action-aware-refresh` and are exported to
  `reproducibility/patches/`.

## Never commit
- HF tokens, credentials, model weights, generated videos, large parquet.
- The `third_party/` clones (they're pinned by SHA in `source_manifest.json`).

## Style
- Typed Python, dataclasses / pydantic. No hidden global state. No silent fallbacks.
- Structured logging (`structlog`). Deterministic seeds where possible.
- No hardcoded personal paths; use `configs/machine.yaml`.

## Shared-machine rule (firesim2)
The GPUs were **idle** when we moved here (2026-08-03) — but two other shared
resources are contended, and both distort measurements:

- **CPU.** Other users run large parallel compiles; the GPUs' CPU affinity is
  cores `0-15,32-47`. This perturbs every CPU-side stage: preprocessing,
  serialization, websocket round-trip, and Isaac Sim physics. Record `loadavg`
  and a process snapshot with **every** timed run, and prefer a quiet window
  for the final compute-anatomy numbers.
- **Disk.** `/scratch` is shared and was 96% full (≈134 GB free) on arrival,
  against a ~100 GB install budget. Check `df -h /scratch` before and after
  every install step; `uv cache prune` after each sync. If space runs out,
  **abort and report** — never let a shared volume fill.

Still check `nvidia-smi` for other users' processes before timing runs, and
note any contention in the result. Correctness runs are fine on a busy box;
timing runs are not.

## Current phase
M0–M2 (audit + baseline + compute anatomy). Do NOT start LoRA, RAFT, v2e, or
spatial modifications until compute anatomy is done and user has approved next phase.
