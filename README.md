# Action-Aware Predictive Refresh for Efficient World-Action Models

Research prototype: measure whether pretrained world-action model inference
(Cosmos3-Nano-Policy-DROID + RoboLab) can be sped up by reusing keyframes,
action chunks, denoising intermediates, and spatial features, while
refreshing only action-relevant changes — without losing closed-loop robot
task success.

The full research spec is checked in as `docs/research_spec.md`. This README
only tells you how to work with the repo.

## Session status

Current milestone: **M0–M2** (source checkout, environment audit, baseline
smoke, compute anatomy). Later milestones require explicit user checkpoints
and GPU-hour approval.

## Prerequisites (this machine: `firesim2`)

- Ubuntu 24.04.4, **4× NVIDIA L40S** (45.0 GiB each, SM 8.9), driver 580.126.18,
  CUDA 13.0. GPU 0 → Cosmos server, GPU 2 → RoboLab/Isaac, GPUs 1/3 spare.
- **No sudo.** Docker and `nvidia-container-toolkit` exist on the host but this
  uid is not in the `docker` group, so all installs are native, user-space, via
  `uv` — which needs no root anyway. `ffmpeg` (RoboLab's only apt prerequisite)
  is already present.
- Python 3.11 via `uv`. HF token as env var only (never on disk) — though the
  Cosmos3-Nano checkpoint is not gated, so no token is required.
- Shared machine: the **GPUs are idle**, but CPU and `/scratch` are contended.
  Record `loadavg` with every timed run and watch `df -h /scratch` (≈134 GB free
  against a ~100 GB install budget). See `docs/environment_report.md`.

## Workflow

```
make audit      # inventory host, write docs/environment_report.md
make sources    # clone third_party/*, write source_manifest.json
make contract   # verify checked-out Cosmos config → baseline_contract.md
make setup      # native uv envs for Cosmos + RoboLab
make test       # unit tests for metrics/profiler/energy
make smoke      # server /healthz + one RoboLab task
make profile    # compute anatomy B0–B4
```

## Picking this up cold

Start with **`docs/handover.md`** — current state, live blockers (including host conditions
that will silently ruin a run), what is sanctioned to do next, and the gotchas that have
already cost time. Then `docs/decision_log.md` (last sections = current direction) and the
per-session reports under `results/reports/`.

## Directory map

- `configs/`      — machine, topology, tasks, methods, sweeps
- `docs/`         — literature notes, baseline contract, anatomy, decision log
- `reproducibility/` — source manifest, env snapshot, command log, patches
- `third_party/`  — pinned clones of Cosmos, RoboLab, cosmos-policy, openpi
- `src/action_refresh/` — measurement + method code
- `scripts/`      — audit, setup, run, profile
- `tests/`        — unit / integration / smoke
- `experiments/`  — registry.yaml, task_sets.yaml
- `results/`      — raw records, processed tables, profiles, plots, reports

## Non-secrets policy

Never commit: HF tokens, model weights, generated videos, large parquet.
See `.gitignore`. Everything under `third_party/*/` is untracked here —
they are pinned by SHA in `reproducibility/source_manifest.json`.
