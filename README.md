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

## Prerequisites (this machine)

- RHEL 9.7, 2× NVIDIA RTX PRO 6000 Blackwell (98 GB VRAM), driver 580.
- **No sudo, no docker, no NVIDIA container toolkit.** All installs are
  native, user-space, via `uv`.
- Python 3.11 via `uv`. HF token as env var only (never on disk).
- Machine is shared with other users — measurement runs must wait for a
  quiet window (see `docs/environment_report.md`).

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
