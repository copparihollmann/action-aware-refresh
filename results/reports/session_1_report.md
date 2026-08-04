# Session 1 report — repo scaffolding

Date: 2026-08-03
Host used for scaffolding: `bwrc-bwell` (no execution performed here — GPUs
were shared with other users at the time).

## What succeeded

- **Machine audit** (`docs/environment_report.md`,
  `reproducibility/environment.json`). Host inventoried without any
  side effects.
- **Repository scaffolded** end-to-end per spec §4:
  - directory tree
  - `pyproject.toml`, Makefile, `.gitignore`, README, CLAUDE.md
  - typed schemas in `src/action_refresh/{metrics,profiler,energy,config,logging}.py`
  - unit tests (`tests/unit/`) — do not require a GPU
  - experiment registry + task-set skeleton
  - reproducibility manifests (`source_manifest.json`,
    `model_revisions.json`, `commands.jsonl`)
- **All shell scripts** written and ready to run on a target machine:
  - `scripts/audit_machine.sh` (runs — verified here)
  - `scripts/clone_sources.sh` + `scripts/update_manifest.py`
  - `scripts/setup_cosmos.sh`, `scripts/setup_robolab.sh`
  - `scripts/start_cosmos_server.sh`, `scripts/run_robolab.sh`
  - `scripts/smoke_test.sh` + `scripts/validate_smoke.py`
  - `scripts/profile_baseline.sh` (+ analysis package modules)
  - `scripts/derive_baseline_contract.py`
  - `scripts/run_sweep.py`, `scripts/build_report.py`
- **Docs seeded**: literature notes, source map, decision log,
  troubleshooting, experiment protocol, compute-anatomy placeholder,
  legal checkpoint, runbook for a new machine.
- **Git repo initialised** with one commit (M0 scaffolding).

## What failed / was deferred

- **Nothing was executed against the GPUs on this host.** The user's
  standing preference is: don't run installers/experiments on shared
  machines when others are actively using them. This includes `uv sync`
  and any HF pull.
- **`third_party/` is empty** — deliberately. `scripts/clone_sources.sh`
  runs on the target machine and populates it with resolved SHAs.
- **`docs/baseline_contract.md`** is a placeholder — `make contract`
  fills it in after cloning.
- **`docs/compute_anatomy.md`** is a placeholder — `make profile` fills
  it in after installing Cosmos + RoboLab.

## Current repo SHAs

```
$(cd /scratch/agustin/robotics/coleman/action-aware-refresh && git log --oneline -1)
```

(Update at commit time.)

## Selected deployment topology

`single_host_multi_gpu` — one Blackwell GPU pinned to Cosmos policy
server, the other to RoboLab/Isaac Sim via `CUDA_VISIBLE_DEVICES`. See
`configs/topology.yaml`. This may need reconfiguring on a different host.

## Exact baseline command (once installed on target machine)

Terminal A:
```
COSMOS_CUDA_DEVICES=0 \
scripts/start_cosmos_server.sh
```

Terminal B (after `curl http://127.0.0.1:8000/healthz` returns 200):
```
export OMNI_KIT_ACCEPT_EULA=Y HF_TOKEN=hf_...
ROBOLAB_CUDA_DEVICES=1 \
make smoke
```

## Baseline smoke result

Not run on `bwrc-bwell`. Report will be at
`results/reports/baseline_smoke.md` on the target machine after
`make smoke`.

## Dominant measured compute component

Unknown — pending `make profile` on the target machine. This is the M2
go/no-go artifact and determines whether the visual-imagination branch
of the project (M6, M7) is prioritized.

## Blockers requiring user action

1. **HF gated-repo access + `HF_TOKEN`.** Accept the Cosmos3-Nano-Policy-
   DROID gate on Hugging Face; export `HF_TOKEN` on the target machine
   (or `uvx hf@latest auth login`).
2. **Isaac Sim EULA.** `OMNI_KIT_ACCEPT_EULA=Y` must be set explicitly
   in the shell that runs the RoboLab client.
3. **Cosmos model license.** Read on the HF model card; accepting the
   gate implies agreement.
4. **Target machine availability.** This host is shared and heavily
   loaded; scaffolding here, but installs + runs happen elsewhere.

## Proposed next three experiments (Session 2, after M1 smoke + M2 anatomy)

Ordered by the rapid-prototype ladder in spec §11 — the ones with the
biggest expected signal for the smallest effort:

1. **Experiment A — Reduced denoising steps** (`baseline_steps_{1,2,3,4}`).
   Strongest trivial baseline; all later caching methods must beat this
   frontier. Offline first (recorded observations), then closed-loop PILOT.
2. **Experiment B — Action-chunk / policy-call reuse**
   (`baseline_fixed_horizon_{8,16,32}`). Requires first verifying the
   client's actual open-loop behaviour from source; then swept.
3. **Experiment E — Action-only / no-imagination baseline**. The Fast-WAM
   challenge. Must be run early to know whether visual caching is worth
   pursuing at all.

## Estimated GPU-hours for Session 2

Given 2× Blackwell 98 GB and a single client per run:

- Compute anatomy (`make profile`): ~1 GPU-hour (setup + warmups + 5
  configs × 30 iters + trace).
- Experiment A: 4 configs × 5 seeds × PILOT ≈ 12–18 tasks × 5 episodes
  ≈ 300 short episodes ≈ 2–4 GPU-hours.
- Experiment B: same order.
- Experiment E: 2–3 GPU-hours (assumes action-only path is straightforward).

Total Session 2: ~8–12 GPU-hours. **Chunk into ≤ 4 GPU-hour batches** with
approval between each; do not launch a single 10-hour run.

## Session summary

Repo is ready to be moved to a quieter, GPU-available machine. On that
machine, follow `docs/runbook_new_machine.md` step-by-step. The scripts
are the source of truth for exact commands.
