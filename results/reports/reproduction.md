# Reproduction — exact commands from a clean machine

Every command below was actually run on `firesim2` (Ubuntu 24.04.4, 4× L40S, driver
580.126.18, no sudo). Where a step is host-specific it says so. Nothing here requires
root, and nothing writes a token to disk.

Read `docs/runbook_new_machine.md` first if the target host differs — it carries the
per-step reasoning and the mistakes worth not repeating. This file is the command list.

---

## 0. Prerequisites

Needed on the host: `git`, `curl`, `ffmpeg` (RoboLab's only apt prerequisite), an NVIDIA
GPU with a driver new enough for the wheel group in step 4, and **~100 GB of free disk on
a non-home volume**.

Not needed: `sudo`, Docker, `git-lfs`, a system Python 3.11 (`uv` supplies its own),
`nsys`, or an `HF_TOKEN` for the policy checkpoint (it is not gated).

```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
export PATH="$HOME/.local/bin:$PATH"
```

## 1. Clone and configure

```bash
git clone <fork-url> action-aware-refresh
cd action-aware-refresh

cp configs/machine.example.yaml configs/machine.yaml
$EDITOR configs/machine.yaml     # paths.hf_cache, paths.uv_cache -> big volume
$EDITOR configs/topology.yaml    # GPU roles by index + UUID, ov_cache_root, ov_data_root
```

**Do not skip the cache paths.** Every launcher resolves `HF_HOME` from
`configs/machine.yaml`; with them unset, Hugging Face writes to `$HOME/.cache/huggingface`
and pages the 34 GB checkpoint over NFS. That cost ~25 minutes on one model load here, and
the scripts now refuse to run rather than let it happen silently.

```bash
make audit          # writes docs/environment_report.md + reproducibility/environment.json
make test           # unit tests (torch-dependent ones skip here by design)
```

## 2. Sources and research patches

```bash
make sources        # clones third_party/*, pins SHAs, creates research branches
```

Then apply the patches — **the server cannot start without 0002/0003**:

```bash
for p in reproducibility/patches/cosmos-framework-*.patch; do
  git -C third_party/cosmos-framework am "$p"
done
for p in reproducibility/patches/robolab-*.patch; do
  git -C third_party/RoboLab am "$p"
done
```

What each one is and whether it affects the numbers: `docs/upstream_patches.md`.

Pinned bases (from `reproducibility/source_manifest.json`): cosmos-framework `a904d2d`,
RoboLab `0aef241`, checkpoint `nvidia/Cosmos3-Nano-Policy-DROID` @
`6706d7680581c255ff61e0f3bb49d90eac55c79e`.

## 3. Licences — yours to accept, not ours

```bash
# Read it first: https://docs.omniverse.nvidia.com/isaacsim/latest/common/NVIDIA_Omniverse_License_Agreement.html
export OMNI_KIT_ACCEPT_EULA=Y     # only if you accept
```

No script sets this. Note that `uv run pytest` inside RoboLab auto-accepts the EULA as a
side effect, which is why acceptance has to precede install verification.

Optional, and only for genuinely gated repos:

```bash
export HF_TOKEN=hf_...            # never written to a file
```

## 4. Install

```bash
make setup          # setup_cosmos.sh then setup_robolab.sh
make test-model     # unit tests in the cosmos venv (includes the torch-dependent ones)
```

Two separate venvs, deliberately — cosmos-framework pins a `cuXXX-train` torch, RoboLab
pins its own `pytorch-cu128` index and Isaac wheels. **Never invoke either with a bare
`uv run`**: it re-resolves to the project's default dependency set, which replaced
torch 2.10.0+cu130 with 2.13.0+cu130 here (breaking flash-attn's ABI) and would uninstall
`isaacsim` in RoboLab. Use `.venv/bin/python` directly, or `uv run --no-sync`.

`setup_robolab.sh` also installs `openpi-client`, which is **not** a declared RoboLab
dependency but *is* imported by `policies/cosmos3/client.py`. Without it `uv sync`
succeeds, RoboLab's own 160-test suite passes, and the closed-loop client dies with
`ModuleNotFoundError` only after Isaac Sim has finished booting.

## 5. M1 — closed-loop smoke test

```bash
OMNI_KIT_ACCEPT_EULA=Y COSMOS_GUARDRAILS=false SMOKE_TIMEOUT_S=1200 \
  bash scripts/smoke_test.sh
```

One command: it sequences server start → `/healthz` wait → primary task → alt task →
validation → process-group teardown. GPU roles come from `configs/topology.yaml` and are
asserted by **UUID** — do not pass `COSMOS_CUDA_DEVICES` / `ROBOLAB_CUDA_DEVICES` by hand,
because indices renumber across driver reloads and a silent swap attributes measurements
to the wrong device.

Result: `results/reports/baseline_smoke.md` (+ `.json`). It also drops a real captured
policy request under `results/raw/`, which the M2 probe uses instead of a synthetic one.

`COSMOS_GUARDRAILS=false` is a **recorded deviation**: upstream defaults guardrails on and
unconditionally downloads the gated `nvidia/Cosmos-Guardrail1`, so the official server
cannot start without approved access. Every latency taken this way excludes guardrail
cost. Set it `true` once access is granted and re-baseline.

## 6. M2 — compute anatomy

```bash
make profile        # B0/B1/B2 sweep, one config per process
```

One config per process because the model is ~31 GiB resident and is not reliably freed —
a second load in the same process OOMs. The probe refuses multi-config runs rather than
dying halfway.

Result: `docs/compute_anatomy.md`. The hand-written go/no-go lives in
`docs/compute_anatomy_conclusion.md` and is inlined verbatim, so regenerating the report
cannot destroy it.

**Measurement discipline that matters for any comparison.** Add `--cooldown-s 8` for
steady-state numbers:

```bash
ANATOMY_CONFIGS=B3_replay_cooled ANATOMY_ITERS=25 \
  ANATOMY_EXTRA="--no-guardrails --cooldown-s 8" bash scripts/run_anatomy_sweep.sh
```

Back-to-back, identical work spans 3,377–5,974 ms on this host; with an idle gap the
median is 3,430.9 ms with a MAD of 0.34% and +0.06% drift. It is **not** thermal or GPU
throttling (`sm_clock` is locked at 1065 MHz, throttle reasons `0x0` throughout) — it is
host-side interference on a shared box. So: median-of-many, idle gap, configs interleaved,
and report medians rather than means.

## 7. M3/M4 — the experiment chain

One resumable command runs everything:

```bash
OMNI_KIT_ACCEPT_EULA=Y COSMOS_GUARDRAILS=false ./.venv/bin/python scripts/run_phases.py
```

```bash
./.venv/bin/python scripts/run_phases.py --status         # what is done / pending
./.venv/bin/python scripts/run_phases.py --only p1_offline
./.venv/bin/python scripts/run_phases.py --retry-failed   # re-attempt errored steps
```

**Killing it is safe.** Every unit of work is claimed through
`src/action_refresh/ledger.py`, which writes `result.json` via atomic rename — a unit is
either absent or complete, never torn. Re-invoking the same command continues from where
it stopped: a mid-sweep kill during development preserved 275 of 704 completed units and
resumed at unit 276. Failed units are recorded distinctly from never-attempted ones and
are not retried unless asked, so a deterministic failure cannot burn GPU-hours in a loop.

The phases, in order:

| phase | what | GPU |
|---|---|---|
| `p0_corpus` | capture real policy requests from closed-loop episodes | ~0.4 h |
| `p1_offline` | Experiments A (denoising steps), E0 (seed spread), E1 (no imagination) | ~1.5 h |
| `p1_offline` | Experiment E2b — shorten the imagined horizon (9→2 latent frames) | ~0.6 h |
| `p2_oracle_offline` | Experiment C feasibility from recorded episodes | none |
| `p3_pareto` | closed-loop screened pilot (9 tasks × 1), baseline + reduced-step | ~1.4 h each |
| `p4_horizon` | Experiment B horizon sweep | ~1.4 h each |
| `p5_report` | interleaved cooled latency, then success vs normalized compute | ~0.4 h |

**Budget warning learned the hard way.** A full 16-task pilot pass costs **~7.3 GPU-h**, not
the ~2.5 first estimated — because Isaac dominates (policy inference was only **16.8%** of
closed-loop wall time) and *failing* tasks run to their full timeout while successes
terminate early. A worse method therefore costs *more* to evaluate, so budgets built from
mean episode length fail in the unsafe direction. The `pilot_screened` set (9 tasks, derived
from measured baseline results) exists for this reason and costs ~1.4 GPU-h per method; see
`experiments/task_sets.yaml` for its derivation and its stated bias.

Individual pieces, if you want them separately:

```bash
# corpus (needed by everything offline; skips tasks already captured)
OMNI_KIT_ACCEPT_EULA=Y bash scripts/capture_corpus.sh

# offline screens (cosmos venv; HF_HOME must be set or it refuses to run)
HF_HOME=<paths.hf_cache> ./third_party/cosmos-framework/.venv/bin/python \
  scripts/offline_action_study.py \
  --corpus-glob 'results/raw/corpus/*/captured_request_*.npz' \
  --steps 1,2,3,4 --seeds 4 --action-only --vision-frames 17,9,5
./.venv/bin/python scripts/analyze_offline_study.py

# Experiment C, offline, no GPU (RoboLab venv — needs h5py)
PYTHONPATH=$PWD/src ./third_party/RoboLab/.venv/bin/python \
  scripts/analyze_oracle_temporal.py

# Experiment F, offline: cross-denoising-step residuals, split by modality (no cache built)
HF_HOME=<paths.hf_cache> ./third_party/cosmos-framework/.venv/bin/python \
  scripts/analyze_denoising_residuals.py

# one closed-loop method. --port matters: run two concurrently on disjoint GPUs and they
# will collide on 8000 otherwise.
OMNI_KIT_ACCEPT_EULA=Y ./.venv/bin/python scripts/run_closed_loop.py \
  --method baseline_full --set pilot_screened --episodes 1
# a second method in parallel, on the spare GPUs:
OMNI_KIT_ACCEPT_EULA=Y COSMOS_CUDA_DEVICES=1 ROBOLAB_CUDA_DEVICES=3 \
  ./.venv/bin/python scripts/run_closed_loop.py \
  --method baseline_steps_2 --set pilot_screened --episodes 1 --port 8001

# authoritative latency: interleaved, cooled, median-of-many. REFUSES to run while another
# process holds the GPU — every speedup quoted in the project comes from here, never from
# the offline study's wall-time column.
HF_HOME=<paths.hf_cache> ./third_party/cosmos-framework/.venv/bin/python \
  scripts/measure_latency.py --repeats 10 --cooldown-s 5

./.venv/bin/python scripts/build_pareto.py
```

## 8. Outputs

| file | what |
|---|---|
| `docs/environment_report.md` | machine inventory |
| `docs/baseline_contract.md` | verified upstream defaults |
| `docs/compute_anatomy.md` | M2: stage breakdown, token census, measurement floor |
| `docs/offline_action_study.md` | Experiments A / E0 / E1 / E2b, action deviation + chunk motion |
| `docs/oracle_temporal.md` | Experiment C feasibility |
| `docs/denoising_residuals.md` | Experiment F: cross-step residuals split by modality |
| `docs/latency.md` | authoritative per-configuration latency (interleaved, cooled) |
| `docs/pareto.md` | M3: success vs normalized total compute |
| `docs/upstream_patches.md` | every patch, why, and whether it moves the numbers |
| `docs/decision_log.md` | decisions and negative results, dated |
| `results/reports/baseline_smoke.md` | M1 pass/fail with real log excerpts |

## 9. Things that are deliberately absent

- **No `nsys`/`ncu`** on this host — PyTorch profiler only. Chrome traces are gitignored
  (432 MB per sweep) but regenerate with `make profile`; their paths are recorded in
  `docs/compute_anatomy.md`.
- **No end-effector-space action deviation.** It needs forward kinematics from the Franka
  model, which lives in Isaac and not in the Cosmos venv the offline sweeps run in. Joint
  deviation is reported instead and is explicitly *not* a substitute.
- **No energy counter.** `total_energy_consumption` is unsupported on these GPUs; energy
  is power-integrated at 10 Hz and always labelled ESTIMATED, with idle measured
  separately.
- **No guardrails**, as above — the one deviation that changes the numbers.
