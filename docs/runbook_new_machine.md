# Runbook — bringing this repo up on a new machine

Rewritten after actually executing it end-to-end on `firesim2` (Session 2).
Session 1 wrote this from the scaffolding without running any of it; several
steps below were wrong in ways that cost real time, and each correction is
marked **⚠ learned the hard way** so it does not get re-optimized away.

Assumes the new machine has:
- Linux, glibc ≥ 2.35 (Ubuntu 24.04 is what the Cosmos image is based on)
- ≥ 1 NVIDIA GPU, SM ≥ 75, driver new enough for the wheel group you pick
- `git`, `curl`, `ffmpeg` (RoboLab's only apt prerequisite)
- **no sudo required** — everything below is user-space

`uv` supplies its own Python, so a system `python3.11` is not needed.
`git-lfs` is also not needed: `hf download` does not use it.

The Cosmos repo documents a `docker build` path. You do not need it — the
image's base is plain `ubuntu24.04` and its only additions over a normal host
are `curl/ffmpeg/git/git-lfs/tree/wget`. Nothing in either install needs root.

## 0. Clone this repo

```bash
git clone <fork-url> action-aware-refresh
cd action-aware-refresh
```

## 1. Install user-space tooling

```bash
curl -LsSf https://astral.sh/uv/install.sh | sh    # -> ~/.local/bin/uv
export PATH="$HOME/.local/bin:$PATH"
```

## 2. Audit the machine

```bash
make audit
```

Read `docs/environment_report.md`. It records driver/CUDA version, per-GPU
VRAM, `nvidia-smi topo -m` affinity, NVML field availability, and free disk.
Three numbers there decide the rest of the run:

- **usable VRAM per GPU** vs the checkpoint's 32.9 GB of bf16 weights.
- **driver CUDA version** → the `cuXXX-train` wheel group in step 6.
- **free disk on the target volume** → budget ~100 GB (see step 3).

## 3. Caches and machine config

```bash
cp configs/machine.example.yaml configs/machine.yaml
$EDITOR configs/machine.yaml   # paths.hf_cache / paths.uv_cache -> big volume
$EDITOR configs/topology.yaml  # GPU roles, ov_cache_root / ov_data_root
```

Budget, measured on this host:

| item | size |
|---|---|
| Cosmos3-Nano checkpoint (HF cache) | 32.9 GB |
| cosmos-framework venv + uv cache | ~25 GB |
| RoboLab venv (`isaacsim[all,extscache]` + torch) | ~25 GB |
| RoboLab assets | ~8 GB |
| Isaac/Omniverse runtime + shader caches | ~5–10 GB |
| **total** | **~100 GB** |

⚠ **learned the hard way — put every cache on the big volume, and make sure
the *run* path exports the same variables the *install* path did.**
`setup_cosmos.sh` exported `HF_HOME` but `start_cosmos_server.sh` did not, so
the server silently re-downloaded all 34 GB into `$HOME/.cache/huggingface` (a
60 GB NFS volume) and then mmap-paged it back at 20 MiB/s — a 23-second model
load became ~25 minutes. Isaac does the same thing with `OMNI_CACHE_ROOT` /
`OMNI_DATA_ROOT`. Both launchers now call `export_cache_env` /
`export_omni_cache_env` from `scripts/lib/common.sh`; keep it that way. **A run
path that diverges from the install path means the thing you measure is not the
thing you installed.**

Keep the uv cache on the *same filesystem* as the venvs so uv hardlinks instead
of copying, and `uv cache prune` after each sync.

## 4. HF authentication

⚠ **learned the hard way — `nvidia/Cosmos3-Nano-Policy-DROID` is NOT gated.**
No `HF_TOKEN`, no click-through. Session 1 recorded a token requirement as a
blocker; it does not exist. (License is `other` = NVIDIA Open Model License —
worth reading, nothing to accept.)

```bash
# Optional, and only for genuinely gated repos. Never write it to a file.
export HF_TOKEN=hf_...
```

One repo *is* gated and does block the official baseline:
**`nvidia/Cosmos-Guardrail1`** (currently ACCESS DENIED for us). See step 7.

## 5. Clone and pin third-party sources

```bash
make sources
```

Populates `third_party/`, writes resolved SHAs into
`reproducibility/source_manifest.json`, and creates the
`research/action-aware-refresh` branch in each clone.

Then apply the research patches — **the server cannot start without 0001–0003**:

```bash
for p in reproducibility/patches/cosmos-framework-*.patch; do
  git -C third_party/cosmos-framework am "$p"
done
git -C third_party/RoboLab am reproducibility/patches/robolab-0001-*.patch
```

What they are and why: `docs/upstream_patches.md`.

## 6. Derive the baseline contract

```bash
make contract
```

Reads the checked-out server + client and writes `docs/baseline_contract.md`.
Its regexes are first-match-wins heuristics — **hand-verify every field against
the clone.** Do not run baseline experiments against a contract containing
`UNKNOWN` or "expected" values.

## 7. Install Cosmos + RoboLab

```bash
make setup    # setup_cosmos.sh then setup_robolab.sh
```

Two **separate** venvs, deliberately: cosmos-framework pins a `cuXXX-train`
torch, RoboLab pins its own `pytorch-cu128` index and Isaac wheels. They must
never share an environment.

⚠ **learned the hard way — never invoke either stack with a bare `uv run`.**
It re-resolves to the project's *default* dependency set: in cosmos-framework
that replaced torch 2.10.0+cu130 with 2.13.0+cu130 and broke flash-attn's ABI;
in RoboLab it uninstalls `isaacsim` right before you try to launch the
simulator. Use `.venv/bin/python` directly, or `uv run --no-sync`.

Do **not** copy the Dockerfile's `TRITON_PTXAS_PATH=/usr/local/cuda/bin/ptxas`.
That line exists because triton's bundled `ptxas` lagged Blackwell; on older
arches the bundled one is correct, and pointing a cu130 torch at a CUDA 12.6
`ptxas` invites a mismatch.

⚠ **learned the hard way — `openpi-client` is not a declared RoboLab
dependency.** `policies/cosmos3/client.py` imports it, but upstream keeps it out
of `pyproject.toml` so non-openpi backends don't pull it in. `uv sync` succeeds,
RoboLab's own 160-test suite passes, and then the closed-loop client dies with
`ModuleNotFoundError` *after* Isaac Sim has finished booting. `setup_robolab.sh`
now installs it, pinned to the version the server side resolved (both ends must
agree on the msgpack wire format).

### EULA — the user must accept, not a script

Isaac Sim will not launch without it, and `uv run pytest` inside RoboLab
auto-accepts it as a side effect. Neither `setup_robolab.sh` nor
`run_robolab.sh` sets it:

1. read https://docs.omniverse.nvidia.com/isaacsim/latest/common/NVIDIA_Omniverse_License_Agreement.html
2. if **you** accept: `export OMNI_KIT_ACCEPT_EULA=Y`
3. record the authorization in `docs/legal_checkpoint.md` (who, when)

### Guardrails — a recorded deviation, not a default

Upstream's `OmniSetupArgs.guardrails` defaults `True` and unconditionally
downloads the gated `nvidia/Cosmos-Guardrail1`, so **the official server cannot
start without approved access.** Research patch 0001 adds a flag;
`COSMOS_GUARDRAILS` defaults to `false` so the smoke test can run at all.

Every latency taken that way **excludes guardrail cost, which is part of the
official baseline.** `start.json` records this under
`deviations_from_official`. Set `COSMOS_GUARDRAILS=true` once access is granted
and re-baseline.

### First VRAM gate

Load the checkpoint and record `peak_allocated` / `peak_reserved` before
anything else. If it OOMs, the honest options are reduced concurrency or
sharding the DiT across two GPUs — recorded as *the* baseline topology and
applied consistently. **Not quantization**: spec §5 requires that be a separate
experiment, not a way to make the baseline fit.

## 8. Smoke test (M1)

```bash
export OMNI_KIT_ACCEPT_EULA=Y
make smoke
```

That is the whole command. ⚠ **learned the hard way — do not hand-run the
server and client in two terminals with `CUDA_VISIBLE_DEVICES=0` / `=1`.** GPU
roles come from `configs/topology.yaml` and are asserted by UUID (indices
renumber across driver reloads; a silent swap attributes every measurement to
the wrong device). The old two-terminal recipe here also pinned the client to
GPU 1, which on this host shares a PCIe switch with the server's GPU 0 — the
pairing the topology deliberately avoids.

`smoke_test.sh` sequences server start → `/healthz` wait → primary task → alt
task → validation → teardown, and kills the whole process group on exit (an
orphaned server pins ~31 GB of VRAM on a shared box). Raise `SMOKE_TIMEOUT_S`
if the weights are cold.

Result: `results/reports/baseline_smoke.md`. The run also drops a real captured
policy request under `results/raw/` for step 9.

## 9. Compute anatomy (M2)

```bash
make profile
```

Writes `docs/compute_anatomy.md` — this is the milestone that answers the
project's go/no-go (is visual imagination a substantial fraction of deployed
cost, or is the cost shared/action computation?). Do **not** start M3+ method
work until that file answers it in writing.

Uses the **in-process** probe, not the wire-level one: the server exposes no
per-stage timings, and `num_steps` / `decode_video` are server-*start* args, so
a wire-level sweep can neither attribute stages nor vary configs. One config
per process — the model is ~31 GiB resident and is not reliably freed.

## 10. End-of-session report

Populate `results/reports/session_<N>_report.md` in the spec §17 format:
what succeeded, what failed, repo SHAs, deployment topology, exact baseline
command, smoke result, dominant measured compute component, next three
experiments, estimated GPU-hours, blockers needing user action.

---

## Common problems

- **`ModuleNotFoundError` for a repo module in a shell script:** the script
  used bare `python3` (the system interpreter, which has PyYAML but not
  pydantic). Use `resolve_repo_py` from `scripts/lib/common.sh`. Note the *repo*
  venv (`.venv`: pydantic/yaml/structlog for our tooling) is a different
  environment from `third_party/*/.venv` (torch, isaacsim). Never conflate them.
- **Model load takes tens of minutes:** you are paging weights over NFS. Check
  `HF_HOME` in the launcher's banner, and process state — `D`/`Dl` with
  `wchan = folio_wait_bit_common` is mmap page-fault I/O. See step 3.
- **`nvidia-smi` shows another user at 100% util:** correctness runs are fine on
  a busy GPU; timing runs are not. Also record `loadavg` — CPU contention
  perturbs preprocessing, serialization, websocket round-trip, and Isaac
  physics, all of which land in end-to-end latency.
- **Single-rank NCCL segfault at server start:** patches 0002/0003. Upstream's
  download and `sync_model_states` paths issue collectives that abort at
  `world_size == 1`.
- **`ImportError: libcudnn.so.9`:** the wheel group does not match the driver.
  Delete `.venv` and re-run with `COSMOS_GROUP=cu128-train`.
- **Isaac Sim exits silently, or `input()` raises `EOFError` at
  `omni/kit_app.py:check_eula`:** the EULA is not accepted. See step 7.
- **Isaac teardown noise** (`Could not find category 'Replicator:Annotators'`,
  `USD stage detach not called`, `Recursive unloadAllPlugins()`): harmless
  shutdown warnings. They appear on successful runs too — do not read them as
  the cause of a failure.
