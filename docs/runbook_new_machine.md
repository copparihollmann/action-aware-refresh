# Runbook — bringing this repo up on a new machine

Assumes the new machine has:
- Linux (Ubuntu 22.04+ preferred, RHEL 9 OK)
- ≥ 1 NVIDIA GPU with recent driver
- `uv`, `git`, `curl`, `git-lfs`, `python3.11`

If the new machine has sudo + docker + nvidia-container-toolkit, you may
prefer the container path documented in the Cosmos Framework repo. This
runbook is for the native `uv` path we set up here (which works with no
sudo).

## 0. Clone your fork of this repo

```bash
git clone <fork-url> action-aware-refresh
cd action-aware-refresh
```

## 1. Audit the machine

```bash
make audit
```

Read `docs/environment_report.md`. If it lists new blockers (e.g. missing
`git-lfs`, no NVML power counter), consult `docs/troubleshooting.md`.

## 2. Set up caches + machine config

```bash
cp configs/machine.example.yaml configs/machine.yaml
$EDITOR configs/machine.yaml   # set paths to a large-disk mount
```

Also edit `configs/topology.yaml` if the GPU assignment differs from the
original host.

## 3. HF authentication (never write the token to disk)

```bash
export HF_TOKEN=hf_...              # your gated-repo token
# or:
uvx hf@latest auth login            # persists to ~/.cache/huggingface
```

Confirm you can access the gated model:

```bash
curl -sfL -H "Authorization: Bearer $HF_TOKEN" \
  https://huggingface.co/api/models/nvidia/Cosmos3-Nano-Policy-DROID >/dev/null && echo OK
```

## 4. Clone third-party sources

```bash
make sources
```

This populates `third_party/` and refreshes
`reproducibility/source_manifest.json` with resolved commit SHAs.

## 5. Derive the baseline contract

```bash
make contract
```

Reads the checked-out server + client and writes
`docs/baseline_contract.md`. If any field is `UNKNOWN`, open the referenced
source file and fill it in manually — do NOT run baseline experiments
against a partially-known contract.

## 6. Install Cosmos + RoboLab

```bash
make setup
```

This runs `scripts/setup_cosmos.sh` (docker path unavailable → native uv
into `third_party/cosmos-framework/.venv`) and `scripts/setup_robolab.sh`
(uv into `third_party/RoboLab/.venv` with the Isaac extra chosen from the
repo's own pyproject).

If Isaac Sim complains about the EULA:
1. read https://docs.omniverse.nvidia.com/isaacsim/latest/common/NVIDIA_Omniverse_License_Agreement.html
2. if you accept: `export OMNI_KIT_ACCEPT_EULA=Y`
3. re-run `make setup`

**Do not commit the venvs — they're gitignored.**

## 7. Smoke test

Terminal A:
```bash
COSMOS_CUDA_DEVICES=0 scripts/start_cosmos_server.sh
```

Terminal B (after `/healthz` responds):
```bash
export OMNI_KIT_ACCEPT_EULA=Y
ROBOLAB_CUDA_DEVICES=1 make smoke
```

`results/reports/baseline_smoke.md` will contain the go/no-go.

## 8. Compute anatomy

```bash
make profile
```

Writes `docs/compute_anatomy.md` — this is the M2 milestone. Do NOT
proceed to M3+ methods until this file answers the go/no-go
(visual imagination cost fraction).

## 9. End-of-session report

Populate `results/reports/session_1_report.md` per spec §17. Send it to
the user for the next-phase decision.

---

## Common problems

- **`nvidia-smi` shows 100% GPU util from another user:** wait or use the
  other GPU via `CUDA_VISIBLE_DEVICES`. Correctness runs are fine on a
  shared GPU; timing runs are not.
- **`libnvidia-ml.so.1` missing:** installer/container issue — either
  install driver-runtime on the host or use a container base that ships
  it.
- **`ImportError: libcudnn.so.9`:** the uv install pulled a torch built
  against a different cuDNN. Delete `.venv`, re-run `make setup` with
  `COSMOS_GROUP=cu128-train` if the host driver only supports CUDA 12.x.
- **Isaac Sim silent exit:** almost always the EULA. See step 6.
