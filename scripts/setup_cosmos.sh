#!/usr/bin/env bash
# scripts/setup_cosmos.sh — install Cosmos Framework natively via `uv`.
#
# Container path is unavailable on this host (no nvidia-container-toolkit).
# So we install into `third_party/cosmos-framework/.venv` using the repo's
# own pyproject/uv config.
#
# Requires: cloned third_party/cosmos-framework (see scripts/clone_sources.sh).
#
# Env:
#   HF_TOKEN        must be exported before any model pull (never written to disk)
#   HF_HOME         set below → persistent cache on /scratch
#   UV_CACHE_DIR    set below → persistent uv cache

set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO_ROOT"

if [ ! -d third_party/cosmos-framework ]; then
  echo "error: third_party/cosmos-framework missing. Run scripts/clone_sources.sh first." >&2
  exit 2
fi

# ---- caches ---------------------------------------------------------------
# Read machine.yaml if it exists, otherwise use defaults from topology.yaml.
if [ -f configs/machine.yaml ]; then
  eval "$(python3 -c '
import yaml, sys
m = yaml.safe_load(open("configs/machine.yaml"))
p = m["paths"]
print(f"HF_HOME_DEFAULT={p[\"hf_cache\"]}")
print(f"UV_CACHE_DEFAULT={p[\"uv_cache\"]}")
')"
else
  HF_HOME_DEFAULT="/scratch/agustin/robotics/coleman/cache/huggingface"
  UV_CACHE_DEFAULT="/scratch/agustin/robotics/coleman/cache/uv"
fi
export HF_HOME="${HF_HOME:-$HF_HOME_DEFAULT}"
export UV_CACHE_DIR="${UV_CACHE_DIR:-$UV_CACHE_DEFAULT}"
mkdir -p "$HF_HOME" "$UV_CACHE_DIR"

echo "HF_HOME     = $HF_HOME"
echo "UV_CACHE_DIR= $UV_CACHE_DIR"

# ---- dep group selection --------------------------------------------------
# Driver on this host reports max CUDA runtime 13.0 → prefer cu130-train.
# If the checked-out cosmos-framework pyproject only ships cu128-train,
# fall back to that (both should be compatible with driver 580).
cd third_party/cosmos-framework

GROUP="${COSMOS_GROUP:-}"
if [ -z "$GROUP" ]; then
  if grep -q 'cu130-train' pyproject.toml 2>/dev/null; then
    GROUP="cu130-train"
  elif grep -q 'cu128-train' pyproject.toml 2>/dev/null; then
    GROUP="cu128-train"
  else
    echo "error: neither cu130-train nor cu128-train group found in pyproject.toml" >&2
    echo "→ inspect and set COSMOS_GROUP env var explicitly." >&2
    exit 3
  fi
fi

echo "cosmos-framework dependency group = $GROUP"
echo "→ uv sync --group=$GROUP --group=policy-server"

# ---- install --------------------------------------------------------------
# NOTE: this will download several GB of PyTorch + CUDA wheels the first
# time. Run this on a machine with good network and disk headroom.
uv sync --group="$GROUP" --group=policy-server

# ---- sanity ---------------------------------------------------------------
uv run python -c "import torch; print('torch', torch.__version__, 'cuda?', torch.cuda.is_available())"
uv run python -c "import cosmos_framework; print('cosmos_framework OK')" 2>&1 | tail -1

# ---- record the resolved group + versions --------------------------------
python3 - <<PY
import json, subprocess
info = {
  "cosmos_group": "$GROUP",
  "installed_utc": __import__("datetime").datetime.utcnow().replace(microsecond=0).isoformat() + "Z",
  "torch_version": subprocess.check_output(["uv", "run", "python", "-c", "import torch;print(torch.__version__)"], text=True).strip(),
}
open("$REPO_ROOT/reproducibility/cosmos_install.json", "w").write(json.dumps(info, indent=2))
PY

echo "setup_cosmos: done"
