#!/usr/bin/env bash
# scripts/setup_robolab.sh — install RoboLab natively via `uv`.
#
# Reads the checked-out repo's README/pyproject to pick the right extra
# (`isaac51` if the repo supports it; fall back to `isaac50`). NEVER
# installs both.
#
# Requires: cloned third_party/RoboLab.
#
# WARNING: Isaac Sim requires driver-level libs + a lot of disk (~15–30 GB).
# You may also need to accept the Omniverse EULA — this script will stop
# and print instructions if that's the case (never sets OMNI_KIT_ACCEPT_EULA=Y
# automatically).

set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO_ROOT"

if [ ! -d third_party/RoboLab ]; then
  echo "error: third_party/RoboLab missing. Run scripts/clone_sources.sh first." >&2
  exit 2
fi

cd third_party/RoboLab

# ---- pick Isaac stack -----------------------------------------------------
EXTRA="${ROBOLAB_EXTRA:-}"
if [ -z "$EXTRA" ]; then
  if grep -q 'isaac51' pyproject.toml 2>/dev/null; then
    EXTRA="isaac51"
  elif grep -q 'isaac50' pyproject.toml 2>/dev/null; then
    EXTRA="isaac50"
  else
    echo "error: neither isaac51 nor isaac50 extra found in RoboLab pyproject.toml" >&2
    echo "→ inspect and set ROBOLAB_EXTRA env var explicitly." >&2
    exit 3
  fi
fi
echo "RoboLab extra = $EXTRA (Blackwell requires 5.1 — confirm before running)"

# ---- env ------------------------------------------------------------------
uv venv --python 3.11 .venv
# shellcheck disable=SC1091
source .venv/bin/activate

# ---- deps -----------------------------------------------------------------
uv sync --extra "$EXTRA"

# ---- EULA check -----------------------------------------------------------
# The Omniverse Isaac Sim launcher aborts if the EULA is not accepted.
# We *do not* set OMNI_KIT_ACCEPT_EULA=Y here. If Isaac needs it, the user
# must set it themselves (implicitly accepting).
if [ -z "${OMNI_KIT_ACCEPT_EULA:-}" ]; then
  cat <<EOF

WARNING: OMNI_KIT_ACCEPT_EULA is not set. Isaac Sim / Omniverse will refuse
to launch without accepting the EULA. Review the EULA here:

  https://docs.omniverse.nvidia.com/isaacsim/latest/common/NVIDIA_Omniverse_License_Agreement.html

If you accept, set  \`export OMNI_KIT_ACCEPT_EULA=Y\`  in your shell before
running \`scripts/run_robolab.sh\`. Setup itself does not require it.

EOF
fi

# ---- record versions ------------------------------------------------------
python3 - <<PY
import json, subprocess, datetime as dt
def try_cmd(cmd):
    try:
        return subprocess.check_output(cmd, text=True, stderr=subprocess.STDOUT).strip()
    except Exception as e:
        return f"ERROR: {e}"
info = {
  "isaac_extra": "$EXTRA",
  "installed_utc": dt.datetime.utcnow().replace(microsecond=0).isoformat() + "Z",
  "isaac_sim_version": try_cmd(["uv", "run", "python", "-c", "import isaacsim, importlib.metadata as m; print(m.version('isaacsim'))"]),
  "isaac_lab_version": try_cmd(["uv", "run", "python", "-c", "import importlib.metadata as m; print(m.version('isaaclab'))"]),
  "robolab_commit": try_cmd(["git", "-C", ".", "rev-parse", "HEAD"]),
}
open("$REPO_ROOT/reproducibility/robolab_install.json", "w").write(json.dumps(info, indent=2))
PY

echo "setup_robolab: done"
