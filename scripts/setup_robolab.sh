#!/usr/bin/env bash
# scripts/setup_robolab.sh — install RoboLab natively via `uv`.
#
# Reads the checked-out repo's pyproject to pick the Isaac extra, defaulting to
# upstream's documented `isaac50` (IsaacSim 5.0 / IsaacLab 2.2.0). The extras are
# mutually exclusive and this NEVER installs both — mixing simulator versions
# would invalidate every cross-method comparison.
#
# Requires: cloned third_party/RoboLab.
#
# WARNING: Isaac Sim requires driver-level libs + a lot of disk (~35 GB with
# assets).
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
# `isaac50` (IsaacSim 5.0 / IsaacLab 2.2.0) is upstream's documented default.
# The two extras are declared mutually exclusive in RoboLab's pyproject
# (`[tool.uv] conflicts`) and MUST NOT share a venv. Whichever we pick, the
# same simulator stack must be used for every comparison in the project.
EXTRA="${ROBOLAB_EXTRA:-}"
if [ -z "$EXTRA" ]; then
  if grep -q 'isaac50' pyproject.toml 2>/dev/null; then
    EXTRA="isaac50"
  elif grep -q 'isaac51' pyproject.toml 2>/dev/null; then
    EXTRA="isaac51"
  else
    echo "error: neither isaac50 nor isaac51 extra found in RoboLab pyproject.toml" >&2
    echo "→ inspect and set ROBOLAB_EXTRA env var explicitly." >&2
    exit 3
  fi
fi
echo "RoboLab extra = $EXTRA (upstream default is isaac50; override with ROBOLAB_EXTRA)"

# ---- caches ---------------------------------------------------------------
# Omniverse defaults its caches into $HOME, which here is a small NFS volume.
# Redirect to the paths recorded in configs/topology.yaml.
if [ -f "$REPO_ROOT/configs/topology.yaml" ]; then
  eval "$(python3 -c '
import yaml
t = yaml.safe_load(open("'"$REPO_ROOT"'/configs/topology.yaml"))
for key, var in (("ov_cache_root", "OV_CACHE"), ("ov_data_root", "OV_DATA"), ("uv_cache", "UVC")):
    v = t.get(key)
    if v:
        print(f"{var}={v}")
')"
fi
if [ -n "${OV_CACHE:-}" ]; then
  mkdir -p "$OV_CACHE" "${OV_DATA:-$OV_CACHE/data}"
  export OMNI_CACHE_ROOT="$OV_CACHE"
  export OMNI_DATA_ROOT="${OV_DATA:-$OV_CACHE/data}"
  echo "OMNI_CACHE_ROOT = $OMNI_CACHE_ROOT"
fi
export UV_CACHE_DIR="${UV_CACHE_DIR:-${UVC:-}}"
[ -n "$UV_CACHE_DIR" ] && mkdir -p "$UV_CACHE_DIR" && echo "UV_CACHE_DIR    = $UV_CACHE_DIR"

# ---- env ------------------------------------------------------------------
# RoboLab requires Python >=3.11; uv downloads it if the host lacks it.
uv venv --python 3.11 .venv
# shellcheck disable=SC1091
source .venv/bin/activate

# ---- deps -----------------------------------------------------------------
# Isaac Sim wheels plus assets are ~35 GB on a shared volume. Check first.
FREE_GB="$(df -BG --output=avail . | tail -1 | tr -dc '0-9')"
echo "free space here: ${FREE_GB} GB"
if [ "${FREE_GB:-0}" -lt 45 ]; then
  echo "error: only ${FREE_GB} GB free — Isaac Sim wheels + assets need ~35 GB" >&2
  echo "plus headroom for shader caches. Free space or relocate." >&2
  exit 6
fi

uv sync --extra "$EXTRA"

# ---- openpi client (separate, deliberate upstream omission) ----------------
# policies/cosmos3/client.py and policies/pi0_family/client.py both do
#   from openpi_client import image_tools, websocket_client_policy
# but `openpi-client` is NOT a declared RoboLab dependency: upstream keeps it out
# so non-openpi backends don't pull it in (see robolab/core/utils/image_utils.py,
# which *vendors* image_tools for exactly that reason), and documents installing
# it by hand in policies/pi0_family/README.md. Without this step `uv sync`
# succeeds, RoboLab's own 160-test suite passes, and the closed-loop client dies
# with ModuleNotFoundError only after Isaac Sim has finished booting.
#
# Upstream's README says `uv pip install -e ../openpi/packages/openpi-client`,
# i.e. clone the whole openpi repo for one pure-python package. We install the
# published wheel instead and pin it to the version the *server* side resolved
# (cosmos-framework's `policy-server` group -> openpi-client 0.1.2), because the
# two ends must agree on the msgpack wire format. Verified with --dry-run to add
# only openpi-client + dm-tree: no numpy/pillow churn, isaacsim untouched.
#
# `uv pip install` (not `uv add`/`uv sync`) on purpose: this must not enter
# pyproject.toml/uv.lock, which belong to upstream and stay clean.
OPENPI_CLIENT_VERSION="${OPENPI_CLIENT_VERSION:-0.1.2}"
uv pip install --python ./.venv/bin/python "openpi-client==${OPENPI_CLIENT_VERSION}"

uv cache prune || true
df -h . | tail -1

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
# Use .venv/bin/python, NOT `uv run`: `uv run` re-syncs to the project's default
# dependency set, which excludes the isaac extra we just installed — it would
# uninstall isaacsim while trying to report its version. Importing isaacsim is
# also avoided here (it initialises Kit and would want the EULA); the version
# comes from package metadata only.
python3 - "$EXTRA" "$REPO_ROOT" <<'PY'
import datetime as dt
import json
import subprocess
import sys

extra, repo_root = sys.argv[1], sys.argv[2]
pybin = ".venv/bin/python"


def try_cmd(cmd: list[str]) -> str:
    try:
        return subprocess.check_output(cmd, text=True, stderr=subprocess.STDOUT).strip()
    except Exception as exc:  # noqa: BLE001
        return f"ERROR: {exc}"


def dist_version(pkg: str) -> str:
    return try_cmd(
        [pybin, "-c", f"import importlib.metadata as m; print(m.version({pkg!r}))"]
    )


info = {
    "isaac_extra": extra,
    "installed_utc": dt.datetime.now(dt.timezone.utc)
    .replace(microsecond=0, tzinfo=None)
    .isoformat()
    + "Z",
    "python_version": try_cmd(
        [pybin, "-c", "import sys; print('.'.join(map(str, sys.version_info[:3])))"]
    ),
    "isaac_sim_version": dist_version("isaacsim"),
    "isaac_lab_version": dist_version("isaaclab"),
    "torch_version": dist_version("torch"),
    # Not a declared RoboLab dependency; installed explicitly above. Must match
    # the server side's openpi-client (msgpack wire format).
    "openpi_client_version": dist_version("openpi-client"),
    "robolab_commit": try_cmd(["git", "-C", ".", "rev-parse", "HEAD"]),
    "note": (
        "isaacsim is not imported here — importing it initialises Omniverse Kit "
        "and requires EULA acceptance. Versions come from package metadata."
    ),
}
open(f"{repo_root}/reproducibility/robolab_install.json", "w").write(
    json.dumps(info, indent=2)
)
print("wrote reproducibility/robolab_install.json")
PY

echo "setup_robolab: done"
