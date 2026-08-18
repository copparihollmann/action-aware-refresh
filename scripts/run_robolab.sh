#!/usr/bin/env bash
# scripts/run_robolab.sh — run the Cosmos3 RoboLab client against a running server.
#
# Env:
#   COSMOS_HOST       default 127.0.0.1
#   COSMOS_PORT       default 8000
#   ROBOLAB_TASK      default BananaInBowlTask
#   ROBOLAB_ENVS      default 1
#   ROBOLAB_HEADLESS  default 1  (set to 0 for a Kit UI window)
#   ROBOLAB_CUDA_DEVICES default from configs/topology.yaml (robolab_client role)
#   ROBOLAB_EXTRA_ARGS  space-separated extras appended to the command
#
# OMNI_KIT_ACCEPT_EULA must be set to Y in your shell before running (never
# set here — user must consent explicitly).

set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO_ROOT"

if [ ! -d third_party/RoboLab ]; then
  echo "error: third_party/RoboLab missing. Run scripts/clone_sources.sh first." >&2
  exit 2
fi

: "${COSMOS_HOST:=127.0.0.1}"
: "${COSMOS_PORT:=8000}"
: "${ROBOLAB_TASK:=BananaInBowlTask}"
: "${ROBOLAB_ENVS:=1}"
: "${ROBOLAB_HEADLESS:=1}"

# Pin from topology rather than a literal. The hardcoded `1` this replaces was
# doubly wrong: topology assigns the client GPU 2, and GPU 1 sits on the *same*
# PCIe switch as the server's GPU 0 — the pairing the topology deliberately
# avoids. Assert the UUID so a driver renumber can't silently move us.
source "$REPO_ROOT/scripts/lib/common.sh"
resolve_repo_py
export_cache_env        # HF_HOME  -> /scratch (not the 60 GB NFS home)
export_omni_cache_env   # OMNI_*   -> /scratch (Isaac defaults these into $HOME)
eval "$(topology_devices robolab_client)"
: "${ROBOLAB_CUDA_DEVICES:=$TOPO_DEVICES}"

if [ "$ROBOLAB_CUDA_DEVICES" = "$TOPO_DEVICES" ]; then
  assert_gpu_uuid robolab_client "$ROBOLAB_CUDA_DEVICES" "$TOPO_UUID"
fi

if [ -z "${OMNI_KIT_ACCEPT_EULA:-}" ]; then
  echo "error: OMNI_KIT_ACCEPT_EULA is not set. Isaac Sim will refuse to launch." >&2
  echo "→ read the EULA at:" >&2
  echo "  https://docs.omniverse.nvidia.com/isaacsim/latest/common/NVIDIA_Omniverse_License_Agreement.html" >&2
  echo "→ export OMNI_KIT_ACCEPT_EULA=Y  if and only if you accept it." >&2
  exit 4
fi

# Health check.
if ! curl -sS -o /dev/null -w '%{http_code}' --max-time 10 "http://$COSMOS_HOST:$COSMOS_PORT/healthz" | grep -q '^200$'; then
  echo "error: Cosmos server at $COSMOS_HOST:$COSMOS_PORT is not healthy." >&2
  exit 5
fi

export CUDA_VISIBLE_DEVICES="$ROBOLAB_CUDA_DEVICES"

RUN_ID="robolab-$(date -u +%Y%m%dT%H%M%SZ)"
OUT_DIR="results/raw/$RUN_ID"
mkdir -p "$OUT_DIR"

HEADLESS_FLAG=""
[ "$ROBOLAB_HEADLESS" = "1" ] && HEADLESS_FLAG="--headless"

# Flag names verified against third_party/RoboLab @ 0aef241:
# policies/cosmos3/run.py declares --remote-host / --remote-port, then adds
# robolab.eval.runner.add_common_eval_args (which supplies --task, nargs="+")
# and isaaclab AppLauncher args (which supply --headless).
cd third_party/RoboLab

# Append to command log.
python3 -c '
import json, sys, datetime as dt
entry = {"kind": "robolab_client",
         "utc": dt.datetime.now(dt.timezone.utc).replace(microsecond=0, tzinfo=None).isoformat() + "Z",
         "args": sys.argv[1:]}
open("../../reproducibility/commands.jsonl", "a").write(json.dumps(entry) + "\n")
' --remote-host "$COSMOS_HOST" --remote-port "$COSMOS_PORT" \
  --task "$ROBOLAB_TASK" --num-envs "$ROBOLAB_ENVS" $HEADLESS_FLAG ${ROBOLAB_EXTRA_ARGS:-}

# Send our own output through tee, THEN true-exec, so this PID is the client.
exec > >(tee "$REPO_ROOT/$OUT_DIR/client.log") 2>&1

# Invoke the venv interpreter directly, NOT `uv run`: a bare `uv run` re-syncs
# the project to its *default* dependency set, which excludes the `isaac50`
# extra — it would uninstall isaacsim/isaaclab right before we try to launch the
# simulator. (Same failure mode cost us the cu130 torch in cosmos-framework.)
exec ./.venv/bin/python policies/cosmos3/run.py \
  --remote-host "$COSMOS_HOST" \
  --remote-port "$COSMOS_PORT" \
  --task "$ROBOLAB_TASK" \
  --num-envs "$ROBOLAB_ENVS" \
  $HEADLESS_FLAG ${ROBOLAB_EXTRA_ARGS:-}
