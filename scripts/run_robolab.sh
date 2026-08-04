#!/usr/bin/env bash
# scripts/run_robolab.sh — run the Cosmos3 RoboLab client against a running server.
#
# Env:
#   COSMOS_HOST       default 127.0.0.1
#   COSMOS_PORT       default 8000
#   ROBOLAB_TASK      default BananaInBowlTask
#   ROBOLAB_ENVS      default 1
#   ROBOLAB_HEADLESS  default 1  (set to 0 for a Kit UI window)
#   ROBOLAB_CUDA_DEVICES default 1
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
: "${ROBOLAB_CUDA_DEVICES:=1}"

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

# NOTE: the exact flag names below reflect the expected current API; if the
# checked-out RoboLab uses different names, adjust here. See
# `third_party/RoboLab/policies/cosmos3/run.py --help`.
cd third_party/RoboLab

# Append to command log.
python3 -c '
import json, sys, datetime as dt
entry = {"kind": "robolab_client", "utc": dt.datetime.utcnow().replace(microsecond=0).isoformat() + "Z",
         "args": sys.argv[1:]}
open("../../reproducibility/commands.jsonl", "a").write(json.dumps(entry) + "\n")
' --remote-host "$COSMOS_HOST" --remote-port "$COSMOS_PORT" \
  --task "$ROBOLAB_TASK" --num-envs "$ROBOLAB_ENVS" $HEADLESS_FLAG ${ROBOLAB_EXTRA_ARGS:-}

exec uv run python policies/cosmos3/run.py \
  --remote-host "$COSMOS_HOST" \
  --remote-port "$COSMOS_PORT" \
  --task "$ROBOLAB_TASK" \
  --num-envs "$ROBOLAB_ENVS" \
  $HEADLESS_FLAG ${ROBOLAB_EXTRA_ARGS:-} \
  2>&1 | tee "$REPO_ROOT/$OUT_DIR/client.log"
