#!/usr/bin/env bash
# scripts/start_cosmos_server.sh — start the Cosmos3 RoboLab policy server.
#
# Foreground process. Log file is written into results/raw/<run_id>/.
# Environment:
#   COSMOS_CHECKPOINT      HF repo id (default: nvidia/Cosmos3-Nano-Policy-DROID)
#   COSMOS_REVISION        HF revision (default: pinned by reproducibility/model_revisions.json if set)
#   COSMOS_DENOISING_STEPS default 4
#   COSMOS_DECODE_VIDEO    "true"/"false"; default false
#   COSMOS_PORT            default 8000
#   COSMOS_SEED            optional deterministic seed
#   COSMOS_CUDA_DEVICES    default "0"
#   HF_TOKEN               required for gated model access; never written to disk
#
# Extra server args can be passed after `--`, e.g.:
#   scripts/start_cosmos_server.sh -- --extra-flag value

set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO_ROOT"

: "${COSMOS_CHECKPOINT:=nvidia/Cosmos3-Nano-Policy-DROID}"
: "${COSMOS_DENOISING_STEPS:=4}"
: "${COSMOS_DECODE_VIDEO:=false}"
: "${COSMOS_PORT:=8000}"
: "${COSMOS_CUDA_DEVICES:=0}"

# Pin revision if we've recorded one.
REVISION=""
if [ -f reproducibility/model_revisions.json ]; then
  REVISION="$(python3 -c '
import json
try:
    d = json.load(open("reproducibility/model_revisions.json"))
    print(d["models"]["nvidia/Cosmos3-Nano-Policy-DROID"].get("revision") or "")
except Exception:
    pass
')"
fi
: "${COSMOS_REVISION:=$REVISION}"

# Run identifiers.
RUN_ID="cosmos-server-$(date -u +%Y%m%dT%H%M%SZ)"
LOG_DIR="results/raw/$RUN_ID"
mkdir -p "$LOG_DIR"

# GPU pin.
export CUDA_VISIBLE_DEVICES="$COSMOS_CUDA_DEVICES"
# Sensible perf defaults.
export CUDA_LAUNCH_BLOCKING=0
export TOKENIZERS_PARALLELISM=false

# Record the command WITHOUT any secret env vars in the log.
CMD_ARGS=(
  --model "$COSMOS_CHECKPOINT"
  --port "$COSMOS_PORT"
  --denoising-steps "$COSMOS_DENOISING_STEPS"
  --decode-video "$COSMOS_DECODE_VIDEO"
)
[ -n "$COSMOS_REVISION" ] && CMD_ARGS+=( --revision "$COSMOS_REVISION" )
[ -n "${COSMOS_SEED:-}" ] && CMD_ARGS+=( --seed "$COSMOS_SEED" )
# Anything after `--` is forwarded to the server.
if [ "$#" -gt 0 ] && [ "$1" = "--" ]; then
  shift
  CMD_ARGS+=( "$@" )
fi

# Log start metadata (safe subset).
python3 - "$LOG_DIR/start.json" "$RUN_ID" "$COSMOS_CHECKPOINT" "$COSMOS_REVISION" \
    "$COSMOS_DENOISING_STEPS" "$COSMOS_DECODE_VIDEO" "$COSMOS_PORT" "$COSMOS_CUDA_DEVICES" <<'PY'
import json, os, sys, datetime as dt
out = sys.argv[1]
run_id, ckpt, rev, steps, decode, port, dev = sys.argv[2:9]
open(out, "w").write(json.dumps({
  "run_id": run_id,
  "timestamp_utc": dt.datetime.utcnow().replace(microsecond=0).isoformat() + "Z",
  "checkpoint": ckpt,
  "revision": rev or None,
  "denoising_steps": int(steps),
  "decode_video": decode.lower() == "true",
  "port": int(port),
  "cuda_visible_devices": dev,
}, indent=2))
PY

# Append to reproducibility/commands.jsonl (no secrets).
python3 -c '
import json, sys, datetime as dt
entry = {"kind": "cosmos_server", "utc": dt.datetime.utcnow().replace(microsecond=0).isoformat() + "Z",
         "args": sys.argv[1:]}
open("reproducibility/commands.jsonl", "a").write(json.dumps(entry) + "\n")
' "${CMD_ARGS[@]}"

echo "starting Cosmos3 policy server on port $COSMOS_PORT (GPU=$COSMOS_CUDA_DEVICES)"
echo "  log: $LOG_DIR/server.log"

# Run the actual server. This binary comes from cosmos-framework's install.
# See docs/generated/cosmos_server_help.txt for the current option spelling —
# adjust CMD_ARGS above if the checked-out source uses different flag names.
cd third_party/cosmos-framework
exec uv run python -m cosmos_framework.scripts.action_policy_server_robolab \
  "${CMD_ARGS[@]}" \
  2>&1 | tee "$REPO_ROOT/$LOG_DIR/server.log"
