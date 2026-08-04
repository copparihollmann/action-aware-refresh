#!/usr/bin/env bash
# scripts/smoke_test.sh — bring server + client up, run one short episode,
# validate pass criteria, tear down. Writes results/reports/baseline_smoke.md.
#
# Requires the Cosmos server binary to be installed (setup_cosmos.sh) AND
# RoboLab (setup_robolab.sh) AND OMNI_KIT_ACCEPT_EULA=Y in the environment.

set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO_ROOT"

: "${COSMOS_HOST:=127.0.0.1}"
: "${COSMOS_PORT:=8000}"
: "${ROBOLAB_TASK:=BananaInBowlTask}"
: "${ROBOLAB_ALT_TASK:=RubiksCubeAndBananaTask}"

REPORT=results/reports/baseline_smoke.md
mkdir -p "$(dirname "$REPORT")"

# ---- start server in background ------------------------------------------
LOG_SERVER=/tmp/cosmos_smoke_server.log
scripts/start_cosmos_server.sh >"$LOG_SERVER" 2>&1 &
SERVER_PID=$!
trap 'echo "stopping server (pid $SERVER_PID)"; kill $SERVER_PID 2>/dev/null || true' EXIT

# Wait for /healthz.
for i in $(seq 1 60); do
  if curl -sS -o /dev/null -w '%{http_code}' --max-time 5 "http://$COSMOS_HOST:$COSMOS_PORT/healthz" 2>/dev/null | grep -q '^200$'; then
    echo "server healthy (after ${i}s)"
    break
  fi
  sleep 1
done

if ! curl -sS -o /dev/null -w '%{http_code}' --max-time 5 "http://$COSMOS_HOST:$COSMOS_PORT/healthz" | grep -q '^200$'; then
  {
    echo "# Baseline smoke — FAIL"
    echo
    echo "Server did not become healthy within 60 s. See server log tail:"
    echo
    echo '```'
    tail -50 "$LOG_SERVER"
    echo '```'
  } >"$REPORT"
  exit 6
fi

# ---- primary task --------------------------------------------------------
export ROBOLAB_TASK ROBOLAB_ENVS=1 ROBOLAB_HEADLESS=1
PRIMARY_LOG=/tmp/robolab_smoke_primary.log
if ! scripts/run_robolab.sh >"$PRIMARY_LOG" 2>&1; then
  {
    echo "# Baseline smoke — FAIL"
    echo
    echo "Primary task $ROBOLAB_TASK client failed. Log tail:"
    echo
    echo '```'
    tail -80 "$PRIMARY_LOG"
    echo '```'
  } >"$REPORT"
  exit 7
fi

# ---- alt task (best-effort) ---------------------------------------------
ROBOLAB_TASK="$ROBOLAB_ALT_TASK"
ALT_LOG=/tmp/robolab_smoke_alt.log
ALT_STATUS="PASS"
if ! scripts/run_robolab.sh >"$ALT_LOG" 2>&1; then
  ALT_STATUS="FAIL (see log)"
fi

# ---- validate ------------------------------------------------------------
# Pass criteria (spec §7.3): checkpoint loaded, connection established,
# actions returned, sim advanced, no NaNs, action dims correct, episode
# termination logged.
python3 scripts/validate_smoke.py \
  --server-log "$LOG_SERVER" \
  --primary-log "$PRIMARY_LOG" \
  --alt-log "$ALT_LOG" \
  --primary-task "$ROBOLAB_TASK" \
  --report "$REPORT"

echo "smoke_test: report at $REPORT"
