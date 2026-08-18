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
# Task IDs MUST come from the checked-out RoboLab registry, not from memory.
# Override with ROBOLAB_TASK / ROBOLAB_ALT_TASK; defaults are resolved from
# experiments/task_sets.yaml (`smoke` set) which is populated from the clone.
: "${SMOKE_TIMEOUT_S:=900}"

if [ -z "${ROBOLAB_TASK:-}" ] || [ -z "${ROBOLAB_ALT_TASK:-}" ]; then
  eval "$(python3 - <<'PY'
import sys

import yaml

sets = yaml.safe_load(open("experiments/task_sets.yaml")) or {}
tasks = ((sets.get("smoke") or {}).get("tasks")) or []
if len(tasks) < 2:
    sys.stderr.write(
        "error: experiments/task_sets.yaml `smoke` needs >=2 task IDs from the\n"
        "checked-out RoboLab registry (third_party/RoboLab/robolab/tasks/benchmark/).\n"
    )
    raise SystemExit(3)
print(f"SMOKE_T0={tasks[0]}")
print(f"SMOKE_T1={tasks[1]}")
PY
)"
  : "${ROBOLAB_TASK:=$SMOKE_T0}"
  : "${ROBOLAB_ALT_TASK:=$SMOKE_T1}"
fi
PRIMARY_TASK="$ROBOLAB_TASK"   # captured before the alt-task reassignment below

REPORT=results/reports/baseline_smoke.md
mkdir -p "$(dirname "$REPORT")"

# ---- start server in background ------------------------------------------
# start_cosmos_server.sh true-execs the server, so $! IS the server process and
# this kill reaches it. Kill the whole process group as a belt-and-braces
# measure: an orphaned server keeps ~31 GB of VRAM pinned on a shared box.
LOG_SERVER=/tmp/cosmos_smoke_server.log
set -m
scripts/start_cosmos_server.sh >"$LOG_SERVER" 2>&1 &
SERVER_PID=$!
set +m
cleanup() {
  echo "stopping server (pid $SERVER_PID)"
  kill -TERM "-$SERVER_PID" 2>/dev/null || kill -TERM "$SERVER_PID" 2>/dev/null || true
  for _ in $(seq 1 20); do
    kill -0 "$SERVER_PID" 2>/dev/null || return 0
    sleep 1
  done
  echo "server did not exit on TERM; sending KILL"
  kill -KILL "-$SERVER_PID" 2>/dev/null || kill -KILL "$SERVER_PID" 2>/dev/null || true
}
trap cleanup EXIT

# Wait for /healthz. Loading ~33 GB of weights takes minutes, so allow
# SMOKE_TIMEOUT_S rather than the previous 60 s.
for i in $(seq 1 "$SMOKE_TIMEOUT_S"); do
  if ! kill -0 "$SERVER_PID" 2>/dev/null; then
    {
      echo "# Baseline smoke — FAIL"
      echo
      echo "Server process exited during startup (after ${i}s). Log tail:"
      echo
      echo '```'
      tail -60 "$LOG_SERVER"
      echo '```'
    } >"$REPORT"
    exit 5
  fi
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
    echo "Server did not become healthy within ${SMOKE_TIMEOUT_S} s. See server log tail:"
    echo
    echo '```'
    tail -50 "$LOG_SERVER"
    echo '```'
  } >"$REPORT"
  exit 6
fi

# ---- primary task --------------------------------------------------------
export ROBOLAB_TASK ROBOLAB_ENVS=1 ROBOLAB_HEADLESS=1

# Dump the first genuine policy request so the M2 in-process probe profiles the
# payload the closed loop actually sends, instead of a synthetic one we invented.
# Absolute path: run_robolab.sh cds into third_party/RoboLab. Primary task only —
# the alt run would otherwise overwrite it with a different prompt.
: "${SMOKE_CAPTURE_REQUEST:=$REPO_ROOT/results/raw/captured_request_$PRIMARY_TASK.npz}"
export ACTION_REFRESH_CAPTURE_REQUEST="$SMOKE_CAPTURE_REQUEST"

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

# ---- alt task (second task, different semantics) -------------------------
# Reassigning ROBOLAB_TASK here is why the report used to mislabel the alt task
# as the primary — the validator is passed $PRIMARY_TASK, captured up top.
export ROBOLAB_TASK="$ROBOLAB_ALT_TASK"
unset ACTION_REFRESH_CAPTURE_REQUEST   # keep the primary task's captured payload
ALT_LOG=/tmp/robolab_smoke_alt.log
ALT_STATUS="PASS"
if ! scripts/run_robolab.sh >"$ALT_LOG" 2>&1; then
  ALT_STATUS="FAIL (see log)"
fi
echo "alt task ($ROBOLAB_ALT_TASK): $ALT_STATUS"

# ---- validate ------------------------------------------------------------
# Pass criteria (spec §7.3): checkpoint loaded, connection established,
# actions returned, sim advanced, no NaNs, action dims correct, episode
# termination logged.
python3 scripts/validate_smoke.py \
  --server-log "$LOG_SERVER" \
  --primary-log "$PRIMARY_LOG" \
  --alt-log "$ALT_LOG" \
  --primary-task "$PRIMARY_TASK" \
  --alt-task "$ROBOLAB_ALT_TASK" \
  --alt-status "$ALT_STATUS" \
  --report "$REPORT"

echo "smoke_test: report at $REPORT"
