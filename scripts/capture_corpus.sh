#!/usr/bin/env bash
# scripts/capture_corpus.sh — record a corpus of REAL policy requests for offline work.
#
# Why this exists: a closed-loop pilot pass costs ~3.7 GPU-h, while replaying one
# captured request in-process costs ~3.5 s. Capturing real trajectories once lets
# Experiments A (denoising steps), E (action-only) and F (cross-step caching) be
# screened offline, so closed-loop episodes are spent only on candidates that
# survived. Spec §11.1 asks for exactly that ordering.
#
# The requests must be *captured* rather than reconstructed: the HDF5 the runner
# writes contains privileged state and actions but no camera images, so there is no
# other route to a real observation.
#
# Resumable: a task whose capture directory already holds requests is skipped, so an
# interrupted run continues where it stopped. Delete a task's directory to redo it.
#
# Env:
#   CORPUS_TASKS   space-separated task names (default: a 4-task competency spread)
#   CORPUS_ROOT    output root (default results/raw/corpus)
#   CORPUS_TIMEOUT_S  server health-wait budget (default 900)
#   COSMOS_GUARDRAILS  see start_cosmos_server.sh (default false — gated repo)
# OMNI_KIT_ACCEPT_EULA=Y must be set by the user.

set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO_ROOT"

# Chosen for competency spread at low cost, using episode_s from RoboLab's own task
# metadata (total 190 sim-seconds ≈ 20 min wall at the measured 6.2x real-time
# factor). Deliberately includes one contact-rich reorientation task and one spatial
# stacking task: a corpus of only pick-and-place would screen methods on the easiest
# dynamics and mislead about contact-sensitive failures.
: "${CORPUS_TASKS:=BananaInBowlTask BowlStackingLeftOnRightTask Stack3RubiksCubeTask ReorientJugTask}"
: "${CORPUS_ROOT:=$REPO_ROOT/results/raw/corpus}"
: "${CORPUS_TIMEOUT_S:=900}"
: "${COSMOS_GUARDRAILS:=false}"
export COSMOS_GUARDRAILS

: "${COSMOS_HOST:=127.0.0.1}"
: "${COSMOS_PORT:=8000}"

if [ -z "${OMNI_KIT_ACCEPT_EULA:-}" ]; then
  echo "error: OMNI_KIT_ACCEPT_EULA is not set — Isaac Sim will refuse to launch." >&2
  echo "→ read the EULA and export it yourself if you accept; scripts never set it." >&2
  exit 4
fi

mkdir -p "$CORPUS_ROOT"

# Work out what is left to do BEFORE paying for a 25 s model load.
TODO=()
for task in $CORPUS_TASKS; do
  d="$CORPUS_ROOT/$task"
  if [ -d "$d" ] && [ -n "$(ls -A "$d"/*.npz 2>/dev/null || true)" ]; then
    n=$(ls -1 "$d"/*.npz 2>/dev/null | wc -l)
    echo "skip  $task — already has $n captured requests"
  else
    TODO+=( "$task" )
  fi
done

if [ "${#TODO[@]}" -eq 0 ]; then
  echo "corpus complete — nothing to capture"
  exit 0
fi
echo "capturing: ${TODO[*]}"

# ---- server lifecycle -----------------------------------------------------
# Same shape as smoke_test.sh, which is proven: start_cosmos_server.sh true-execs the
# server so $! is the server itself, and we kill the whole process group because an
# orphaned server keeps ~31 GB of VRAM pinned on a shared box.
LOG_SERVER=/tmp/cosmos_corpus_server.log
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
  kill -KILL "-$SERVER_PID" 2>/dev/null || kill -KILL "$SERVER_PID" 2>/dev/null || true
}
trap cleanup EXIT

for i in $(seq 1 "$CORPUS_TIMEOUT_S"); do
  if ! kill -0 "$SERVER_PID" 2>/dev/null; then
    echo "error: server exited during startup after ${i}s. Tail:" >&2
    tail -40 "$LOG_SERVER" >&2
    exit 5
  fi
  if curl -sS -o /dev/null -w '%{http_code}' --max-time 5 \
      "http://$COSMOS_HOST:$COSMOS_PORT/healthz" 2>/dev/null | grep -q '^200$'; then
    echo "server healthy after ${i}s"
    break
  fi
  sleep 1
done
if ! curl -sS -o /dev/null -w '%{http_code}' --max-time 5 \
    "http://$COSMOS_HOST:$COSMOS_PORT/healthz" | grep -q '^200$'; then
  echo "error: server not healthy within ${CORPUS_TIMEOUT_S}s. Tail:" >&2
  tail -40 "$LOG_SERVER" >&2
  exit 6
fi

# ---- capture, one task at a time ------------------------------------------
FAILED=()
for task in "${TODO[@]}"; do
  outdir="$CORPUS_ROOT/$task"
  # Capture into a staging dir and only publish on success, so a crashed episode
  # cannot leave a half-trajectory that the skip-check above would treat as done.
  staging="$CORPUS_ROOT/.staging_$task"
  rm -rf "$staging"; mkdir -p "$staging"
  log="/tmp/corpus_${task}.log"
  echo "=== $task -> $outdir"
  # A trailing directory (no .npz suffix) selects capture-all mode in the client
  # patch; --video-mode none skips mp4 encode and viewport render, which we do not
  # need for a corpus and which cost real time.
  if ROBOLAB_TASK="$task" ROBOLAB_ENVS=1 ROBOLAB_HEADLESS=1 \
     ROBOLAB_EXTRA_ARGS="--video-mode none" \
     ACTION_REFRESH_CAPTURE_REQUEST="$staging/" \
     scripts/run_robolab.sh >"$log" 2>&1; then
    n=$(ls -1 "$staging"/*.npz 2>/dev/null | wc -l)
    if [ "$n" -eq 0 ]; then
      echo "  WARNING: episode succeeded but captured 0 requests — capture not wired?" >&2
      FAILED+=( "$task(0 requests)" )
      continue
    fi
    mv "$staging" "$outdir"
    echo "  captured $n requests"
  else
    echo "  FAILED (see $log):" >&2
    tail -20 "$log" >&2
    FAILED+=( "$task" )
    # Keep going: a corpus of 3 tasks is far better than none, and the failure is
    # recorded rather than hidden.
  fi
done

TOTAL=$(find "$CORPUS_ROOT" -name '*.npz' 2>/dev/null | wc -l)
echo "corpus now holds $TOTAL requests across $(ls -1d "$CORPUS_ROOT"/*/ 2>/dev/null | wc -l) tasks"
du -sh "$CORPUS_ROOT" 2>/dev/null || true
if [ "${#FAILED[@]}" -gt 0 ]; then
  echo "tasks that failed to capture: ${FAILED[*]}" >&2
  exit 7
fi
