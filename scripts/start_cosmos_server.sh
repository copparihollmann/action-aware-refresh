#!/usr/bin/env bash
# scripts/start_cosmos_server.sh — start the Cosmos3 RoboLab policy server.
#
# Foreground process: after setup this script *becomes* the server (true exec),
# so the caller's $! is the server's own PID and `kill` reaches it. Do not
# reintroduce a `| tee` pipeline here — that leaves the server as a child of a
# subshell and orphans it on teardown, holding the GPU.
#
# Flag names below are verified against the checked-out source
# (third_party/cosmos-framework @ a904d2d, `RobolabServerArgs`, parsed by tyro:
# snake_case -> --kebab-case). The model is a pydantic BaseModel with
# extra="forbid", so a wrong flag fails loudly rather than being ignored.
# Regenerate docs/generated/cosmos_server_help.txt after any upstream bump.
#
# Environment:
#   COSMOS_CHECKPOINT      HF repo id or local dir (default: nvidia/Cosmos3-Nano-Policy-DROID)
#   COSMOS_REVISION        HF revision (default: pinned by reproducibility/model_revisions.json;
#                          upstream's own default is the moving ref "main")
#   COSMOS_DENOISING_STEPS default 4          -> --num-steps
#   COSMOS_DECODE_VIDEO    "true"/"false"; default false
#   COSMOS_PORT            default 8000
#   COSMOS_HOST            bind address; default 127.0.0.1 (upstream default is
#                          0.0.0.0 — we bind loopback since client and server
#                          share this host and the box is multi-user)
#   COSMOS_CUDA_DEVICES    default from configs/topology.yaml (cosmos_server role)
#   COSMOS_SEED            base seed (default 0 upstream)
#   COSMOS_DETERMINISTIC   "true" -> --deterministic-seed (same seed every request;
#                          required for the B3 deterministic-replay config)
#   COSMOS_GUARDRAILS      "true"/"false"; default false. DEVIATION FROM OFFICIAL:
#                          upstream OmniSetupArgs.guardrails defaults True and
#                          unconditionally downloads nvidia/Cosmos-Guardrail1,
#                          which is GATED and currently ACCESS DENIED for us, so
#                          the official server cannot start. Research patch 0001
#                          adds the flag; we default it false so the smoke test
#                          can run at all. Every number taken this way EXCLUDES
#                          guardrail cost, which is part of the official
#                          baseline — set this true once access is granted.
#   HF_TOKEN               only needed for *gated* models; this checkpoint is not
#                          gated. Never written to disk.
#
# Extra server args can be passed after `--`, e.g.:
#   scripts/start_cosmos_server.sh -- --guidance 1.0

set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO_ROOT"

: "${COSMOS_CHECKPOINT:=nvidia/Cosmos3-Nano-Policy-DROID}"
: "${COSMOS_DENOISING_STEPS:=4}"
: "${COSMOS_DECODE_VIDEO:=false}"
: "${COSMOS_PORT:=8000}"
: "${COSMOS_HOST:=127.0.0.1}"
: "${COSMOS_DETERMINISTIC:=false}"
: "${COSMOS_GUARDRAILS:=false}"

# ---- which source tree serves the policy -----------------------------------
# COSMOS_SUBSTRATE selects it (default from configs/substrates.yaml); everything
# below refers to $COSMOS_SRC / $COSMOS_PY / $COSMOS_SERVER_FILE rather than to a
# literal path, so the same launcher can serve upstream or the efficiency fork and
# the start.json record always names which one ran.
source "$REPO_ROOT/scripts/lib/common.sh"
resolve_repo_py
resolve_substrate
export_cache_env
eval "$(topology_devices cosmos_server)"
: "${COSMOS_CUDA_DEVICES:=$TOPO_DEVICES}"

if [ "$COSMOS_CUDA_DEVICES" = "$TOPO_DEVICES" ]; then
  assert_gpu_uuid cosmos_server "$COSMOS_CUDA_DEVICES" "$TOPO_UUID"
fi

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
if [ -z "$COSMOS_REVISION" ]; then
  echo "warning: no pinned HF revision — upstream will resolve the moving ref 'main'." >&2
  echo "         Record a SHA in reproducibility/model_revisions.json for reproducibility." >&2
fi

# Run identifiers.
RUN_ID="cosmos-server-$(date -u +%Y%m%dT%H%M%SZ)"
LOG_DIR="results/raw/$RUN_ID"
mkdir -p "$LOG_DIR"

export CUDA_VISIBLE_DEVICES="$COSMOS_CUDA_DEVICES"
export CUDA_LAUNCH_BLOCKING=0
export TOKENIZERS_PARALLELISM=false

# ---- build the argument list (verified spellings) --------------------------
CMD_ARGS=(
  --checkpoint-path "$COSMOS_CHECKPOINT"
  --port "$COSMOS_PORT"
  --host "$COSMOS_HOST"
  --num-steps "$COSMOS_DENOISING_STEPS"
)
if [ "${COSMOS_DECODE_VIDEO,,}" = "true" ]; then
  CMD_ARGS+=( --decode-video )
else
  CMD_ARGS+=( --no-decode-video )
fi
[ -n "$COSMOS_REVISION" ] && CMD_ARGS+=( --hf-revision "$COSMOS_REVISION" )
[ -n "${COSMOS_SEED:-}" ] && CMD_ARGS+=( --seed "$COSMOS_SEED" )
[ "${COSMOS_DETERMINISTIC,,}" = "true" ] && CMD_ARGS+=( --deterministic-seed )
# COSMOS_PROMPT_JSON — serve structured-JSON prompts, matching training.
#
# This is a REAL SETUP BUG we shipped for three sessions. The server tries to load the
# checkpoint's training config to decide the prompt format, but that requires a local
# `checkpoint.json`; with an HF repo id as --checkpoint-path there is none, so it logs
#   "could not load training config for transforms: Missing key _type"
#   "no training action dataset config found; using default ActionTransformPipeline
#    (format_prompt_as_json=False)"
# and silently serves PLAIN-TEXT prompts. But the recipe this checkpoint was trained with,
# cosmos_framework/configs/.../action_policy_droid_nano.py:219, sets
# format_prompt_as_json=True. So the conditioning text was off-distribution for every
# request, baseline included. Note the two formats are mutually exclusive: with JSON on,
# the viewpoint and duration/FPS text augmentors are disabled because that metadata moves
# into the JSON structure instead.
#
# Spelling note: the field is `bool | None`, so tyro expects a VALUE
# (`--format-prompt-as-json True`), not a `--flag/--no-flag` pair. Passing the pair form
# aborts with "Expected one of ('None', 'True', 'False')" — loudly, which is why the
# mistake cost one startup rather than a silently mis-served run.
if [ -n "${COSMOS_PROMPT_JSON:-}" ]; then
  if [ "${COSMOS_PROMPT_JSON,,}" = "true" ]; then
    CMD_ARGS+=( --format-prompt-as-json True )
  else
    CMD_ARGS+=( --format-prompt-as-json False )
  fi
fi
# --vision-frames exists only on the research branch (patch 0004). Experiment E2b
# shortens the imagined horizon, which is 85.3% of the sequence. Fail loudly rather
# than dropping the flag: silently running the full 33-frame horizon under an E2b
# label would fabricate the comparison.
if [ -n "${COSMOS_VISION_FRAMES:-}" ]; then
  if grep -q 'vision_frames' "$COSMOS_SERVER_FILE"; then
    CMD_ARGS+=( --vision-frames "$COSMOS_VISION_FRAMES" )
  else
    echo "error: COSMOS_VISION_FRAMES=$COSMOS_VISION_FRAMES, but the checked-out" >&2
    echo "       $COSMOS_SUBSTRATE substrate has no --vision-frames flag (patch 0004" >&2
    echo "       not applied to $COSMOS_SRC)." >&2
    echo "→ apply reproducibility/patches/cosmos-framework-0004-vision-frames.patch" >&2
    exit 11
  fi
fi
# --guardrails / --no-guardrails exist only on the research branch (patch 0001).
# Fail loudly rather than silently dropping the flag if we're on stock upstream:
# a silent drop would try to download the gated repo and die in a confusing way.
if grep -q 'guardrails: bool' "$COSMOS_SERVER_FILE"; then
  if [ "${COSMOS_GUARDRAILS,,}" = "true" ]; then
    CMD_ARGS+=( --guardrails )
  else
    CMD_ARGS+=( --no-guardrails )
  fi
elif [ "${COSMOS_GUARDRAILS,,}" != "true" ]; then
  echo "error: COSMOS_GUARDRAILS=false, but the $COSMOS_SUBSTRATE substrate has no" >&2
  echo "       --no-guardrails flag (research patch 0001 is not applied)." >&2
  echo "→ apply reproducibility/patches/cosmos-framework-0001-guardrails-flag.patch," >&2
  echo "  or set COSMOS_GUARDRAILS=true if you have access to nvidia/Cosmos-Guardrail1." >&2
  exit 10
fi
# Anything after `--` is forwarded to the server.
if [ "$#" -gt 0 ] && [ "$1" = "--" ]; then
  shift
  CMD_ARGS+=( "$@" )
fi

# ---- record start metadata + contention snapshot --------------------------
# The contention snapshot matters as much as the timings: this host's CPU is
# shared, and host-side stages are sensitive to it.
python3 - "$LOG_DIR/start.json" "$RUN_ID" "$COSMOS_CHECKPOINT" "$COSMOS_REVISION" \
    "$COSMOS_DENOISING_STEPS" "$COSMOS_DECODE_VIDEO" "$COSMOS_PORT" "$COSMOS_CUDA_DEVICES" \
    "${TOPO_UUID:-}" "$COSMOS_GUARDRAILS" \
    "$COSMOS_SUBSTRATE" "$COSMOS_SUBSTRATE_COMMIT" "$COSMOS_SUBSTRATE_DIRTY" <<'PY'
import datetime as dt
import json
import os
import subprocess
import sys

out = sys.argv[1]
run_id, ckpt, rev, steps, decode, port, dev, uuid, guardrails = sys.argv[2:11]
substrate, substrate_commit, substrate_dirty = sys.argv[11:14]


def sh(cmd: str) -> str:
    try:
        return subprocess.run(
            cmd, shell=True, capture_output=True, text=True, timeout=15
        ).stdout.strip()
    except Exception:
        return ""


try:
    load = list(os.getloadavg())
except OSError:
    load = None

open(out, "w").write(
    json.dumps(
        {
            "run_id": run_id,
            "timestamp_utc": dt.datetime.now(dt.timezone.utc)
            .replace(microsecond=0, tzinfo=None)
            .isoformat()
            + "Z",
            "checkpoint": ckpt,
            "revision": rev or None,
            "denoising_steps": int(steps),
            "decode_video": decode.lower() == "true",
            "port": int(port),
            "cuda_visible_devices": dev,
            "gpu_uuid": uuid or None,
            "guardrails": guardrails.lower() == "true",
            # Which source tree served this. Recorded unconditionally, including for
            # the default: a result that does not name its substrate cannot be
            # compared with one that does.
            "substrate": {
                "name": substrate,
                "commit": substrate_commit or None,
                "dirty": substrate_dirty.lower() == "true",
            },
            "deviations_from_official": (
                []
                if guardrails.lower() == "true"
                else [
                    "guardrails disabled (nvidia/Cosmos-Guardrail1 is gated and "
                    "access is denied); latencies EXCLUDE guardrail cost"
                ]
            )
            + (
                []
                if substrate == "upstream"
                else [
                    f"substrate={substrate} (not NVIDIA upstream cosmos-framework); "
                    "numbers are not directly comparable to the published baseline"
                ]
            )
            + (
                []
                if substrate_dirty.lower() != "true"
                else [f"substrate {substrate} tree is DIRTY — uncommitted local changes"]
            ),
            "contention": {
                "loadavg_1_5_15": load,
                "gpu_compute_apps": sh(
                    "nvidia-smi --query-compute-apps=pid,process_name,used_memory "
                    "--format=csv,noheader"
                ),
                "top_cpu": sh(
                    "ps -eo user,pcpu,comm --sort=-pcpu --no-headers | head -5"
                ),
            },
        },
        indent=2,
    )
)
PY

# Append to reproducibility/commands.jsonl (no secrets).
python3 -c '
import datetime as dt, json, sys
entry = {"kind": "cosmos_server",
         "utc": dt.datetime.now(dt.timezone.utc).replace(microsecond=0, tzinfo=None).isoformat() + "Z",
         "args": sys.argv[1:]}
open("reproducibility/commands.jsonl", "a").write(json.dumps(entry) + "\n")
' "${CMD_ARGS[@]}"

echo "starting Cosmos3 policy server on ${COSMOS_HOST}:${COSMOS_PORT} (GPU=$COSMOS_CUDA_DEVICES)"
echo "  substrate:  $COSMOS_SUBSTRATE @ ${COSMOS_SUBSTRATE_COMMIT:0:12} (dirty=$COSMOS_SUBSTRATE_DIRTY)"
echo "              $COSMOS_SRC"
echo "  checkpoint: $COSMOS_CHECKPOINT @ ${COSMOS_REVISION:-main}"
echo "  HF_HOME:    $HF_HOME"
echo "  guardrails: $COSMOS_GUARDRAILS"
echo "  pg backend: ${COSMOS_PG_BACKEND:-gloo}"
echo "  log: $LOG_DIR/server.log"

# Send our own stdout/stderr through tee, THEN exec the server so this PID
# becomes the server process.
exec > >(tee "$REPO_ROOT/$LOG_DIR/server.log") 2>&1

cd "$COSMOS_SRC"
# Invoke the venv interpreter directly. A bare `uv run` re-syncs the project to
# its default dependency set and will silently replace the cuXXX-group torch
# (observed: 2.10.0+cu130 -> 2.13.0+cu130, which breaks flash-attn's ABI).
# Re-resolving dependencies at server-start would also mean the thing we profile
# is not necessarily the thing we installed.
#
# The entry wrapper only creates the one-rank process group that upstream's
# maybe_init_distributed() would otherwise create with NCCL — which core-dumps on this
# stack — and then runs upstream's own __main__ with argv untouched. That is what lets
# us drop the two patches that used to skip upstream collectives; see
# `src/action_refresh/server/process_group.py` for the measured equivalence
# (bitwise-identical actions). COSMOS_PG_BACKEND=none restores the old route, which
# requires cosmos-framework-0002/-0003 to be applied.
exec "$COSMOS_PY" "$REPO_ROOT/scripts/cosmos_server_entry.py" "${CMD_ARGS[@]}"
