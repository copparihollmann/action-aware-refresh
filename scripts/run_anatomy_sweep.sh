#!/usr/bin/env bash
# scripts/run_anatomy_sweep.sh — M2 sweep, ONE CONFIG PER PROCESS.
#
# The policy model is ~31 GiB resident on a 45 GiB card and is not reliably
# freed inside a process (references held by profiler/hooks outlive `del`), so
# loading a second config in the same process OOMs. Each config therefore gets a
# fresh interpreter; ~40 s of model-load cost per config is the price.
set -euo pipefail
REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO_ROOT"

: "${ANATOMY_OUT:=$REPO_ROOT/results/profiles/m2-anatomy}"
: "${ANATOMY_WARMUP:=3}"
: "${ANATOMY_ITERS:=10}"
: "${ANATOMY_CONFIGS:=B0 B1 B2_steps_1 B2_steps_2 B2_steps_3 B2_steps_4}"
: "${ANATOMY_EXTRA:=--no-guardrails}"

# HF_HOME from configs/machine.yaml, not a literal: a hardcoded personal path is
# both against repo rules and the exact bug that made the server re-download
# 34 GB onto an NFS home volume. Pin the GPU from topology and assert its UUID —
# without CUDA_VISIBLE_DEVICES this lands on whatever the driver calls device 0,
# which on a shared box is how you attribute timings to someone else's card.
source "$REPO_ROOT/scripts/lib/common.sh"
resolve_repo_py
# Which source tree is being profiled. Compute anatomy is the measurement most
# likely to differ between substrates — it is the whole reason to compare them — so
# it must never be taken against an assumed tree.
resolve_substrate
export_cache_env
eval "$(topology_devices cosmos_server)"
: "${ANATOMY_CUDA_DEVICES:=$TOPO_DEVICES}"
if [ "$ANATOMY_CUDA_DEVICES" = "$TOPO_DEVICES" ]; then
  assert_gpu_uuid cosmos_server "$ANATOMY_CUDA_DEVICES" "$TOPO_UUID"
fi
export CUDA_VISIBLE_DEVICES="$ANATOMY_CUDA_DEVICES"
export TOKENIZERS_PARALLELISM=false
export PYTORCH_ALLOC_CONF=expandable_segments:True

# Profile the payload the closed loop actually sends, when we have one. The
# capture is written by scripts/smoke_test.sh (research patch robolab-0001);
# absent it, the probe falls back to a synthetic observation and says so in every
# record. Point ANATOMY_REQUEST_NPZ elsewhere to use a different task's capture.
if [ -z "${ANATOMY_REQUEST_NPZ:-}" ]; then
  ANATOMY_REQUEST_NPZ="$(ls -1t "$REPO_ROOT"/results/raw/captured_request_*.npz 2>/dev/null | head -1 || true)"
fi
REQUEST_ARGS=()
if [ -n "${ANATOMY_REQUEST_NPZ:-}" ] && [ -f "$ANATOMY_REQUEST_NPZ" ]; then
  REQUEST_ARGS=( --request-npz "$ANATOMY_REQUEST_NPZ" )
  echo "input: REAL captured request $ANATOMY_REQUEST_NPZ"
else
  echo "input: SYNTHETIC (no results/raw/captured_request_*.npz — run 'make smoke' first)"
fi

mkdir -p "$ANATOMY_OUT"
echo "substrate: $COSMOS_SUBSTRATE @ ${COSMOS_SUBSTRATE_COMMIT:0:12} (dirty=$COSMOS_SUBSTRATE_DIRTY)"
cd "$COSMOS_SRC"
for cfg in $ANATOMY_CONFIGS; do
  echo "=============== $cfg ==============="
  "$COSMOS_PY" "$REPO_ROOT/scripts/probe_compute_anatomy.py" \
    --out-dir "$ANATOMY_OUT" --configs "$cfg" \
    --warmup "$ANATOMY_WARMUP" --iters "$ANATOMY_ITERS" \
    "${REQUEST_ARGS[@]}" \
    $ANATOMY_EXTRA 2>&1 \
    | grep -E "wall |peak VRAM|wrote |tokens:|thermal|throttle|census|OutOfMemory|Traceback|Error" || true
done
cd "$REPO_ROOT"
# Render the report only for a FULL sweep into the canonical output dir.
#
# This used to render unconditionally, which meant `ANATOMY_CONFIGS=B0` into a
# scratch dir silently overwrote docs/compute_anatomy.md — the M2 deliverable —
# with a one-config stub. The doc is fully derived from the summary JSONs, so it
# was recoverable, but a side effect that destroys a milestone artifact should
# not be reachable from an exploratory one-config run.
: "${ANATOMY_CANONICAL_OUT:=$REPO_ROOT/results/profiles/m2-anatomy}"
# The real-payload cross-check lives in its own dir (one config, captured input).
# Pass it through so a regeneration keeps the input-validation section instead of
# silently dropping it.
: "${ANATOMY_COMPARE_OUT:=$REPO_ROOT/results/profiles/m2-anatomy-real}"
if [ "$ANATOMY_OUT" = "$ANATOMY_CANONICAL_OUT" ]; then
  echo "=== sweep done; rendering report ==="
  COMPARE_ARGS=()
  [ -d "$ANATOMY_COMPARE_OUT" ] && COMPARE_ARGS=( --compare-dir "$ANATOMY_COMPARE_OUT" )
  "$REPO_ROOT/.venv/bin/python" "$REPO_ROOT/scripts/write_compute_anatomy.py" \
    --summary-dir "$ANATOMY_OUT" "${COMPARE_ARGS[@]}" \
    --out "$REPO_ROOT/docs/compute_anatomy.md"
else
  echo "=== sweep done ==="
  echo "note: ANATOMY_OUT ($ANATOMY_OUT) is not the canonical dir, so"
  echo "      docs/compute_anatomy.md was NOT regenerated. Render explicitly with:"
  echo "  .venv/bin/python scripts/write_compute_anatomy.py \\"
  echo "      --summary-dir $ANATOMY_OUT --out <some-other-path.md>"
fi
