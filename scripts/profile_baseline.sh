#!/usr/bin/env bash
# scripts/profile_baseline.sh — M2 compute anatomy.
#
# Assumes: Cosmos server is running, RoboLab client is installed. Drives
# `src/action_refresh/analysis/profile_runner.py` to execute B0..B4 from
# spec §9, collecting stage-level latency, kernel time, memory, FLOPs, and
# a PyTorch profiler chrome trace for one representative request per config.
#
# Writes:
#   results/profiles/<run_id>/{stages.jsonl, chrome_trace_*.json, summary.md}
#   docs/compute_anatomy.md   (replaced with a data-driven version)

set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO_ROOT"

: "${COSMOS_HOST:=127.0.0.1}"
: "${COSMOS_PORT:=8000}"
: "${PROFILE_WARMUP:=5}"
: "${PROFILE_ITERS:=30}"

RUN_ID="profile-$(date -u +%Y%m%dT%H%M%SZ)"
OUT_DIR="results/profiles/$RUN_ID"
mkdir -p "$OUT_DIR"

echo "compute anatomy → $OUT_DIR"

uv run python -m action_refresh.analysis.profile_runner \
  --cosmos-host "$COSMOS_HOST" --cosmos-port "$COSMOS_PORT" \
  --warmup "$PROFILE_WARMUP" --iters "$PROFILE_ITERS" \
  --out-dir "$OUT_DIR" \
  --configs B0,B1,B2,B3,B4

uv run python -m action_refresh.analysis.compute_anatomy_report \
  --profile-dir "$OUT_DIR" \
  --out docs/compute_anatomy.md

echo "compute_anatomy: report at docs/compute_anatomy.md"
