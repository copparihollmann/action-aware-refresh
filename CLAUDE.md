# CLAUDE.md — repo-local operating rules

## Environment invariants
- Host: `bwrc-bwell`, RHEL 9.7. No sudo. No docker. No NVIDIA container toolkit.
  → Container path is blocked. All installs are native, user-space, via `uv`.
- 2× RTX PRO 6000 Blackwell (SM 12.0, 98 GB VRAM), driver 580.82.09, CUDA runtime max 13.0.
- Host CUDA toolkit is 12.6 (`/usr/local/cuda`). PyTorch wheels: prefer cu128 (well-tested) or cu130.
- Machine is shared with other users (checked once — user `ken_ho` had python on both GPUs).
  Timing benchmarks MUST wait for a quiet window; note contention in every result.
- HF token via `HF_TOKEN` env var only. Never write to disk. `hf auth login` OK.

## Research constraints (from spec)
- Do NOT retrain Cosmos3 from scratch. LoRA / adapters / small gates only.
- Do NOT optimize primarily for image quality.
- Task success vs total compute (with all overhead counted) is the main metric.
- Every claimed speedup requires: normalized FLOPs, measured GPU time, end-to-end latency, task success.
- Never accept licenses on the user's behalf (Cosmos, HF gated, Isaac EULA). Stop and ask.
- No run > 4 GPU-hours without explicit approval (report estimate first).
- Preserve upstream repos — modifications land as local branches under
  `third_party/*/research/action-aware-refresh` and are exported to
  `reproducibility/patches/`.

## Never commit
- HF tokens, credentials, model weights, generated videos, large parquet.
- The `third_party/` clones (they're pinned by SHA in `source_manifest.json`).

## Style
- Typed Python, dataclasses / pydantic. No hidden global state. No silent fallbacks.
- Structured logging (`structlog`). Deterministic seeds where possible.
- No hardcoded personal paths; use `configs/machine.yaml`.

## Current phase
M0–M2 (audit + baseline + compute anatomy). Do NOT start LoRA, RAFT, v2e, or
spatial modifications until compute anatomy is done and user has approved next phase.
