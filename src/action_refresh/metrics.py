"""Typed request/episode schemas + JSONL writers.

Every experiment run must populate a `RunMeta`, one `EpisodeRecord` per
episode, and many `RequestRecord`s. These are the canonical schemas from
spec §8.

Design choices:
- pydantic models so extras are caught at construction time, not at analysis;
- Optional fields default to None so we don't invent zeros;
- writers append JSONL (streamable, safe under crash); Parquet is a batch
  conversion in `analysis/`.
"""
from __future__ import annotations

import json
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from pydantic import BaseModel, ConfigDict, Field


class RunMeta(BaseModel):
    """One per experiment run — written to results/raw/<run_id>/meta.json."""

    model_config = ConfigDict(extra="forbid")

    run_id: str
    timestamp_utc: str
    git_sha_repo: str
    git_shas_third_party: dict[str, str]
    model_revision: str
    simulator_version: str
    gpu_model: str
    method: str
    method_config: dict[str, Any]
    tasks: list[str]
    seeds: list[int]
    num_envs: int
    warmup_count: int
    episode_count: int
    command: str
    output_paths: dict[str, str]


class RequestRecord(BaseModel):
    """One row per policy request. All fields from spec §8."""

    model_config = ConfigDict(extra="forbid")

    run_id: str
    task: str
    env_id: int
    episode_id: int
    control_step: int
    policy_request_index: int
    seed: int
    method: str
    cache_mode: str | None = None
    cache_age: int | None = None
    gate_decision: str | None = None
    gate_features: dict[str, float] | None = None

    policy_call_performed: bool
    full_refresh: bool = False
    partial_refresh: bool = False
    keyframe_refresh: bool = False

    number_of_denoising_steps: int | None = None
    number_of_transformer_forwards: int | None = None
    blocks_recomputed: int | None = None
    visual_tokens_processed: int | None = None
    action_tokens_processed: int | None = None
    tokens_recomputed: int | None = None
    cache_hits: int | None = None
    cache_misses: int | None = None

    preprocessing_ms: float | None = None
    serialization_ms: float | None = None
    network_roundtrip_ms: float | None = None
    server_queue_ms: float | None = None
    model_wall_ms: float | None = None
    model_cuda_ms: float | None = None
    vision_encode_ms: float | None = None
    context_ms: float | None = None
    denoising_ms: float | None = None
    vision_decode_ms: float | None = None
    postprocess_ms: float | None = None

    peak_allocated_vram_mb: float | None = None
    peak_reserved_vram_mb: float | None = None
    gpu_power_w_mean: float | None = None
    gpu_energy_j: float | None = None
    estimated_flops: float | None = None
    # Fraction of FLOPs that came from a real counter vs analytic fallback.
    measured_flop_coverage: float | None = None

    observation_hash: str | None = None
    action_chunk_hash: str | None = None
    contact_state: str | None = None
    subtask_progress: float | None = None
    failure_reason: str | None = None


class EpisodeRecord(BaseModel):
    model_config = ConfigDict(extra="forbid")

    run_id: str
    task: str
    env_id: int
    episode_id: int
    seed: int
    method: str

    success: bool
    progress: float | None = None
    steps: int
    duration_s: float

    policy_calls: int
    keyframe_refreshes: int = 0
    full_refreshes: int = 0
    partial_refreshes: int = 0
    total_denoising_steps: int = 0
    total_visual_tokens: int = 0
    total_action_tokens: int = 0
    total_estimated_flops: float | None = None
    total_gpu_time_ms: float | None = None
    total_wall_time_ms: float | None = None
    total_auxiliary_overhead_ms: float | None = None
    total_energy_j: float | None = None

    mean_action_latency_ms: float | None = None
    p50_action_latency_ms: float | None = None
    p90_action_latency_ms: float | None = None
    p95_action_latency_ms: float | None = None
    p99_action_latency_ms: float | None = None

    mean_cache_age: float | None = None
    max_cache_age: int | None = None
    missed_refresh_count: int | None = None
    unnecessary_refresh_count: int | None = None
    task_competency: str | None = None
    task_difficulty: str | None = None


@dataclass
class JsonlWriter:
    """Append-only JSONL writer. One process per file; fsync on close."""

    path: Path
    _fp: Any = None

    def __post_init__(self) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._fp = self.path.open("a", buffering=1)  # line-buffered

    def write(self, model: BaseModel) -> None:
        self._fp.write(json.dumps(model.model_dump(), separators=(",", ":")) + "\n")

    def close(self) -> None:
        if self._fp is not None:
            try:
                self._fp.flush()
            finally:
                self._fp.close()
                self._fp = None

    def __enter__(self) -> "JsonlWriter":
        return self

    def __exit__(self, *exc: object) -> None:
        self.close()


def new_run_id(prefix: str = "run") -> str:
    """Monotonic run id: <prefix>-YYYYmmddTHHMMSS-<millis>."""
    t = time.time()
    return time.strftime(f"{prefix}-%Y%m%dT%H%M%S", time.gmtime(t)) + f"-{int((t % 1) * 1000):03d}"
