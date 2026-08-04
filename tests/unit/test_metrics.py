"""Sanity tests for metrics schemas + JSONL writer."""
from __future__ import annotations

import json
from pathlib import Path

import pytest

from action_refresh.metrics import (
    EpisodeRecord,
    JsonlWriter,
    RequestRecord,
    RunMeta,
    new_run_id,
)


def _make_request() -> RequestRecord:
    return RequestRecord(
        run_id="r1",
        task="BananaInBowlTask",
        env_id=0,
        episode_id=0,
        control_step=0,
        policy_request_index=0,
        seed=42,
        method="baseline_full",
        policy_call_performed=True,
    )


def test_request_record_extra_forbidden() -> None:
    with pytest.raises(Exception):
        RequestRecord(  # type: ignore[call-arg]
            run_id="r1",
            task="t",
            env_id=0,
            episode_id=0,
            control_step=0,
            policy_request_index=0,
            seed=0,
            method="m",
            policy_call_performed=True,
            nonsense=1,
        )


def test_jsonl_writer_roundtrip(tmp_path: Path) -> None:
    p = tmp_path / "out.jsonl"
    with JsonlWriter(p) as w:
        w.write(_make_request())
        w.write(_make_request())
    rows = [json.loads(x) for x in p.read_text().splitlines()]
    assert len(rows) == 2
    assert rows[0]["run_id"] == "r1"
    assert rows[0]["method"] == "baseline_full"


def test_new_run_id_format() -> None:
    r = new_run_id("test")
    assert r.startswith("test-")
    # 8 chars date + T + 6 chars time + "-" + 3 millis
    tail = r.split("-", 1)[1]
    assert "T" in tail
    assert tail.split("-")[-1].isdigit()


def test_episode_record_defaults() -> None:
    e = EpisodeRecord(
        run_id="r1",
        task="t",
        env_id=0,
        episode_id=0,
        seed=0,
        method="m",
        success=True,
        steps=100,
        duration_s=1.5,
        policy_calls=4,
    )
    assert e.full_refreshes == 0
    assert e.partial_refreshes == 0


def test_run_meta_serializes() -> None:
    m = RunMeta(
        run_id="r1",
        timestamp_utc="2026-08-03T00:00:00Z",
        git_sha_repo="deadbeef",
        git_shas_third_party={"cosmos": "abc123"},
        model_revision="rev0",
        simulator_version="isaac-5.1",
        gpu_model="RTX PRO 6000",
        method="baseline_full",
        method_config={"steps": 4},
        tasks=["BananaInBowlTask"],
        seeds=[0, 1, 2],
        num_envs=1,
        warmup_count=2,
        episode_count=5,
        command="python foo.py",
        output_paths={"records": "results/raw/r1/requests.jsonl"},
    )
    d = m.model_dump()
    assert d["method"] == "baseline_full"
    assert d["seeds"] == [0, 1, 2]
