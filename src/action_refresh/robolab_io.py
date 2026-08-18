"""Read RoboLab's run outputs. Stdlib only — no numpy, no torch.

Deliberately dependency-free so every consumer can share it: `scripts/validate_smoke.py`
runs under the *system* python3 (it is invoked from `smoke_test.sh`), the closed-loop
runner runs in the repo venv, and analysis runs in either. Duplicating the parsing in
each is how the two would drift apart and start disagreeing about whether a run
succeeded.

Everything here is anchored to strings and paths verified against
`third_party/RoboLab @ 0aef241`, with the source location named at each site.
"""
from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any

# VERIFIED: robolab/core/utils/print_utils.py:30 prints "  Output         : <dir>"
_OUTPUT_RE = re.compile(r"^\s*Output\s+:\s+(\S+)", re.MULTILINE)

# VERIFIED: robolab/core/logging/results.py:606
EPISODE_RESULTS_FILENAME = "episode_results.jsonl"


def find_output_dir(client_log_text: str) -> Path | None:
    """Recover a run's output directory from the client's stdout."""
    m = _OUTPUT_RE.search(client_log_text)
    return Path(m.group(1)) if m else None


def read_episode_results(output_dir: Path) -> list[dict[str, Any]]:
    """Parse `episode_results.jsonl`, one record per episode.

    Skips blank lines and raises on malformed JSON rather than silently dropping a
    record: a quietly missing episode would understate a method's failure rate.
    """
    path = Path(output_dir) / EPISODE_RESULTS_FILENAME
    if not path.is_file():
        return []
    episodes: list[dict[str, Any]] = []
    for i, line in enumerate(path.read_text().splitlines(), 1):
        line = line.strip()
        if not line:
            continue
        try:
            episodes.append(json.loads(line))
        except json.JSONDecodeError as exc:
            raise ValueError(f"{path}:{i} is not valid JSON: {exc}") from exc
    return episodes


def episode_summary(episodes: list[dict[str, Any]], horizon: int) -> dict[str, Any]:
    """Aggregate a task's episodes into the fields the Pareto analysis needs.

    Success is counted from the runner's own boolean. `policy_calls` is *derived* from
    the step count and the client's open-loop horizon rather than guessed: the client
    calls the policy once per `horizon` control steps, which the M1 smoke run confirmed
    (34 requests for 1,045 control steps at horizon 32).

    `horizon` is required rather than defaulted because Experiment B varies it, and a
    stale default would silently misreport the compute of every horizon-sweep run.
    """
    if horizon <= 0:
        raise ValueError(f"horizon must be positive, got {horizon}")
    if not episodes:
        return {"n_episodes": 0}
    successes = [bool(e.get("success")) for e in episodes]
    steps = [int(e.get("episode_step") or 0) for e in episodes]

    def total(key: str) -> float:
        return float(sum((e.get("timing") or {}).get(key, 0.0) or 0.0 for e in episodes))

    return {
        "n_episodes": len(episodes),
        "n_success": sum(successes),
        "success_rate": sum(successes) / len(successes),
        "scores": [e.get("score") for e in episodes],
        "steps": steps,
        "total_steps": sum(steps),
        # Per-episode then summed. ceil(sum/h) != sum(ceil(s_i/h)) once there is more
        # than one episode: each episode starts a fresh chunk, so the partial final
        # chunk is paid once *per episode*. Aggregating first would undercount calls
        # and therefore understate the method's compute.
        "policy_calls": sum(-(-s // horizon) if s else 0 for s in steps),
        "open_loop_horizon": horizon,
        # Timing straight from the runner: end-to-end, including serialization and the
        # websocket round trip, which is the level spec §8 requires claims be made at.
        "policy_inference_s": total("policy_inference_s"),
        "env_step_s": total("env_step_s"),
        "video_write_s": total("video_write_s"),
        "wall_total_s": total("wall_total_s"),
        "reasons": [e.get("reason") for e in episodes],
        # Event counts (drops, wrong grabs, contact) — the contact-sensitive failures
        # spec §14 asks about, which a bare success rate hides.
        "events": _merge_events(episodes),
    }


def _merge_events(episodes: list[dict[str, Any]]) -> dict[str, Any]:
    merged: dict[str, Any] = {}
    for e in episodes:
        for key, val in (e.get("events") or {}).items():
            if isinstance(val, (int, float)):
                merged[key] = merged.get(key, 0) + val
            else:
                merged.setdefault(key, []).append(val)
    return merged


def policy_calls(total_steps: int, horizon: int) -> int:
    """Server round trips implied by a step count at a given open-loop horizon."""
    if horizon <= 0:
        raise ValueError(f"horizon must be positive, got {horizon}")
    return -(-total_steps // horizon)
