"""Tests for parsing RoboLab run outputs.

Every number in the Pareto table flows through here, so the cases below are the ones
that would silently corrupt a comparison: miscounted policy calls, a dropped episode,
or a hardcoded horizon surviving into a horizon sweep.
"""
from __future__ import annotations

import json

import pytest

from action_refresh.robolab_io import (
    EPISODE_RESULTS_FILENAME,
    episode_summary,
    find_output_dir,
    policy_calls,
    read_episode_results,
)

# Real shape of the line RoboLab prints (print_utils.py:30).
CLIENT_LOG = """\
[INFO]: Completed setting up the environment...
  Policy         : cosmos3
  Output         : /scratch/x/third_party/RoboLab/output/2026-08-03_22-13-27_cosmos3
  Instr. Type    : default
"""


def episode(success=True, steps=145, policy_s=44.852, env_s=32.394, score=1.0, events=None):
    return {
        "success": success,
        "episode_step": steps,
        "score": score,
        "reason": "Completed subtask" if success else "Condition not satisfied",
        "events": events or {},
        "timing": {
            "policy_inference_s": policy_s,
            "env_step_s": env_s,
            "video_write_s": 3.1,
            "wall_total_s": policy_s + env_s + 3.1,
        },
    }


def test_finds_output_dir():
    d = find_output_dir(CLIENT_LOG)
    assert d is not None
    assert d.name == "2026-08-03_22-13-27_cosmos3"


def test_missing_output_line_returns_none():
    """Callers must be able to refuse to record a result rather than guess a path."""
    assert find_output_dir("nothing useful here") is None


def test_reads_episode_records(tmp_path):
    (tmp_path / EPISODE_RESULTS_FILENAME).write_text(
        json.dumps(episode()) + "\n\n" + json.dumps(episode(success=False)) + "\n"
    )
    eps = read_episode_results(tmp_path)
    assert len(eps) == 2, "blank lines must be skipped, records must not be"


def test_missing_results_file_is_empty_not_an_error(tmp_path):
    assert read_episode_results(tmp_path) == []


def test_malformed_record_raises_rather_than_dropping(tmp_path):
    """A silently dropped episode would understate a method's failure rate."""
    (tmp_path / EPISODE_RESULTS_FILENAME).write_text(json.dumps(episode()) + "\n{oops\n")
    with pytest.raises(ValueError, match="not valid JSON"):
        read_episode_results(tmp_path)


def test_summary_counts_success_and_sums_timing():
    s = episode_summary([episode(success=True), episode(success=False)], horizon=32)
    assert s["n_episodes"] == 2
    assert s["n_success"] == 1
    assert s["success_rate"] == 0.5
    assert s["policy_inference_s"] == pytest.approx(2 * 44.852)
    assert s["env_step_s"] == pytest.approx(2 * 32.394)


def test_policy_calls_counted_per_episode_not_on_the_total():
    """ceil(sum/h) != sum(ceil(s_i/h)).

    Each episode starts a fresh chunk, so the partial final chunk is paid once per
    episode. Aggregating first undercounts calls and understates the method's compute —
    and undercounting the *denominator* would inflate every later speedup.
    """
    # Two episodes of 33 steps at horizon 32: 2 calls each = 4, not ceil(66/32) = 3.
    s = episode_summary([episode(steps=33), episode(steps=33)], horizon=32)
    assert s["policy_calls"] == 4
    assert policy_calls(66, 32) == 3, "the aggregate-first form is the wrong one"


def test_horizon_is_honoured_not_hardcoded():
    """A stale default would misreport every Experiment B run."""
    eps = [episode(steps=64)]
    assert episode_summary(eps, horizon=32)["policy_calls"] == 2
    assert episode_summary(eps, horizon=8)["policy_calls"] == 8
    assert episode_summary(eps, horizon=64)["policy_calls"] == 1
    assert episode_summary(eps, horizon=8)["open_loop_horizon"] == 8


def test_zero_step_episode_contributes_no_calls():
    s = episode_summary([episode(steps=0)], horizon=32)
    assert s["policy_calls"] == 0


def test_invalid_horizon_is_refused():
    with pytest.raises(ValueError, match="positive"):
        episode_summary([episode()], horizon=0)
    with pytest.raises(ValueError, match="positive"):
        policy_calls(10, 0)


def test_events_merge_across_episodes():
    """Contact-sensitive failures (drops, wrong grabs) must survive aggregation —
    a bare success rate hides exactly the failures spec §14 asks about."""
    s = episode_summary(
        [
            episode(events={"TARGET_OBJECT_DROPPED": 2}),
            episode(events={"TARGET_OBJECT_DROPPED": 11, "WRONG_OBJECT_GRABBED": 2}),
        ],
        horizon=32,
    )
    assert s["events"]["TARGET_OBJECT_DROPPED"] == 13
    assert s["events"]["WRONG_OBJECT_GRABBED"] == 2


def test_non_numeric_events_are_collected_not_summed():
    s = episode_summary(
        [episode(events={"wrong_objects_grabbed": ["banana"]})], horizon=32
    )
    assert s["events"]["wrong_objects_grabbed"] == [["banana"]]


def test_empty_episode_list():
    assert episode_summary([], horizon=32) == {"n_episodes": 0}
