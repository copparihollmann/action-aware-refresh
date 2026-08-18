"""Validate experiments/registry.yaml and task_sets.yaml structurally.

A YAML quoting slip here is silent and expensive: a missing space produced the key
`"keyframe_oracle_refresh:{ parent"`, so the entry existed under a name no runner
would ever match and `keyframe_oracle_refresh` simply did not exist. Nothing failed —
it would just have been unrunnable when we got to M7.
"""
from __future__ import annotations

from pathlib import Path

import pytest
import yaml

REPO_ROOT = Path(__file__).resolve().parents[2]
REGISTRY = REPO_ROOT / "experiments" / "registry.yaml"
TASK_SETS = REPO_ROOT / "experiments" / "task_sets.yaml"


@pytest.fixture(scope="module")
def registry() -> dict:
    return yaml.safe_load(REGISTRY.read_text()) or {}


def test_no_malformed_keys(registry):
    """`:` or `{` in a key means a quoting bug swallowed the next mapping."""
    bad = [k for k in registry if any(ch in k for ch in ":{}[]")]
    assert bad == [], f"malformed registry keys (YAML quoting bug): {bad}"


def test_every_entry_is_a_mapping(registry):
    non_mappings = {k: type(v).__name__ for k, v in registry.items() if not isinstance(v, dict)}
    assert non_mappings == {}, f"registry entries must be mappings: {non_mappings}"


def test_parents_resolve(registry):
    """A dangling parent means the comparison baseline does not exist."""
    missing = {
        k: v["parent"]
        for k, v in registry.items()
        if v.get("parent") and v["parent"] not in registry
    }
    assert missing == {}, f"entries reference unknown parents: {missing}"


def test_spec_required_entries_present(registry):
    """Spec §12 names the entries the registry must define."""
    required = {
        "baseline_full",
        "baseline_decode_video",
        "baseline_steps_1",
        "baseline_steps_2",
        "baseline_steps_3",
        "baseline_steps_4",
        "baseline_action_only",
        "baseline_fixed_horizon_8",
        "baseline_fixed_horizon_16",
        "baseline_fixed_horizon_32",
        "oracle_temporal",
        "event_threshold",
        "flow_threshold",
        "event_flow_fused",
        "learned_gate",
        "cache_uniform",
        "cache_teacache_style",
        "cache_physical_aware",
        "spatial_random",
        "spatial_oracle",
        "keyframe_fixed_refresh",
        "keyframe_oracle_refresh",
        "keyframe_event_refresh",
        "combined_best",
    }
    assert required - set(registry) == set(), f"missing: {sorted(required - set(registry))}"


def test_baseline_full_is_the_official_configuration(registry):
    """The reference every Pareto plot is normalized against must not drift.

    4 denoising steps and no video decode is the upstream default; a change here
    silently redefines every reported speedup.
    """
    sa = registry["baseline_full"].get("server_args") or {}
    assert sa.get("denoising_steps") == 4
    assert sa.get("decode_video") is False
    assert (registry["baseline_full"].get("client_args") or {}).get("open_loop_horizon") == 32


def test_step_entries_match_their_names(registry):
    """`baseline_steps_2` must actually run 2 steps.

    A mismatch would label results with the wrong config — the same class of bug as
    run_sweep.py discarding server_args, which made steps_1 and steps_4 run identically.
    """
    for n in (1, 2, 3, 4):
        entry = registry[f"baseline_steps_{n}"]
        got = (entry.get("server_args") or {}).get("denoising_steps")
        assert got == n, f"baseline_steps_{n} declares denoising_steps={got}"


def test_horizon_entries_match_their_names(registry):
    for n in (8, 16, 32):
        entry = registry[f"baseline_fixed_horizon_{n}"]
        got = (entry.get("client_args") or {}).get("open_loop_horizon")
        assert got == n, f"baseline_fixed_horizon_{n} declares open_loop_horizon={got}"


def test_task_sets_are_well_formed():
    sets = yaml.safe_load(TASK_SETS.read_text()) or {}
    for name in ("smoke", "pilot", "full"):
        assert name in sets, f"task set `{name}` missing"
        tasks = (sets[name] or {}).get("tasks")
        assert tasks, f"task set `{name}` has no tasks"
        assert len(tasks) == len(set(tasks)), f"task set `{name}` has duplicates"


def test_task_sets_reference_real_tasks():
    """Every task name must exist in RoboLab's own metadata.

    A typo here fails only after Isaac has booted, minutes into a run.
    """
    import json

    meta_path = (
        REPO_ROOT / "third_party" / "RoboLab" / "robolab" / "tasks" / "_metadata" / "task_metadata.json"
    )
    if not meta_path.exists():
        pytest.skip("RoboLab clone not present")
    known = {t["task_name"] for t in json.loads(meta_path.read_text())}
    sets = yaml.safe_load(TASK_SETS.read_text()) or {}
    for name, spec in sets.items():
        tasks = (spec or {}).get("tasks") or []
        unknown = [t for t in tasks if t not in known]
        assert unknown == [], f"task set `{name}` references unknown tasks: {unknown}"
