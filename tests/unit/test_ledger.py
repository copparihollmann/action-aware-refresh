"""Tests for the resume ledger.

These matter more than most: the ledger is what stands between a killed 18-GPU-hour
chain and having to redo it. The cases below are the ones that would actually cost
us — a unit silently re-run, a crashed unit mistaken for a result, or two configs in
a sweep colliding onto one identity.
"""
from __future__ import annotations

import json

import pytest

from action_refresh.ledger import Ledger, WorkUnit


def test_identity_distinguishes_variants(tmp_path):
    """A step sweep must not collapse to one unit.

    This is the bug that would quietly destroy a sweep: if `variant` did not
    participate in the hash, num_steps 1..4 would share a directory, the first
    result would be reused for all four, and the report would show four identical
    rows without any error.
    """
    units = [WorkUnit(phase="p2", kind="offline", variant=f"steps={n}") for n in (1, 2, 3, 4)]
    assert len({u.key for u in units}) == 4
    led = Ledger(tmp_path)
    assert len({led.unit_dir(u) for u in units}) == 4


def test_identity_is_stable_across_processes(tmp_path):
    """Content-addressed, not positional: same identity -> same key, always."""
    a = WorkUnit(phase="p3", kind="closed_loop", method="baseline", task="BananaInBowlTask", episode=2)
    b = WorkUnit(phase="p3", kind="closed_loop", method="baseline", task="BananaInBowlTask", episode=2)
    assert a.key == b.key
    # Changing any field changes identity.
    assert a.key != WorkUnit(**{**a.as_dict(), "episode": 3}).key
    assert a.key != WorkUnit(**{**a.as_dict(), "method": "steps_1"}).key


def test_completed_unit_is_skipped_and_result_readable(tmp_path):
    led = Ledger(tmp_path)
    unit = WorkUnit(phase="p0", kind="census")
    assert led.should_run(unit)
    with led.claim(unit) as slot:
        slot.result = {"tokens": 3184}
    assert led.is_done(unit)
    assert not led.should_run(unit)
    assert led.state(unit).result == {"tokens": 3184}
    (payload,) = list(led.results(phase="p0"))
    assert payload["result"]["tokens"] == 3184


def test_work_runs_exactly_once_across_restarts(tmp_path):
    """The property the whole module exists for."""
    calls = []

    def drive():
        led = Ledger(tmp_path)
        for n in (1, 2, 3):
            unit = WorkUnit(phase="p2", kind="offline", variant=f"steps={n}")
            if not led.should_run(unit):
                continue
            with led.claim(unit) as slot:
                calls.append(n)
                slot.result = {"n": n}

    drive()
    drive()  # simulates a resume after the first pass completed
    assert calls == [1, 2, 3]


def test_partial_progress_survives_a_crash(tmp_path):
    """Kill the driver mid-sweep; finished units must not be recomputed."""
    calls = []

    class Boom(RuntimeError):
        pass

    def drive(fail_on: int | None):
        led = Ledger(tmp_path)
        for n in (1, 2, 3):
            unit = WorkUnit(phase="p2", kind="offline", variant=f"steps={n}")
            if not led.should_run(unit):
                continue
            with led.claim(unit) as slot:
                calls.append(n)
                if n == fail_on:
                    raise Boom("simulated kill")
                slot.result = {"n": n}

    with pytest.raises(Boom):
        drive(fail_on=2)
    assert calls == [1, 2]

    # Resume with retry enabled: unit 1 is done and skipped, 2 is retried, 3 runs.
    calls.clear()
    led = Ledger(tmp_path, retry_failed=True)
    for n in (1, 2, 3):
        unit = WorkUnit(phase="p2", kind="offline", variant=f"steps={n}")
        if not led.should_run(unit):
            continue
        with led.claim(unit) as slot:
            calls.append(n)
            slot.result = {"n": n}
    assert calls == [2, 3], "finished work must not be recomputed"


def test_exception_records_failure_not_success(tmp_path):
    led = Ledger(tmp_path)
    unit = WorkUnit(phase="p1", kind="e2a")
    with pytest.raises(ValueError):
        with led.claim(unit):
            raise ValueError("model blew up")
    st = led.state(unit)
    assert st.failed and not st.done
    assert "model blew up" in (st.error or "")
    # Default policy: do not burn GPU retrying a deterministic failure.
    assert not led.should_run(unit)
    assert Ledger(tmp_path, retry_failed=True).should_run(unit)


def test_falling_through_without_result_is_a_failure(tmp_path):
    """A unit cannot be marked done by accident."""
    led = Ledger(tmp_path)
    unit = WorkUnit(phase="p1", kind="e0")
    with led.claim(unit):
        pass  # forgot to set .result
    st = led.state(unit)
    assert st.failed and not st.done
    assert "no result" in (st.error or "")


def test_interrupted_unit_is_neither_done_nor_failed(tmp_path):
    """A hard kill leaves started.json; that must not read as a result."""
    led = Ledger(tmp_path)
    unit = WorkUnit(phase="p3", kind="closed_loop", task="T")
    claim = led.claim(unit)
    claim.__enter__()  # never exits: emulates SIGKILL mid-unit
    st = led.state(unit)
    assert st.interrupted and not st.done and not st.failed
    assert led.should_run(unit), "interrupted work must be re-attempted"


def test_corrupt_result_is_not_treated_as_done(tmp_path):
    """Defensive: a truncated result must not cause the work to be skipped."""
    led = Ledger(tmp_path)
    unit = WorkUnit(phase="p0", kind="census")
    d = led.unit_dir(unit)
    d.mkdir(parents=True)
    (d / "result.json").write_text("{not json")
    assert not led.is_done(unit)
    assert led.should_run(unit)


def test_result_write_is_atomic_leaving_no_temp_files(tmp_path):
    led = Ledger(tmp_path)
    unit = WorkUnit(phase="p0", kind="b3")
    with led.claim(unit) as slot:
        slot.result = {"std_ms": 30.0}
    d = led.unit_dir(unit)
    assert (d / "result.json").is_file()
    assert not list(d.glob("*.tmp")), "temp files must be renamed away, not left behind"
    assert not (d / "started.json").exists(), "started marker must be cleared on success"
    assert not (d / "claim.lock").exists(), "lock must be released"


def test_summary_counts_every_state(tmp_path):
    led = Ledger(tmp_path)
    done = WorkUnit(phase="p", kind="k", variant="done")
    failed = WorkUnit(phase="p", kind="k", variant="failed")
    pending = WorkUnit(phase="p", kind="k", variant="pending")
    with led.claim(done) as slot:
        slot.result = {}
    with led.claim(failed):
        pass
    assert led.summary([done, failed, pending]) == {
        "done": 1,
        "failed": 1,
        "interrupted": 0,
        "pending": 1,
    }


def test_audit_log_records_order(tmp_path):
    led = Ledger(tmp_path)
    with led.claim(WorkUnit(phase="p", kind="a")) as slot:
        slot.result = {}
    with led.claim(WorkUnit(phase="p", kind="b")):
        pass
    events = [json.loads(l)["event"] for l in led.audit.read_text().splitlines()]
    assert events == ["start", "done", "start", "error"]
