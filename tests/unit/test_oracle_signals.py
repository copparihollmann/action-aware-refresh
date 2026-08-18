"""Tests for the privileged-state oracle (Experiment C).

Built on synthetic `EpisodeState`s so they need no h5py and no recorded episode. The
cases are the ones that would distort the oracle's upper bound: a dropped terminal
event, static scenery diluting the object-motion signal, or `max` silently becoming
`sum` across objects.
"""
from __future__ import annotations

import numpy as np
import pytest

from action_refresh.oracle.signals import EpisodeState, compute_signals, critical_steps


def make_state(t=20, events=None, objects=None) -> EpisodeState:
    objects = objects if objects is not None else {"banana": 0.0, "table": 0.0}
    return EpisodeState(
        n_steps=t,
        object_pose={
            name: np.tile(np.array([x, 0, 0, 0, 0, 0, 1.0]), (t, 1))
            for name, x in objects.items()
        },
        object_velocity={name: np.zeros((t, 6)) for name in objects},
        ee_position=np.zeros((t, 3)),
        ee_orientation=np.tile(np.array([0, 0, 0, 1.0]), (t, 1)),
        ee_linear_velocity=np.zeros((t, 3)),
        ee_angular_velocity=np.zeros((t, 3)),
        joint_position=np.zeros((t, 13)),
        actions=np.zeros((t, 8)),
        subtask_score=np.zeros(3),
        success=False,
        events=events or [],
    )


def test_signal_arrays_all_align_to_step_count():
    """Misaligned signals would attribute a refresh decision to the wrong step."""
    st = make_state(t=25)
    sig = compute_signals(st)
    assert sig
    assert all(len(v) == 25 for v in sig.values()), {k: len(v) for k, v in sig.items()}


def test_terminal_event_is_captured_not_dropped():
    """Events are logged at step == n_steps; a strict bound dropped them.

    That silently discarded OBJECT_IN_CONTAINER_SUCCESS — the event marking the task as
    succeeded — so the oracle's most important moment was invisible.
    """
    t = 10
    st = make_state(t=t, events=[{"step": t, "name": "OBJECT_IN_CONTAINER_SUCCESS", "score": 1.0}])
    sig = compute_signals(st)
    assert sig["subtask_progress_event"].sum() == 1.0
    assert sig["subtask_progress_event"][-1] == 1.0, "should land on the final step"


def test_out_of_range_events_are_ignored():
    st = make_state(t=10, events=[{"step": 99, "name": "OBJECT_GRABBED_SUCCESS"}, {"step": -1, "name": "OBJECT_BUMPED"}])
    sig = compute_signals(st)
    assert sig["contact_event_proxy"].sum() == 0.0


def test_contact_events_mark_their_step():
    st = make_state(
        t=10,
        events=[
            {"step": 3, "name": "OBJECT_GRABBED_SUCCESS"},
            {"step": 7, "name": "TARGET_OBJECT_DROPPED"},
            {"step": 5, "name": "SOMETHING_UNRELATED"},
        ],
    )
    sig = compute_signals(st)
    assert list(np.flatnonzero(sig["contact_event_proxy"])) == [3, 7]


def test_static_scenery_is_excluded_by_default():
    """The table never moves; including it drags every object statistic toward zero."""
    st = make_state(t=10, objects={"banana": 0.0, "table": 0.0})
    # Move only the banana.
    st.object_pose["banana"][5:, 0] = 1.0
    sig = compute_signals(st)
    assert sig["object_max_step_translation"][5] == pytest.approx(1.0)
    # Explicitly restricting to the table alone gives no motion, proving the default
    # exclusion is what surfaced the banana.
    sig_table = compute_signals(st, task_objects=["table"])
    assert sig_table["object_max_step_translation"].max() == 0.0


def test_object_motion_uses_max_not_sum():
    """One object moving is the event; summing lets many small jitters imitate it."""
    st = make_state(t=6, objects={"a": 0.0, "b": 0.0})
    st.object_pose["a"][3:, 0] = 0.10
    st.object_pose["b"][3:, 0] = 0.10
    sig = compute_signals(st)
    # max -> 0.10, sum would have been 0.20.
    assert sig["object_max_step_translation"][3] == pytest.approx(0.10)


def test_no_task_objects_yields_zero_signals_not_a_crash():
    st = make_state(t=8, objects={"table": 0.0})
    sig = compute_signals(st, task_objects=[])
    for key in ("object_max_step_translation", "object_max_speed", "object_gripper_motion_mismatch"):
        assert sig[key].shape == (8,)
        assert sig[key].max() == 0.0


def test_critical_steps_dilate_around_events():
    """An event is logged when it completes; the useful refresh moment is just before,
    so a gate firing one step late has effectively missed it."""
    st = make_state(t=12, events=[{"step": 6, "name": "OBJECT_GRABBED_SUCCESS"}])
    sig = compute_signals(st)
    crit = critical_steps(sig, dilate=2)
    assert list(np.flatnonzero(crit)) == [4, 5, 6, 7, 8]


def test_critical_steps_without_dilation():
    st = make_state(t=12, events=[{"step": 6, "name": "OBJECT_GRABBED_SUCCESS"}])
    crit = critical_steps(compute_signals(st), dilate=0)
    assert list(np.flatnonzero(crit)) == [6]


def test_critical_steps_dilation_clamps_at_boundaries():
    st = make_state(t=5, events=[{"step": 0, "name": "OBJECT_BUMPED"}])
    crit = critical_steps(compute_signals(st), dilate=3)
    assert crit[0] and crit[3]
    assert len(crit) == 5, "dilation must not change the array length"


def test_ee_motion_signals_track_velocity():
    st = make_state(t=6)
    st.ee_linear_velocity[2:] = np.array([0.3, 0.4, 0.0])  # norm 0.5
    sig = compute_signals(st)
    assert sig["ee_speed"][3] == pytest.approx(0.5)
    assert sig["ee_speed"][0] == 0.0
