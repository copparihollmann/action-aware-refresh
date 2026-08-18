"""Tests for action-deviation metrics.

These gate every offline screening decision in Phases 1-2, so the properties that
matter are: identical chunks read as zero deviation, the gripper is compared *after*
binarization, and a shape mismatch is refused rather than silently truncated.
"""
from __future__ import annotations

import numpy as np
import pytest

from action_refresh.deviation import (
    GRIPPER_THRESHOLD,
    action_deviation,
    chunk_spread,
    ee_deviation_available,
)


def chunk(t: int = 32, joints: float = 0.0, gripper: float = 0.0) -> np.ndarray:
    a = np.zeros((t, 8), dtype=np.float32)
    a[:, :7] = joints
    a[:, -1] = gripper
    return a


def test_identical_chunks_have_zero_deviation():
    """The identity check: a harness that cannot report zero here is broken."""
    a = chunk(joints=0.3, gripper=0.9)
    d = action_deviation(a, a.copy())
    assert d.joint_l2_mean == 0.0
    assert d.joint_l2_max == 0.0
    assert d.joint_rmse == 0.0
    assert d.endpoint_joint_l2 == 0.0
    assert d.gripper_disagreement_rate == 0.0
    assert d.gripper_first_disagreement_step is None
    assert d.cosine_similarity_mean == pytest.approx(1.0)


def test_joint_error_magnitude_is_reported_in_radians():
    ref = chunk(joints=0.0)
    cand = chunk(joints=0.0)
    cand[:, 0] = 0.1  # one joint off by 0.1 rad throughout
    d = action_deviation(cand, ref)
    assert d.joint_l2_mean == pytest.approx(0.1)
    assert d.joint_linf_max == pytest.approx(0.1)
    assert d.endpoint_joint_l2 == pytest.approx(0.1)


def test_gripper_compared_after_binarization():
    """Raw-float differences that survive thresholding must not count.

    The robot only ever sees the binarized command (`_postprocess_chunk`), so a
    0.9-vs-0.6 difference is invisible to it. Reporting that as disagreement would
    inflate every deviation number.
    """
    ref = chunk(gripper=0.9)
    cand = chunk(gripper=0.6)  # both above threshold -> same command
    assert action_deviation(cand, ref).gripper_disagreement_rate == 0.0

    # Straddling the threshold is a real disagreement.
    cand2 = chunk(gripper=GRIPPER_THRESHOLD - 0.01)
    assert action_deviation(cand2, ref).gripper_disagreement_rate == 1.0


def test_first_gripper_disagreement_step_is_located():
    """Early flips matter more than late ones, so the step is reported."""
    ref = chunk(t=10, gripper=0.0)
    cand = chunk(t=10, gripper=0.0)
    cand[7:, -1] = 1.0
    d = action_deviation(cand, ref)
    assert d.gripper_first_disagreement_step == 7
    assert d.gripper_disagreement_rate == pytest.approx(0.3)


def test_shape_mismatch_is_refused():
    """Truncating to the overlap would flatter a method returning a short chunk."""
    with pytest.raises(ValueError, match="shape mismatch"):
        action_deviation(chunk(t=16), chunk(t=32))


def test_malformed_input_is_refused():
    with pytest.raises(ValueError, match=r"\[T, D\]"):
        action_deviation(np.zeros(8), chunk())
    with pytest.raises(ValueError, match="expected at least"):
        action_deviation(np.zeros((32, 3)), np.zeros((32, 3)))


def test_cosine_similarity_detects_opposite_motion():
    t = 10
    ref = np.zeros((t, 8), dtype=np.float32)
    ref[:, 0] = np.linspace(0, 1, t)  # joint 0 sweeping up
    cand = np.zeros((t, 8), dtype=np.float32)
    cand[:, 0] = np.linspace(0, -1, t)  # sweeping down
    assert action_deviation(cand, ref).cosine_similarity_mean == pytest.approx(-1.0)


def test_stationary_chunks_report_agreement_not_nan():
    """Zero motion must not produce a divide-by-zero NaN."""
    d = action_deviation(chunk(joints=0.5), chunk(joints=0.5))
    assert d.cosine_similarity_mean == pytest.approx(1.0)
    assert not np.isnan(d.cosine_similarity_mean)


def test_chunk_spread_is_zero_for_identical_samples():
    """E0: if varying the seed changes nothing, spread is zero."""
    chunks = [chunk(joints=0.2, gripper=1.0) for _ in range(4)]
    s = chunk_spread(chunks)
    assert s["joint_spread_rms"] == 0.0
    assert s["joint_spread_max"] == 0.0
    assert s["gripper_unstable_step_rate"] == 0.0
    assert s["n_chunks"] == 4.0


def test_chunk_spread_detects_gripper_instability():
    a = chunk(t=10, gripper=1.0)
    b = chunk(t=10, gripper=1.0)
    b[3:6, -1] = 0.0
    s = chunk_spread([a, b])
    assert s["gripper_unstable_step_rate"] == pytest.approx(0.3)
    assert s["joint_spread_rms"] == 0.0  # joints identical


def test_chunk_spread_needs_two_chunks():
    with pytest.raises(ValueError, match="at least 2"):
        chunk_spread([chunk()])


def test_ee_deviation_is_declared_unavailable():
    """Spec §11.1 wants EE deviation; it needs Isaac FK. Absence must be explicit,
    not silent, so callers report the limitation instead of omitting the metric."""
    assert ee_deviation_available() is False


# --- reference-free chunk motion (added after E2b's 0/9 closed-loop failure) --------
from action_refresh.deviation import chunk_motion  # noqa: E402


def test_stationary_chunk_reports_no_motion():
    """A policy emitting a constant pose must be detectable.

    Deviation from a reference cannot see this — a stationary chunk can sit at an
    ordinary L2 distance from the teacher — which is exactly how `vision_frames_9`
    passed the offline screen and then scored 0/9 closed-loop.
    """
    m = chunk_motion(chunk(t=32, joints=0.5))
    assert m.per_step_motion == 0.0
    assert m.joint_range == 0.0
    assert m.net_displacement == 0.0
    assert m.path_length == 0.0
    assert m.straightness == 0.0, "stationary must read as going nowhere, not as NaN"


def test_straight_chunk_is_maximally_straight():
    t = 10
    a = np.zeros((t, 8), dtype=np.float32)
    a[:, 0] = np.linspace(0.0, 1.0, t)
    m = chunk_motion(a)
    assert m.net_displacement == pytest.approx(1.0)
    assert m.path_length == pytest.approx(1.0)
    assert m.straightness == pytest.approx(1.0)


def test_meandering_chunk_has_low_straightness():
    """Moves a lot, gets nowhere — the signature of a policy that will not reach."""
    t = 11
    a = np.zeros((t, 8), dtype=np.float32)
    a[:, 0] = [0, .3, 0, .3, 0, .3, 0, .3, 0, .3, 0]
    m = chunk_motion(a)
    assert m.net_displacement == pytest.approx(0.0, abs=1e-6)
    assert m.path_length > 2.5, "it definitely moved"
    assert m.straightness < 0.05, "but got nowhere"


def test_range_distinguishes_travel_from_rate():
    """The measured E2b signature: normal per-step rate, reduced range.

    Two chunks with the same per-step motion can cover very different amounts of joint
    space, which is what deviation-from-teacher hides.
    """
    t = 33
    far = np.zeros((t, 8), dtype=np.float32)
    far[:, 0] = np.linspace(0, 1.0, t)          # travels 1.0
    near = np.zeros((t, 8), dtype=np.float32)
    step = 1.0 / (t - 1)
    near[:, 0] = [(i % 2) * step for i in range(t)]  # same per-step rate, no travel
    mf, mn = chunk_motion(far), chunk_motion(near)
    assert mf.per_step_motion == pytest.approx(mn.per_step_motion, rel=1e-6)
    assert mf.joint_range > 10 * mn.joint_range


def test_single_step_chunk_is_handled():
    m = chunk_motion(np.zeros((1, 8), dtype=np.float32))
    assert m.n_steps == 1 and m.path_length == 0.0
