"""Privileged-state oracle signals (Experiment C, spec §11.3).

The point of an oracle is to establish an **upper bound** before building anything
learned: if refresh scheduling driven by perfect knowledge of the simulator's state
does not beat a fixed cadence, then optical flow, event cameras and learned gates are
not worth their overhead, and that is a result worth having cheaply.

Signals are computed from what RoboLab actually records. Verified present in
`data/demo_0/` of a run's `run_0.hdf5` (and still written when `--video-mode none`, so
the cheap capture mode does not cost us the oracle):

| signal | source |
|---|---|
| task-object / target pose change | `states/rigid_object/<obj>/root_pose` (7: xyz + quat) |
| unexpected object motion | `states/rigid_object/<obj>/root_velocity` (6) |
| object centroids | `bbox/centroid/<obj>` (3) |
| gripper pose and motion | `ee_pose/{position,orientation,linear_velocity,angular_velocity}` |
| arm state | `states/articulation/robot/{joint_position,joint_velocity}` (13) |
| subtask progress / transition | `subtask/{score,status,completed}` |
| executed actions | `actions` (T, 8) |

**What is NOT available, stated rather than approximated.** Spec §11.3 lists "contact
onset" and "contact loss" as oracle signals. There is no per-step contact-force array
in the HDF5. What exists is a per-episode *event log* (`log_0_env0.json`) carrying
step-indexed events — `OBJECT_GRABBED_SUCCESS`, `TARGET_OBJECT_DROPPED`,
`WRONG_OBJECT_GRABBED`, `GRIPPER_HIT_OBJECT`, `OBJECT_BUMPED` — which is a *proxy* for
contact transitions, not contact state. It is used as such and labelled as such.
Fabricating a contact signal from object velocities would look like the requested
metric while measuring something else.

**These are offline-derived.** Everything here reads a finished episode's recording,
which makes it the right tool for the oracle upper bound (spec §11.3 explicitly allows
privileged state) but *not* a deployable gate: an online oracle would need the same
quantities at runtime, which is a separate question flagged in the plan and not
assumed here.
"""
from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np


@dataclass(frozen=True)
class EpisodeState:
    """Per-step privileged state for one episode, as arrays of length T."""

    n_steps: int
    object_pose: dict[str, np.ndarray]  # name -> [T,7]
    object_velocity: dict[str, np.ndarray]  # name -> [T,6]
    ee_position: np.ndarray  # [T,3]
    ee_orientation: np.ndarray  # [T,4]
    ee_linear_velocity: np.ndarray  # [T,3]
    ee_angular_velocity: np.ndarray  # [T,3]
    joint_position: np.ndarray  # [T,13]
    actions: np.ndarray  # [T,8]
    subtask_score: np.ndarray  # [K] — per subtask, NOT per step (see note below)
    success: bool
    events: list[dict[str, Any]]

    @property
    def object_names(self) -> list[str]:
        return sorted(self.object_pose)


def load_episode_state(hdf5_path: Path, demo: str = "demo_0") -> EpisodeState:
    """Read one episode's privileged state.

    Requires h5py, which lives in the RoboLab venv (it is RoboLab's own dependency),
    so oracle analysis runs there rather than in the Cosmos venv.
    """
    import h5py  # noqa: PLC0415

    with h5py.File(hdf5_path, "r") as f:
        g = f[f"data/{demo}"]
        states = g["states"]
        rigid = states["rigid_object"]
        object_pose = {k: np.asarray(rigid[k]["root_pose"]) for k in rigid}
        object_velocity = {k: np.asarray(rigid[k]["root_velocity"]) for k in rigid}
        ee = g["ee_pose"]
        out = EpisodeState(
            n_steps=int(g.attrs.get("num_samples", len(np.asarray(g["actions"])))),
            object_pose=object_pose,
            object_velocity=object_velocity,
            ee_position=np.asarray(ee["position"]),
            ee_orientation=np.asarray(ee["orientation"]),
            ee_linear_velocity=np.asarray(ee["linear_velocity"]),
            ee_angular_velocity=np.asarray(ee["angular_velocity"]),
            joint_position=np.asarray(states["articulation"]["robot"]["joint_position"]),
            actions=np.asarray(g["actions"]),
            # NOTE: subtask/score has length K (number of subtasks), not T. Treating it
            # as per-step would silently misalign progress with time.
            subtask_score=np.asarray(g["subtask"]["score"]),
            success=bool(g.attrs.get("success", False)),
            events=[],
        )
    # The step-indexed event stream lives beside the HDF5, not inside it.
    log = hdf5_path.parent / "log_0_env0.json"
    events: list[dict[str, Any]] = []
    if log.is_file():
        try:
            events = json.loads(log.read_text()).get("events") or []
        except json.JSONDecodeError:
            events = []
    return EpisodeState(**{**out.__dict__, "events": events})


def _pose_delta(pose: np.ndarray) -> np.ndarray:
    """Per-step translation magnitude of a [T,7] pose track (xyz + quat)."""
    xyz = pose[:, :3]
    d = np.zeros(len(xyz))
    if len(xyz) > 1:
        d[1:] = np.linalg.norm(np.diff(xyz, axis=0), axis=1)
    return d


def compute_signals(state: EpisodeState, *, task_objects: list[str] | None = None) -> dict[str, np.ndarray]:
    """Per-step oracle features, each an array of length T.

    `task_objects` restricts object-motion signals to the manipulated objects. Without
    it, static scenery (a table) dilutes the signal — the table's pose never changes, so
    including it drags every mean toward zero and makes the oracle look uninformative.
    """
    names = task_objects or [n for n in state.object_names if n != "table"]
    t = len(state.ee_position)

    signals: dict[str, np.ndarray] = {}

    # Gripper motion: how fast the arm is actually moving.
    signals["ee_speed"] = np.linalg.norm(state.ee_linear_velocity, axis=1)
    signals["ee_angular_speed"] = np.linalg.norm(state.ee_angular_velocity, axis=1)
    signals["ee_step_translation"] = np.concatenate(
        [[0.0], np.linalg.norm(np.diff(state.ee_position, axis=0), axis=1)]
    )

    # Object motion, aggregated over task objects. `max` rather than `sum`: one object
    # moving is the event of interest, and summing lets many small numerical jitters
    # imitate one real motion.
    if names:
        per_obj_delta = np.stack([_pose_delta(state.object_pose[n])[:t] for n in names])
        per_obj_speed = np.stack(
            [np.linalg.norm(state.object_velocity[n][:t, :3], axis=1) for n in names]
        )
        signals["object_max_step_translation"] = per_obj_delta.max(axis=0)
        signals["object_max_speed"] = per_obj_speed.max(axis=0)
        # Relative motion: object moving while the gripper is not (or vice versa) is the
        # mismatch that usually means something slipped or was knocked.
        signals["object_gripper_motion_mismatch"] = np.abs(
            signals["object_max_step_translation"] - signals["ee_step_translation"]
        )
    else:
        for key in (
            "object_max_step_translation",
            "object_max_speed",
            "object_gripper_motion_mismatch",
        ):
            signals[key] = np.zeros(t)

    # Contact-transition PROXY from the step-indexed event log. Not contact state — see
    # the module docstring. A 1.0 marks a step where a contact-related event fired.
    contact = np.zeros(t)
    contact_event_names = {
        "OBJECT_GRABBED_SUCCESS",
        "OBJECT_GRABBED_FAILURE",
        "TARGET_OBJECT_DROPPED",
        "WRONG_OBJECT_GRABBED",
        "GRIPPER_HIT_OBJECT",
        "OBJECT_BUMPED",
        "GRIPPER_FULLY_CLOSED",
    }
    subtask = np.zeros(t)
    for ev in state.events:
        step = ev.get("step")
        if not isinstance(step, int) or step < 0 or step > t:
            continue
        # Terminal events are logged at step == n_steps (one past the last recorded
        # state), so a strict `step < t` bound silently dropped them — including
        # OBJECT_IN_CONTAINER_SUCCESS, i.e. the event that marks the task succeeding.
        # Clamp to the final step instead: the event did happen, at the end.
        idx = min(step, t - 1)
        if ev.get("name") in contact_event_names:
            contact[idx] = 1.0
        if (ev.get("score") or 0) > 0:
            subtask[idx] = 1.0  # a scoring event = subtask progress
    signals["contact_event_proxy"] = contact
    signals["subtask_progress_event"] = subtask

    lengths = {k: len(v) for k, v in signals.items()}
    if len(set(lengths.values())) != 1:
        raise ValueError(f"signal length mismatch, cannot align to steps: {lengths}")
    return signals


def critical_steps(signals: dict[str, np.ndarray], *, dilate: int = 2) -> np.ndarray:
    """Steps where a refresh is plausibly *necessary*, as a boolean mask.

    Union of contact transitions and subtask progress, dilated by `dilate` steps on
    each side. Dilation is deliberate: an event is detected at the step it *completes*,
    while the useful moment to refresh is slightly before, and a gate that refreshes one
    step late has effectively missed it. Reported as recall-on-critical-steps rather
    than folded into an overall accuracy, because missing a contact transition and
    firing one unnecessary refresh are not comparable errors (spec §11.4).
    """
    base = (signals["contact_event_proxy"] > 0) | (signals["subtask_progress_event"] > 0)
    if dilate <= 0:
        return base
    out = base.copy()
    for shift in range(1, dilate + 1):
        out[shift:] |= base[:-shift]  # earlier steps flagged for a later event
        out[:-shift] |= base[shift:]
    return out
