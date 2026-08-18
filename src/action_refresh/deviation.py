"""Action-deviation metrics: how far a cheap policy drifts from the full-compute one.

Spec §11.1 asks for action L2, end-effector translation/rotation deviation, gripper
disagreement and endpoint deviation. This module implements what the DROID
`joint_pos` action space actually permits, and is explicit about what it does not.

**What the action vector is.** `action_dim = 8` for `action_space="joint_pos"`:
7 joint positions plus 1 gripper channel. The server already un-flips the gripper
(`action_np[:, -1] = 1.0 - action_np[:, -1]`) and the RoboLab client binarizes it at
0.5 (`Cosmos3Client._postprocess_chunk`), so gripper agreement is measured *after*
that same threshold — comparing raw gripper floats would report disagreement the
robot never sees.

**What is deliberately absent.** End-effector translation and rotation deviation need
forward kinematics from the Franka model, which lives in Isaac (RoboLab's venv), not
in the Cosmos venv where these offline sweeps run. Rather than fabricate an EE number
from joint values, `ee_deviation_available()` returns False and the caller reports the
joint-space metrics with that limitation stated. Joint deviation is not a substitute:
a small wrist error and a small shoulder error move the gripper by very different
amounts. Treat these as a *screen*, and let closed-loop task success be the arbiter.

All functions take `[T, D]` arrays for a single action chunk and are pure.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any

import numpy as np

# Layout of the joint_pos action vector.
N_JOINTS = 7
GRIPPER_INDEX = -1
GRIPPER_THRESHOLD = 0.5  # matches Cosmos3Client._postprocess_chunk


@dataclass(frozen=True)
class ActionDeviation:
    """Deviation of one candidate action chunk from a reference chunk."""

    n_steps: int
    joint_l2_mean: float
    """Mean over timesteps of the per-timestep L2 norm across the 7 joints (rad)."""
    joint_l2_max: float
    joint_linf_max: float
    """Largest single-joint error anywhere in the chunk (rad) — the worst case."""
    joint_rmse: float
    endpoint_joint_l2: float
    """Deviation of the FINAL commanded pose. The chunk's last action is where an
    open-loop horizon leaves the robot, so this is the error the next observation
    inherits."""
    gripper_disagreement_rate: float
    """Fraction of timesteps where the binarized gripper command differs. A single
    flip can decide grasp vs no-grasp, so this is reported separately from joint
    error rather than folded into one score."""
    gripper_first_disagreement_step: int | None
    """Earliest disagreeing timestep, or None. Early flips matter far more than late
    ones because the rest of the chunk executes from a different contact state."""
    cosine_similarity_mean: float
    """Direction agreement of the per-step joint delta, ignoring magnitude."""

    def as_dict(self) -> dict[str, Any]:
        return asdict(self)


def _as_2d(a: np.ndarray, name: str) -> np.ndarray:
    arr = np.asarray(a, dtype=np.float64)
    if arr.ndim != 2:
        raise ValueError(f"{name} must be [T, D], got shape {arr.shape}")
    if arr.shape[1] < N_JOINTS + 1:
        raise ValueError(
            f"{name} has width {arr.shape[1]}, expected at least {N_JOINTS + 1} "
            "(7 joints + gripper). Is this really a joint_pos chunk?"
        )
    return arr


def action_deviation(candidate: np.ndarray, reference: np.ndarray) -> ActionDeviation:
    """Compare a candidate action chunk against a reference (teacher) chunk.

    Raises rather than truncating on a length mismatch: silently comparing the
    overlap would understate the deviation of a method that returns a shorter chunk,
    which is exactly the kind of quiet favouritism that invalidates a comparison.
    """
    cand = _as_2d(candidate, "candidate")
    ref = _as_2d(reference, "reference")
    if cand.shape != ref.shape:
        raise ValueError(
            f"chunk shape mismatch: candidate {cand.shape} vs reference {ref.shape}. "
            "Comparing only the overlap would flatter the shorter chunk."
        )

    cj, rj = cand[:, :N_JOINTS], ref[:, :N_JOINTS]
    diff = cj - rj
    per_step_l2 = np.linalg.norm(diff, axis=1)

    cg = (cand[:, GRIPPER_INDEX] > GRIPPER_THRESHOLD).astype(np.int8)
    rg = (ref[:, GRIPPER_INDEX] > GRIPPER_THRESHOLD).astype(np.int8)
    disagree = cg != rg
    first = int(np.argmax(disagree)) if disagree.any() else None

    # Per-step joint *delta* direction: what the arm is being told to do next,
    # independent of how far. Guard the zero-motion case rather than emitting NaN.
    if cand.shape[0] >= 2:
        cd = np.diff(cj, axis=0)
        rd = np.diff(rj, axis=0)
        cn = np.linalg.norm(cd, axis=1)
        rn = np.linalg.norm(rd, axis=1)
        valid = (cn > 1e-9) & (rn > 1e-9)
        if valid.any():
            cos = np.sum(cd[valid] * rd[valid], axis=1) / (cn[valid] * rn[valid])
            cos_mean = float(np.clip(cos, -1.0, 1.0).mean())
        else:
            cos_mean = 1.0  # both chunks stationary: perfectly in agreement
    else:
        cos_mean = 1.0

    return ActionDeviation(
        n_steps=int(cand.shape[0]),
        joint_l2_mean=float(per_step_l2.mean()),
        joint_l2_max=float(per_step_l2.max()),
        joint_linf_max=float(np.abs(diff).max()),
        joint_rmse=float(np.sqrt((diff**2).mean())),
        endpoint_joint_l2=float(np.linalg.norm(diff[-1])),
        gripper_disagreement_rate=float(disagree.mean()),
        gripper_first_disagreement_step=first,
        cosine_similarity_mean=cos_mean,
    )


@dataclass(frozen=True)
class ChunkMotion:
    """How much a chunk actually *commits to moving*, independent of any reference.

    Added after a failure the deviation metrics could not see. `vision_frames_9`
    deviated only 1.14x the model's own sampling noise from the teacher, yet scored
    **0/9** closed-loop and produced a single contact event across nine episodes, versus
    96 for the baseline. The robot was not mis-manipulating; it was barely engaging.

    Deviation from a reference cannot detect that, because a chunk that meanders near its
    start point can sit at a perfectly ordinary L2 distance from the teacher. What
    separates them is *net travel*: `vision_frames_9` moved at a normal per-step rate
    (0.050 vs the teacher's 0.042 rad) but covered only 56% of the teacher's joint range
    (0.365 vs 0.655). Mechanistically that is what a shortened imagined horizon should do
    — the imagination *is* the plan, so a model that can only see three latent frames
    ahead commits to less displacement.

    Stated honestly: this is a *candidate* diagnostic, not a validated predictor. It does
    not cleanly separate the failure from the baseline's own seed variation (`seed_2`
    reaches only 0.424 and still succeeds), so it should be reported alongside deviation
    and task success rather than trusted as a screen on its own. The real lesson is that
    open-loop metrics did not predict closed-loop outcomes here at all.
    """

    n_steps: int
    per_step_motion: float
    """Median per-step joint displacement (rad) — the instantaneous rate."""
    joint_range: float
    """L2 norm of the per-joint (max - min) over the chunk: how much of joint space the
    chunk actually sweeps."""
    net_displacement: float
    """Distance from first to last commanded pose. A chunk that returns to where it
    started has near-zero net displacement no matter how much it wandered."""
    path_length: float
    """Total path travelled. `path_length >> net_displacement` means meandering."""
    straightness: float
    """net_displacement / path_length, in [0, 1]. Low means the chunk wanders without
    getting anywhere — the signature of a policy that will not reach an object."""

    def as_dict(self) -> dict[str, Any]:
        return asdict(self)


def chunk_motion(chunk: np.ndarray) -> ChunkMotion:
    """Reference-free motion statistics for one action chunk."""
    a = _as_2d(chunk, "chunk")[:, :N_JOINTS]
    if a.shape[0] < 2:
        return ChunkMotion(int(a.shape[0]), 0.0, 0.0, 0.0, 0.0, 0.0)
    steps = np.linalg.norm(np.diff(a, axis=0), axis=1)
    path = float(steps.sum())
    net = float(np.linalg.norm(a[-1] - a[0]))
    return ChunkMotion(
        n_steps=int(a.shape[0]),
        per_step_motion=float(np.median(steps)),
        joint_range=float(np.linalg.norm(a.max(axis=0) - a.min(axis=0))),
        net_displacement=net,
        path_length=path,
        # Guard the stationary case rather than emitting NaN: a chunk that does not move
        # is maximally "straight" in the useless sense, so report 0 (goes nowhere).
        straightness=(net / path) if path > 1e-9 else 0.0,
    )


def ee_deviation_available() -> bool:
    """Whether end-effector-space deviation can be computed in this environment.

    Always False in the Cosmos venv: forward kinematics needs the Franka model from
    Isaac. Callers must report joint-space metrics *and* say EE metrics are missing,
    instead of quietly omitting a spec-required number.
    """
    return False


def chunk_spread(chunks: list[np.ndarray]) -> dict[str, float]:
    """Dispersion across several chunks generated for the *same* observation.

    This is the Experiment E0 measurement. Varying only the diffusion seed changes
    the imagined future; if the resulting actions barely move, the sampled
    imagination is not carrying action-relevant information. Compared against the
    spread across *different* observations, it says whether the model's action output
    is driven by what it sees or by what it invents.
    """
    if len(chunks) < 2:
        raise ValueError("need at least 2 chunks to measure spread")
    stack = np.stack([_as_2d(c, "chunk")[:, :N_JOINTS] for c in chunks])  # [K,T,J]
    mean = stack.mean(axis=0)
    # RMS distance of each sample from the mean chunk: a single spread number in the
    # same units (rad) as the deviation metrics, so the two are directly comparable.
    dev = np.sqrt(((stack - mean) ** 2).sum(axis=2))  # [K,T]
    gripper = np.stack(
        [(_as_2d(c, "chunk")[:, GRIPPER_INDEX] > GRIPPER_THRESHOLD).astype(np.int8) for c in chunks]
    )
    # Fraction of timesteps where the K samples do not unanimously agree.
    unstable = (gripper.min(axis=0) != gripper.max(axis=0)).mean()
    return {
        "n_chunks": float(len(chunks)),
        "joint_spread_rms": float(dev.mean()),
        "joint_spread_max": float(dev.max()),
        "gripper_unstable_step_rate": float(unstable),
    }
