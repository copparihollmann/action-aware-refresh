"""The refresh decision interface shared by every gate.

Spec §11.10 requires one high-level decision interface so the same mechanism can be
transferred to a second model family, and §11.2 requires separating three things that
are easy to conflate:

1. how many already-computed actions get executed,
2. whether a stale visual/keyframe representation is reused,
3. whether a whole new model invocation is skipped.

These four decisions encode exactly that separation:

``REUSE``
    Execute the next action from the cached chunk. No server call, no model compute.
    This is what the *baseline already does* for 31 of every 32 control steps
    (`OPEN_LOOP_HORIZON = 32`, confirmed: 34 server requests for 1,045 control steps).
    Any proposal framed as "skip redundant frames" must be measured against this, not
    against a per-control-step strawman the baseline never was.

``ACTION_REFRESH``
    Call the policy for a fresh action chunk, but reuse the cached visual state where
    the implementation allows it. Distinct from FULL_REFRESH because the token census
    shows 85.3% of the sequence is imagined future video — so *if* it can be reused,
    the saving is most of the request.

``PARTIAL_WORLD_REFRESH``
    Recompute only the action-relevant part of the visual representation (Experiment
    G's spatial masking). Kept in the interface even before it is implemented so gates
    and analysis speak one vocabulary; `NotImplementedError` at the execution site is
    honest, a silent downgrade to FULL_REFRESH is not.

``FULL_REFRESH``
    Everything recomputed — the official baseline's behaviour at a chunk boundary.

A gate returns a `Decision`, never a bare bool: "should I refresh?" cannot express the
partial cases, and a boolean interface would quietly collapse the distinction the whole
project rests on.
"""
from __future__ import annotations

import enum
from dataclasses import dataclass, field
from typing import Any


class RefreshAction(str, enum.Enum):
    """What to do at this control step. `str` mixin so it serializes readably."""

    REUSE = "REUSE"
    ACTION_REFRESH = "ACTION_REFRESH"
    PARTIAL_WORLD_REFRESH = "PARTIAL_WORLD_REFRESH"
    FULL_REFRESH = "FULL_REFRESH"

    @property
    def calls_policy(self) -> bool:
        """Whether this incurs a server round trip — i.e. costs real compute."""
        return self is not RefreshAction.REUSE


@dataclass(frozen=True)
class Decision:
    """A gate's output, with the evidence that produced it.

    `reason` and `features` are not decoration: refresh precision/recall and the
    "missed critical refresh" analysis (spec §11.4) need to know *why* each decision
    was made, and reconstructing that after the fact is impossible. They are recorded
    per decision so a gate can be audited rather than merely scored.
    """

    action: RefreshAction
    reason: str = ""
    features: dict[str, float] = field(default_factory=dict)
    # Control steps since the cached chunk/visual state was produced. Reported by every
    # gate because "action deviation vs cache age" is a required plot (spec §13.14).
    cache_age: int = 0

    @property
    def calls_policy(self) -> bool:
        return self.action.calls_policy

    def as_dict(self) -> dict[str, Any]:
        return {
            "action": self.action.value,
            "reason": self.reason,
            "features": dict(self.features),
            "cache_age": self.cache_age,
            "calls_policy": self.calls_policy,
        }


class Gate:
    """Base class: decide what to do at each control step.

    **The gate owns its cache age.** An earlier version made the caller pass `cache_age`
    in and reset it on refresh, which is a trap: a refresh *consumes an action
    immediately*, so the age after refreshing is 1, not 0. Getting that wrong gives a
    refresh period of `horizon + 1` — a 3% compute error at horizon 32, silently in the
    method's favour, and every gate would have had to rediscover it. Callers now just
    call `decide(step, context)` and the bookkeeping lives in one place.

    Subclasses implement `_decide`. The base class deliberately has no default policy —
    an accidental fallthrough returning REUSE would produce a method that never
    refreshes and looks extremely cheap.
    """

    name = "gate"

    def __init__(self) -> None:
        self._cache_age = 0
        self._calls = 0

    def reset(self) -> None:
        """Called at episode start. Subclasses with state must call super().reset()."""
        self._cache_age = 0
        self._calls = 0

    @property
    def cache_age(self) -> int:
        """Actions consumed from the current chunk (0 = no chunk yet)."""
        return self._cache_age

    @property
    def policy_calls(self) -> int:
        return self._calls

    def decide(self, step: int, context: dict[str, Any] | None = None) -> Decision:
        """Decide, then update the cache-age bookkeeping. Do not override."""
        decision = self._decide(step, self._cache_age, context or {})
        if decision.calls_policy:
            self._calls += 1
            # A fresh chunk, whose first action is consumed by this very step.
            self._cache_age = 1
        else:
            self._cache_age += 1
        return decision

    def _decide(self, step: int, cache_age: int, context: dict[str, Any]) -> Decision:
        raise NotImplementedError

    # Overhead accounting is mandatory, not optional: spec §7 requires auxiliary cost
    # (flow, event generation, gate inference, cache management) to be counted against
    # any claimed saving. A gate that cannot report its own cost cannot be evaluated.
    def overhead_ms(self) -> float:
        return 0.0


class FixedCadenceGate(Gate):
    """Refresh every `horizon` steps. The mandatory comparator.

    Spec §11.2/§11.3: an adaptive gate is only interesting if it beats *this* at equal
    compute. Reproducing the baseline exactly at `horizon=32` also makes it a sanity
    check on the harness — if the fixed gate at 32 does not match the unmodified
    baseline's call count, the harness is wrong, not the method.
    """

    name = "fixed_cadence"

    def __init__(self, horizon: int = 32) -> None:
        super().__init__()
        if horizon < 1:
            raise ValueError(f"horizon must be >= 1, got {horizon}")
        self.horizon = horizon

    def _decide(self, step: int, cache_age: int, context: dict[str, Any]) -> Decision:
        # cache_age == 0 means no chunk exists yet (episode start), which is a refresh
        # for the same reason a boundary is: there is nothing to reuse.
        if cache_age == 0 or cache_age >= self.horizon:
            return Decision(
                RefreshAction.FULL_REFRESH,
                reason=(
                    "no cached chunk"
                    if cache_age == 0
                    else f"cache_age {cache_age} >= horizon {self.horizon}"
                ),
                cache_age=cache_age,
            )
        return Decision(
            RefreshAction.REUSE,
            reason=f"cache_age {cache_age} < horizon {self.horizon}",
            cache_age=cache_age,
        )
