"""Oracle temporal refresh gate (Experiment C, spec §11.3).

Uses privileged simulator state to place refreshes at the moments that matter, to
establish an **upper bound** before anything learned is built. If perfect knowledge
does not beat a fixed cadence, then optical flow, event cameras and learned gates are
not worth their overhead — a negative result worth having for the price of an offline
analysis rather than a training run.

One measurement reframes this experiment before it starts. Contact-critical steps, as
derived from the recorded event stream, are **5.5–10.1% of an episode's steps** (two
real episodes), while the baseline already refreshes at only **1/32 = 3.1%**. So the
oracle cannot save compute by "refreshing only when necessary" — refreshing at every
critical moment would cost *more* than the baseline. The honest question, which is also
exactly what spec §11.3 asks, is therefore:

    at a matched policy-call budget, does placing calls at oracle-chosen moments beat
    spacing them uniformly?

Hence two gates:

``OracleTemporalGate``
    Threshold form: refresh when a critical moment is imminent, with a hard cache-age
    cap so it can never drift arbitrarily far from a fresh plan. Its call count depends
    on the episode, so it is *not* directly comparable to fixed cadence.

``BudgetMatchedOracleGate``
    Given a fixed number of calls — the number the baseline would have made — spend
    them on the highest-priority moments. This is the comparison that isolates
    *placement* from *frequency*, which is the only way to attribute a difference to
    the oracle's knowledge rather than to it simply refreshing more.

Both are **oracles, not deployable gates**: they read a mask derived from a completed
episode's privileged state. That is legitimate for an upper bound (spec §11.3 permits
privileged state) and inadmissible as a method. `overhead_ms()` returns 0 for the same
reason — there is no real feature extraction to charge for, so an oracle's compute
advantage is an *over*-estimate of any learned gate's.
"""
from __future__ import annotations

from typing import Any, Sequence

import numpy as np

from action_refresh.gates.interface import Decision, Gate, RefreshAction


class OracleTemporalGate(Gate):
    """Refresh when a critical moment is imminent, capped by a maximum cache age.

    `lookahead` exists because a refresh only helps if it happens *before* the moment it
    is meant to handle: events are logged when they complete, so refreshing on the event
    step is already too late.

    `max_cache_age` is a safety cap, not a cadence. Without it the gate would sail
    through long quiet stretches on an arbitrarily stale plan, which is not a
    controlled experiment — it conflates "refresh at good moments" with "refresh very
    rarely".
    """

    name = "oracle_temporal"

    def __init__(
        self,
        critical: Sequence[bool] | np.ndarray,
        *,
        lookahead: int = 4,
        max_cache_age: int = 32,
    ) -> None:
        super().__init__()
        if lookahead < 0:
            raise ValueError(f"lookahead must be >= 0, got {lookahead}")
        if max_cache_age < 1:
            raise ValueError(f"max_cache_age must be >= 1, got {max_cache_age}")
        self.critical = np.asarray(critical, dtype=bool)
        self.lookahead = lookahead
        self.max_cache_age = max_cache_age

    def _critical_ahead(self, step: int) -> bool:
        hi = min(step + self.lookahead + 1, len(self.critical))
        if step >= len(self.critical):
            return False
        return bool(self.critical[step:hi].any())

    def _decide(self, step: int, cache_age: int, context: dict[str, Any]) -> Decision:
        if cache_age == 0:
            return Decision(RefreshAction.FULL_REFRESH, reason="no cached chunk", cache_age=cache_age)
        if self._critical_ahead(step):
            return Decision(
                RefreshAction.FULL_REFRESH,
                reason=f"critical moment within {self.lookahead} steps",
                features={"critical_ahead": 1.0},
                cache_age=cache_age,
            )
        if cache_age >= self.max_cache_age:
            return Decision(
                RefreshAction.FULL_REFRESH,
                reason=f"cache_age cap {self.max_cache_age} reached",
                features={"critical_ahead": 0.0},
                cache_age=cache_age,
            )
        return Decision(
            RefreshAction.REUSE,
            reason="no critical moment ahead",
            features={"critical_ahead": 0.0},
            cache_age=cache_age,
        )


class BudgetMatchedOracleGate(Gate):
    """Spend exactly `budget` refreshes, placed at the highest-priority steps.

    The comparison that isolates *placement* from *frequency*. Priority is the distance
    to the nearest critical step (nearer is higher), so the budget is spent covering
    critical moments first and filling quiet stretches only afterwards.

    Step 0 always consumes one refresh — there is no chunk to reuse at episode start —
    so a budget of N leaves N-1 to place. If the budget cannot cover every critical
    moment, `uncovered_critical` records how many were missed, which is the quantity
    that makes a negative result interpretable rather than merely disappointing.
    """

    name = "oracle_budget_matched"

    def __init__(
        self,
        critical: Sequence[bool] | np.ndarray,
        *,
        budget: int,
        n_steps: int,
        lookahead: int = 4,
    ) -> None:
        super().__init__()
        if budget < 1:
            raise ValueError(f"budget must be >= 1, got {budget}")
        self.critical = np.asarray(critical, dtype=bool)
        self.budget = budget
        self.n_steps = n_steps
        self.lookahead = lookahead
        self.refresh_steps = self._plan()
        self.uncovered_critical = self._count_uncovered()

    def _plan(self) -> set[int]:
        """Choose refresh steps up front — legitimate for an oracle, which may see the
        whole episode."""
        chosen = {0}
        # Target step for each critical moment: `lookahead` before it, so the fresh plan
        # is in hand when the moment arrives.
        targets = [
            max(0, int(i) - self.lookahead) for i in np.flatnonzero(self.critical)
        ]
        # Deduplicate while preserving order, so nearby critical moments do not each
        # consume a refresh at the same step.
        for t in targets:
            if len(chosen) >= self.budget:
                break
            chosen.add(t)
        # Any leftover budget goes to the largest remaining gaps: a plan is only valid
        # for so long, and leaving budget unspent would understate what the oracle can do.
        while len(chosen) < self.budget:
            ordered = sorted(chosen)
            gaps = [
                (b - a, a + (b - a) // 2)
                for a, b in zip(ordered, ordered[1:] + [self.n_steps])
                if b - a > 1
            ]
            if not gaps:
                break
            gaps.sort(reverse=True)
            chosen.add(gaps[0][1])
        return chosen

    def _count_uncovered(self) -> int:
        """Critical moments with no refresh in the `lookahead` window before them."""
        uncovered = 0
        for i in np.flatnonzero(self.critical):
            lo = max(0, int(i) - self.lookahead)
            if not any(s in self.refresh_steps for s in range(lo, int(i) + 1)):
                uncovered += 1
        return uncovered

    def _decide(self, step: int, cache_age: int, context: dict[str, Any]) -> Decision:
        if cache_age == 0 or step in self.refresh_steps:
            return Decision(
                RefreshAction.FULL_REFRESH,
                reason="planned refresh" if cache_age else "no cached chunk",
                cache_age=cache_age,
            )
        return Decision(RefreshAction.REUSE, reason="not a planned step", cache_age=cache_age)
