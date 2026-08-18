"""Tests for the refresh-decision interface and the fixed-cadence comparator.

The fixed-cadence gate is the mandatory comparator for every adaptive method
(spec §11.2/§11.3), so it has to reproduce the baseline *exactly* at horizon 32. If it
does not, an adaptive gate could beat it for reasons that have nothing to do with the
gate.
"""
from __future__ import annotations

import pytest

from action_refresh.gates.interface import (
    Decision,
    FixedCadenceGate,
    Gate,
    RefreshAction,
)
from action_refresh.gates.oracle_temporal import (
    BudgetMatchedOracleGate,
    OracleTemporalGate,
)


def test_only_reuse_avoids_a_policy_call():
    """The compute accounting hinges on this: everything except REUSE costs a round trip."""
    assert RefreshAction.REUSE.calls_policy is False
    for action in (
        RefreshAction.ACTION_REFRESH,
        RefreshAction.PARTIAL_WORLD_REFRESH,
        RefreshAction.FULL_REFRESH,
    ):
        assert action.calls_policy is True


def test_actions_serialize_readably():
    """Decisions land in JSONL result files; opaque enum reprs would be unreadable."""
    d = Decision(RefreshAction.FULL_REFRESH, reason="boundary", features={"x": 1.0}, cache_age=32)
    assert d.as_dict() == {
        "action": "FULL_REFRESH",
        "reason": "boundary",
        "features": {"x": 1.0},
        "cache_age": 32,
        "calls_policy": True,
    }


def test_base_gate_has_no_default_decision():
    """A fallthrough returning REUSE would produce a method that never refreshes and
    looks extremely cheap — the most flattering possible bug."""
    with pytest.raises(NotImplementedError):
        Gate().decide(0)


def drive(gate, steps: int) -> int:
    """Run a gate over `steps` control steps and return the policy-call count."""
    gate.reset()
    for step in range(steps):
        gate.decide(step)
    return gate.policy_calls


def test_fixed_cadence_reproduces_the_baseline_at_32():
    """One call per 32 control steps, matching the measured baseline.

    The M1 smoke run served 34 requests for 145 + 900 control steps across two
    episodes: ceil(145/32) + ceil(900/32) = 5 + 29 = 34. Reproducing that exactly is
    the harness sanity check.
    """
    assert drive(FixedCadenceGate(horizon=32), 145) == 5
    assert drive(FixedCadenceGate(horizon=32), 900) == 29


def test_fixed_cadence_refreshes_at_step_zero():
    """There is no cached chunk at episode start, so the first step must call."""
    gate = FixedCadenceGate(horizon=32)
    assert gate.decide(0).calls_policy is True
    assert gate.cache_age == 1, "the refreshing step consumes the chunk's first action"


def test_cache_age_advances_by_one_per_reused_step():
    gate = FixedCadenceGate(horizon=32)
    gate.decide(0)  # refresh -> age 1
    for expected in range(2, 33):
        d = gate.decide(expected)
        assert d.action is RefreshAction.REUSE
        assert gate.cache_age == expected


def test_fixed_cadence_refreshes_exactly_at_the_horizon():
    """The off-by-one this interface exists to prevent: refresh period must be
    `horizon`, not `horizon + 1`."""
    gate = FixedCadenceGate(horizon=8)
    actions = [gate.decide(s).action for s in range(17)]
    refresh_steps = [i for i, a in enumerate(actions) if a is RefreshAction.FULL_REFRESH]
    assert refresh_steps == [0, 8, 16]


@pytest.mark.parametrize("horizon", [1, 4, 8, 16, 32])
def test_call_rate_matches_the_horizon(horizon):
    """Experiment B's whole premise: halving the horizon doubles the calls."""
    steps = 640
    assert drive(FixedCadenceGate(horizon=horizon), steps) == steps // horizon


def test_reset_clears_state_between_episodes():
    """Leaked state would make episode 2 of a task behave differently from episode 1."""
    gate = FixedCadenceGate(horizon=8)
    drive(gate, 64)
    assert gate.policy_calls == 8
    gate.reset()
    assert gate.cache_age == 0 and gate.policy_calls == 0
    assert gate.decide(0).calls_policy is True


def test_invalid_horizon_is_refused():
    with pytest.raises(ValueError, match=">= 1"):
        FixedCadenceGate(horizon=0)


def test_gate_reports_its_own_overhead():
    """Spec §7 requires auxiliary cost counted against any saving; a gate that cannot
    report its cost cannot be evaluated. Fixed cadence genuinely costs nothing."""
    assert FixedCadenceGate().overhead_ms() == 0.0


# --- oracle temporal gates (Experiment C) ----------------------------------


def test_oracle_refreshes_before_a_critical_moment_not_on_it():
    """Refreshing on the event step is already too late: events are logged when they
    complete, so the plan must be fresh beforehand."""
    critical = [False] * 20
    critical[10] = True
    gate = OracleTemporalGate(critical, lookahead=3, max_cache_age=100)
    actions = [gate.decide(s).action for s in range(20)]
    refreshes = [i for i, a in enumerate(actions) if a is RefreshAction.FULL_REFRESH]
    assert 0 in refreshes, "must refresh at episode start"
    assert 7 in refreshes, "should fire lookahead steps before the critical moment"
    assert all(i <= 10 or i not in refreshes for i in range(11, 20)), "no late churn"


def test_oracle_cache_age_cap_prevents_unbounded_staleness():
    """Without the cap the gate would coast on an arbitrarily stale plan through quiet
    stretches, which conflates 'good placement' with 'almost never refresh'."""
    gate = OracleTemporalGate([False] * 100, lookahead=2, max_cache_age=10)
    assert drive(gate, 100) == 10


def test_oracle_with_no_critical_steps_reduces_to_the_cap():
    gate = OracleTemporalGate([False] * 64, lookahead=4, max_cache_age=32)
    assert drive(gate, 64) == 2


def test_oracle_rejects_bad_parameters():
    with pytest.raises(ValueError, match="lookahead"):
        OracleTemporalGate([False], lookahead=-1)
    with pytest.raises(ValueError, match="max_cache_age"):
        OracleTemporalGate([False], max_cache_age=0)


def test_budget_matched_oracle_spends_exactly_its_budget():
    """The comparison isolating placement from frequency only works if the budget is
    honoured exactly — otherwise the oracle wins by refreshing more."""
    critical = [False] * 320
    for i in (50, 120, 200):
        critical[i] = True
    gate = BudgetMatchedOracleGate(critical, budget=10, n_steps=320, lookahead=4)
    assert drive(gate, 320) == 10


def test_budget_matched_oracle_covers_critical_moments_first():
    critical = [False] * 100
    critical[40] = True
    critical[80] = True
    gate = BudgetMatchedOracleGate(critical, budget=3, n_steps=100, lookahead=5)
    assert gate.refresh_steps == {0, 35, 75}
    assert gate.uncovered_critical == 0


def test_budget_matched_oracle_reports_uncovered_critical_moments():
    """A negative result is only interpretable if we know the budget was the binding
    constraint rather than the placement."""
    critical = [False] * 100
    for i in (10, 20, 30, 40, 50):
        critical[i] = True
    gate = BudgetMatchedOracleGate(critical, budget=2, n_steps=100, lookahead=2)
    assert gate.uncovered_critical > 0


def test_budget_matched_oracle_spends_leftover_budget_on_gaps():
    """Leaving budget unspent would understate what the oracle can achieve."""
    gate = BudgetMatchedOracleGate([False] * 100, budget=4, n_steps=100, lookahead=2)
    assert len(gate.refresh_steps) == 4
    assert drive(gate, 100) == 4


def test_budget_matched_oracle_rejects_zero_budget():
    with pytest.raises(ValueError, match="budget"):
        BudgetMatchedOracleGate([False] * 10, budget=0, n_steps=10)


def test_oracles_report_zero_overhead_because_they_are_not_deployable():
    """An oracle charges no feature-extraction cost, so its compute advantage is an
    OVER-estimate of any learned gate's. Stated in code so it cannot be forgotten."""
    assert OracleTemporalGate([False]).overhead_ms() == 0.0
    assert BudgetMatchedOracleGate([False], budget=1, n_steps=1).overhead_ms() == 0.0
