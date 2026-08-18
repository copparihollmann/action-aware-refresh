#!/usr/bin/env python3
"""Experiment C, offline: can an oracle schedule beat a fixed cadence at all?

Runs in the RoboLab venv (needs h5py). Costs no GPU.

Why offline first. A closed-loop oracle run needs a client-side gate patch plus ~3
GPU-h, and spec §11.3 sets a hard continue/stop gate on it. But two of the three
quantities that gate turns on are computable from recorded episodes alone:

1. **How many refreshes would an oracle want?** If contact-critical moments are *more*
   frequent than the baseline's fixed cadence, the oracle cannot save compute by
   refreshing less — it would refresh more. That is settled by counting, not by running.
2. **Can a matched budget even cover the critical moments?** If yes, the interesting
   question is placement and a closed-loop run is warranted. If the budget cannot cover
   them, an oracle at matched compute is structurally unable to help and the negative
   result is already in hand.

What this cannot settle is whether better placement *improves task success*. That needs
closed-loop episodes, and this analysis exists to decide whether they are worth buying.

    scripts/analyze_oracle_temporal.py --output-root third_party/RoboLab/output
"""
from __future__ import annotations

import argparse
import json
import statistics
import sys
from pathlib import Path
from typing import Any

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

from action_refresh.ledger import Ledger  # noqa: E402
from action_refresh.gates.interface import FixedCadenceGate  # noqa: E402
from action_refresh.gates.oracle_temporal import (  # noqa: E402
    BudgetMatchedOracleGate,
    OracleTemporalGate,
)
from action_refresh.oracle.signals import (  # noqa: E402
    compute_signals,
    critical_steps,
    load_episode_state,
)

BASELINE_HORIZON = 32


def drive(gate: Any, n_steps: int) -> int:
    gate.reset()
    for step in range(n_steps):
        gate.decide(step)
    return gate.policy_calls


def analyse_episode(hdf5: Path, *, lookahead: int, dilate: int) -> dict[str, Any]:
    state = load_episode_state(hdf5)
    signals = compute_signals(state)
    crit = critical_steps(signals, dilate=dilate)
    n = len(crit)

    baseline_calls = drive(FixedCadenceGate(horizon=BASELINE_HORIZON), n)
    threshold_calls = drive(
        OracleTemporalGate(crit, lookahead=lookahead, max_cache_age=BASELINE_HORIZON), n
    )
    budget_gate = BudgetMatchedOracleGate(
        crit, budget=baseline_calls, n_steps=n, lookahead=lookahead
    )
    budget_calls = drive(budget_gate, n)

    return {
        "episode": str(hdf5.parent.name),
        "path": str(hdf5),
        "n_steps": n,
        "success": state.success,
        "n_critical_steps": int(crit.sum()),
        "critical_rate": float(crit.mean()),
        "baseline_calls": baseline_calls,
        "baseline_call_rate": baseline_calls / n if n else 0.0,
        "oracle_threshold_calls": threshold_calls,
        "oracle_budget_calls": budget_calls,
        "oracle_budget_uncovered_critical": budget_gate.uncovered_critical,
        "n_events": len(state.events),
    }


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--output-root", default=str(REPO_ROOT / "third_party" / "RoboLab" / "output"))
    ap.add_argument(
        "--method",
        default="baseline_full",
        help="restrict to this method's episodes, resolved via its closed-loop ledger. "
        "Pooling every method's episodes is a confound: a degraded method barely touches "
        "anything (vision_frames_9 produced ONE contact event in nine episodes), so its "
        "episodes contribute almost no 'critical' steps and drag the rate down. The oracle "
        "question is about the BASELINE's refresh needs. Pass '' to pool everything.",
    )
    ap.add_argument("--lookahead", type=int, default=4)
    ap.add_argument("--dilate", type=int, default=2)
    ap.add_argument("--out-json", default=str(REPO_ROOT / "results" / "processed" / "oracle_temporal.json"))
    ap.add_argument("--out-md", default=str(REPO_ROOT / "docs" / "oracle_temporal.md"))
    args = ap.parse_args()

    hdf5s = sorted(Path(args.output_root).glob("*/*/run_*.hdf5"))
    if not hdf5s:
        raise SystemExit(f"no run_*.hdf5 under {args.output_root} — run a closed-loop episode first")

    if args.method:
        # Each closed-loop unit records the output directory it produced, so the ledger is
        # the authoritative method->episodes mapping; the directory names are only
        # timestamps and carry no method label.
        led_root = REPO_ROOT / "results" / "ledger" / "closed_loop" / args.method
        if not led_root.is_dir():
            raise SystemExit(
                f"no ledger for method {args.method!r} at {led_root} — run it closed-loop "
                "first, or pass --method '' to pool every episode (and accept the confound)."
            )
        wanted = set()
        for payload in Ledger(led_root).results():
            out = (payload.get("result") or {}).get("output_dir")
            if out:
                wanted.add(Path(out).resolve())
        before = len(hdf5s)
        hdf5s = [h for h in hdf5s if h.parent.parent.resolve() in wanted]
        print(
            f"restricted to {args.method}: {len(hdf5s)} of {before} episodes "
            f"({len(wanted)} output dirs)",
            flush=True,
        )
        if not hdf5s:
            raise SystemExit(f"no episodes matched method {args.method!r}")

    episodes: list[dict[str, Any]] = []
    for h in hdf5s:
        try:
            episodes.append(analyse_episode(h, lookahead=args.lookahead, dilate=args.dilate))
        except Exception as exc:  # noqa: BLE001
            # One unreadable episode must not lose the rest; record and continue.
            print(f"  skip {h}: {type(exc).__name__}: {exc}", file=sys.stderr)

    if not episodes:
        raise SystemExit("no episode could be analysed")

    report = {
        "method": args.method or "(all methods pooled)",
        "lookahead": args.lookahead,
        "dilate": args.dilate,
        "baseline_horizon": BASELINE_HORIZON,
        "n_episodes": len(episodes),
        "episodes": episodes,
        "aggregate": {
            "median_critical_rate": statistics.median(e["critical_rate"] for e in episodes),
            "median_baseline_call_rate": statistics.median(e["baseline_call_rate"] for e in episodes),
            "median_threshold_calls_ratio": statistics.median(
                e["oracle_threshold_calls"] / e["baseline_calls"]
                for e in episodes
                if e["baseline_calls"]
            ),
            "episodes_where_budget_covers_all_critical": sum(
                1 for e in episodes if e["oracle_budget_uncovered_critical"] == 0
            ),
        },
    }
    out_json = Path(args.out_json)
    out_json.parent.mkdir(parents=True, exist_ok=True)
    out_json.write_text(json.dumps(report, indent=2, default=str))
    Path(args.out_md).write_text(render(report))
    print(f"wrote {out_json}")
    print(f"wrote {args.out_md}")
    return 0


def render(r: dict[str, Any]) -> str:
    L: list[str] = []
    agg = r["aggregate"]
    L.append("# Experiment C (offline): oracle temporal refresh feasibility")
    L.append("")
    L.append(
        f"Generated by `scripts/analyze_oracle_temporal.py` over **{r['n_episodes']}** "
        f"recorded episodes. Critical steps are contact/subtask events from the "
        f"step-indexed event log, dilated by ±{r['dilate']} steps; the oracle is given "
        f"{r['lookahead']} steps of lead time. No GPU was used."
    )
    L.append("")
    L.append(
        "| episode | steps | success | critical steps | critical rate | baseline calls | "
        "oracle (threshold) | oracle (budget-matched) | uncovered critical |"
    )
    L.append("|---|---|---|---|---|---|---|---|---|")
    for e in r["episodes"]:
        L.append(
            f"| `{e['episode']}` | {e['n_steps']} | {e['success']} | "
            f"{e['n_critical_steps']} | {100 * e['critical_rate']:.1f}% | "
            f"{e['baseline_calls']} | {e['oracle_threshold_calls']} | "
            f"{e['oracle_budget_calls']} | {e['oracle_budget_uncovered_critical']} |"
        )
    L.append("")

    crit_rate = agg["median_critical_rate"]
    base_rate = agg["median_baseline_call_rate"]
    ratio = agg["median_threshold_calls_ratio"]
    L.append("## The finding that reframes this experiment")
    L.append("")
    L.append(
        f"- Contact-critical steps: **{100 * crit_rate:.1f}%** of an episode (median)."
    )
    L.append(
        f"- The baseline already refreshes at only **{100 * base_rate:.1f}%** of steps "
        f"(one call per {r['baseline_horizon']} control steps)."
    )
    L.append(
        f"- A threshold oracle that refreshes whenever a critical moment is imminent "
        f"therefore makes **{ratio:.2f}x** the baseline's calls (median)."
    )
    L.append("")
    if crit_rate > base_rate:
        L.append(
            "> **An oracle cannot save compute by refreshing only when necessary.** "
            "Critical moments are *more* frequent than the baseline's fixed cadence, so "
            "covering them all costs more, not less. This is a structural fact about the "
            "baseline — it already reuses each action chunk for 32 control steps — and no "
            "amount of gate cleverness changes it."
        )
        L.append("")
        L.append(
            "So spec §11.3's ‘≥20–25% total compute reduction from temporal scheduling’ "
            "is **not reachable by refreshing less often**. The only remaining way an "
            "oracle temporal gate can earn its place is by *improving success at matched "
            "compute* — spending the same number of calls at better-chosen moments."
        )
    else:
        L.append(
            "> Critical moments are rarer than the baseline cadence, so an oracle could "
            "in principle refresh less often and save compute. Worth a closed-loop run."
        )
    L.append("")

    covered = agg["episodes_where_budget_covers_all_critical"]
    L.append("## Can a matched budget cover the critical moments?")
    L.append("")
    L.append(
        f"In **{covered}/{r['n_episodes']}** episodes, a budget-matched oracle (exactly "
        "as many calls as the baseline) covers every critical moment with the required "
        "lead time."
    )
    L.append("")
    if covered == r["n_episodes"]:
        L.append(
            "Placement is therefore *not* budget-constrained: the baseline's own call "
            "budget is enough to be fresh at every contact transition, if only it knew "
            "where they were. That makes the matched-budget closed-loop comparison the "
            "informative experiment — and it is a success question, not a compute one."
        )
    else:
        L.append(
            "In the remaining episodes the budget is the binding constraint, so a "
            "matched-budget oracle is structurally unable to be fresh at every critical "
            "moment. A negative closed-loop result there would say more about the budget "
            "than about oracle scheduling, and must be reported that way."
        )
    L.append("")

    L.append("## What this does and does not establish")
    L.append("")
    L.append(
        "- **Established, at no GPU cost:** the direction and magnitude of the oracle's "
        "*call-count* effect, and whether a matched budget can cover critical moments. "
        "That is enough to rule out the compute-reduction framing of Experiment C."
    )
    L.append(
        "- **Not established:** whether better placement improves task success. Only "
        "closed-loop episodes can show that, and this analysis is what decides whether "
        "to buy them."
    )
    L.append(
        "- **Contact signal is a proxy.** There is no per-step contact-force array in "
        "the recordings; what exists is a step-indexed event log "
        "(`OBJECT_GRABBED_SUCCESS`, `TARGET_OBJECT_DROPPED`, `GRIPPER_HIT_OBJECT`, …). "
        "It is used as a proxy for contact transitions and labelled as one rather than "
        "fabricating a contact signal from object velocities."
    )
    L.append(
        "- **These are oracles, not gates.** They read privileged state from a completed "
        "episode and are charged zero feature-extraction overhead, so they bound any "
        "learned gate from above — a learned gate must pay for flow/event computation "
        "out of whatever saving remains (spec §7)."
    )
    L.append("")
    return "\n".join(L) + "\n"


if __name__ == "__main__":
    raise SystemExit(main())
