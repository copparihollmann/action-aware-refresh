#!/usr/bin/env python3
"""Build the success-vs-normalized-total-compute frontier — the M3 deliverable.

Reads every closed-loop ledger under `results/ledger/closed_loop/<method>/` and emits
`docs/pareto.md` plus `results/processed/pareto.json`.

The compute measure is the one spec §10 defines:

    normalized_total_compute = method_total_compute_with_overhead
                             / official_baseline_total_compute

Three things this script is deliberately careful about, because each is a way to
accidentally publish a flattering number:

1. **Compute is measured, not counted.** The denominator is the baseline's *measured*
   policy-inference wall time, not a FLOP estimate. Spec §8 forbids inferring a
   speedup from fewer tokens or theoretical FLOPs without measured latency, and this
   project already has a case in point: the wire costs ~1,197 ms per call regardless
   of what the model does.

2. **Per-task pairing, and macro before micro.** A method that fails a task runs to
   the timeout and therefore burns *more* compute than one that succeeds, so summing
   raw totals across tasks lets a bad method look expensive-but-honest while a
   selectively-failing one looks cheap. Compute is normalized per task and then
   averaged (macro), with the micro figure reported alongside.

3. **Simulator time is excluded from the compute ratio.** Isaac stepping dominated
   wall-clock in the smoke run (216.8 s of 370.3 s), but it is not part of a real
   deployment. Including it would dilute every speedup toward 1.0 and flatter any
   method. It is reported separately instead.

With one episode per task there are no meaningful confidence intervals — that is
stated in the output rather than papered over with an interval computed from n=1.
"""
from __future__ import annotations

import argparse
import json
import statistics
import sys
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

from action_refresh.ledger import Ledger  # noqa: E402

BASELINE = "baseline_full"


#: fields summed when a task has several units (e.g. one per seed/episode).
_ADDITIVE = (
    "n_episodes",
    "n_success",
    "total_steps",
    "policy_calls",
    "policy_inference_s",
    "env_step_s",
    "video_write_s",
    "wall_total_s",
)


def collect(root: Path) -> dict[str, dict[str, dict[str, Any]]]:
    """{method: {task: aggregated summary}} from every closed-loop ledger present.

    A task can have MULTIPLE units — one per seed, since seed is part of the work-unit
    identity so that additional episodes can be collected without the ledger skipping
    them as already done. Those units must be **summed**, not overwritten: assigning
    `per_task[task] = result` kept only whichever unit the glob happened to yield last
    and silently discarded the rest, which halved the episode count and turned a
    2-episode comparison back into a 1-episode one without any warning.
    """
    out: dict[str, dict[str, dict[str, Any]]] = {}
    if not root.is_dir():
        return out
    for method_dir in sorted(root.iterdir()):
        if not method_dir.is_dir():
            continue
        led = Ledger(method_dir)
        per_task: dict[str, dict[str, Any]] = {}
        for payload in led.results():
            unit = payload.get("unit") or {}
            task = unit.get("task")
            res = payload.get("result") or {}
            if not task or not res:
                continue
            acc = per_task.setdefault(task, {"n_units": 0, "variants": []})
            acc["n_units"] += 1
            acc["variants"].append(unit.get("variant", ""))
            for key in _ADDITIVE:
                if res.get(key) is not None:
                    acc[key] = (acc.get(key) or 0) + res[key]
            # Non-additive context: keep the first unit's value.
            for key in ("open_loop_horizon", "output_dir"):
                acc.setdefault(key, res.get(key))
        for task, acc in per_task.items():
            if acc.get("n_episodes"):
                acc["success_rate"] = acc["n_success"] / acc["n_episodes"]
        if per_task:
            out[method_dir.name] = per_task
    return out


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--ledger-root", default=str(REPO_ROOT / "results" / "ledger" / "closed_loop"))
    ap.add_argument("--out-json", default=str(REPO_ROOT / "results" / "processed" / "pareto.json"))
    ap.add_argument("--out-md", default=str(REPO_ROOT / "docs" / "pareto.md"))
    args = ap.parse_args()

    data = collect(Path(args.ledger_root))
    if not data:
        raise SystemExit(
            f"no closed-loop results under {args.ledger_root} — run "
            "scripts/run_closed_loop.py first"
        )
    if BASELINE not in data:
        raise SystemExit(
            f"`{BASELINE}` has no results. Every normalized-compute number divides by "
            "it, so there is nothing to normalize against. Run the baseline first."
        )

    base = data[BASELINE]
    rows: list[dict[str, Any]] = []
    for method, per_task in sorted(data.items()):
        # Only tasks measured under BOTH the method and the baseline can be paired.
        shared = sorted(set(per_task) & set(base))
        if not shared:
            continue
        successes = [per_task[t]["n_success"] for t in shared]
        episodes = [per_task[t]["n_episodes"] for t in shared]
        base_succ = [base[t]["n_success"] for t in shared]
        base_eps = [base[t]["n_episodes"] for t in shared]

        # Per-task normalized compute, then macro-averaged. Totals are divided by that
        # task's EPISODE COUNT first, because methods no longer share one: the baseline has
        # 2 episodes on the screened tasks while some methods have 1. Ratioing raw totals
        # made a method with half the episodes look ~2x cheaper — an artefact of the
        # denominator, not a saving. Guard the zero case rather than emitting inf.
        def per_episode(src: dict[str, dict[str, Any]], task: str, key: str) -> float:
            d = src[task]
            eps = d.get("n_episodes") or 0
            return ((d.get(key) or 0.0) / eps) if eps else 0.0

        ratios = []
        for t in shared:
            b = per_episode(base, t, "policy_inference_s")
            m = per_episode(per_task, t, "policy_inference_s")
            if b > 0:
                ratios.append(m / b)
        micro_num = sum(per_episode(per_task, t, "policy_inference_s") for t in shared)
        micro_den = sum(per_episode(base, t, "policy_inference_s") for t in shared)

        # A second, noise-immune compute measure. Measured wall time on this host has
        # an intermittent host-side stall (B3: identical work spanning 3,377-5,974 ms
        # under sustained load), and closed-loop runs of different methods happen at
        # different times, so a wall-time-only ratio can be perturbed by something that
        # has nothing to do with the method. Policy-call count cannot be: it is
        # determined by the trajectory and the horizon. Where the two disagree, the call
        # count says what the method *did* and the wall time says what it *cost* — and
        # for a horizon sweep, calls are the primary signal (halving the horizon doubles
        # the calls) with wall time as confirmation.
        call_ratios = []
        for t in shared:
            b = per_episode(base, t, "policy_calls")
            m = per_episode(per_task, t, "policy_calls")
            if b > 0:
                call_ratios.append(m / b)

        succ_rate = sum(successes) / max(sum(episodes), 1)
        base_rate = sum(base_succ) / max(sum(base_eps), 1)

        # Decompose total compute. Spec §10 defines normalized_total_compute as a ratio of
        # totals, and that is what is reported — but a total is
        # (per-call cost) x (calls per episode), and calls per episode depends on how long
        # the episode ran, i.e. on whether the method SUCCEEDED. A method that fails runs
        # to the timeout, makes more calls, and therefore shows *higher* total compute even
        # when each of its calls is cheaper. Reporting only the total would credit that to
        # the method's efficiency instead of to its failure. So report per-call cost too:
        # it is the part attributable to the configuration.
        def per_call(src: dict[str, dict[str, Any]]) -> float | None:
            t = sum(src[k].get("policy_inference_s") or 0.0 for k in shared)
            c = sum(src[k].get("policy_calls") or 0 for k in shared)
            return (t / c) if c else None

        m_per_call = per_call(per_task)
        b_per_call = per_call(base)

        rows.append(
            {
                "method": method,
                "n_tasks": len(shared),
                "n_episodes": sum(episodes),
                "success_rate": succ_rate,
                # The baseline's rate ON THE SAME TASKS. Without this the table compares a
                # method's rate over 9 shared tasks against the baseline's over its own 15,
                # and the delta reads as if it were computed from the headline numbers.
                "baseline_success_rate_on_shared": base_rate,
                "success_points_vs_baseline": 100.0 * (succ_rate - base_rate),
                "policy_s_per_call": m_per_call,
                "baseline_policy_s_per_call": b_per_call,
                "normalized_per_call": (
                    (m_per_call / b_per_call) if (m_per_call and b_per_call) else None
                ),
                "normalized_compute_macro": statistics.mean(ratios) if ratios else None,
                "normalized_compute_micro": (micro_num / micro_den) if micro_den else None,
                "normalized_calls_macro": (
                    statistics.mean(call_ratios) if call_ratios else None
                ),
                "policy_inference_s": micro_num,
                "policy_calls": sum(per_task[t].get("policy_calls") or 0 for t in shared),
                "total_steps": sum(per_task[t].get("total_steps") or 0 for t in shared),
                # Reported, never folded into the ratio.
                "env_step_s": sum(per_task[t].get("env_step_s") or 0.0 for t in shared),
                "per_task": {
                    t: {
                        "success": per_task[t]["n_success"],
                        "episodes": per_task[t]["n_episodes"],
                        "baseline_success": base[t]["n_success"],
                        "policy_inference_s": per_task[t].get("policy_inference_s"),
                        "baseline_policy_inference_s": base[t].get("policy_inference_s"),
                        "steps": per_task[t].get("total_steps"),
                        "policy_calls": per_task[t].get("policy_calls"),
                    }
                    for t in shared
                },
            }
        )

    report = {"baseline": BASELINE, "rows": rows}
    out_json = Path(args.out_json)
    out_json.parent.mkdir(parents=True, exist_ok=True)
    out_json.write_text(json.dumps(report, indent=2, default=str))
    Path(args.out_md).write_text(render(report))
    print(f"wrote {out_json}")
    print(f"wrote {args.out_md}")
    return 0


def render(r: dict[str, Any]) -> str:
    L: list[str] = []
    rows = r["rows"]
    L.append("# Task success vs normalized total compute (M3)")
    L.append("")
    L.append(
        f"Generated by `scripts/build_pareto.py`. Compute is normalized against "
        f"`{r['baseline']}` per task, then macro-averaged across tasks."
    )
    L.append("")
    L.append(
        "| method | tasks | success | baseline on same tasks | Δ | **norm. per-call** | "
        "norm. total | norm. calls | policy calls |"
    )
    L.append("|---|---|---|---|---|---|---|---|---|")
    for row in sorted(rows, key=lambda x: (x["normalized_per_call"] or 9e9)):

        def f(v: Any, nd: int = 3) -> str:
            return "—" if v is None else f"{v:.{nd}f}"

        is_base = row["method"] == r["baseline"]
        delta = "—" if is_base else f"{row['success_points_vs_baseline']:+.1f} pts"
        L.append(
            f"| `{row['method']}` | {row['n_tasks']} | "
            f"{100 * row['success_rate']:.1f}% | "
            f"{100 * row['baseline_success_rate_on_shared']:.1f}% | "
            f"{delta} | "
            f"**{f(row['normalized_per_call'])}** | {f(row['normalized_compute_macro'])} | "
            f"{f(row['normalized_calls_macro'])} | {row['policy_calls']} |"
        )
    L.append("")
    L.append(
        "**Read `norm. per-call` as the method's efficiency, and `norm. total` as its "
        "cost on this workload — they are different questions.** Total compute is "
        "`per-call cost x calls per episode`, and calls per episode depends on how long "
        "the episode ran, i.e. on whether the method *succeeded*. A method that fails runs "
        "to the timeout, makes more calls, and so shows **higher total compute even when "
        "each call is cheaper**. Spec §10 defines the headline measure as the ratio of "
        "totals, so that is reported — but crediting a failure-inflated total to the "
        "configuration's efficiency would be wrong, which is why the per-call column sits "
        "beside it."
    )
    L.append("")
    L.append(
        "`norm. calls` is the policy-call count: immune to this host's intermittent stalls "
        "(B3: identical work spanning 3,377–5,974 ms under sustained load) but blind to "
        "per-call cost. `success` and `baseline on same tasks` are both restricted to the "
        "tasks the two methods share, so the Δ is properly paired even when the baseline "
        "was measured on a larger set."
    )
    L.append("")
    L.append(
        "⚠ Per-call figures here come from a *closed-loop* run under sustained load, which "
        "B3 showed is noisy on this host. The authoritative per-configuration latency, "
        "measured interleaved and cooled, is in `docs/latency.md`; use that for any "
        "speedup claim and treat this column as corroboration."
    )
    L.append("")

    # Significance, stated in the table rather than left to the reader. A one-sided
    # binomial against the baseline's rate on the shared tasks is crude but sufficient to
    # separate "measured" from "not measured", and at these episode counts almost nothing
    # is measured — which is the honest headline.
    import math

    L.append("## Is any success difference statistically real?")
    L.append("")
    L.append("| method | score | baseline rate | P(<= observed) | verdict |")
    L.append("|---|---|---|---|---|")
    for row in sorted(rows, key=lambda x: x["success_rate"]):
        if row["method"] == r["baseline"]:
            continue
        k = int(round(row["success_rate"] * row["n_episodes"]))
        n = row["n_episodes"]
        pb = row["baseline_success_rate_on_shared"]
        pv = sum(math.comb(n, i) * pb**i * (1 - pb) ** (n - i) for i in range(k + 1))
        L.append(
            f"| `{row['method']}` | {k}/{n} | {pb:.3f} | **{pv:.3f}** | "
            + ("**significant**" if pv < 0.05 else "not established")
            + " |"
        )
    L.append("")
    L.append(
        "⚠ **Config matching.** Every row is normalised against `baseline_full`, which was run "
        "with the *default plain-text* prompt format. Rows suffixed `_promptjson` use the format "
        "the checkpoint was trained with, so they should be read against "
        "`baseline_full_promptjson` instead: `vision_frames_9_promptjson` is **0/9 against that "
        "baseline's 3/9, P = 0.026** — still significant. Note also that "
        "`baseline_full_promptjson` itself is indistinguishable from the plain-text baseline "
        "(P = 0.51), which is why the un-matched rows remain informative rather than void."
    )
    L.append("")
    L.append(
        "> **At these episode counts almost nothing is measurable.** The baseline disagrees "
        "with *itself* on 3 of 9 tasks between two seeds. A power calculation on the observed "
        "~11-point difference gives **~31 episodes per task per arm (~87 GPU-h for one "
        "pairwise comparison)** for 80% power — and the project's §14 gate is **2** absolute "
        "points, which would need thousands. Any claim of non-inferior success on this "
        "benchmark must either buy that many episodes or move to a lower-variance signal "
        "(task progress score, subtask counts) instead of binary success."
    )
    L.append("")

    ep_per_task = min((x["n_episodes"] / max(x["n_tasks"], 1)) for x in rows) if rows else 0
    if ep_per_task < 2:
        L.append(
            "> **No confidence intervals, deliberately.** At one episode per task each "
            "per-task outcome is a single Bernoulli draw, and closed-loop runs of the "
            "*identical* configuration are not reproducible here (the same task and "
            "settings ended at 145 steps once and ran to the 750-step timeout another "
            "time — the sampler and simulator are not bitwise deterministic). So this "
            "table shows **where methods differ enough to be worth more episodes**, not "
            "a statistically supported success comparison. Quoting an interval from n=1 "
            "would be inventing precision."
        )
        L.append("")

    L.append("## What is and is not in the compute number")
    L.append("")
    L.append(
        "- **In:** the client's measured end-to-end policy time, which includes image "
        "composition, msgpack serialization and the websocket round trip. Roughly "
        "1,197 ms per call of that is outside the model and invariant to anything done "
        "inside it, so it caps every speedup — Amdahl, measured rather than assumed."
    )
    L.append(
        "- **Out:** simulator stepping. It dominated wall-clock in the smoke run "
        "(216.8 s of 370.3 s) but is not part of a real deployment; including it would "
        "pull every ratio toward 1.0 and flatter every method. Reported separately:"
    )
    for row in rows:
        L.append(
            f"  - `{row['method']}`: env step {row['env_step_s']:.0f} s vs policy "
            f"{row['policy_inference_s']:.0f} s"
        )
    L.append(
        "- **Out:** FLOP counts. Spec §8 forbids treating fewer tokens or theoretical "
        "FLOPs as a speedup unless measured latency confirms it, so they inform "
        "hypotheses and never the frontier."
    )
    L.append("")

    L.append("## Per-task detail")
    L.append("")
    for row in rows:
        if row["method"] == r["baseline"]:
            continue
        L.append(f"### `{row['method']}` vs `{r['baseline']}`")
        L.append("")
        L.append("| task | success | baseline | steps | policy calls | policy time | baseline time |")
        L.append("|---|---|---|---|---|---|---|")
        for task, d in sorted(row["per_task"].items()):
            L.append(
                f"| `{task}` | {d['success']}/{d['episodes']} | "
                f"{d['baseline_success']}/{d['episodes']} | {d['steps']} | "
                f"{d['policy_calls']} | {(d['policy_inference_s'] or 0):.0f} s | "
                f"{(d['baseline_policy_inference_s'] or 0):.0f} s |"
            )
        L.append("")
        # Saturated tasks carry no information about a method difference, and §10 asks
        # that they be excluded from the screened set. Name them instead of leaving
        # them to dilute the average.
        both_zero = [
            t for t, d in row["per_task"].items() if d["success"] == 0 and d["baseline_success"] == 0
        ]
        both_full = [
            t
            for t, d in row["per_task"].items()
            if d["success"] == d["episodes"] and d["baseline_success"] == d["episodes"]
        ]
        if both_zero or both_full:
            L.append(
                f"Saturated (uninformative for this comparison): "
                f"{len(both_zero)} task(s) failed under both "
                f"({', '.join(f'`{t}`' for t in both_zero) or '—'}); "
                f"{len(both_full)} succeeded under both "
                f"({', '.join(f'`{t}`' for t in both_full) or '—'}). Spec §10 wants these "
                "out of the screened set; they are listed rather than silently averaged in."
            )
            L.append("")

    L.append("## Deviations from the official baseline")
    L.append("")
    L.append(
        "Guardrails were disabled throughout (`nvidia/Cosmos-Guardrail1` is gated and "
        "access is denied), so **every** latency here understates the official baseline "
        "by whatever the guardrail runners cost. This affects numerator and denominator "
        "alike, so the *ratios* are internally consistent, but the absolute figures are "
        "not the official ones. See `docs/upstream_patches.md`."
    )
    L.append("")
    return "\n".join(L) + "\n"


if __name__ == "__main__":
    raise SystemExit(main())
