#!/usr/bin/env python3
"""Summarize the offline V×A grid: did asymmetric scheduling survive on our checkpoint?

Reads `results/processed/split_schedule.jsonl` and writes
`results/reports/split_schedule.json` plus a table on stdout.

The only interpretation rule, fixed before the numbers were seen: every cell's deviation
from the 4-step joint teacher is read against the **stock-repeat control**
(`teacher_repeat`), which is the same configuration run twice. A cell at or below that
floor is indistinguishable from the pipeline's own nondeterminism; a cell far above it has
changed the action. Cosine similarity carries the qualitative verdict — near 1 means a
perturbed version of the teacher's chunk, near 0 means an unrelated one, which is collapse
rather than degradation.

No success rate is claimed or implied. This decides which cells, if any, deserve
closed-loop episodes.
"""
from __future__ import annotations

import argparse
import json
import statistics as stats
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[1]

METRICS = ("joint_l2_mean", "joint_linf_max", "endpoint_joint_l2", "gripper_disagreement_rate")


def med(xs: list[float]) -> float | None:
    return stats.median(xs) if xs else None


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--jsonl", default=str(REPO_ROOT / "results" / "processed" / "split_schedule.jsonl")
    )
    ap.add_argument(
        "--out", default=str(REPO_ROOT / "results" / "reports" / "split_schedule.json")
    )
    args = ap.parse_args()

    path = Path(args.jsonl)
    if not path.exists():
        raise SystemExit(f"error: {path} not found — run scripts/offline_split_schedule.py")
    recs = [json.loads(x) for x in path.read_text().splitlines() if x.strip()]
    recs = [r for r in recs if r.get("kind") == "split_schedule"]
    if not recs:
        raise SystemExit("error: no split_schedule records")

    by_cell: dict[str, list[dict]] = defaultdict(list)
    for r in recs:
        by_cell[r["cell"]].append(r)

    floor = [
        r["deviation_vs_teacher"]["joint_l2_mean"]
        for r in by_cell.get("teacher_repeat", [])
        if r.get("deviation_vs_teacher")
    ]
    floor_med = med(floor)
    if floor_med is None:
        print(
            "warning: no teacher_repeat control in this file — deviations have no floor to "
            "be read against, so 'small' is not decidable. Re-run with the control.",
            file=sys.stderr,
        )

    rows: list[dict[str, Any]] = []
    for cell, rs in sorted(by_cell.items()):
        devs = [r["deviation_vs_teacher"] for r in rs if r.get("deviation_vs_teacher")]
        row: dict[str, Any] = {
            "cell": cell,
            "video_steps": rs[0].get("video_steps"),
            "action_steps": rs[0].get("action_steps"),
            "n": len(rs),
            "n_engaged": sum(1 for r in rs if r.get("sampler_engaged")),
            "cuda_ms_median": med([r["cuda_ms"] for r in rs]),
            "all_finite": all(r.get("action_finite", True) for r in rs),
            "tasks": sorted({r.get("task") for r in rs if r.get("task")}),
        }
        for m in METRICS:
            row[f"{m}_median"] = med([d[m] for d in devs if d.get(m) is not None])
        row["cosine_median"] = med(
            [d["cosine_similarity_mean"] for d in devs if d.get("cosine_similarity_mean") is not None]
        )
        dev_med = row.get("joint_l2_mean_median")
        if floor_med is not None and dev_med is not None:
            # A floor of exactly 0.0 is the good case, not a missing value: repeat calls
            # are bit-identical, so *any* nonzero deviation is signal. `if floor_med:`
            # would treat that as absent and silently drop the column.
            if floor_med > 0:
                row["ratio_to_noise_floor"] = dev_med / floor_med
            else:
                row["ratio_to_noise_floor"] = 0.0 if dev_med == 0 else float("inf")
        rows.append(row)

    print(f"\nnoise floor (stock repeat, median joint_l2_mean): {floor_med}")
    if floor_med == 0.0:
        print(
            "  → repeat calls are BIT-IDENTICAL. deterministic_seed=True holds for the whole\n"
            "    pipeline, not just the seed, so every nonzero deviation below is signal and\n"
            "    no minimum-detectable-effect threshold is needed for this comparison."
        )
    print(
        f"\n{'cell':16s} {'V':>2s} {'A':>3s} {'n':>3s} {'eng':>4s} "
        f"{'joint_l2':>9s} {'xfloor':>7s} {'linf':>7s} {'grip':>6s} {'cos':>6s} {'cuda_ms':>8s}"
    )
    for r in rows:
        if r["cell"] == "teacher_cold4":
            continue

        def f(v: Any, spec: str = "9.4g") -> str:
            return format(v, spec) if isinstance(v, (int, float)) else "—"

        print(
            f"{r['cell']:16s} {r['video_steps'] or 0:2d} {r['action_steps'] or 0:3d} "
            f"{r['n']:3d} {r['n_engaged']:4d} "
            f"{f(r.get('joint_l2_mean_median')):>9s} "
            f"{f(r.get('ratio_to_noise_floor'), '7.1f'):>7s} "
            f"{f(r.get('joint_linf_max_median'), '7.3g'):>7s} "
            f"{f(r.get('gripper_disagreement_rate_median'), '6.3f'):>6s} "
            f"{f(r.get('cosine_median'), '6.3f'):>6s} "
            f"{f(r.get('cuda_ms_median'), '8.0f'):>8s}"
        )

    teacher = by_cell.get("teacher_cold4", [])
    verdict_rows = [r for r in rows if r["cell"].startswith("knfv_") and r["video_steps"] != r["action_steps"]]
    collapsed = [r for r in verdict_rows if (r.get("cosine_median") or 0) < 0.3]
    summary = {
        "kind": "split_schedule_summary",
        "n_records": len(recs),
        "noise_floor_joint_l2_mean_median": floor_med,
        "teacher_cuda_ms_median": med([r["cuda_ms"] for r in teacher]),
        "cells": rows,
        "asymmetric_cells": len(verdict_rows),
        "asymmetric_cells_collapsed": len(collapsed),
        "verdict": (
            "COLLAPSE"
            if verdict_rows and len(collapsed) == len(verdict_rows)
            else "MIXED"
            if collapsed
            else "GRACEFUL"
        ),
        "interpretation": (
            "cosine ~0 means the emitted chunk is unrelated to the teacher's, not a degraded "
            "version of it. Read together with the noise floor: a ratio in the hundreds with "
            "near-zero cosine is the hard collapse their own sampler documents as the untested "
            "about for independent_action_schedule=False checkpoints, not graceful degradation."
        ),
        "provenance": {
            k: recs[0].get(k)
            for k in ("substrate", "commit", "vendor_source", "vendor_commit")
            if k in recs[0]
        },
    }
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(summary, indent=2))
    print(f"\nverdict: {summary['verdict']}  ({len(collapsed)}/{len(verdict_rows)} asymmetric cells collapsed)")
    print(f"wrote {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
