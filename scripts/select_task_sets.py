#!/usr/bin/env python3
"""Generate experiments/task_sets.yaml from the checked-out RoboLab registry.

Task IDs are never hand-written: they come from
``third_party/RoboLab/robolab/tasks/_metadata/task_metadata.json`` (the same
metadata that produces upstream's task table), so the file cannot drift from the
pinned RoboLab commit.

Three sets, per spec §10:

- ``smoke``  2 tasks, 1-2 episodes. One is upstream's documented example so our
             smoke path matches the published one.
- ``pilot``  12-18 tasks chosen to span the benchmark's competency attributes
             and difficulty strata. **Provisional**: spec §10 requires the pilot
             be based on a screening run and requires that baseline success be
             neither always-0 nor always-1. We cannot know that before M3
             screening, so this script emits stratified *candidates* and marks
             them for filtering. Do not treat the pilot as final.
- ``full``   all 120 benchmark tasks (RoboLab-120).

Deterministic: candidates are sorted, and ties break on task name. No RNG.
"""
from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
METADATA = (
    REPO_ROOT
    / "third_party/RoboLab/robolab/tasks/_metadata/task_metadata.json"
)
OUT = REPO_ROOT / "experiments/task_sets.yaml"

# Upstream's documented example task, used in both the Cosmos server docs and
# RoboLab's README. Keeping it first makes our smoke path directly comparable.
SMOKE_PRIMARY = "BananaInBowlTask"
SMOKE_ALT = "RubiksCubeAndBananaTask"

# Competency axes we want the pilot to span, expressed as RoboLab `attributes`
# values, plus a structural axis (multi-stage) that is not an attribute.
# Ordered: rarer/more-specific axes first so they get picked before the
# extremely common `semantics` soaks up the budget.
ATTRIBUTE_AXES = [
    "stacking",       # contact-rich placement / alignment
    "reorientation",  # in-hand or regrasp-style reorientation
    "counting",       # requires tracking how many objects were moved
    "conjunction",    # multiple predicates in one instruction
    "sorting",        # relational multi-object assignment
    "size",           # size-discriminative grounding
    "vague",          # underspecified instruction
    "affordance",     # object-function grounding
    "spatial",        # relational spatial placement
    "color",          # colour-discriminative grounding, distractor-heavy
    "semantics",      # the broad default category
]

PILOT_TARGET = 16  # spec §10 asks for 12-18


def load_tasks() -> list[dict]:
    if not METADATA.exists():
        raise SystemExit(
            f"error: {METADATA} not found.\n"
            "→ run `make sources` first so third_party/RoboLab is checked out."
        )
    return json.loads(METADATA.read_text())


def attrs_of(task: dict) -> list[str]:
    return [a.strip() for a in str(task.get("attributes", "")).split(",") if a.strip()]


def select_pilot(tasks: list[dict]) -> tuple[list[str], list[str]]:
    """Return (task_names, rationale_lines).

    Greedy stratified cover: walk the competency axes in order, and for each,
    take the not-yet-selected task that best improves difficulty balance.
    """
    by_name = {t["task_name"]: t for t in tasks}
    chosen: list[str] = []
    rationale: list[str] = []
    diff_count: Counter[str] = Counter()

    def pick(candidates: list[dict], why: str) -> bool:
        # Prefer the difficulty label we currently have least of, so the pilot
        # does not collapse onto `simple` (64 of 120 tasks).
        pool = [t for t in candidates if t["task_name"] not in chosen]
        if not pool:
            return False
        pool.sort(
            key=lambda t: (
                diff_count[t["difficulty_label"]],
                -int(t.get("num_subtasks", 1)),
                t["task_name"],
            )
        )
        t = pool[0]
        chosen.append(t["task_name"])
        diff_count[t["difficulty_label"]] += 1
        rationale.append(
            f"{t['task_name']}: {why}; difficulty={t['difficulty_label']}"
            f"(score {t.get('difficulty_score')}), subtasks={t.get('num_subtasks')}, "
            f"stages={t.get('num_sequential_stages')}, episode_s={t.get('episode_s')}"
        )
        return True

    # 1. Structural axis first: genuinely multi-stage / procedural tasks are
    #    rare (7 of 120) and are the ones most likely to expose stale-cache
    #    failures at subtask transitions.
    multistage = [t for t in tasks if int(t.get("num_sequential_stages", 1)) > 1]
    for t in sorted(multistage, key=lambda x: (-int(x.get("num_sequential_stages", 1)), x["task_name"]))[:3]:
        pick([t], f"multi-stage ({t.get('num_sequential_stages')} sequential stages)")

    # 2. One task per competency attribute.
    for axis in ATTRIBUTE_AXES:
        if len(chosen) >= PILOT_TARGET:
            break
        pick([t for t in tasks if axis in attrs_of(t)], f"covers attribute `{axis}`")

    # 3. Top up with the most complex remaining tasks — contact-rich, many
    #    subtasks — since those are where reuse is most likely to break.
    if len(chosen) < PILOT_TARGET:
        rest = sorted(
            (t for t in tasks if t["task_name"] not in chosen),
            key=lambda t: (-int(t.get("difficulty_score", 0)), -int(t.get("num_subtasks", 1)), t["task_name"]),
        )
        for t in rest:
            if len(chosen) >= PILOT_TARGET:
                break
            pick([t], "top-up: high difficulty score / many subtasks")

    return chosen, rationale


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=str(OUT))
    args = ap.parse_args()

    tasks = load_tasks()
    names = sorted(t["task_name"] for t in tasks)
    for required in (SMOKE_PRIMARY, SMOKE_ALT):
        if required not in names:
            raise SystemExit(
                f"error: expected smoke task {required!r} is not in the checked-out "
                "registry. Inspect task_metadata.json and update SMOKE_* above — "
                "do not silently substitute another task."
            )

    pilot, rationale = select_pilot(tasks)
    by_name = {t["task_name"]: t for t in tasks}
    diff = Counter(by_name[n]["difficulty_label"] for n in pilot)
    covered: Counter[str] = Counter()
    for n in pilot:
        for a in attrs_of(by_name[n]):
            covered[a] += 1

    lines: list[str] = []
    lines.append("# Task sets — GENERATED by scripts/select_task_sets.py. Do not hand-edit.")
    lines.append(f"# Source: third_party/RoboLab @ {_robolab_sha()}")
    lines.append("#         robolab/tasks/_metadata/task_metadata.json (120 benchmark tasks)")
    lines.append("# Regenerate with: python scripts/select_task_sets.py")
    lines.append("")
    lines.append("smoke:")
    lines.append("  # spec §10: 2 tasks, 1-2 episodes. Both verified present in the registry.")
    lines.append("  # BananaInBowlTask is the example used by upstream's own server docs.")
    lines.append("  episodes_per_task: 2")
    lines.append("  tasks:")
    lines.append(f"    - {SMOKE_PRIMARY}")
    lines.append(f"    - {SMOKE_ALT}")
    lines.append("")
    lines.append("pilot:")
    lines.append("  # PROVISIONAL — stratified candidates, NOT a screened set.")
    lines.append("  # spec §10 requires the pilot exclude tasks whose baseline success is")
    lines.append("  # always 0 or always 1. That needs an M3 screening run (5-10 paired")
    lines.append("  # episodes/task); until then treat this as the screening candidate list.")
    lines.append(f"  # difficulty mix: {dict(diff)}")
    lines.append(f"  # attributes covered: {dict(covered)}")
    lines.append("  episodes_per_task: 10  # screening; promote to 20-30 for kept configs")
    lines.append("  screened: false")
    lines.append("  tasks:")
    for n in pilot:
        lines.append(f"    - {n}")
    lines.append("")
    lines.append("  rationale:")
    for r in rationale:
        lines.append(f"    # {r}")
    lines.append("")
    lines.append("full:")
    lines.append("  # RoboLab-120: every benchmark task.")
    lines.append("  # COST WARNING: upstream README quotes ~30 GPU-hours per 100 tasks, so")
    lines.append("  # this is ~36 GPU-hours PER CONFIGURATION. Requires explicit approval")
    lines.append("  # (CLAUDE.md: no run >4 GPU-hours without it) and must be chunked.")
    lines.append("  episodes_per_task: null  # use the benchmark's standard protocol")
    lines.append("  tasks:")
    for n in names:
        lines.append(f"    - {n}")
    lines.append("")

    Path(args.out).write_text("\n".join(lines))
    print(f"wrote {args.out}")
    print(f"  smoke: 2   pilot: {len(pilot)}   full: {len(names)}")
    print(f"  pilot difficulty mix: {dict(diff)}")
    return 0


def _robolab_sha() -> str:
    import subprocess

    try:
        return subprocess.run(
            ["git", "-C", str(REPO_ROOT / "third_party/RoboLab"), "rev-parse", "HEAD"],
            capture_output=True,
            text=True,
            timeout=10,
        ).stdout.strip() or "unknown"
    except Exception:
        return "unknown"


if __name__ == "__main__":
    raise SystemExit(main())
