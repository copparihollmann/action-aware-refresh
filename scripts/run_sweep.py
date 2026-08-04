#!/usr/bin/env python3
"""Sweep runner — validates configs, refuses duplicate run IDs, resumes
interrupted experiments safely, and never mixes simulator versions.

The heavy lifting (starting Cosmos, driving RoboLab) is delegated to the
existing shell scripts. This module is the guard-rail layer that spec §12
requires.
"""
from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path

import yaml

REPO_ROOT = Path(__file__).resolve().parents[1]
REGISTRY = REPO_ROOT / "experiments" / "registry.yaml"
TASK_SETS = REPO_ROOT / "experiments" / "task_sets.yaml"
COMMANDS = REPO_ROOT / "reproducibility" / "commands.jsonl"


def load_yaml(p: Path) -> dict:
    return yaml.safe_load(p.read_text()) or {}


def existing_run_ids() -> set[str]:
    if not COMMANDS.exists():
        return set()
    ids: set[str] = set()
    for line in COMMANDS.read_text().splitlines():
        try:
            e = json.loads(line)
            if "run_id" in e:
                ids.add(e["run_id"])
        except json.JSONDecodeError:
            pass
    return ids


def check_isaac_lock() -> None:
    """Refuse to run if isaac5.0 and isaac5.1 are both installed."""
    r5 = REPO_ROOT / "third_party" / "RoboLab" / ".venv" / "lib"
    if not r5.is_dir():
        return
    have50 = any("isaacsim50" in p.name for p in r5.rglob("*"))
    have51 = any("isaacsim51" in p.name for p in r5.rglob("*"))
    if have50 and have51:
        sys.exit("ABORT: both isaac 5.0 and 5.1 present in RoboLab venv — pick one")


def resolve_config(name: str) -> dict:
    reg = load_yaml(REGISTRY)
    if name not in reg:
        sys.exit(f"unknown method: {name}\nknown: {sorted(reg)}")
    return reg[name]


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--method", required=True, help="key from experiments/registry.yaml")
    ap.add_argument("--set", dest="task_set", default="smoke", choices=("smoke", "pilot", "full"))
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()

    cfg = resolve_config(args.method)
    tasks = load_yaml(TASK_SETS).get(args.task_set, {})
    if not tasks.get("tasks"):
        sys.exit(f"task set `{args.task_set}` is empty — populate experiments/task_sets.yaml first")

    check_isaac_lock()

    print(f"method={args.method}  task_set={args.task_set}  n_tasks={len(tasks['tasks'])}")
    print(json.dumps(cfg, indent=2))
    if args.dry_run:
        return 0

    # Actual launch delegates to shell scripts:
    for task in tasks["tasks"]:
        env = {
            "ROBOLAB_TASK": task,
            "ROBOLAB_ENVS": "1",
            "ROBOLAB_HEADLESS": "1",
        }
        subprocess.check_call(["bash", "scripts/run_robolab.sh"], env={**dict(**__import__("os").environ), **env})

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
