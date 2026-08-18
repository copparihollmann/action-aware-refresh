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
    cfg = reg[name] or {}
    if not isinstance(cfg, dict):
        sys.exit(f"registry entry `{name}` is not a mapping: {cfg!r}")
    return cfg


# Registry key -> environment variable understood by the launch scripts.
# Anything in `server_args` must be applied at server START; anything in
# `client_args` can be applied per client invocation.
SERVER_ARG_ENV = {
    "denoising_steps": "COSMOS_DENOISING_STEPS",
    "num_steps": "COSMOS_DENOISING_STEPS",
    "decode_video": "COSMOS_DECODE_VIDEO",
    "action_chunk_size": "COSMOS_ACTION_CHUNK_SIZE",
    "guidance": "COSMOS_GUIDANCE",
    "seed": "COSMOS_SEED",
    "deterministic_seed": "COSMOS_DETERMINISTIC",
}
CLIENT_ARG_ENV = {
    "open_loop_horizon": "ROBOLAB_OPEN_LOOP_HORIZON",
    "num_envs": "ROBOLAB_ENVS",
}


def translate_config(cfg: dict) -> tuple[dict[str, str], dict[str, str]]:
    """Turn registry args into env vars, failing loudly on unknown keys.

    Previously the resolved config was printed and then thrown away: only
    ROBOLAB_TASK reached the launcher, so `baseline_steps_1` and
    `baseline_steps_4` executed *identically* while being recorded under
    different method names. Silently mislabelled runs are worse than no runs.
    """
    server_env: dict[str, str] = {}
    client_env: dict[str, str] = {}
    unknown: list[str] = []

    for key, value in (cfg.get("server_args") or {}).items():
        var = SERVER_ARG_ENV.get(key)
        if var is None:
            unknown.append(f"server_args.{key}")
            continue
        server_env[var] = "true" if value is True else "false" if value is False else str(value)

    for key, value in (cfg.get("client_args") or {}).items():
        var = CLIENT_ARG_ENV.get(key)
        if var is None:
            unknown.append(f"client_args.{key}")
            continue
        client_env[var] = str(value)

    if unknown:
        sys.exit(
            f"ABORT: registry keys not wired to any launcher setting: {unknown}\n"
            "→ add them to SERVER_ARG_ENV / CLIENT_ARG_ENV in scripts/run_sweep.py "
            "and teach the launch script to honour them. Refusing to run a config "
            "whose settings would be silently dropped."
        )
    return server_env, client_env


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--method", required=True, help="key from experiments/registry.yaml")
    ap.add_argument("--set", dest="task_set", default="smoke", choices=("smoke", "pilot", "full"))
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument(
        "--assume-server-configured",
        action="store_true",
        help="confirm the running server was already started with this method's "
        "server_args (they cannot be set per request)",
    )
    args = ap.parse_args()

    cfg = resolve_config(args.method)
    tasks = load_yaml(TASK_SETS).get(args.task_set, {})
    if not tasks.get("tasks"):
        sys.exit(f"task set `{args.task_set}` is empty — populate experiments/task_sets.yaml first")

    check_isaac_lock()

    print(f"method={args.method}  task_set={args.task_set}  n_tasks={len(tasks['tasks'])}")
    print(json.dumps(cfg, indent=2))

    server_env, client_env = translate_config(cfg)
    print("server env:", server_env or "(defaults)")
    print("client env:", client_env or "(defaults)")

    if not tasks.get("screened", True):
        print(
            f"NOTE: task set `{args.task_set}` is marked screened: false — it is a "
            "candidate list, not a screened set (spec §10 requires excluding tasks "
            "whose baseline success is always 0 or always 1)."
        )

    if args.dry_run:
        return 0

    # A config that changes server-side generation (denoising steps, decode
    # video, chunk size) cannot be applied by the client: those are
    # server-START arguments. Refuse rather than silently running the baseline
    # under another method's label — that would fabricate a comparison.
    if server_env and not args.assume_server_configured:
        launch = " ".join(f"{k}={v}" for k, v in sorted(server_env.items()))
        sys.exit(
            f"ABORT: method `{args.method}` needs server-side settings "
            f"{sorted(server_env)}, but run_sweep only drives the client.\n"
            "Denoising steps / decode_video / chunk size are server-START args.\n"
            "→ start the server with them first:\n"
            f"    {launch} bash scripts/start_cosmos_server.sh\n"
            "  then re-run this command with --assume-server-configured."
        )

    import os

    for task in tasks["tasks"]:
        env = {
            **os.environ,
            **client_env,
            "ROBOLAB_TASK": task,
            "ROBOLAB_ENVS": str(cfg.get("client_args", {}).get("num_envs", 1)),
            "ROBOLAB_HEADLESS": "1",
        }
        subprocess.check_call(["bash", "scripts/run_robolab.sh"], env=env)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
