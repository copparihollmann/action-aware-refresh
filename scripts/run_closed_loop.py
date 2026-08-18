#!/usr/bin/env python3
"""Run one registry method closed-loop over a task set, resumably.

This is the expensive path: a 16-task pilot pass costs ~3.7 GPU-h worst case, and
because RoboLab episodes terminate early on success but run to a per-task timeout on
failure, **a worse method costs more**. So the unit of resumable work is one *task*,
and an interrupted sweep never repeats a task it already finished.

    scripts/run_closed_loop.py --method baseline_full --set pilot --episodes 2

Why per-task rather than per-episode: RoboLab's `--num-runs` runs several episodes in
one client invocation, and each invocation pays ~20-60 s of Isaac startup. Per-episode
units would multiply that startup by the episode count for no benefit.

Server lifecycle lives here because denoising steps, decode-video and chunk size are
server-*start* arguments — they cannot be varied per request. The previous runner
required a manually pre-started server and took the operator's word (via
`--assume-server-configured`) that it matched the method; that is exactly how a
mislabelled comparison happens. Here the method's `server_args` determine the server
the method runs against, by construction.

Determinism: `COSMOS_SEED` is fixed so the server's seed *sequence* is reproducible
(`self._rng = np.random.default_rng(cfg.seed)`), which is the strongest pairing
available. It is not full reproducibility — closed-loop trajectories still diverge
between runs of the identical config (observed: the same task and settings ended at
145 steps once and ran to the 750-step timeout another time), because the sampler and
the simulator are not bitwise deterministic. That is why per-task episode counts, not
single episodes, decide anything.
"""
from __future__ import annotations

import argparse
import json
import os
import signal
import subprocess
import sys
import time
import urllib.error
import urllib.request
from pathlib import Path
from typing import Any

import yaml

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

from action_refresh.ledger import Ledger, WorkUnit  # noqa: E402
from action_refresh.robolab_io import (  # noqa: E402
    episode_summary,
    find_output_dir,
    read_episode_results,
)

REGISTRY = REPO_ROOT / "experiments" / "registry.yaml"
TASK_SETS = REPO_ROOT / "experiments" / "task_sets.yaml"

# Registry key -> server env var. Kept in sync with start_cosmos_server.sh.
SERVER_ARG_ENV = {
    "denoising_steps": "COSMOS_DENOISING_STEPS",
    "num_steps": "COSMOS_DENOISING_STEPS",
    "decode_video": "COSMOS_DECODE_VIDEO",
    "guidance": "COSMOS_GUIDANCE",
    "seed": "COSMOS_SEED",
    "deterministic_seed": "COSMOS_DETERMINISTIC",
    # Experiment E2b: shorten the imagined horizon (patch cosmos-framework-0004).
    # A server-START arg like the others, so it cannot be varied per request.
    "vision_frames": "COSMOS_VISION_FRAMES",
    # Prompt format must match training; see start_cosmos_server.sh for why the
    # upstream default is wrong when --checkpoint-path is an HF repo id.
    "format_prompt_as_json": "COSMOS_PROMPT_JSON",
}
CLIENT_ARG_ENV = {
    "open_loop_horizon": "ROBOLAB_OPEN_LOOP_HORIZON",
    "num_envs": "ROBOLAB_ENVS",
}


def load_yaml(p: Path) -> dict:
    return yaml.safe_load(p.read_text()) or {}


def resolve_method(name: str) -> dict[str, Any]:
    """Resolve a registry entry, inheriting `parent` args.

    Inheritance matters: most entries only declare their delta, so without merging the
    parent a method like `baseline_steps_1` would silently lose the baseline's
    open_loop_horizon and run against different client settings than the reference.
    """
    reg = load_yaml(REGISTRY)
    if name not in reg:
        raise SystemExit(f"unknown method {name!r}\nknown: {sorted(reg)}")
    chain: list[dict[str, Any]] = []
    seen: set[str] = set()
    cur: str | None = name
    while cur:
        if cur in seen:
            raise SystemExit(f"cyclic parent chain at {cur!r}")
        seen.add(cur)
        entry = reg.get(cur)
        if entry is None:
            raise SystemExit(f"method {name!r} has unknown parent {cur!r}")
        chain.append(entry)
        cur = entry.get("parent")
    merged: dict[str, Any] = {"server_args": {}, "client_args": {}}
    for entry in reversed(chain):  # root first, so the child overrides
        merged["server_args"].update(entry.get("server_args") or {})
        merged["client_args"].update(entry.get("client_args") or {})
        for k, v in entry.items():
            if k not in ("server_args", "client_args"):
                merged[k] = v
    return merged


def translate_args(cfg: dict[str, Any]) -> tuple[dict[str, str], dict[str, str]]:
    """Registry args -> env vars, failing loudly on anything unwired.

    A silently dropped setting is the worst failure mode available here: the run would
    complete, be recorded under the method's name, and actually be the baseline.
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
        raise SystemExit(
            f"ABORT: registry keys not wired to any launcher setting: {unknown}\n"
            "Refusing to run a config whose settings would be silently dropped — the "
            "results would be mislabelled. Wire them in SERVER_ARG_ENV / CLIENT_ARG_ENV "
            "and teach the launcher to honour them."
        )
    return server_env, client_env


class CosmosServer:
    """Start the policy server, wait for health, and guarantee teardown.

    Uses a new session so the whole process group can be signalled: the server spawns
    children and an orphan keeps ~31 GiB of VRAM pinned on a shared box. This mirrors
    what `smoke_test.sh` does, which is proven in practice.
    """

    def __init__(self, env: dict[str, str], host: str, port: int, log_path: Path, timeout_s: int):
        self.env = env
        self.host = host
        self.port = port
        self.log_path = log_path
        self.timeout_s = timeout_s
        self.proc: subprocess.Popen | None = None

    def _wait_for_free_port(self, timeout_s: int = 120) -> None:
        """Don't start until the port is actually free.

        Consecutive steps in an unattended chain each start a server, and a previous
        one can still be releasing the port (or a stale server can still own it). The
        new server would then fail to bind, minutes into a step, for a reason that
        looks nothing like the real cause. So check first and say so plainly.
        """
        import socket

        deadline = time.monotonic() + timeout_s
        while time.monotonic() < deadline:
            with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
                s.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
                try:
                    s.bind((self.host, self.port))
                    return
                except OSError:
                    pass
            if self._healthy():
                raise RuntimeError(
                    f"another Cosmos server is already serving {self.host}:{self.port}. "
                    "Refusing to run against a server this script did not configure — "
                    "its denoising steps and decode-video settings are unknown, and a "
                    "mismatch would silently mislabel every result. Stop it first."
                )
            print(f"  waiting for {self.host}:{self.port} to free up...", flush=True)
            time.sleep(3.0)
        raise RuntimeError(
            f"{self.host}:{self.port} still in use after {timeout_s}s; not starting a server."
        )

    def __enter__(self) -> "CosmosServer":
        self._wait_for_free_port()
        self.log_path.parent.mkdir(parents=True, exist_ok=True)
        self._log = self.log_path.open("w")
        self.proc = subprocess.Popen(
            ["bash", "scripts/start_cosmos_server.sh"],
            cwd=REPO_ROOT,
            env=self.env,
            stdout=self._log,
            stderr=subprocess.STDOUT,
            start_new_session=True,  # own process group, so killpg reaches every child
        )
        deadline = time.monotonic() + self.timeout_s
        while time.monotonic() < deadline:
            if self.proc.poll() is not None:
                tail = "\n".join(self.log_path.read_text(errors="replace").splitlines()[-30:])
                raise RuntimeError(
                    f"server exited with {self.proc.returncode} during startup.\n{tail}"
                )
            if self._healthy():
                print(f"  server healthy on {self.host}:{self.port}", flush=True)
                return self
            time.sleep(1.0)
        raise RuntimeError(f"server not healthy within {self.timeout_s}s; see {self.log_path}")

    def _healthy(self) -> bool:
        try:
            with urllib.request.urlopen(
                f"http://{self.host}:{self.port}/healthz", timeout=5
            ) as resp:
                return resp.status == 200
        except (urllib.error.URLError, OSError, ValueError):
            return False

    def __exit__(self, *exc: object) -> bool:
        if self.proc is not None and self.proc.poll() is None:
            pgid = os.getpgid(self.proc.pid)
            os.killpg(pgid, signal.SIGTERM)
            for _ in range(30):
                if self.proc.poll() is not None:
                    break
                time.sleep(1.0)
            else:
                print("  server ignored SIGTERM; sending SIGKILL", flush=True)
                os.killpg(pgid, signal.SIGKILL)
                self.proc.wait(timeout=30)
        self._log.close()
        return False


def contention_snapshot() -> dict[str, Any]:
    def sh(cmd: str) -> str:
        try:
            return subprocess.run(
                cmd, shell=True, capture_output=True, text=True, timeout=15
            ).stdout.strip()
        except Exception:  # noqa: BLE001
            return ""

    try:
        load = list(os.getloadavg())
    except OSError:
        load = None
    return {
        "loadavg_1_5_15": load,
        "gpu_compute_apps": sh(
            "nvidia-smi --query-compute-apps=pid,process_name,used_memory --format=csv,noheader"
        ),
    }


OFFICIAL_NUM_ENVS = 10
"""`policies/cosmos3/README.md`'s own example, and 1200 episodes / 120 tasks."""


def deviations_from_official(env: dict[str, str], args: argparse.Namespace) -> list[str]:
    """Everything about this run that differs from upstream's published recipe.

    Recorded per unit, because a deviation nobody wrote down is a deviation nobody can
    correct for later. Each entry states its materiality: the 2026-08-04 audit found that
    the deviation we had been flagging loudest (guardrails) changes no measured quantity,
    while one we had not flagged at all (`--num-envs 1` against an official 10) changed
    the episode count per task by 10×.
    """
    out: list[str] = []
    if env.get("COSMOS_GUARDRAILS") != "true":
        out.append(
            "guardrails disabled (Cosmos-Guardrail1 weights not fetched; gated=auto). "
            "IMMATERIAL: the runners are never invoked on the RoboLab policy-server path "
            "— generate_samples_from_batch bypasses OmniInference.generate_batch, and "
            "decode_video is False. Costs startup download + VRAM only."
        )
    backend = env.get("COSMOS_PG_BACKEND", "gloo")
    if backend != "nccl":
        out.append(
            f"one-rank process group pre-created with {backend!r} instead of upstream's "
            "NCCL (which core-dumps at world_size 1 on this stack). Upstream code runs "
            "unmodified; actions verified bitwise identical "
            "(results/processed/pg_backend_ab.jsonl)."
        )
    envs = int(env.get("ROBOLAB_ENVS", "1"))
    if envs != OFFICIAL_NUM_ENVS:
        out.append(
            f"--num-envs {envs} vs the official {OFFICIAL_NUM_ENVS}. MATERIAL: episodes "
            "per task = num_runs * num_envs, so this changes the statistical power of "
            "every per-task success rate."
        )
    if not args.video:
        out.append(
            "--video-mode none. Affects wall-clock and disk, not success; /scratch has no "
            "room for per-episode video at this episode count."
        )
    return out


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--method", required=True)
    ap.add_argument("--set", dest="task_set", default="smoke")
    ap.add_argument("--episodes", type=int, default=1, help="episodes per task (--num-runs)")
    ap.add_argument("--ledger", default=None, help="default: results/ledger/closed_loop/<method>")
    ap.add_argument("--host", default="127.0.0.1")
    ap.add_argument("--port", type=int, default=8000)
    ap.add_argument("--server-timeout-s", type=int, default=900)
    ap.add_argument("--seed", type=int, default=0, help="COSMOS_SEED, fixed for pairing")
    ap.add_argument(
        "--video",
        action="store_true",
        help="keep video output (default off: mp4 encode and viewport render cost real "
        "time and are not needed for success/compute numbers)",
    )
    ap.add_argument("--retry-failed", action="store_true")
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()

    if not os.environ.get("OMNI_KIT_ACCEPT_EULA"):
        raise SystemExit(
            "OMNI_KIT_ACCEPT_EULA is not set — Isaac Sim will refuse to launch. Export it "
            "yourself if you accept the EULA; this script never sets it for you."
        )

    cfg = resolve_method(args.method)
    server_env, client_env = translate_args(cfg)
    tasks = (load_yaml(TASK_SETS).get(args.task_set) or {}).get("tasks") or []
    if not tasks:
        raise SystemExit(f"task set {args.task_set!r} is empty")

    horizon = int((cfg.get("client_args") or {}).get("open_loop_horizon", 32))
    ledger_root = Path(
        args.ledger or REPO_ROOT / "results" / "ledger" / "closed_loop" / args.method
    )
    led = Ledger(ledger_root, retry_failed=args.retry_failed)

    units = [
        WorkUnit(
            phase="p3_closed_loop",
            kind="task",
            method=args.method,
            task=t,
            # Seed is part of the identity. Closed-loop runs of an identical config are
            # not reproducible here, so re-running with a different seed is a legitimate
            # *additional* sample rather than a repeat — and without the seed in the key
            # the ledger would skip it as already done, making n=2 impossible to collect.
            # Seed 0 keeps the original key so existing results are not orphaned.
            variant=(f"ep{args.episodes}" if args.seed == 0 else f"ep{args.episodes}_seed{args.seed}"),
        )
        for t in tasks
    ]
    counts = led.summary(units)
    print(f"method={args.method} set={args.task_set} tasks={len(tasks)} episodes/task={args.episodes}")
    print(f"  server_args={server_env or '(defaults)'}  client_args={client_env or '(defaults)'}")
    print(f"  state: {counts}")

    todo = [(u, t) for u, t in zip(units, tasks) if led.should_run(u)]
    if args.dry_run:
        for _, t in todo:
            print(f"  TODO {t}")
        return 0
    if not todo:
        print("nothing to do — every task already has a result")
        return 0

    base_env = {
        **os.environ,
        **server_env,
        "COSMOS_SEED": str(args.seed),
        "COSMOS_GUARDRAILS": os.environ.get("COSMOS_GUARDRAILS", "false"),
        # `--port` MUST reach the server, not just the health check and the client.
        # `start_cosmos_server.sh` reads COSMOS_PORT and defaults it to 8000, so without
        # this the server bound 8000 while everything else talked to --port. On a free
        # machine that merely works by accident; running a second method concurrently it
        # collided with the first server and the run died at startup. The port-free check
        # did not catch it because it checked --port, which really was free.
        "COSMOS_HOST": args.host,
        "COSMOS_PORT": str(args.port),
    }
    run_stamp = time.strftime("%Y%m%dT%H%M%SZ", time.gmtime())
    server_log = REPO_ROOT / "results" / "raw" / f"closed-loop-{args.method}-{run_stamp}" / "server.log"

    extra = [f"--num-runs {args.episodes}"]
    if not args.video:
        extra.append("--video-mode none")

    with CosmosServer(base_env, args.host, args.port, server_log, args.server_timeout_s):
        for i, (unit, task) in enumerate(todo, 1):
            print(f"[{i}/{len(todo)}] {task}", flush=True)
            with led.claim(unit) as slot:
                client_log = slot.dir / "client.log"
                env = {
                    **base_env,
                    **client_env,
                    "ROBOLAB_TASK": task,
                    "ROBOLAB_HEADLESS": "1",
                    # Registry wins, then an explicitly exported value, then 1.
                    # Hardcoding "1" here silently overrode an ambient ROBOLAB_ENVS, so a
                    # deliberate `ROBOLAB_ENVS=10` ran a single env and reported "1/1
                    # success" — the same silent-override class as the COSMOS_PORT bug.
                    # num_envs matters: RoboLab's runner says "Total episodes = num_runs *
                    # num_envs. Prefer increasing --num-envs for more episodes", and the
                    # official recipe uses --num-envs 10, which is how the leaderboard's
                    # 10 episodes/task is obtained.
                    "ROBOLAB_ENVS": client_env.get(
                        "ROBOLAB_ENVS", os.environ.get("ROBOLAB_ENVS", "1")
                    ),
                    "ROBOLAB_EXTRA_ARGS": " ".join(extra),
                    "COSMOS_HOST": args.host,
                    "COSMOS_PORT": str(args.port),
                }
                slot.meta = {
                    "method": args.method,
                    "config": cfg,
                    "server_env": server_env,
                    "client_env": client_env,
                    "contention": contention_snapshot(),
                    "guardrails": base_env["COSMOS_GUARDRAILS"] == "true",
                    "pg_backend": env.get("COSMOS_PG_BACKEND", "gloo"),
                    "deviations_from_official": deviations_from_official(env, args),
                }
                t0 = time.perf_counter()
                with client_log.open("w") as log:
                    proc = subprocess.run(
                        ["bash", "scripts/run_robolab.sh"],
                        cwd=REPO_ROOT,
                        env=env,
                        stdout=log,
                        stderr=subprocess.STDOUT,
                        check=False,
                    )
                elapsed = time.perf_counter() - t0
                text = client_log.read_text(errors="replace")
                if proc.returncode != 0:
                    tail = "\n".join(text.splitlines()[-20:])
                    raise RuntimeError(f"client exited {proc.returncode} on {task}\n{tail}")

                out_dir = find_output_dir(text)
                if out_dir is None:
                    raise RuntimeError(
                        f"no `Output :` line in {client_log} — cannot locate results, so "
                        "this task's outcome is unknown. Refusing to record a result."
                    )
                episodes = read_episode_results(out_dir)
                if not episodes:
                    raise RuntimeError(
                        f"{out_dir} has no episode records; the episode did not complete."
                    )
                summary = episode_summary(episodes, horizon)
                summary["wall_elapsed_s"] = round(elapsed, 1)
                summary["output_dir"] = str(out_dir)
                summary["episodes"] = episodes
                slot.result = summary
            print(
                f"    {summary['n_success']}/{summary['n_episodes']} success, "
                f"{summary['total_steps']} steps, {summary['policy_calls']} policy calls, "
                f"{elapsed / 60:.1f} min",
                flush=True,
            )

    final = led.summary(units)
    print(f"done: {final}")
    (ledger_root / "method.json").write_text(
        json.dumps({"method": args.method, "config": cfg, "horizon": horizon}, indent=2)
    )
    return 0 if final["failed"] == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
