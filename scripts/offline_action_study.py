#!/usr/bin/env python3
"""Offline action study — Experiments A and E without the simulator.

Runs inside `third_party/cosmos-framework/.venv` (needs torch + cosmos_framework).

Why offline first: a closed-loop pilot pass costs ~3.7 GPU-h, while one in-process
request costs ~3.5 s. Screening candidate configurations here and spending episodes
only on survivors is the difference between a plan that fits the budget and one that
does not. Spec §11.1 asks for exactly this ordering, and §14 prefers rapid
falsification.

What it produces, per (condition, corpus request): the emitted action chunk plus
timing and token counts. Deviations against the 4-step teacher are computed by
`scripts/analyze_offline_study.py`, so generation (expensive, GPU) is separated from
analysis (cheap, rerunnable) — otherwise a change to a metric definition would mean
re-running the GPU sweep.

Resumable by construction: every unit is claimed through `action_refresh.ledger`, so
a kill at any point loses at most the in-flight request. Re-invoke the same command
to continue.

Two model settings make this cheap, both verified against the pinned source:
  - `num_steps` is a per-call argument of `generate_samples_from_batch`, so the
    denoising sweep needs no reload.
  - `RobolabServerArgs` is a non-frozen pydantic model and `_next_seed()` returns
    `cfg.seed` verbatim when `deterministic_seed` is set, so seeds are controllable
    and runs are bit-reproducible.
One 31 GiB model load therefore serves every condition.
"""
from __future__ import annotations

import argparse
import dataclasses
import json
import os
import platform
import subprocess
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np
import torch

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

from action_refresh.ledger import Ledger, WorkUnit  # noqa: E402

# Reference condition. Everything is measured against this, so it is named once.
TEACHER = "teacher_steps4"
BASE_SEED = 0


def assert_cache_env() -> str:
    """Refuse to run unless HF_HOME points at the configured cache.

    Defence at the point of use, not just in the caller. This project has now hit the
    same bug three times: without HF_HOME, Hugging Face resolves
    `$HOME/.cache/huggingface` — a 60 GB NFS volume here — and pages tens of GB over
    the network, turning a 34 s model load into ~25 minutes and quietly filling a
    quota-limited home directory. Every caller *should* export it; this makes a caller
    that forgets fail immediately and loudly instead of running slowly against the
    wrong files.

    Not a fallback: we do not silently set it. A wrong cache path means the thing being
    measured is not the thing that was installed.
    """
    hf_home = os.environ.get("HF_HOME")
    machine = REPO_ROOT / "configs" / "machine.yaml"
    expected = None
    if machine.is_file():
        try:
            import yaml  # noqa: PLC0415

            expected = ((yaml.safe_load(machine.read_text()) or {}).get("paths") or {}).get(
                "hf_cache"
            )
        except Exception:  # noqa: BLE001
            expected = None
    if not hf_home:
        raise SystemExit(
            "HF_HOME is not set. Hugging Face would resolve $HOME/.cache/huggingface, "
            "which on this host is a 60 GB NFS volume — the model would be paged over "
            "the network and the timings would be meaningless.\n"
            + (f"→ export HF_HOME={expected}\n" if expected else "")
            + "  (or run this through scripts/run_phases.py, which sets it from "
            "configs/machine.yaml)"
        )
    if expected and Path(hf_home).resolve() != Path(expected).resolve():
        print(
            f"warning: HF_HOME={hf_home} differs from configs/machine.yaml "
            f"paths.hf_cache={expected}. Continuing because it was set deliberately, "
            "but the cache being read is not the configured one.",
            flush=True,
        )
    return hf_home


def load_corpus(paths: list[Path]) -> list[dict[str, Any]]:
    """Load captured requests into replayable observations.

    Accepts the single-request capture from `smoke_test.sh` and the per-step corpus
    from `robolab-0002`. The prompt lives in the sidecar `.json` because npz cannot
    hold a str.
    """
    corpus: list[dict[str, Any]] = []
    for p in sorted(paths):
        meta_path = p.with_suffix(".json")
        if not meta_path.exists():
            raise SystemExit(f"error: {p} has no sidecar {meta_path.name} (holds the prompt)")
        meta = json.loads(meta_path.read_text())
        prompt = (meta.get("strings") or {}).get("prompt")
        if not prompt:
            raise SystemExit(f"error: {meta_path} has no strings.prompt")
        npz = np.load(p)
        obs = {k: npz[k] for k in npz.files}
        obs["prompt"] = prompt
        missing = {
            "observation/image",
            "observation/joint_position",
            "observation/gripper_position",
        } - set(obs)
        if missing:
            raise SystemExit(f"error: {p} missing {sorted(missing)}")
        corpus.append(
            {
                # Stable id: the ledger keys on this, so it must not depend on
                # filesystem order or absolute path.
                "id": p.stem.replace("captured_request_", ""),
                "path": str(p),
                "obs": obs,
                "control_step": meta.get("control_step"),
                "task": meta.get("task"),
            }
        )
    if not corpus:
        raise SystemExit(
            "error: empty corpus. Run `make smoke` (or the capture phase) to record "
            "real requests into results/raw/captured_request_*.npz first."
        )
    return corpus


def build_conditions(
    steps: list[int],
    n_seeds: int,
    include_action_only: bool,
    vision_frame_sweep: list[int] | None = None,
) -> dict[str, dict[str, Any]]:
    """The conditions to measure, each a set of per-call overrides.

    - `TEACHER` and `steps_N` are Experiment A: identical input and seed, only the
      denoising budget varies, so the deviation is attributable to steps alone.
    - `seed_K` is Experiment E0: identical input, *different* diffusion seed. Because
      vision and action are denoised jointly, changing the seed changes the imagined
      future. If the action barely moves while the imagination does, the imagination
      is not carrying action-relevant information.
    - `action_only*` is E2, and only appears once the attention patch is present.
    """
    vision_frame_sweep = vision_frame_sweep or []
    conds: dict[str, dict[str, Any]] = {
        TEACHER: {"num_steps": 4, "seed": BASE_SEED, "experiment": "A/reference"},
    }
    for n in sorted(steps):
        if n == 4:
            continue  # that is the teacher; a duplicate would just cost GPU time
        conds[f"steps_{n}"] = {"num_steps": n, "seed": BASE_SEED, "experiment": "A"}
    # Seed 0 is the teacher, so alternates start at 1.
    for k in range(1, n_seeds):
        conds[f"seed_{k}"] = {"num_steps": 4, "seed": BASE_SEED + k, "experiment": "E0"}
    if include_action_only:
        # Compute is unchanged in `freeze` (same token count), so this condition
        # answers accuracy only. Pair it with steps_1 to see whether ablating the
        # imagination costs more or less accuracy than simply denoising less.
        conds["no_imagination_freeze"] = {
            "num_steps": 4,
            "seed": BASE_SEED,
            "action_only": "freeze",
            "experiment": "E1/E2a",
        }
    # E2b: actually shorten the imagined horizon so the sequence shrinks. This is the
    # only variant whose latency means anything — `freeze` keeps every token. Frame
    # counts are the ones that keep the action stream aligned (32 % (N-1) == 0), giving
    # 9 -> 1 latent frames and 3,060 -> 340 vision tokens.
    for frames in vision_frame_sweep:
        conds[f"vision_frames_{frames}"] = {
            "num_steps": 4,
            "seed": BASE_SEED,
            "vision_frames": frames,
            "experiment": "E2b",
        }
    return conds


def gpu_snapshot() -> dict[str, Any]:
    """Contention snapshot. Recorded per unit, not per run: an 18-GPU-h chain will
    span quiet and busy periods, and a single run-level snapshot would misattribute
    them."""

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
            "nvidia-smi --query-compute-apps=pid,process_name,used_memory "
            "--format=csv,noheader"
        ),
    }


class Runner:
    """Owns the loaded model and executes one condition x request at a time."""

    def __init__(self, checkpoint: str, revision: str, guardrails: bool) -> None:
        # Create the one-rank process group ourselves, with gloo, so upstream's
        # maybe_init_distributed() finds one already there and its collectives run
        # unmodified. Without this the NCCL group upstream builds core-dumps on this
        # stack. See src/action_refresh/server/process_group.py.
        from action_refresh.server.process_group import ensure_single_rank_group  # noqa: PLC0415

        print(f"[pg] {ensure_single_rank_group(os.environ.get('COSMOS_PG_BACKEND', 'gloo'))}", flush=True)

        from cosmos_framework.scripts.action_policy_server_robolab import (  # noqa: PLC0415
            RobolabPolicyService,
            RobolabServerArgs,
        )

        args = RobolabServerArgs(
            checkpoint_path=checkpoint,
            hf_revision=revision,
            num_steps=4,
            decode_video=False,
            guardrails=guardrails,
            # Determinism is the whole point: identical input must give identical
            # output, otherwise a deviation cannot be attributed to the condition.
            deterministic_seed=True,
            seed=BASE_SEED,
        )
        t0 = time.perf_counter()
        self.service = RobolabPolicyService(args)
        self.load_s = time.perf_counter() - t0
        self.action_dim = int(self.service.cfg.action_dim)
        print(f"model loaded in {self.load_s:.1f}s (action_dim={self.action_dim})", flush=True)

    def run_one(self, obs: dict[str, Any], overrides: dict[str, Any]) -> dict[str, Any]:
        """Execute one request under one condition and return actions + timings."""
        # `service.cfg` is a FROZEN dataclass (RobolabPolicyConfig), not the pydantic
        # RobolabServerArgs — attribute assignment on it raises. Swap in a replaced
        # copy instead of forcing a write through object.__setattr__: a frozen type is
        # frozen for a reason, and mutating a shared instance in place would leak the
        # override into anything else holding a reference.
        #
        # Note the seed still has to be set here rather than passed per call: infer()
        # reads cfg.seed via _next_seed(), and with deterministic_seed=True that makes
        # the request bit-reproducible, which is what lets a deviation be attributed
        # to the condition.
        prev_cfg = self.service.cfg
        replace_kw: dict[str, Any] = {
            "num_steps": int(overrides["num_steps"]),
            "seed": int(overrides["seed"]),
        }
        if overrides.get("vision_frames"):
            # E2b: shorten the imagined horizon. Validated by the server's own check at
            # construction, but that ran for the *default*, so re-check here — a
            # misaligned plan yields plausible-looking actions from the wrong layout.
            vf = int(overrides["vision_frames"])
            n_actions = prev_cfg.action_chunk_size + (1 if prev_cfg.use_state else 0)
            if vf < 2 or (n_actions - 1) % (vf - 1) != 0:
                raise ValueError(
                    f"vision_frames={vf} misaligns the action stream for "
                    f"{n_actions} action steps"
                )
            replace_kw["vision_frames"] = vf
        self.service.cfg = dataclasses.replace(prev_cfg, **replace_kw)

        mode = overrides.get("action_only")
        ctx = _action_only_context(self.service, mode) if mode else _null_context()
        intervention: dict[str, Any] | None = None
        try:
            torch.cuda.synchronize()
            torch.cuda.reset_peak_memory_stats()
            start_ev = torch.cuda.Event(enable_timing=True)
            end_ev = torch.cuda.Event(enable_timing=True)
            t0 = time.perf_counter_ns()
            start_ev.record()
            with ctx as stats:
                out = self.service.infer(obs)
                # Captured inside the block: the context manager validates on exit
                # that the intervention actually fired, and we want its evidence in
                # the record either way.
                intervention = dict(stats) if isinstance(stats, dict) else None
            end_ev.record()
            torch.cuda.synchronize()
            wall_ms = (time.perf_counter_ns() - t0) / 1e6
            cuda_ms = start_ev.elapsed_time(end_ev)
        finally:
            self.service.cfg = prev_cfg

        action = np.asarray(out["action"], dtype=np.float32)
        return {
            "action": action,
            "wall_ms": wall_ms,
            "cuda_ms": cuda_ms,
            "action_shape": list(action.shape),
            "action_finite": bool(np.isfinite(action).all()),
            "peak_reserved_mib": torch.cuda.max_memory_reserved() / 2**20,
            "num_steps": cfg_value(overrides, "num_steps"),
            "seed": cfg_value(overrides, "seed"),
            "vision_frames": cfg_value(overrides, "vision_frames"),
            # Evidence the ablation ran (hook call count, vision latent shape acted
            # on). None for unmodified conditions.
            "intervention": intervention,
        }


def cfg_value(overrides: dict[str, Any], key: str) -> Any:
    return overrides.get(key)


class _null_context:
    def __enter__(self):
        return None

    def __exit__(self, *exc):
        return False


def _action_only_context(service: Any, mode: str):
    """Enable the action-only path for the duration of one request.

    Implemented in `action_refresh.server.action_only`; imported lazily so the A/E0
    conditions, which need no patch, still run when the patch is absent.
    """
    from action_refresh.server.action_only import action_only  # noqa: PLC0415

    return action_only(service.model, mode=mode)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--ledger", default=str(REPO_ROOT / "results" / "ledger" / "offline_study"))
    ap.add_argument(
        "--corpus-glob",
        default=str(REPO_ROOT / "results" / "raw" / "captured_request_*.npz"),
        help="captured real requests to replay",
    )
    ap.add_argument("--steps", default="1,2,3,4", help="denoising steps to sweep (Experiment A)")
    ap.add_argument(
        "--seeds",
        type=int,
        default=4,
        help="how many diffusion seeds per request for E0 (1 = teacher only, no E0)",
    )
    ap.add_argument(
        "--action-only",
        action="store_true",
        help="include the E1/E2a no-imagination condition (vision velocity frozen; token "
        "count and therefore compute unchanged, so it answers accuracy only)",
    )
    ap.add_argument(
        "--vision-frames",
        default="",
        help="E2b: comma-separated imagined-horizon lengths to sweep, e.g. '17,9,5,3'. "
        "Must keep the action stream aligned (32 %% (N-1) == 0 for a 32-action chunk), "
        "so valid values are 33,17,9,5,3 -> 9,5,3,2,1 latent frames. This is the only "
        "variant that actually shrinks the sequence, so it is the only one whose latency "
        "is a speedup.",
    )
    ap.add_argument("--checkpoint-path", default="nvidia/Cosmos3-Nano-Policy-DROID")
    ap.add_argument("--hf-revision", default="6706d7680581c255ff61e0f3bb49d90eac55c79e")
    ap.add_argument(
        "--guardrails",
        action="store_true",
        help="enable guardrail runners (requires access to the gated Cosmos-Guardrail1)",
    )
    ap.add_argument("--retry-failed", action="store_true")
    ap.add_argument("--dry-run", action="store_true", help="list the work and exit")
    args = ap.parse_args()

    hf_home = assert_cache_env()
    corpus = load_corpus(_glob(args.corpus_glob))
    steps = [int(s) for s in args.steps.split(",") if s.strip()]
    frames = [int(s) for s in args.vision_frames.split(",") if s.strip()]
    conditions = build_conditions(steps, args.seeds, args.action_only, frames)

    led = Ledger(args.ledger, retry_failed=args.retry_failed)
    units: list[tuple[WorkUnit, str, dict[str, Any]]] = []
    for cname, overrides in conditions.items():
        for item in corpus:
            unit = WorkUnit(
                phase="p1_p2_offline",
                kind="action_chunk",
                method=cname,
                task=item["id"],
                variant=f"steps{overrides['num_steps']}_seed{overrides['seed']}"
                + (f"_{overrides['action_only']}" if overrides.get("action_only") else ""),
            )
            units.append((unit, cname, overrides))

    counts = led.summary([u for u, _, _ in units])
    print(
        f"corpus={len(corpus)} conditions={len(conditions)} units={len(units)} "
        f"-> done={counts['done']} failed={counts['failed']} "
        f"interrupted={counts['interrupted']} pending={counts['pending']}",
        flush=True,
    )
    todo = [(u, c, o) for (u, c, o) in units if led.should_run(u)]
    if args.dry_run:
        for u, c, _ in todo:
            print(f"  TODO {c:20s} {u.task}")
        return 0
    if not todo:
        print("nothing to do — every unit already has a result", flush=True)
        return 0

    # Load the model only when there is real work: a resume that finds everything
    # done should not pay 25 s and 31 GiB to discover that.
    runner = Runner(args.checkpoint_path, args.hf_revision, args.guardrails)
    by_id = {item["id"]: item for item in corpus}

    env = {
        "python": platform.python_version(),
        "torch": torch.__version__,
        "gpu_name": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
        "model_load_s": runner.load_s,
        # Recorded because a load time far above ~35 s is the signature of reading the
        # checkpoint over NFS, which makes every timing in the run suspect.
        "hf_home": hf_home,
        "guardrails": bool(args.guardrails),
        "deviations_from_official": (
            []
            if args.guardrails
            else ["guardrails disabled (nvidia/Cosmos-Guardrail1 gated); excludes guardrail cost"]
        ),
    }

    for i, (unit, cname, overrides) in enumerate(todo, 1):
        item = by_id[unit.task]
        print(f"[{i}/{len(todo)}] {cname} on {unit.task}", flush=True)
        with led.claim(unit) as slot:
            slot.meta = {"env": env, "condition": cname, "overrides": overrides,
                         "contention": gpu_snapshot()}
            res = runner.run_one(item["obs"], overrides)
            action = res.pop("action")
            # The array lives beside result.json rather than inside it: JSON floats
            # would lose precision and bloat the ledger.
            action_path = slot.dir / "action.npy"
            np.save(action_path, action)
            # Repo-relative when the ledger lives inside the repo (keeps results
            # portable), absolute otherwise — a ledger placed on scratch or /tmp must
            # not make the record unreadable. `REPO_ROOT / <abs>` resolves to the
            # absolute path, so the analyzer handles both without branching.
            try:
                res["action_path"] = str(action_path.relative_to(REPO_ROOT))
            except ValueError:
                res["action_path"] = str(action_path)
            res["corpus_path"] = item["path"]
            res["experiment"] = overrides.get("experiment")
            slot.result = res
        print(
            f"    wall {res['wall_ms']:.0f} ms  cuda {res['cuda_ms']:.0f} ms  "
            f"action {res['action_shape']}  finite={res['action_finite']}",
            flush=True,
        )

    final = led.summary([u for u, _, _ in units])
    print(f"done: {final}", flush=True)
    return 0 if final["failed"] == 0 else 1


def _glob(pattern: str) -> list[Path]:
    from glob import glob  # noqa: PLC0415

    return [Path(p) for p in sorted(glob(pattern))]


if __name__ == "__main__":
    raise SystemExit(main())
