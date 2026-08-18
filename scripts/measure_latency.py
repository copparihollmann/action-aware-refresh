#!/usr/bin/env python3
"""Trustworthy latency per configuration: interleaved, cooled, median-of-many.

Runs in the cosmos venv. This exists because the offline action study's `wall ms` column
is **not** a latency measurement and must not be quoted as one: it sweeps conditions in
back-to-back blocks, and B3 showed that is exactly how not to time anything on this host —
identical work spans 3,377–5,974 ms under sustained load, so whichever condition happens to
run during a busy stretch looks slower.

Three properties make the numbers here defensible, each answering a specific way the naive
measurement goes wrong:

**Interleaved.** Conditions are measured round-robin, one repeat each per round, rather
than one condition to completion then the next. Any drift in host load then hits every
condition roughly equally instead of landing on whichever ran last. This is the single most
important difference from the sweep.

**Cooled.** An idle gap before each timed request. Back-to-back inference on this box
produces an intermittent latency tail with no change in the work done, and the gap removes
it (measured: MAD 0.34% with a gap, 3,377–5,974 ms without).

**Median-of-many, with MAD.** The mean of a run with a tail describes the tail. Medians and
MAD describe the model. Both are reported, along with the min, because the minimum is the
closest thing to an uncontended measurement this host offers.

Every repeat is appended to JSONL as it completes, so a kill loses at most one request and
`--out-jsonl` can be re-analysed without re-measuring.

    scripts/measure_latency.py --repeats 12 --cooldown-s 6
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

TEACHER = "teacher_steps4"


def conditions() -> dict[str, dict[str, Any]]:
    """The configurations whose cost the Pareto frontier needs.

    Includes the *combination* of the two independent levers: reducing denoising steps cuts
    work per token, shortening the imagined horizon cuts tokens. They act on different
    factors of the same matmul-bound cost, so the combined point is the obvious frontier
    candidate and measuring it is not optional — assuming the speedups multiply would be
    exactly the kind of unmeasured inference spec §8 forbids.
    """
    conds: dict[str, dict[str, Any]] = {
        TEACHER: {"num_steps": 4},
        "steps_3": {"num_steps": 3},
        "steps_2": {"num_steps": 2},
        "steps_1": {"num_steps": 1},
        "vision_frames_17": {"num_steps": 4, "vision_frames": 17},
        "vision_frames_9": {"num_steps": 4, "vision_frames": 9},
        "vision_frames_5": {"num_steps": 4, "vision_frames": 5},
        "steps_1_vision_frames_9": {"num_steps": 1, "vision_frames": 9},
        "steps_1_vision_frames_5": {"num_steps": 1, "vision_frames": 5},
    }
    return conds


def gpu_state() -> dict[str, Any]:
    def sh(cmd: str) -> str:
        try:
            return subprocess.run(
                cmd, shell=True, capture_output=True, text=True, timeout=10
            ).stdout.strip()
        except Exception:  # noqa: BLE001
            return ""

    try:
        load = list(os.getloadavg())
    except OSError:
        load = None
    # Exclude our own PID. `nvidia-smi --query-compute-apps` lists the *measuring* process
    # too, so an unfiltered check flagged all 90 samples as contended and would have
    # discredited perfectly good data with a warning about itself.
    raw = sh(
        "nvidia-smi --query-compute-apps=pid,process_name,used_memory --format=csv,noheader"
    )
    mine = str(os.getpid())
    others = "\n".join(
        line
        for line in raw.splitlines()
        if line.strip() and line.split(",")[0].strip() != mine
    )
    return {
        "loadavg_1_5_15": load,
        "gpu_compute_apps": others,
        "gpu_compute_apps_including_self": raw,
        "sm_clock_temp_power": sh(
            "nvidia-smi --query-gpu=clocks.sm,temperature.gpu,power.draw "
            "--format=csv,noheader --id=0"
        ),
    }


def robust(xs: list[float]) -> dict[str, float]:
    a = np.asarray(xs, dtype=float)
    med = float(np.median(a))
    mad = float(np.median(np.abs(a - med)))
    return {
        "n": int(a.size),
        "median": med,
        "mad": mad,
        "mad_pct": (100.0 * mad / med) if med else 0.0,
        "mean": float(a.mean()),
        "std": float(a.std(ddof=1)) if a.size > 1 else 0.0,
        "min": float(a.min()),
        "max": float(a.max()),
    }


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--repeats", type=int, default=12, help="timed repeats per condition")
    ap.add_argument("--warmup", type=int, default=2)
    ap.add_argument("--cooldown-s", type=float, default=6.0)
    ap.add_argument(
        "--request",
        default=None,
        help="captured request .npz to replay (default: first in the corpus)",
    )
    ap.add_argument("--out-jsonl", default=str(REPO_ROOT / "results" / "processed" / "latency.jsonl"))
    ap.add_argument("--out-json", default=str(REPO_ROOT / "results" / "processed" / "latency.json"))
    ap.add_argument("--out-md", default=str(REPO_ROOT / "docs" / "latency.md"))
    ap.add_argument("--checkpoint-path", default="nvidia/Cosmos3-Nano-Policy-DROID")
    ap.add_argument("--hf-revision", default="6706d7680581c255ff61e0f3bb49d90eac55c79e")
    ap.add_argument("--guardrails", action="store_true")
    ap.add_argument(
        "--analyze-only",
        action="store_true",
        help="re-render the report from an existing JSONL without measuring",
    )
    args = ap.parse_args()

    out_jsonl = Path(args.out_jsonl)
    out_jsonl.parent.mkdir(parents=True, exist_ok=True)

    if not args.analyze_only:
        if not os.environ.get("HF_HOME"):
            raise SystemExit(
                "HF_HOME is not set — the model would be paged from the NFS home volume "
                "and every number here would be meaningless. Export it (see "
                "configs/machine.yaml) or run via scripts/run_phases.py."
            )
        # Refuse to measure against a busy GPU. Latency is the whole output; taking it
        # while another process shares the device produces a number that describes the
        # contention, not the configuration.
        apps = gpu_state()["gpu_compute_apps"]
        if apps.strip() and not os.environ.get("LATENCY_ALLOW_BUSY_GPU"):
            raise SystemExit(
                "another process is using the GPU:\n"
                f"  {apps}\n"
                "Latency measured against a busy device describes the contention, not the "
                "configuration. Wait for a quiet window, or set "
                "LATENCY_ALLOW_BUSY_GPU=1 to override and have the contention recorded "
                "alongside every sample."
            )
        measure(args, out_jsonl)

    render(args, out_jsonl)
    return 0


def measure(args: argparse.Namespace, out_jsonl: Path) -> None:
    from action_refresh.ledger import _atomic_write_json  # noqa: F401, PLC0415  (import check)
    from cosmos_framework.scripts.action_policy_server_robolab import (  # noqa: PLC0415
        RobolabPolicyService,
        RobolabServerArgs,
    )

    req = Path(args.request) if args.request else _first_corpus_request()
    obs = _load_request(req)

    # Create the one-rank process group ourselves, with gloo, so upstream's
    # maybe_init_distributed() finds one already there and its collectives run
    # unmodified. Without this the NCCL group upstream builds core-dumps on this stack.
    # See src/action_refresh/server/process_group.py.
    from action_refresh.server.process_group import ensure_single_rank_group  # noqa: PLC0415

    print(f"[pg] {ensure_single_rank_group(os.environ.get('COSMOS_PG_BACKEND', 'gloo'))}", flush=True)

    service = RobolabPolicyService(
        RobolabServerArgs(
            checkpoint_path=args.checkpoint_path,
            hf_revision=args.hf_revision,
            num_steps=4,
            decode_video=False,
            guardrails=args.guardrails,
            deterministic_seed=True,
            seed=0,
        )
    )
    base_cfg = service.cfg
    conds = conditions()
    print(f"request={req.name}  conditions={len(conds)}  repeats={args.repeats}", flush=True)

    with out_jsonl.open("a") as fh:
        # Warmup is per-condition and untimed: the first call in any new shape pays
        # allocator and kernel-selection costs that a steady-state number should exclude.
        for name, ov in conds.items():
            service.cfg = dataclasses.replace(base_cfg, **_replace_kw(base_cfg, ov))
            for _ in range(args.warmup):
                service.infer(obs)
            service.cfg = base_cfg
        # Interleaved: round-robin, so drift hits every condition equally.
        for rep in range(args.repeats):
            for name, ov in conds.items():
                if args.cooldown_s > 0:
                    time.sleep(args.cooldown_s)
                service.cfg = dataclasses.replace(base_cfg, **_replace_kw(base_cfg, ov))
                try:
                    rec = _time_one(service, obs, name, ov, rep)
                finally:
                    service.cfg = base_cfg
                fh.write(json.dumps(rec) + "\n")
                fh.flush()  # a kill loses at most this one request
            print(
                f"  round {rep + 1}/{args.repeats} done",
                flush=True,
            )


def _replace_kw(base_cfg: Any, ov: dict[str, Any]) -> dict[str, Any]:
    kw: dict[str, Any] = {"num_steps": int(ov["num_steps"])}
    if ov.get("vision_frames"):
        kw["vision_frames"] = int(ov["vision_frames"])
    return kw


def _time_one(
    service: Any, obs: dict[str, Any], name: str, ov: dict[str, Any], rep: int
) -> dict[str, Any]:
    torch.cuda.synchronize()
    torch.cuda.reset_peak_memory_stats()
    ev0 = torch.cuda.Event(enable_timing=True)
    ev1 = torch.cuda.Event(enable_timing=True)
    ev0.record()
    t0 = time.perf_counter_ns()
    out = service.infer(obs)
    wall_ms = (time.perf_counter_ns() - t0) / 1e6
    ev1.record()
    torch.cuda.synchronize()
    cuda_ms = ev0.elapsed_time(ev1)
    action = np.asarray(out["action"], dtype=np.float32)
    return {
        "condition": name,
        "repeat": rep,
        "overrides": ov,
        "wall_ms": wall_ms,
        "cuda_ms": cuda_ms,
        "host_overhead_ms": wall_ms - cuda_ms,
        "action_shape": list(action.shape),
        "action_finite": bool(np.isfinite(action).all()),
        "peak_reserved_mib": torch.cuda.max_memory_reserved() / 2**20,
        "utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "contention": gpu_state(),
    }


def _first_corpus_request() -> Path:
    from glob import glob  # noqa: PLC0415

    for pattern in (
        "results/raw/corpus/*/captured_request_*.npz",
        "results/raw/captured_request_*.npz",
    ):
        hits = sorted(glob(str(REPO_ROOT / pattern)))
        if hits:
            return Path(hits[0])
    raise SystemExit("no captured request found — run scripts/capture_corpus.sh first")


def _load_request(p: Path) -> dict[str, Any]:
    meta = json.loads(p.with_suffix(".json").read_text())
    prompt = (meta.get("strings") or {}).get("prompt")
    if not prompt:
        raise SystemExit(f"{p} sidecar has no strings.prompt")
    npz = np.load(p)
    obs: dict[str, Any] = {k: npz[k] for k in npz.files}
    obs["prompt"] = prompt
    return obs


def render(args: argparse.Namespace, out_jsonl: Path) -> None:
    if not out_jsonl.is_file():
        raise SystemExit(f"{out_jsonl} not found — measure first")
    by_cond: dict[str, list[dict[str, Any]]] = {}
    for line in out_jsonl.read_text().splitlines():
        line = line.strip()
        if not line:
            continue
        rec = json.loads(line)
        by_cond.setdefault(rec["condition"], []).append(rec)
    if TEACHER not in by_cond:
        raise SystemExit(f"no `{TEACHER}` samples — nothing to normalize against")

    stats = {
        c: {
            "wall_ms": robust([r["wall_ms"] for r in recs]),
            "cuda_ms": robust([r["cuda_ms"] for r in recs]),
            "host_overhead_ms": robust([r["host_overhead_ms"] for r in recs]),
            "overrides": recs[0]["overrides"],
            "peak_reserved_mib": max(r["peak_reserved_mib"] for r in recs),
            "all_finite": all(r["action_finite"] for r in recs),
        }
        for c, recs in by_cond.items()
    }
    base = stats[TEACHER]["cuda_ms"]["median"]
    for c, s in stats.items():
        s["speedup_vs_teacher"] = base / s["cuda_ms"]["median"] if s["cuda_ms"]["median"] else None

    contention = [r for recs in by_cond.values() for r in recs if (r.get("contention") or {}).get("gpu_compute_apps", "").strip()]
    report = {
        "teacher": TEACHER,
        "n_conditions": len(stats),
        "stats": stats,
        "samples_with_other_gpu_processes": len(contention),
        "env": {
            "python": platform.python_version(),
            "torch": torch.__version__ if "torch" in sys.modules else None,
        },
    }
    Path(args.out_json).write_text(json.dumps(report, indent=2, default=str))
    Path(args.out_md).write_text(_markdown(report))
    print(f"wrote {args.out_json}")
    print(f"wrote {args.out_md}")


def _markdown(r: dict[str, Any]) -> str:
    L: list[str] = []
    stats = r["stats"]
    L.append("# Measured latency per configuration")
    L.append("")
    L.append(
        "Interleaved (round-robin over conditions), cooled (idle gap before each timed "
        "request), median-of-many. This protocol exists because the offline study's "
        "`wall ms` column is not a latency measurement — it sweeps conditions in "
        "back-to-back blocks, and identical work spans 3,377–5,974 ms on this host under "
        "sustained load. **Speedups quoted anywhere in this project come from here.**"
    )
    L.append("")
    L.append(
        "| condition | steps | frames | CUDA median | MAD | wall median | host overhead | "
        "speedup (CUDA) | peak VRAM |"
    )
    L.append("|---|---|---|---|---|---|---|---|---|")

    def order(kv):
        s = kv[1]["cuda_ms"]["median"]
        return s

    for c, s in sorted(stats.items(), key=order, reverse=True):
        ov = s["overrides"]
        cu, wa, ho = s["cuda_ms"], s["wall_ms"], s["host_overhead_ms"]
        sp = s["speedup_vs_teacher"]
        L.append(
            f"| `{c}` | {ov.get('num_steps')} | {ov.get('vision_frames') or 33} | "
            f"**{cu['median']:,.0f} ms** | {cu['mad']:.0f} ({cu['mad_pct']:.2f}%) | "
            f"{wa['median']:,.0f} ms | {ho['median']:.0f} ms | "
            f"{'—' if c == r['teacher'] else f'{sp:.2f}x'} | "
            f"{s['peak_reserved_mib']:,.0f} MiB |"
        )
    L.append("")
    if r["samples_with_other_gpu_processes"]:
        L.append(
            f"⚠ **{r['samples_with_other_gpu_processes']} samples were taken while another "
            "process shared the GPU** (recorded per sample in the JSONL). Treat the "
            "affected medians as upper bounds and re-measure in a quiet window before "
            "quoting them."
        )
    else:
        L.append(
            "No other GPU compute process was present for any sample — the script refuses "
            "to measure otherwise unless explicitly overridden."
        )
    L.append("")
    L.append("## How to read this")
    L.append("")
    L.append(
        "- **CUDA median is the model's cost**; wall median includes host-side work. Their "
        "difference (`host overhead`) is in-process only and does *not* include the "
        "~1,197 ms of client-side composition, msgpack and websocket round trip measured "
        "in the closed loop — that sits on top of every number here and is invariant to "
        "anything done inside the model."
    )
    L.append(
        "- **Speedups are CUDA-time ratios against the 4-step baseline.** They say nothing "
        "about task success: `docs/offline_action_study.md` prices the action deviation and "
        "`docs/pareto.md` the closed-loop success."
    )
    L.append(
        "- **A combined condition is measured, not inferred.** Reducing denoising steps and "
        "shortening the imagined horizon act on different factors of the same matmul-bound "
        "cost, so their speedups might or might not multiply; spec §8 forbids assuming."
    )
    L.append("")
    return "\n".join(L) + "\n"


if __name__ == "__main__":
    raise SystemExit(main())
