#!/usr/bin/env python3
"""Does asymmetric video/action denoising survive on OUR checkpoint? Offline, no simulator.

The question. The group's efficiency repo (registered as the private source
`cosmos3-efficient-imagination` in reproducibility/source_manifest.json) established on a
different benchmark and a differently-trained checkpoint that closed-loop success falls off
with *video* denoising steps far more slowly than with *action* steps. If that transfers to
RoboLab/DROID it is the single largest inference-only lever available to us, because the
video is ~98% of the latent.

  NOTE ON CITATIONS. Their repo is private and carries no licence, and this repo's `origin`
  is public, so their measured numbers, their internal file/line references and their
  wording are deliberately NOT reproduced here. They are in `private/` (untracked; see
  private/README.md), which is where the quantitative comparison against their published
  cells lives. Only interface names their code requires — the `warm_start` package, the
  `KNFV_*` / `COSMOS_WARMSTART_*` environment contract — appear below, because this script
  cannot call their sampler without them.

Why it might not transfer. Their result is on checkpoints trained for it. Ours is not, and
that is established from PUBLIC NVIDIA source: `nano_model_config.py:90` sets
`independent_action_schedule=False` and `action_policy_droid_nano.py` never overrides it, so
`Cosmos3-Nano-Policy-DROID` was trained with video and action sharing **one** σ schedule.
Their own sampler documents this as the untested condition, distinguishing graceful
degradation from hard collapse. A second argument points the same way: action σ is sampled
`logitnormal`, so almost no training mass sits at low σ, and large action-step counts visit
σ values this checkpoint never saw. Widening that floor is training we are not doing.

So this script answers the cheap version of the question before any closed-loop hours:
replay the captured request corpus over a (V, A) grid and measure how far each cell's action
chunk lands from the 4-step joint teacher — the sampler that produced `baseline_official`
(28/90 = 31.1%). Deviation is necessary, not sufficient: a cell that matches the teacher
closely is *worth* closed-loop episodes, a cell that diverges wildly is not. Nothing here
claims a success rate.

Cells use their frozen-video sampler in its NO-PRIOR configuration
(`KNFV_NO_PRIOR=1`, `KNFV_VIDEO_ENTRY_SIGMA=1.0`): the video denoises cold from the current
observation in K steps and then freezes for the remaining N−K action steps. That has no
rolling state, which is what makes single-request replay a faithful measurement of it.

One honest limit, which their own write-up records: the frozen video's forward pass is still
recomputed and discarded, so the CUDA times here do **not** show the large theoretical
saving — that needs a KV cache nobody has built yet. Latency is recorded to establish the
floor such a cache would have to beat, not to claim a speedup.

Run under a substrate venv:

    third_party/cosmos-framework/.venv/bin/python scripts/offline_split_schedule.py \
        --grid 4x4,1x4,2x4,1x8,2x8,1x16 --limit 8
"""
from __future__ import annotations

import argparse
import contextlib
import dataclasses
import io
import json
import os
import sys
import time
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))
sys.path.insert(0, str(REPO_ROOT / "scripts"))

TEACHER = "teacher_cold4"


def parse_grid(spec: str) -> list[tuple[int, int]]:
    """`"1x8,2x8"` -> [(1, 8), (2, 8)] as (video_steps K, action_steps N)."""
    cells: list[tuple[int, int]] = []
    for part in spec.split(","):
        part = part.strip()
        if not part:
            continue
        try:
            k, n = (int(x) for x in part.lower().split("x", 1))
        except ValueError as exc:
            raise SystemExit(f"bad grid cell {part!r}; expected VxA e.g. 1x8") from exc
        if not (1 <= k <= n):
            raise SystemExit(f"grid cell {part!r}: need 1 <= V <= A")
        cells.append((k, n))
    if not cells:
        raise SystemExit("empty --grid")
    return cells


def cell_env(k: int, n: int, horizon: int, seed: int) -> dict[str, str]:
    """The env their patch reads. Every name below was verified against their source
    rather than inferred; the file/line citations are kept in `private/` because their
    repo is private and unlicensed and this repo's origin is public.
    """
    return {
        "COSMOS_WARMSTART_MODE": "ofp",
        "COSMOS_WARMSTART_SAMPLER": "knfv",
        "COSMOS_WARMSTART_VIDEO_STEPS": str(k),
        "COSMOS_WARMSTART_STEPS": str(n),
        "COSMOS_WARMSTART_H": str(horizon),
        # No prior: the video is cold from the current observation, so a single request
        # is a complete measurement. Entry sigma 1.0 = full cold entry.
        "KNFV_NO_PRIOR": "1",
        "KNFV_VIDEO_ENTRY_SIGMA": "1.0",
        "KNFV_EPS_SEED": str(seed),
        # Their `[knfv]` traces are gated behind this (gated in their sampler module), and they
        # are the only positive evidence the cell engaged rather than silently falling
        # back to the stock sampler. Cheap host-side prints; on by design, not oversight.
        "KNFV_DEBUG": "1",
        # Rolling prior OFF: it would carry one request's imagination into the next, and
        # the corpus is a set of independent replays, not a trajectory. Leaving it on
        # (their default when MODE=ofp) makes every record depend on iteration order.
        "COSMOS_WARMSTART_ROLLING": "0",
    }


@contextlib.contextmanager
def env_patch(values: dict[str, str]):
    """Set env for one call and restore it — their config is read per call, not per process."""
    saved = {k: os.environ.get(k) for k in values}
    os.environ.update(values)
    try:
        yield
    finally:
        for k, old in saved.items():
            if old is None:
                os.environ.pop(k, None)
            else:
                os.environ[k] = old


def main() -> int:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument("--grid", default="4x4,1x4,2x4,1x8,2x8,1x16", help="comma list of VxA")
    ap.add_argument("--corpus-glob", default="results/raw/corpus/*/captured_request_*.npz")
    ap.add_argument("--limit", type=int, default=0, help="0 = whole corpus")
    ap.add_argument(
        "--per-task",
        type=int,
        default=0,
        help=(
            "take at most N requests from EACH task instead of the first N overall. "
            "The corpus is sorted by task directory, so a bare --limit silently samples one "
            "task — which cannot support a claim about the checkpoint in general."
        ),
    )
    ap.add_argument("--teacher-steps", type=int, default=4)
    ap.add_argument("--horizon", type=int, default=32, help="OPEN_LOOP_HORIZON of the client")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--checkpoint-path", default="nvidia/Cosmos3-Nano-Policy-DROID")
    ap.add_argument("--hf-revision", default="6706d7680581c255ff61e0f3bb49d90eac55c79e")
    ap.add_argument(
        "--out-jsonl", default=str(REPO_ROOT / "results" / "processed" / "split_schedule.jsonl")
    )
    args = ap.parse_args()

    if not os.environ.get("HF_HOME"):
        raise SystemExit(
            "HF_HOME is not set — the checkpoint would be paged from the NFS home volume. "
            "Export it (see configs/machine.yaml)."
        )

    cells = parse_grid(args.grid)

    # Vendor patch BEFORE the service is built: patch.py monkeypatches OmniMoTModel at
    # import and its guardrail bypass only matters pre-construction.
    from action_refresh.config import resolve_substrate
    from action_refresh.server.warm_start_vendor import apply as apply_vendor

    sub = resolve_substrate(repo_root=REPO_ROOT, require_venv=False)
    prov = apply_vendor(REPO_ROOT)
    print(f"[vendor] {prov.source_name} @ {(prov.commit or '?')[:12]} from {prov.sampler_dir}")
    print(f"[vendor] modules: {', '.join(prov.modules)}")
    print(f"[substrate] {sub.name} @ {(sub.commit or '?')[:12]} dirty={sub.dirty}")

    import numpy as np
    import torch
    from measure_latency import gpu_state
    from offline_action_study import load_corpus

    from action_refresh.deviation import action_deviation, chunk_motion
    from action_refresh.server.process_group import ensure_single_rank_group

    corpus = load_corpus(sorted(REPO_ROOT.glob(args.corpus_glob)))
    if args.per_task:
        seen: dict[str, int] = {}
        kept = []
        for item in corpus:
            task = item.get("task") or "?"
            if seen.get(task, 0) < args.per_task:
                seen[task] = seen.get(task, 0) + 1
                kept.append(item)
        corpus = kept
        print(f"stratified: {seen}")
    elif args.limit:
        corpus = corpus[: args.limit]

    print(f"[pg] {ensure_single_rank_group(os.environ.get('COSMOS_PG_BACKEND', 'gloo'))}")

    from cosmos_framework.scripts.action_policy_server_robolab import (
        RobolabPolicyService,
        RobolabServerArgs,
    )

    t0 = time.perf_counter()
    service = RobolabPolicyService(
        RobolabServerArgs(
            checkpoint_path=args.checkpoint_path,
            hf_revision=args.hf_revision,
            num_steps=args.teacher_steps,
            decode_video=False,
            guardrails=False,
            deterministic_seed=True,
            seed=args.seed,
        )
    )
    load_s = time.perf_counter() - t0
    print(f"service constructed in {load_s:.1f}s; corpus={len(corpus)} cells={len(cells)}")

    out_path = Path(args.out_jsonl)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    utc = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
    base_cfg = service.cfg

    def run_once(obs: dict[str, Any], num_steps: int, env: dict[str, str] | None):
        """One inference, timed, with the engagement trace captured.

        stdout is captured because their sampler's `[knfv]` line is the only evidence the
        cell engaged. Their own CLAUDE.md warns that a missing trace means the mode is off —
        and a silently-cold cell would look like "asymmetric scheduling is free", the most
        expensive possible wrong conclusion here.
        """
        service.cfg = dataclasses.replace(base_cfg, num_steps=int(num_steps))
        buf = io.StringIO()
        try:
            torch.cuda.synchronize()
            torch.cuda.reset_peak_memory_stats()
            ev0 = torch.cuda.Event(enable_timing=True)
            ev1 = torch.cuda.Event(enable_timing=True)
            ctx = env_patch(env) if env else contextlib.nullcontext()
            with ctx:
                ev0.record()
                t = time.perf_counter_ns()
                with contextlib.redirect_stdout(buf):
                    out = service.infer(obs)
                wall_ms = (time.perf_counter_ns() - t) / 1e6
                ev1.record()
            torch.cuda.synchronize()
        finally:
            service.cfg = base_cfg
        return (
            np.asarray(out["action"], dtype=np.float32),
            wall_ms,
            ev0.elapsed_time(ev1),
            buf.getvalue(),
        )

    n_written = 0
    engaged_any = False
    with out_path.open("a") as fh:

        def emit(rec: dict[str, Any]) -> None:
            nonlocal n_written
            fh.write(json.dumps(rec) + "\n")
            n_written += 1

        for item in corpus:
            ref, w_ms, c_ms, _ = run_once(item["obs"], args.teacher_steps, None)
            emit(
                {
                    "kind": "split_schedule",
                    "utc": utc,
                    "cell": TEACHER,
                    "video_steps": args.teacher_steps,
                    "action_steps": args.teacher_steps,
                    "synchronized": True,
                    "request_id": item["id"],
                    "task": item.get("task"),
                    "control_step": item.get("control_step"),
                    "wall_ms": w_ms,
                    "cuda_ms": c_ms,
                    "action_finite": bool(np.isfinite(ref).all()),
                    "motion": chunk_motion(ref).as_dict(),
                    "peak_reserved_mib": torch.cuda.max_memory_reserved() / 2**20,
                    **sub.provenance(),
                    **prov.as_dict(),
                    "contention": gpu_state(),
                }
            )

            # Control: the SAME stock config run a second time. Every deviation below is
            # read against this, not against zero. Without it there is no way to tell a
            # real schedule effect from run-to-run nondeterminism — and `deterministic_seed`
            # is a claim about the seed, not a measurement of the whole pipeline.
            rep, w_ms, c_ms, _ = run_once(item["obs"], args.teacher_steps, None)
            rep_dev = action_deviation(rep, ref).as_dict()
            emit(
                {
                    "kind": "split_schedule",
                    "utc": utc,
                    "cell": "teacher_repeat",
                    "video_steps": args.teacher_steps,
                    "action_steps": args.teacher_steps,
                    "synchronized": True,
                    "request_id": item["id"],
                    "task": item.get("task"),
                    "control_step": item.get("control_step"),
                    "wall_ms": w_ms,
                    "cuda_ms": c_ms,
                    "action_finite": bool(np.isfinite(rep).all()),
                    "sampler_engaged": False,
                    "deviation_vs_teacher": rep_dev,
                    **sub.provenance(),
                    **prov.as_dict(),
                }
            )
            print(
                f"  {item['id'][:30]:30s} NOISE-FLOOR (stock repeat) "
                f"joint_l2_mean={rep_dev['joint_l2_mean']:.4g} "
                f"cos={rep_dev['cosine_similarity_mean']:.3f}",
                flush=True,
            )

            for k, n in cells:
                act, w_ms, c_ms, trace = run_once(
                    item["obs"], n, cell_env(k, n, args.horizon, args.seed)
                )
                engaged = "[knfv]" in trace
                engaged_any = engaged_any or engaged
                dev = action_deviation(act, ref).as_dict()
                emit(
                    {
                        "kind": "split_schedule",
                        "utc": utc,
                        "cell": f"knfv_V{k}_A{n}",
                        "video_steps": k,
                        "action_steps": n,
                        "synchronized": k == n,
                        "request_id": item["id"],
                        "task": item.get("task"),
                        "control_step": item.get("control_step"),
                        "wall_ms": w_ms,
                        "cuda_ms": c_ms,
                        "action_finite": bool(np.isfinite(act).all()),
                        # Without this the whole record is unusable: a cold fallback
                        # produces teacher-identical actions and reads as success.
                        "sampler_engaged": engaged,
                        "deviation_vs_teacher": dev,
                        "motion": chunk_motion(act).as_dict(),
                        "peak_reserved_mib": torch.cuda.max_memory_reserved() / 2**20,
                        **sub.provenance(),
                        **prov.as_dict(),
                        "contention": gpu_state(),
                    }
                )
                print(
                    f"  {item['id'][:30]:30s} V{k}/A{n} engaged={engaged} "
                    f"joint_l2_mean={dev['joint_l2_mean']:.4g} "
                    f"linf={dev['joint_linf_max']:.4g} "
                    f"grip_disagree={dev['gripper_disagreement_rate']:.3f} "
                    f"cos={dev['cosine_similarity_mean']:.3f} cuda={c_ms:.0f}ms",
                    flush=True,
                )

    print(f"\nwrote {n_written} records to {out_path}")
    if not engaged_any:
        print(
            "\nERROR: the [knfv] trace never appeared — their sampler did not engage in ANY "
            "cell, so every 'deviation' above is the stock sampler compared with itself. "
            "Fix the wiring before reading anything into these numbers.",
            file=sys.stderr,
        )
        return 3
    print("summarize with: scripts/analyze_split_schedule.py")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
