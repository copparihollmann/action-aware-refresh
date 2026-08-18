#!/usr/bin/env python3
"""Can we drop the two NCCL workaround patches by pre-initializing a gloo group?

Why this exists. Two of our six upstream patches
(`cosmos-framework-0002-single-rank-download`, `-0003-sync-model-states-single-rank`)
*skip* upstream collectives, because `maybe_init_distributed()` builds a **one-rank
NCCL** process group for the standalone server and two collectives then segfault on
this stack (torch 2.10.0+cu130, L40S sm89). Skipping a collective is a semantic no-op
at world_size 1, but it is still our code running instead of upstream's.

`maybe_init_distributed()` returns early when a process group already exists. So if a
**gloo** one-rank group is created before the service is constructed, upstream's
collectives run *unmodified* and the two patches become unnecessary. A 40-line probe
already showed both collectives survive under gloo and that NCCL core-dumps. What that
probe cannot answer is whether gloo is safe for the *model*: `sync_model_states` is
called from the VAE tokenizer constructors, and the DiT is FSDP-aware, so a CPU-side
backend could in principle move per-forward collectives onto the host and wreck latency.

This script answers exactly that, by timing identical requests under a chosen backend:

    # arm A — what we ship today (patched tree, NCCL group built by upstream)
    scripts/validate_pg_backend.py --pre-init none --tag patched_nccl

    # arm B — patches reverted (PYTHONPATH shadow), gloo group pre-initialized
    PYTHONPATH=.worktrees/cf-no-nccl-patch \
      scripts/validate_pg_backend.py --pre-init gloo --tag vanilla_gloo

Compare the medians. If arm B matches arm A within the measurement floor, gloo is a
strictly better deviation than skipping collectives and the two patches are dropped.
Any latency difference means the backend touches the hot path and the patches stay.

This is a *relative* comparison and is valid on a busy machine as long as both arms run
under comparable load — which is why the contention snapshot is recorded per sample and
printed with the result. It is not a source of absolute latency numbers; `docs/latency.md`
owns those.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
import time
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))
sys.path.insert(0, str(REPO_ROOT / "scripts"))


def pre_init(backend: str) -> None:
    """Create the one-rank group the way upstream would, but with a chosen backend.

    Delegates to the shipped helper rather than reimplementing it: this script is the
    evidence for that helper's behaviour, so timing a *different* implementation would
    make the evidence describe code nobody runs.
    """
    from action_refresh.server.process_group import ensure_single_rank_group  # noqa: PLC0415

    print(f"[pre-init] {ensure_single_rank_group(backend)}", flush=True)  # type: ignore[arg-type]


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--pre-init", choices=("none", "gloo", "nccl"), default="none")
    ap.add_argument("--tag", required=True, help="label for this arm in the output")
    ap.add_argument("--repeats", type=int, default=10)
    ap.add_argument("--warmup", type=int, default=2)
    ap.add_argument("--checkpoint-path", default="nvidia/Cosmos3-Nano-Policy-DROID")
    ap.add_argument("--hf-revision", default="6706d7680581c255ff61e0f3bb49d90eac55c79e")
    ap.add_argument(
        "--out-jsonl",
        default=str(REPO_ROOT / "results" / "processed" / "pg_backend_ab.jsonl"),
    )
    args = ap.parse_args()

    if not os.environ.get("HF_HOME"):
        raise SystemExit(
            "HF_HOME is not set — the checkpoint would be paged from the NFS home volume. "
            "Export it (see configs/machine.yaml)."
        )

    pre_init(args.pre_init)

    import numpy as np  # noqa: PLC0415
    import torch  # noqa: PLC0415

    from measure_latency import _first_corpus_request, _load_request, gpu_state, robust  # noqa: PLC0415
    from cosmos_framework.scripts.action_policy_server_robolab import (  # noqa: PLC0415
        RobolabPolicyService,
        RobolabServerArgs,
    )

    import cosmos_framework

    src = Path(cosmos_framework.__file__).resolve().parents[1]
    print(f"[arm {args.tag}] cosmos_framework from {src}", flush=True)

    req = _first_corpus_request()
    obs = _load_request(req)

    t_load0 = time.perf_counter()
    service = RobolabPolicyService(
        RobolabServerArgs(
            checkpoint_path=args.checkpoint_path,
            hf_revision=args.hf_revision,
            num_steps=4,
            decode_video=False,
            guardrails=False,
            deterministic_seed=True,
            seed=0,
        )
    )
    load_s = time.perf_counter() - t_load0
    print(f"[arm {args.tag}] service constructed in {load_s:.1f}s", flush=True)

    for _ in range(args.warmup):
        service.infer(obs)

    samples: list[dict[str, Any]] = []
    for rep in range(args.repeats):
        torch.cuda.synchronize()
        ev0, ev1 = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
        ev0.record()
        t0 = time.perf_counter_ns()
        out = service.infer(obs)
        wall_ms = (time.perf_counter_ns() - t0) / 1e6
        ev1.record()
        torch.cuda.synchronize()
        action = np.asarray(out["action"], dtype=np.float32)
        samples.append(
            {
                "tag": args.tag,
                "pre_init": args.pre_init,
                "source": str(src),
                "repeat": rep,
                "wall_ms": wall_ms,
                "cuda_ms": ev0.elapsed_time(ev1),
                "action_shape": list(action.shape),
                # Full-array digest, not a sampled element: the claim this A/B has to
                # support is that switching the process-group backend leaves the *result*
                # untouched, so no experiment needs re-running. One matching float would
                # not support that; a digest of every byte does.
                "action_sha256": hashlib.sha256(action.tobytes()).hexdigest(),
                "action_sha_head": float(action.reshape(-1)[0]),
                "action_finite": bool(np.isfinite(action).all()),
                "load_s": load_s,
                "utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
                "contention": gpu_state(),
            }
        )
        print(f"  rep {rep}: wall {wall_ms:.1f} ms  cuda {samples[-1]['cuda_ms']:.1f} ms", flush=True)

    out_jsonl = Path(args.out_jsonl)
    out_jsonl.parent.mkdir(parents=True, exist_ok=True)
    with out_jsonl.open("a") as fh:
        for s in samples:
            fh.write(json.dumps(s) + "\n")

    wall = robust([s["wall_ms"] for s in samples])
    cuda = robust([s["cuda_ms"] for s in samples])
    print(
        f"\n[arm {args.tag}] wall median {wall['median']:.1f} ms (MAD {wall['mad_pct']:.2f}%)  "
        f"cuda median {cuda['median']:.1f} ms (MAD {cuda['mad_pct']:.2f}%)  "
        f"n={wall['n']}  load {load_s:.1f}s",
        flush=True,
    )
    print(f"appended {len(samples)} samples to {out_jsonl}")
    # Deterministic seed + identical request => identical actions across arms. Printing a
    # scalar of the action lets the two arms be compared for numerical equivalence, not
    # just speed: a backend that changed the result would be a correctness problem, not a
    # performance one.
    digests = {s["action_sha256"] for s in samples}
    print(f"[arm {args.tag}] action sha256 (deterministic seed, {len(digests)} distinct): "
          f"{sorted(digests)[0]}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
