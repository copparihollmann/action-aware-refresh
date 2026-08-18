"""One-rank process group setup, so upstream's collectives can run unmodified.

## The problem this replaces

`cosmos_framework.scripts.action_policy_server_utils.maybe_init_distributed()` builds a
**one-rank NCCL** process group when the policy server is launched outside `torchrun`
(the documented standalone path). On this stack — torch 2.10.0+cu130, L40S sm89 — two
collectives then abort the process:

- `broadcast_object_list()` inside `_download_on_rank0()` → core dump
- `_verify_param_shape_across_processes()` / `_sync_module_states()` inside
  `cosmos_framework.utils.distributed.sync_model_states()`, called from the VAE
  tokenizer constructors → core dump

Reproduce in isolation with `scripts/validate_pg_backend.py`'s companion probe: a
40-line script with no model at all shows NCCL dumping core and gloo surviving.

We first worked around this by *patching upstream to skip both collectives*
(`cosmos-framework-0002`, `-0003`). At `world_size == 1` skipping is a semantic no-op —
there is no peer to talk to — but it is still our code running instead of upstream's,
and it is two patches a reader has to audit before trusting any number we publish.

## The fix

`maybe_init_distributed()` returns early if a process group already exists. So creating
the one-rank group ourselves — with **gloo** instead of NCCL — leaves every line of
upstream's code untouched *and* lets the collectives actually execute. Fewer deviations,
not more.

Measured (`results/processed/pg_backend_ab.jsonl`, 2026-08-04, 4 arms × 10 requests plus
2 arms × 3):

- actions are **bitwise identical** — sha256 `5c01880496f1f666…` under both routes, so
  adopting this re-runs nothing
- latency is indistinguishable: 2510.6 ms (patched/NCCL) vs 2558.1 ms (unpatched/gloo),
  against a between-instance spread of 4.6–6.9% measured by replicating each arm

Backend choice is deliberate and narrow: gloo carries the *startup* collectives (a
self-copy of the VAE weights at world_size 1) and nothing on the per-request path, which
is why the timing above does not move. Do not read this as a claim that gloo is fine for
multi-rank work — at `world_size > 1` this module refuses to act at all.
"""
from __future__ import annotations

import socket
from typing import Literal

Backend = Literal["gloo", "nccl", "none"]

_VALID: tuple[str, ...] = ("gloo", "nccl", "none")


def _free_port() -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.bind(("127.0.0.1", 0))
        return int(s.getsockname()[1])


def ensure_single_rank_group(backend: Backend = "gloo") -> str:
    """Create the one-rank group upstream expects, before upstream can create it.

    Returns a short description of what happened, for logging and provenance records.
    Idempotent: a second call is a no-op.

    ``backend="none"`` deliberately does nothing, leaving upstream to build its own NCCL
    group — which requires `cosmos-framework-0002`/`-0003` to be applied, or the process
    will abort. It exists so the A/B in `scripts/validate_pg_backend.py` can express the
    old route, not as a normal setting.

    Raises rather than falling back:

    - an unknown backend name is an error, not a silent default
    - a group that already exists with a *different* backend is an error, because the
      caller's intent could not be honoured
    - ``world_size > 1`` is an error, because this helper reasons only about the
      single-rank case
    """
    if backend not in _VALID:
        raise ValueError(f"backend must be one of {_VALID}, got {backend!r}")
    if backend == "none":
        return "none (upstream will build its own NCCL group)"

    import torch
    import torch.distributed as dist

    if not dist.is_available():
        raise RuntimeError(
            "torch.distributed is unavailable, but the policy server calls "
            "maybe_init_distributed() unconditionally — it cannot start on this build."
        )
    if dist.is_initialized():
        world = dist.get_world_size()
        if world != 1:
            raise RuntimeError(
                f"a process group with world_size={world} already exists; this helper "
                "only reasons about the single-rank standalone-server case."
            )
        existing = dist.get_backend()
        if existing != backend:
            raise RuntimeError(
                f"a one-rank {existing!r} group already exists, but {backend!r} was "
                "requested — refusing to pretend the request was honoured."
            )
        return f"reused existing one-rank {existing} group"

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required: the policy service refuses to run without it.")
    # Mirror upstream: it calls torch.cuda.set_device(0) on the standalone path. Under
    # CUDA_VISIBLE_DEVICES pinning, index 0 is the pinned physical device.
    torch.cuda.set_device(0)
    dist.init_process_group(
        backend=backend,
        init_method=f"tcp://127.0.0.1:{_free_port()}",
        rank=0,
        world_size=1,
    )
    return f"initialized one-rank {backend} group"
