#!/usr/bin/env python3
"""Do two Cosmos substrates compute the same policy? Answer it for ~0.2 GPU-h.

Why this exists. We measured `baseline_official` (28/90 = 31.1%, 5.95 GPU-h) on NVIDIA
upstream `cosmos-framework`. The group's own baseline is a *different* tree
(`chooper1/Cosmos3-Efficient-Imagination`), and our results are only comparable to
theirs if they are taken on theirs. The expensive way to switch is to re-run the 9-task
closed loop on the fork (~6 GPU-h). The cheap way is to ask whether the fork emits the
*same actions* on inputs we have already captured — 89 real requests in
`results/raw/corpus/`. If it does, the closed-loop result transfers unchanged and the
6 GPU-h is not owed.

That question is decidable offline and bit-exactly, because `deterministic_seed=True`
makes a request reproducible: same observation, same seed, same actions. So each arm
replays the corpus and records a sha256 of every action array; the comparison is then
digest equality, with `action_deviation` supplying magnitude only where digests differ.

Consequences are fixed in advance so the verdict cannot be negotiated after seeing it:

  identical on all N   -> baseline_official transfers; annotate it and move on
  close but not exact  -> report deviation against the B3 noise floor; a 2-task
                          closed-loop spot check (~0.6 GPU-h) decides
  materially different -> the fork changes the policy; the 9-task baseline must be
                          re-run (~6 GPU-h) and that needs explicit approval first

Usage — one process per arm, because each substrate has its own venv and the two must
never share an interpreter:

    scripts/validate_substrate.py run --substrate upstream
    scripts/validate_substrate.py run --substrate efficient_imagination
    scripts/validate_substrate.py compare --substrates upstream,efficient_imagination

`run` re-executes itself under the substrate's own interpreter and then *verifies* that
the imported `cosmos_framework` actually lives inside that substrate's tree. That check
is the load-bearing one: running arm B with arm A's interpreter would produce identical
digests and a confident, wrong verdict of "equivalent".

Latency is recorded but is NOT the point and is not comparable across arms here (the
arms run sequentially, on a shared box, and this script does no interleaving);
`docs/latency.md` owns latency claims.
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

from action_refresh.config import ResolvedSubstrate, resolve_substrate  # noqa: E402

DEFAULT_JSONL = REPO_ROOT / "results" / "processed" / "substrate_ab.jsonl"
DEFAULT_CORPUS_GLOB = "results/raw/corpus/*/captured_request_*.npz"
#: Set by the parent before re-exec, so the child knows not to re-exec again.
REEXEC_SENTINEL = "ACTION_REFRESH_SUBSTRATE_REEXEC"


# ---------------------------------------------------------------------------
# arm: replay the corpus under one substrate
# ---------------------------------------------------------------------------


def _reexec_under(sub: ResolvedSubstrate) -> None:
    """Replace this process with the substrate's own interpreter, once.

    Not a convenience: the venvs pin different wheel stacks, and an arm run under the
    wrong interpreter is indistinguishable from a genuine "equivalent" result.
    """
    if os.environ.get(REEXEC_SENTINEL):
        return
    if Path(sys.executable).resolve() == sub.python.resolve():
        return
    env = {**os.environ, REEXEC_SENTINEL: sub.name, "COSMOS_SUBSTRATE": sub.name}
    print(f"[arm {sub.name}] re-exec under {sub.python}", flush=True)
    os.execve(str(sub.python), [str(sub.python), str(Path(__file__).resolve()), *sys.argv[1:]], env)


def _assert_import_comes_from(sub: ResolvedSubstrate) -> Path:
    """Fail unless the imported package is the substrate's own copy."""
    pkg_name = sub.server_module.split(".", 1)[0]
    pkg = __import__(pkg_name)
    got = Path(pkg.__file__).resolve().parents[1]
    if got != sub.root.resolve():
        raise SystemExit(
            f"error: substrate {sub.name!r} lives in {sub.root}, but `import {pkg_name}` "
            f"resolved to {got}. This arm would measure the wrong tree, and because both "
            "arms are deterministic it would look like agreement. Check that "
            f"{sub.python} has {pkg_name} installed from its own source."
        )
    return got


def run_arm(args: argparse.Namespace) -> int:
    sub = resolve_substrate(args.substrate, repo_root=REPO_ROOT)
    _reexec_under(sub)

    if not os.environ.get("HF_HOME"):
        raise SystemExit(
            "HF_HOME is not set — the checkpoint would be paged from the NFS home volume. "
            "Export it (see configs/machine.yaml)."
        )

    src = _assert_import_comes_from(sub)
    print(f"[arm {sub.name}] {sub.server_module.split('.', 1)[0]} from {src}", flush=True)
    print(f"[arm {sub.name}] commit {(sub.commit or '?')[:12]} dirty={sub.dirty}", flush=True)

    import numpy as np  # noqa: PLC0415
    import torch  # noqa: PLC0415
    from measure_latency import gpu_state  # noqa: PLC0415
    from offline_action_study import load_corpus  # noqa: PLC0415

    corpus = load_corpus(sorted(REPO_ROOT.glob(args.corpus_glob)))
    if args.limit:
        corpus = corpus[: args.limit]
    print(f"[arm {sub.name}] corpus={len(corpus)} requests", flush=True)

    # Import through the resolved module name rather than a literal: the fork need not
    # keep upstream's module path, and configs/substrates.yaml is where that is recorded.
    import importlib  # noqa: PLC0415

    server_mod = importlib.import_module(sub.server_module)
    from action_refresh.server.process_group import ensure_single_rank_group  # noqa: PLC0415

    print(f"[pg] {ensure_single_rank_group(os.environ.get('COSMOS_PG_BACKEND', 'gloo'))}", flush=True)

    t0 = time.perf_counter()
    service = server_mod.RobolabPolicyService(
        server_mod.RobolabServerArgs(
            checkpoint_path=args.checkpoint_path,
            hf_revision=args.hf_revision,
            num_steps=args.num_steps,
            decode_video=False,
            guardrails=False,
            deterministic_seed=True,
            seed=args.seed,
        )
    )
    load_s = time.perf_counter() - t0
    print(f"[arm {sub.name}] service constructed in {load_s:.1f}s", flush=True)

    out_path = Path(args.out_jsonl)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    utc = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())

    n = 0
    with out_path.open("a") as fh:
        for warm in range(args.warmup):
            service.infer(corpus[warm % len(corpus)]["obs"])

        for item in corpus:
            torch.cuda.synchronize()
            ev0 = torch.cuda.Event(enable_timing=True)
            ev1 = torch.cuda.Event(enable_timing=True)
            ev0.record()
            t_req = time.perf_counter_ns()
            out = service.infer(item["obs"])
            wall_ms = (time.perf_counter_ns() - t_req) / 1e6
            ev1.record()
            torch.cuda.synchronize()

            action = np.asarray(out["action"], dtype=np.float32)
            fh.write(
                json.dumps(
                    {
                        "kind": "substrate_ab",
                        "utc": utc,
                        "arm": sub.name,
                        # Full provenance per record, not once per file: these lines get
                        # concatenated across sessions and a record that cannot name its
                        # own tree is unusable.
                        **sub.provenance(),
                        "request_id": item["id"],
                        "task": item.get("task"),
                        "control_step": item.get("control_step"),
                        "num_steps": args.num_steps,
                        "seed": args.seed,
                        # Digest of every byte, not a sampled element: the claim is that
                        # the substrates agree on the whole action array.
                        "action_sha256": hashlib.sha256(action.tobytes()).hexdigest(),
                        "action_shape": list(action.shape),
                        "action_finite": bool(np.isfinite(action).all()),
                        # Kept so a digest mismatch can be sized without a re-run.
                        "action": action.tolist(),
                        "wall_ms": wall_ms,
                        "cuda_ms": ev0.elapsed_time(ev1),
                        "load_s": load_s,
                        "contention": gpu_state(),
                    }
                )
                + "\n"
            )
            n += 1
            if n % 10 == 0 or n == len(corpus):
                print(f"  {n}/{len(corpus)}", flush=True)

    print(f"[arm {sub.name}] appended {n} records to {out_path}")
    print(f"next: scripts/validate_substrate.py compare --substrates <a>,{sub.name}")
    return 0


# ---------------------------------------------------------------------------
# compare: pair the arms and issue the verdict
# ---------------------------------------------------------------------------


def _latest_per_request(records: list[dict], arm: str) -> dict[str, dict]:
    """Most recent record per request id for one arm (the file is append-only)."""
    latest: dict[str, dict] = {}
    for r in records:
        if r.get("arm") != arm:
            continue
        key = r["request_id"]
        if key not in latest or r.get("utc", "") >= latest[key].get("utc", ""):
            latest[key] = r
    return latest


def compare(args: argparse.Namespace) -> int:
    import numpy as np  # noqa: PLC0415

    from action_refresh.deviation import action_deviation  # noqa: PLC0415

    path = Path(args.out_jsonl)
    if not path.exists():
        raise SystemExit(f"error: {path} does not exist — run both arms first")
    records = [json.loads(line) for line in path.read_text().splitlines() if line.strip()]

    arm_a, arm_b = (s.strip() for s in args.substrates.split(","))
    a, b = _latest_per_request(records, arm_a), _latest_per_request(records, arm_b)
    if not a or not b:
        raise SystemExit(
            f"error: need records for both arms; have {arm_a}={len(a)}, {arm_b}={len(b)}"
        )

    shared = sorted(set(a) & set(b))
    only_a, only_b = sorted(set(a) - set(b)), sorted(set(b) - set(a))
    if not shared:
        raise SystemExit(
            f"error: no request ids in common between {arm_a} ({len(a)}) and {arm_b} "
            f"({len(b)}). The arms replayed different corpora; re-run both with the same "
            "--corpus-glob."
        )
    if only_a or only_b:
        # Reported, never silently intersected: comparing 12 of 89 requests and calling
        # it equivalence is exactly the kind of quiet truncation that reads as coverage.
        print(
            f"warning: request sets differ — {len(only_a)} only in {arm_a}, "
            f"{len(only_b)} only in {arm_b}; comparing the {len(shared)} in common",
            file=sys.stderr,
        )

    identical = [k for k in shared if a[k]["action_sha256"] == b[k]["action_sha256"]]
    differing = [k for k in shared if k not in set(identical)]

    print(f"\nsubstrate A: {arm_a} @ {(a[shared[0]].get('commit') or '?')[:12]}")
    print(f"substrate B: {arm_b} @ {(b[shared[0]].get('commit') or '?')[:12]}")
    print(f"requests compared: {len(shared)}  identical: {len(identical)}  differing: {len(differing)}")

    devs: list[dict[str, Any]] = []
    for k in differing:
        ca = np.asarray(a[k]["action"], dtype=np.float32)
        cb = np.asarray(b[k]["action"], dtype=np.float32)
        if ca.shape != cb.shape:
            print(f"  {k}: SHAPE MISMATCH {ca.shape} vs {cb.shape}")
            devs.append({"request_id": k, "shape_mismatch": [list(ca.shape), list(cb.shape)]})
            continue
        d = action_deviation(cb, ca).as_dict()
        devs.append({"request_id": k, **d})
        print(
            f"  {k}: mean_l2={d.get('mean_l2'):.6g} max_abs={d.get('max_abs'):.6g} "
            f"gripper_disagree={d.get('gripper_disagreement')}"
        )

    if not differing:
        verdict = "IDENTICAL"
        consequence = (
            "The substrates compute the same policy on every compared request. "
            "results/reports/baseline_official.json transfers unchanged — annotate it with "
            "this evidence. No closed-loop re-run is owed."
        )
    else:
        worst = max((d.get("mean_l2") or 0.0) for d in devs)
        verdict = "DIFFERENT"
        consequence = (
            f"{len(differing)}/{len(shared)} requests differ (worst mean_l2={worst:.6g}). "
            "Compare that against the B3 noise floor in docs/compute_anatomy.md before "
            "sizing it: below the floor, confirm with a 2-task closed-loop spot check "
            "(~0.6 GPU-h); above it, the fork changes the policy and the 9-task baseline "
            "must be re-run (~6 GPU-h) — which needs explicit approval first."
        )

    print(f"\nverdict: {verdict}\n{consequence}")

    summary = {
        "kind": "substrate_ab_summary",
        "utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "arm_a": arm_a,
        "arm_b": arm_b,
        "commit_a": a[shared[0]].get("commit"),
        "commit_b": b[shared[0]].get("commit"),
        "n_compared": len(shared),
        "n_identical": len(identical),
        "n_differing": len(differing),
        "only_in_a": only_a,
        "only_in_b": only_b,
        "verdict": verdict,
        "consequence": consequence,
        "deviations": devs,
    }
    summary_path = Path(args.summary_json)
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    summary_path.write_text(json.dumps(summary, indent=2))
    print(f"wrote {summary_path}")
    # Exit non-zero on DIFFERENT so a caller cannot treat it as a pass by accident.
    return 0 if verdict == "IDENTICAL" else 1


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)

    r = sub.add_parser("run", help="replay the corpus under one substrate")
    r.add_argument("--substrate", required=True)
    r.add_argument("--corpus-glob", default=DEFAULT_CORPUS_GLOB)
    r.add_argument("--limit", type=int, default=0, help="0 = whole corpus")
    r.add_argument("--warmup", type=int, default=2)
    r.add_argument("--num-steps", type=int, default=4)
    r.add_argument("--seed", type=int, default=0)
    r.add_argument("--checkpoint-path", default="nvidia/Cosmos3-Nano-Policy-DROID")
    r.add_argument("--hf-revision", default="6706d7680581c255ff61e0f3bb49d90eac55c79e")
    r.add_argument("--out-jsonl", default=str(DEFAULT_JSONL))
    r.set_defaults(func=run_arm)

    c = sub.add_parser("compare", help="pair two arms and issue the verdict")
    c.add_argument("--substrates", required=True, help="comma-separated, e.g. upstream,efficient_imagination")
    c.add_argument("--out-jsonl", default=str(DEFAULT_JSONL))
    c.add_argument(
        "--summary-json",
        default=str(REPO_ROOT / "results" / "reports" / "substrate_equivalence.json"),
    )
    c.set_defaults(func=compare)

    args = ap.parse_args()
    return int(args.func(args))


if __name__ == "__main__":
    raise SystemExit(main())
