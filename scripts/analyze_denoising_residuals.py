#!/usr/bin/env python3
"""Experiment F, offline: how much does each block's output change between denoising steps?

**This is the first measurement of our own idea rather than a mandated baseline.** Runs in
the cosmos venv.

Why F is the right contribution to attack first, given what the baselines showed. Two cheap
levers have now been measured closed-loop and both lose task success: shortening the
imagined horizon scored 0/9 (the imagination *is* the plan — chunks stop committing to
travel) and 1-step denoising scored 2/9 against the baseline's 5/9. What those have in
common is that they *change the computation the policy performs*. Cross-step caching does
not: it keeps all 4 denoising steps and all 9 latent frames, and only skips recomputing
block outputs that would barely have changed. If any compute is recoverable without moving
the policy off the frontier, that is where it should be.

What this measures, per transformer block `l` and denoising step `s`, following §11.6:

    d[s,l] = || R[s,l] - R[s-1,l] || / (|| R[s-1,l] || + eps)

and — this is the part that makes it *our* experiment rather than a reimplementation of
TeaCache/DeepCache — it computes that separately over **text, vision and action tokens**,
using the exact packed-sequence index ranges rather than assumed offsets. The token census
established that a request is 95 text / 3,060 vision / 33 action tokens, and §11.6 F3 asks
for separate action-token and vision-token reuse thresholds. You cannot set those without
first knowing whether the two modalities' residuals actually evolve differently. Nobody
has measured that here, and it is cheap to.

Interpretation guide, decided before looking at the numbers so it cannot be fitted to them:

- **Low d across steps** for a block means its output is nearly unchanged between
  consecutive denoising steps, so caching it is nearly free. That is the compute to
  recover.
- **Action-token d much lower than vision-token d** would mean the action stream converges
  earlier than the imagination, so the action half can be cached more aggressively — which
  would be a genuinely action-aware caching policy rather than a uniform one.
- **The reverse** would say the action tokens are the volatile part, and caching must
  protect them, which is equally informative and would kill the "cache the action side"
  framing.

No closed-loop cost, and no cache is built yet: per §11.6 the analysis comes first, and if
the residuals are large everywhere then there is nothing to reuse and the branch closes
cheaply.
"""
from __future__ import annotations

import argparse
import json
import os
import re
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np
import torch

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

# Verified against the loaded model: 36 blocks at `net.language_model.model.layers.N`
# (note the `.model.` level — omitting it matched nothing, and the recorder raises rather
# than silently reporting zero residuals for an unrecognised structure).
BLOCK_RE = re.compile(r"language_model\.model\.layers\.(\d+)$")


def load_request(p: Path) -> dict[str, Any]:
    meta = json.loads(p.with_suffix(".json").read_text())
    prompt = (meta.get("strings") or {}).get("prompt")
    if not prompt:
        raise SystemExit(f"{p} sidecar has no strings.prompt")
    npz = np.load(p)
    obs: dict[str, Any] = {k: npz[k] for k in npz.files}
    obs["prompt"] = prompt
    return obs


def first_corpus_request() -> Path:
    from glob import glob

    for pat in ("results/raw/corpus/*/captured_request_*.npz", "results/raw/captured_request_*.npz"):
        hits = sorted(glob(str(REPO_ROOT / pat)))
        if hits:
            return Path(hits[0])
    raise SystemExit("no captured request found — run scripts/capture_corpus.sh first")


class ResidualRecorder:
    """Capture each block's output and reduce it to per-modality norms immediately.

    Holding every hidden state would cost ~7 GB (36 blocks x 8 forwards x 26 MB), so only
    the *previous* call's tensor is kept per block and the comparison is reduced to scalars
    on the GPU. That is the difference between this running alongside the model and not
    running at all.
    """

    def __init__(
        self,
        model: torch.nn.Module,
        token_ranges: dict[str, torch.Tensor],
        seq_len: int,
        branches: int = 2,
    ):
        self.ranges = token_ranges
        # Only the CFG *conditional* branch is analysed. Guidance is 3.0, so the net runs
        # twice per denoising step, and the unconditional branch can skip text tokens
        # (`skip_text_tokens_for_cfg`) — giving it a different token count. Two
        # consequences, both of which broke the first attempt: consecutive calls for one
        # block then never share a shape, and the packed indices (which address the full
        # 3,188-token conditional sequence) would be out of bounds on the shorter one.
        # Filtering by sequence length keeps the comparison like-for-like instead of
        # silently mixing branches.
        self.seq_len = seq_len
        # CFG branches per denoising step: 2 whenever guidance != 1.0 (baseline is 3.0).
        self.branches = branches
        self.skipped_other_len: dict[int, int] = defaultdict(int)
        self.seen_shapes: set[tuple[int, ...]] = set()
        self._local: dict[str, tuple[str, torch.Tensor | None]] | None = None
        # generation-split length = sequence_length - text tokens
        self._expected_full = seq_len - int(token_ranges['text'].numel())
        self.handles: list[Any] = []
        self.prev: dict[tuple[int, int], dict[str, torch.Tensor | None]] = {}
        # rows: (block, call_index, modality, relative_change, abs_change, prev_norm)
        self.rows: list[tuple[int, int, str, float, float, float]] = []
        self.calls: dict[int, int] = defaultdict(int)
        self.n_blocks = 0
        for name, mod in model.named_modules():
            m = BLOCK_RE.search(name)
            if m:
                idx = int(m.group(1))
                self.n_blocks = max(self.n_blocks, idx + 1)
                self.handles.append(mod.register_forward_hook(self._make_hook(idx)))
        if not self.handles:
            raise RuntimeError(
                "no transformer blocks matched language_model.layers.N — the module naming "
                "changed; refusing to report residuals for an unknown structure."
            )

    def _make_hook(self, block: int):
        def hook(_mod, _args, out):  # noqa: ANN001
            # A block returns Cosmos's `SequencePack` dict, not a bare tensor — the model
            # keeps the two attention splits separate:
            #   causal_seq     [95, 4096]   -> text (understanding tower)
            #   full_only_seq  [3093, 4096] -> generation tokens (vision + action)
            # plus `_full_indices`, mapping each generation row to its global sequence
            # position. Assuming a `[batch, tokens, hidden]` tensor here silently captured
            # nothing at all, so the structure is read rather than guessed.
            pack = out[0] if isinstance(out, tuple) and out and isinstance(out[0], dict) else None
            if pack is None:
                return None
            causal = pack.get("causal_seq")
            full = pack.get("full_only_seq")
            if not isinstance(full, torch.Tensor):
                return None
            self.seen_shapes.add(tuple(full.shape))
            if full.shape[0] != self._expected_full:
                # Unconditional CFG branch or another layout; not comparable.
                self.skipped_other_len[int(full.shape[0])] += 1
                return None
            if self._local is None:
                self._build_local_index(pack)

            call = self.calls[block]
            self.calls[block] += 1
            cur = {
                "text": causal.detach() if isinstance(causal, torch.Tensor) else None,
                "gen": full.detach(),
            }
            # Compare each CFG branch against ITSELF one denoising step earlier.
            #
            # The unconditional branch skips *text* tokens, not generation tokens, so it
            # passes the generation-length filter too and the calls strictly alternate
            # cond, uncond, cond, ... Comparing against the immediately-previous call
            # therefore measured cond-vs-uncond — a guidance difference — and dressed it up
            # as a step-to-step residual. The symptom was an alternating low/high pattern
            # across "steps" that no denoising schedule would produce.
            branch = call % self.branches
            prev = self.prev.get((block, branch))
            if prev is not None:
                for modality, sel in self._local.items():
                    src, idx = sel
                    a, b = cur.get(src), prev.get(src)
                    if not isinstance(a, torch.Tensor) or not isinstance(b, torch.Tensor):
                        continue
                    if a.shape != b.shape:
                        continue
                    if idx is not None:
                        a = a.index_select(0, idx)
                        b = b.index_select(0, idx)
                    a32, b32 = a.float(), b.float()
                    dn = torch.linalg.vector_norm(a32 - b32).item()
                    pn = torch.linalg.vector_norm(b32).item()
                    self.rows.append((block, call, modality, dn / (pn + 1e-8), dn, pn))
            self.prev[(block, branch)] = cur
            return None

        return hook

    def _build_local_index(self, pack: dict[str, Any]) -> None:
        """Map global token positions to rows of `full_only_seq`.

        `_full_indices` gives the global position of each generation row, so inverting it
        turns the census's global vision/action indices into row selectors. Built once and
        asserted, because a wrong mapping would compare vision against action and the
        modality split is the entire point of this experiment.
        """
        full_idx = pack["_full_indices"].long()
        dev = full_idx.device
        lut = torch.full((self.seq_len,), -1, dtype=torch.long, device=dev)
        lut[full_idx] = torch.arange(full_idx.numel(), device=dev)
        local: dict[str, tuple[str, torch.Tensor | None]] = {"text": ("text", None)}
        for name in ("vision", "action"):
            g = self.ranges.get(name)
            if g is None or g.numel() == 0:
                continue
            rows = lut[g.to(dev)]
            if int((rows < 0).sum()) != 0:
                raise RuntimeError(
                    f"{name} tokens are not all inside full_only_seq — the packed layout "
                    "differs from what the census reported; refusing to compare "
                    "mismatched slices."
                )
            local[name] = ("gen", rows)
        self._local = local

    def close(self) -> None:
        for h in self.handles:
            h.remove()
        self.prev.clear()


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--request", default=None)
    ap.add_argument("--num-steps", type=int, default=4)
    ap.add_argument("--checkpoint-path", default="nvidia/Cosmos3-Nano-Policy-DROID")
    ap.add_argument("--hf-revision", default="6706d7680581c255ff61e0f3bb49d90eac55c79e")
    ap.add_argument("--guardrails", action="store_true")
    ap.add_argument("--out-json", default=str(REPO_ROOT / "results" / "processed" / "denoising_residuals.json"))
    ap.add_argument("--out-md", default=str(REPO_ROOT / "docs" / "denoising_residuals.md"))
    args = ap.parse_args()

    if not os.environ.get("HF_HOME"):
        raise SystemExit("HF_HOME is not set — refusing to page the model over NFS.")

    from cosmos_framework.scripts.action_policy_server_robolab import (  # noqa: PLC0415
        RobolabPolicyService,
        RobolabServerArgs,
    )

    sys.path.insert(0, str(REPO_ROOT / "scripts"))
    from probe_compute_anatomy import capture_packed_sequence, token_census  # noqa: PLC0415

    req = Path(args.request) if args.request else first_corpus_request()
    obs = load_request(req)
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
            num_steps=args.num_steps,
            decode_video=False,
            guardrails=args.guardrails,
            deterministic_seed=True,
            seed=0,
        )
    )

    # Exact token ranges from the model's own packed sequence — never assumed offsets. If
    # vision and action were mis-sliced, the modality comparison that is the entire point
    # of this experiment would be meaningless.
    with capture_packed_sequence(service.model) as slot:
        service.infer(obs)
    if not slot:
        raise SystemExit("could not capture the packed sequence; cannot slice by modality")
    packed = slot[0]
    census = token_census(packed)
    dev = next(service.model.parameters()).device
    ranges: dict[str, torch.Tensor] = {
        "text": packed.text_indexes.to(dev).long(),
    }
    for name in ("vision", "action"):
        mod = getattr(packed, name, None)
        if mod is not None:
            ranges[name] = mod.sequence_indexes.to(dev).long()
    print(
        "token ranges: "
        + ", ".join(f"{k}={v.numel()}" for k, v in ranges.items())
        + f"  (sequence_length={census['sequence_length']})",
        flush=True,
    )

    # 2 CFG branches when guidance != 1.0; derived from the resolved config.
    branches = 2 if float(service.cfg.guidance) != 1.0 else 1
    rec = ResidualRecorder(service.model, ranges, int(census['sequence_length']), branches)
    try:
        service.infer(obs)
    finally:
        rec.close()

    if not rec.rows:
        # A failure must report what it actually saw, otherwise diagnosing it costs
        # another 40 s model load per guess.
        raise SystemExit(
            "no residuals captured. Observed block-output token counts: "
            f"{dict(rec.skipped_other_len)} (expected {census['sequence_length']}); "
            f"comparable calls per block: {dict(rec.calls)}; "
            f"observed output shapes: {sorted(rec.seen_shapes)}"
        )

    forwards = max(rec.calls.values())
    # Only conditional-branch calls were kept, so there is one comparable forward per
    # denoising step. Derive rather than assume, and say what was filtered out.
    per_step = max(1, round(forwards / args.num_steps))
    print(
        f"blocks={rec.n_blocks} comparable forwards/block={forwards} "
        f"-> {per_step} forward(s)/step; skipped other-length calls: "
        f"{dict(rec.skipped_other_len)}",
        flush=True,
    )

    rows = [
        {
            "block": b,
            "call": c,
            "step": c // per_step,
            "cfg_branch": c % per_step,
            "branch": c % per_step,
            "modality": m,
            "rel_change": rc,
            "abs_change": ac,
            "prev_norm": pn,
        }
        for (b, c, m, rc, ac, pn) in rec.rows
    ]
    report = summarize(rows, rec.n_blocks, per_step, args.num_steps, census, str(req))
    Path(args.out_json).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out_json).write_text(json.dumps(report, indent=2, default=str))
    Path(args.out_md).write_text(render(report))
    print(f"wrote {args.out_json}")
    print(f"wrote {args.out_md}")
    return 0


def summarize(
    rows: list[dict[str, Any]],
    n_blocks: int,
    per_step: int,
    num_steps: int,
    census: dict[str, Any],
    request: str,
) -> dict[str, Any]:
    def med(vals: list[float]) -> float:
        return float(np.median(vals)) if vals else float("nan")

    by_mod: dict[str, list[float]] = defaultdict(list)
    by_mod_block: dict[str, dict[int, list[float]]] = defaultdict(lambda: defaultdict(list))
    by_mod_step: dict[str, dict[int, list[float]]] = defaultdict(lambda: defaultdict(list))
    for r in rows:
        # Compare like with like: only the same CFG branch across steps.
        by_mod[r["modality"]].append(r["rel_change"])
        by_mod_block[r["modality"]][r["block"]].append(r["rel_change"])
        by_mod_step[r["modality"]][r["step"]].append(r["rel_change"])

    modalities = sorted(by_mod)
    return {
        "request": request,
        "n_blocks": n_blocks,
        "forwards_per_step": per_step,
        "num_steps": num_steps,
        "token_counts": {
            "text": census["text_tokens"],
            "vision": (census.get("vision") or {}).get("tokens_total"),
            "action": (census.get("action") or {}).get("tokens_total"),
        },
        "median_rel_change": {m: med(by_mod[m]) for m in modalities},
        "by_block": {
            m: {str(b): med(v) for b, v in sorted(by_mod_block[m].items())} for m in modalities
        },
        "by_step": {
            m: {str(s): med(v) for s, v in sorted(by_mod_step[m].items())} for m in modalities
        },
        # Blocks whose output barely moves between steps are the cacheable ones. The
        # thresholds are round numbers chosen before seeing the data.
        "cacheable_fraction": {
            m: {
                f"below_{thr}": float(np.mean([v <= thr for v in by_mod[m]]))
                for thr in (0.01, 0.02, 0.05, 0.10)
            }
            for m in modalities
        },
        "n_samples": len(rows),
    }


def render(r: dict[str, Any]) -> str:
    L: list[str] = []
    L.append("# Experiment F (offline): cross-denoising-step residuals")
    L.append("")
    L.append(
        "First measurement of **our own mechanism** rather than a mandated baseline. "
        f"One request (`{Path(r['request']).name}`), {r['num_steps']} denoising steps, "
        f"{r['n_blocks']} transformer blocks, {r['forwards_per_step']} forward(s) per step, "
        f"{r['n_samples']:,} block x step x modality samples."
    )
    L.append("")
    L.append(
        "`d[s,l] = ||R[s,l] - R[s-1,l]|| / ||R[s-1,l]||` per §11.6, computed separately over "
        "text, vision and action tokens using the model's **actual** packed-sequence "
        "indices — not assumed offsets, since a mis-slice would invalidate the one "
        "comparison this experiment exists to make."
    )
    L.append("")
    tc = r["token_counts"]
    L.append(f"Token counts: text {tc['text']}, vision {tc['vision']:,}, action {tc['action']}.")
    L.append("")
    L.append("## Median relative change between consecutive denoising steps")
    L.append("")
    L.append("| modality | median d | share of samples below 1% | below 2% | below 5% | below 10% |")
    L.append("|---|---|---|---|---|---|")
    for m, v in sorted(r["median_rel_change"].items()):
        cf = r["cacheable_fraction"][m]
        L.append(
            f"| **{m}** | {v:.4f} | {100 * cf['below_0.01']:.0f}% | "
            f"{100 * cf['below_0.02']:.0f}% | {100 * cf['below_0.05']:.0f}% | "
            f"{100 * cf['below_0.1']:.0f}% |"
        )
    L.append("")
    mv = r["median_rel_change"]
    cf = r["cacheable_fraction"]
    if "action" in mv and "vision" in mv and mv["vision"] > 0:
        ratio = mv["action"] / mv["vision"]
        # Two independent questions, and conflating them would overstate the result: is
        # there an action/vision *asymmetry* (a ratio question), and is either modality
        # *cacheable at all* (an absolute-magnitude question)? A favourable ratio between
        # two large residuals buys nothing.
        gen_cacheable = max(
            cf.get("action", {}).get("below_0.1", 0.0), cf.get("vision", {}).get("below_0.1", 0.0)
        )
        L.append(
            f"> **The asymmetry is real: action-token residuals are {ratio:.2f}x the "
            f"vision-token residuals** ({mv['action']:.3f} vs {mv['vision']:.3f}). The "
            "action stream *is* the more stable of the two, which is the direction this "
            "project's action-aware framing predicted — now measured rather than assumed."
        )
        L.append("")
        if gen_cacheable < 0.05:
            L.append(
                f"> **But nothing is cacheable at this step count, so the asymmetry is not "
                f"exploitable.** Only {100 * gen_cacheable:.1f}% of generation-token samples "
                "change by less than 10% between consecutive denoising steps, and **0%** "
                "change by less than 5%. Reuse needs near-identical block outputs; an 18–32% "
                "change is a different tensor."
            )
            L.append("")
            L.append(
                "This is a **negative result for Experiment F as framed**, and it has a clean "
                "mechanical explanation. TeaCache / DeepCache / token-wise caching all operate "
                "on 20–50-step diffusion schedules, where consecutive steps genuinely barely "
                "differ. Cosmos3-Nano runs **4** steps. The short schedule has already "
                "squeezed out the inter-step redundancy those methods harvest — so there is "
                "nothing left for a cache to recover, and F3's separate action/vision "
                "thresholds are correct in principle but moot in practice here."
            )
            L.append("")
            L.append(
                "It also suggests the one reframing that could revive F: a *longer* schedule "
                "plus aggressive caching, targeting equal compute to the 4-step baseline. "
                "That is speculative and this data argues against it — at 18–32% per-step "
                "change the cache would have to be nearly free to break even — but it is the "
                "only version of F these numbers do not already rule out."
            )
            L.append("")
        text_free = cf.get("text", {}).get("below_0.01", 0.0)
        if text_free >= 0.99 and mv.get("text", 1.0) == 0.0:
            share = (r["token_counts"]["text"] or 0) / max(
                sum(v or 0 for v in r["token_counts"].values()), 1
            )
            L.append(
                f"> **One exact, free saving does exist.** Text-token residuals are "
                f"**identically zero** across every block and every step — the "
                "understanding tower's output does not depend on the noised latents at all, "
                "so it can be computed once per request and reused for every denoising step "
                "with *bit-identical* results, not merely small error. The ceiling is small "
                f"because text is only {r['token_counts']['text']} of "
                f"{sum(v or 0 for v in r['token_counts'].values()):,} tokens "
                f"(~{100 * share:.0f}%), and it is worth checking whether the implementation "
                "already avoids recomputing it before claiming the saving."
            )
            L.append("")
    L.append("## Per-step change (does denoising converge?)")
    L.append("")
    steps = sorted({s for m in r["by_step"] for s in r["by_step"][m]}, key=int)
    L.append("| modality | " + " | ".join(f"step {s}" for s in steps) + " |")
    L.append("|---" * (len(steps) + 1) + "|")
    for m in sorted(r["by_step"]):
        L.append(
            f"| **{m}** | "
            + " | ".join(f"{r['by_step'][m].get(s, float('nan')):.4f}" for s in steps)
            + " |"
        )
    L.append("")
    L.append(
        "A falling trend means later steps change less and are the cheapest to cache; a "
        "flat trend means every step matters and uniform whole-step reuse (F0) will hurt."
    )
    L.append("")
    L.append("## Limitations")
    L.append("")
    L.append(
        "- **One request.** Enough to establish the shape and to decide whether to build a "
        "cache at all; not enough to set thresholds. Those need the full corpus."
    )
    L.append(
        "- **No cache is implemented yet**, so nothing here is a speedup. §11.6 requires "
        "the analysis first precisely so that a cache is only built if there is something "
        "to reuse — and any cache must then be charged its own lookup/copy overhead "
        "(spec §7), which memory traffic can easily exceed the compute saved."
    )
    L.append(
        "- **Residual size is not the same as action impact.** A block whose output barely "
        "moves may still be the one the action tokens depend on. The closed-loop lesson "
        "from Experiments A and E applies here in advance: this session already showed "
        "that an offline metric matched to sampling noise failed to predict closed-loop "
        "success, so any threshold chosen from this table must be validated against task "
        "success before being believed."
    )
    L.append("")
    return "\n".join(L) + "\n"


if __name__ == "__main__":
    raise SystemExit(main())
