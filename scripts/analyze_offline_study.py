#!/usr/bin/env python3
"""Analyse the offline action study: deviation vs the 4-step teacher, per condition.

Runs in the repo venv (numpy only) — deliberately separate from
`offline_action_study.py`, which owns the GPU. Generation is expensive and
re-runnable only at cost; analysis is cheap and gets iterated on. Keeping them apart
means changing a metric definition never means re-running the sweep.

Reads completed units from the ledger and writes:
  - `results/processed/offline_study.json`  machine-readable, per (condition, request)
  - `docs/offline_action_study.md`          the report

Answers two questions:

**Experiment A** — how much does the action move when denoising steps are cut? The
compute side is already known (1 step is 2.98x faster than 4). What was unknown is
the accuracy cost, and that is what decides which step counts deserve closed-loop
episodes.

**Experiment E0** — how much does the action move when only the *diffusion seed*
changes? Because vision and action are denoised jointly, a different seed means a
different imagined future. If the action barely moves across seeds while it moves a
lot across observations, the specific imagined content is not what drives the action.
That comparison is the point: seed-spread alone means nothing without the
across-observation scale to compare it against.

The two are then read together. If cutting steps perturbs the action *less* than
merely reshuffling the seed does, then reduced steps are within the model's own
sampling noise — a much stronger statement than "the deviation looks small".
"""
from __future__ import annotations

import argparse
import json
import statistics
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

from action_refresh.deviation import (  # noqa: E402
    action_deviation,
    chunk_motion,
    chunk_spread,
    ee_deviation_available,
)
from action_refresh.ledger import Ledger  # noqa: E402

TEACHER = "teacher_steps4"


def load_units(ledger_root: Path) -> dict[tuple[str, str], dict[str, Any]]:
    """Completed results keyed by (condition, request_id)."""
    led = Ledger(ledger_root)
    out: dict[tuple[str, str], dict[str, Any]] = {}
    for payload in led.results():
        unit = payload.get("unit") or {}
        cond, req = unit.get("method"), unit.get("task")
        if not cond or not req:
            continue
        res = payload.get("result") or {}
        path = res.get("action_path")
        if not path:
            continue
        p = REPO_ROOT / path
        if not p.exists():
            continue
        res = dict(res)
        res["action"] = np.load(p)
        res["elapsed_s"] = payload.get("elapsed_s")
        res["condition"] = cond
        res["request"] = req
        out[(cond, req)] = res
    return out


def analyse_chunk_boundaries(units: dict[tuple[str, str], dict[str, Any]]) -> dict[str, Any]:
    """How much the plan jumps when the policy re-plans at a chunk boundary.

    Corpus request ids look like `<Task>_reqNNNN`, consecutive within a task and one
    open-loop horizon apart, so teacher chunk k's last action and chunk k+1's first
    action describe the *same instant* from either side of a re-plan.

    Two numbers per boundary:
      - `jump`: |first(k+1) - last(k)| in joint space — the discontinuity the robot
        would actually experience.
      - `intra_step`: the median step-to-step change *within* a chunk, as the scale to
        judge the jump against. A jump comparable to normal intra-chunk motion is not a
        discontinuity at all; one much larger means re-planning materially changed the
        intention, and executing the stale chunk further would have diverged.
    """
    from action_refresh.deviation import N_JOINTS  # noqa: PLC0415

    by_task: dict[str, list[tuple[int, np.ndarray]]] = {}
    for (cond, req), res in units.items():
        if cond != TEACHER:
            continue
        if "_req" not in req:
            continue  # single-capture corpus item; no sequence to walk
        task, _, idx = req.rpartition("_req")
        try:
            by_task.setdefault(task, []).append((int(idx), res["action"]))
        except ValueError:
            continue

    jumps: list[float] = []
    intra: list[float] = []
    per_task: dict[str, dict[str, float]] = {}
    for task, items in by_task.items():
        items.sort(key=lambda x: x[0])
        t_jumps: list[float] = []
        for (i0, a0), (i1, a1) in zip(items, items[1:]):
            if i1 != i0 + 1:
                continue  # gap in the corpus; not adjacent in time
            jump = float(np.linalg.norm(a1[0, :N_JOINTS] - a0[-1, :N_JOINTS]))
            t_jumps.append(jump)
            jumps.append(jump)
        for _, a in items:
            if len(a) > 1:
                intra.append(
                    float(np.median(np.linalg.norm(np.diff(a[:, :N_JOINTS], axis=0), axis=1)))
                )
        if t_jumps:
            per_task[task] = {
                "n_boundaries": len(t_jumps),
                "median_jump": float(np.median(t_jumps)),
                "max_jump": float(np.max(t_jumps)),
            }
    if not jumps:
        return {}
    med_jump = float(np.median(jumps))
    med_intra = float(np.median(intra)) if intra else None
    return {
        "n_boundaries": len(jumps),
        "median_jump_rad": med_jump,
        "max_jump_rad": float(np.max(jumps)),
        "median_intra_chunk_step_rad": med_intra,
        "jump_vs_intra_ratio": (med_jump / med_intra) if med_intra else None,
        "per_task": per_task,
    }


def summarize(values: list[float]) -> dict[str, float] | None:
    if not values:
        return None
    a = np.asarray(values, dtype=float)
    return {
        "n": int(a.size),
        "median": float(np.median(a)),
        "mean": float(a.mean()),
        "max": float(a.max()),
        "p95": float(np.percentile(a, 95)) if a.size > 1 else float(a[0]),
    }


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--ledger", default=str(REPO_ROOT / "results" / "ledger" / "offline_study"))
    ap.add_argument("--out-json", default=str(REPO_ROOT / "results" / "processed" / "offline_study.json"))
    ap.add_argument("--out-md", default=str(REPO_ROOT / "docs" / "offline_action_study.md"))
    args = ap.parse_args()

    units = load_units(Path(args.ledger))
    if not units:
        raise SystemExit(
            f"no completed units in {args.ledger} — run scripts/offline_action_study.py first"
        )

    conditions = sorted({c for c, _ in units})
    requests = sorted({r for _, r in units})
    teacher_requests = [r for r in requests if (TEACHER, r) in units]
    if not teacher_requests:
        raise SystemExit(
            f"no `{TEACHER}` results — every deviation is measured against it, so there "
            "is nothing to compare. Re-run the study including the teacher condition."
        )

    # --- per-condition deviation vs teacher --------------------------------
    per_condition: dict[str, dict[str, Any]] = {}
    rows: list[dict[str, Any]] = []
    for cond in conditions:
        if cond == TEACHER:
            continue
        devs: list[Any] = []
        for req in teacher_requests:
            cand = units.get((cond, req))
            if cand is None:
                continue  # not measured on this request; absence is reported below
            try:
                d = action_deviation(cand["action"], units[(TEACHER, req)]["action"])
            except ValueError as exc:
                rows.append({"condition": cond, "request": req, "error": str(exc)})
                continue
            devs.append(d)
            rows.append({"condition": cond, "request": req, **d.as_dict()})
        if not devs:
            continue
        per_condition[cond] = {
            "n_requests": len(devs),
            "coverage": f"{len(devs)}/{len(teacher_requests)}",
            "joint_l2_mean": summarize([d.joint_l2_mean for d in devs]),
            "joint_linf_max": summarize([d.joint_linf_max for d in devs]),
            "endpoint_joint_l2": summarize([d.endpoint_joint_l2 for d in devs]),
            "gripper_disagreement_rate": summarize([d.gripper_disagreement_rate for d in devs]),
            "cosine_similarity_mean": summarize([d.cosine_similarity_mean for d in devs]),
            "n_with_any_gripper_flip": sum(1 for d in devs if d.gripper_disagreement_rate > 0),
            "experiment": next(
                (
                    units[(cond, r)].get("experiment")
                    for r in teacher_requests
                    if (cond, r) in units
                ),
                None,
            ),
            "latency_median_ms": summarize(
                [units[(cond, r)]["wall_ms"] for r in teacher_requests if (cond, r) in units]
            ),
            "cuda_median_ms": summarize(
                [units[(cond, r)]["cuda_ms"] for r in teacher_requests if (cond, r) in units]
            ),
        }

    # --- E0: seed spread, and the scale it must be compared against --------
    seed_conditions = [TEACHER] + [c for c in conditions if c.startswith("seed_")]
    seed_spread: dict[str, Any] = {}
    if len(seed_conditions) >= 2:
        per_request: list[dict[str, float]] = []
        for req in requests:
            chunks = [units[(c, req)]["action"] for c in seed_conditions if (c, req) in units]
            if len(chunks) >= 2:
                per_request.append(chunk_spread(chunks))
        if per_request:
            seed_spread = {
                "n_requests": len(per_request),
                "n_seeds": len(seed_conditions),
                "joint_spread_rms": summarize([s["joint_spread_rms"] for s in per_request]),
                "joint_spread_max": summarize([s["joint_spread_max"] for s in per_request]),
                "gripper_unstable_step_rate": summarize(
                    [s["gripper_unstable_step_rate"] for s in per_request]
                ),
            }

    # Across-observation scale: how different are teacher actions for *different*
    # observations? Without this, a seed spread of 0.05 rad is uninterpretable.
    across_obs: dict[str, Any] = {}
    if len(teacher_requests) >= 2:
        teacher_chunks = [units[(TEACHER, r)]["action"] for r in teacher_requests]
        shapes = {c.shape for c in teacher_chunks}
        if len(shapes) == 1:
            across_obs = chunk_spread(teacher_chunks)
        else:
            across_obs = {"error": f"teacher chunks have differing shapes: {sorted(shapes)}"}

    # The noise reference every other condition is judged against. It MUST be the same
    # kind of quantity as the deviations in `per_condition` — a pairwise deviation from
    # the teacher — or the ratios are meaningless. (An earlier version compared
    # pairwise deviations against the seeds' spread about their own mean, which for K=2
    # is half the pairwise distance and therefore inflated every ratio by ~2x.)
    seed_devs = [
        d["joint_l2_mean"]["median"]
        for c, d in per_condition.items()
        if c.startswith("seed_")
    ]
    noise_ref = (
        {
            "value": float(statistics.median(seed_devs)),
            "n": len(seed_devs),
            "definition": (
                "median over seed_* conditions of the median joint-L2 deviation from "
                "the teacher — same quantity as the per-condition table"
            ),
        }
        if seed_devs
        else None
    )

    # --- chunk-boundary discontinuity (Experiment B, spec §11.2) -----------
    # Free: it reuses the teacher chunks already computed. Consecutive corpus requests
    # are one open-loop horizon apart, so chunk k ends where chunk k+1 begins. The jump
    # between them is how much the policy *changes its mind* when it re-plans from a
    # fresh observation. A small jump means the plan was stable and refreshing less
    # often is plausible; a large one means re-planning matters and extending the
    # horizon would execute a stale intention.
    boundary = analyse_chunk_boundaries(units)

    # --- reference-free motion, per condition -------------------------------
    # Added after `vision_frames_9` deviated only 1.14x sampling noise offline and then
    # scored 0/9 closed-loop with one contact event in nine episodes. Deviation from a
    # teacher cannot see a chunk that meanders near its start point; net travel can.
    motion: dict[str, dict[str, float]] = {}
    for cond in conditions:
        chunks = [units[(cond, rq)]["action"] for rq in requests if (cond, rq) in units]
        if not chunks:
            continue
        ms = [chunk_motion(c) for c in chunks]
        motion[cond] = {
            "n": len(ms),
            "per_step_motion": float(np.median([m.per_step_motion for m in ms])),
            "joint_range": float(np.median([m.joint_range for m in ms])),
            "net_displacement": float(np.median([m.net_displacement for m in ms])),
            "straightness": float(np.median([m.straightness for m in ms])),
        }

    report = {
        "sampling_noise_reference": noise_ref,
        "chunk_motion": motion,
        "chunk_boundary_discontinuity": boundary,
        "conditions": conditions,
        "n_requests": len(requests),
        "requests": requests,
        "teacher": TEACHER,
        "per_condition": per_condition,
        "seed_spread_within_observation": seed_spread,
        "spread_across_observations": across_obs,
        "ee_metrics_available": ee_deviation_available(),
        "rows": rows,
    }
    out_json = Path(args.out_json)
    out_json.parent.mkdir(parents=True, exist_ok=True)
    out_json.write_text(json.dumps(report, indent=2, default=str))

    Path(args.out_md).write_text(render_markdown(report))
    print(f"wrote {out_json}")
    print(f"wrote {args.out_md}")
    return 0


def render_markdown(r: dict[str, Any]) -> str:
    L: list[str] = []
    L.append("# Offline action study — Experiments A and E0")
    L.append("")
    L.append(
        f"Generated by `scripts/analyze_offline_study.py` from "
        f"{r['n_requests']} captured requests. Every deviation is measured against "
        f"`{r['teacher']}` (4 denoising steps, seed 0) on the *same* observation, so "
        "differences are attributable to the condition and nothing else."
    )
    L.append("")
    L.append(
        "All input is **real**: requests captured from live closed-loop episodes "
        "(research patch `robolab-0001`/`0002`), not synthesised."
    )
    L.append("")

    # -- the scale first: numbers below are meaningless without it -----------
    across = r.get("spread_across_observations") or {}
    seed = r.get("seed_spread_within_observation") or {}
    L.append("## The scale to judge everything against")
    L.append("")
    if across and not across.get("error"):
        L.append(
            f"- **Across different observations** (teacher vs teacher): joint spread RMS "
            f"**{across['joint_spread_rms']:.4f} rad**, max {across['joint_spread_max']:.4f}. "
            "This is how much the action *should* change when the world changes — the "
            "signal we must not destroy."
        )
    elif across.get("error"):
        L.append(f"- Across observations: not computed — {across['error']}")
    noise = r.get("sampling_noise_reference")
    if noise:
        L.append(
            f"- **Across diffusion seeds, same observation** (Experiment E0, "
            f"{noise['n']} seed condition(s)): median joint deviation from the teacher "
            f"**{noise['value']:.4f} rad**. Changing the seed changes the imagined "
            "future while keeping it fully in-distribution, so this is the model's own "
            "sampling noise — the amount the baseline already disagrees with itself "
            "between runs. Any method deviating *less* than this is not measurably "
            "changing the policy."
        )
    if seed:
        ss = seed["joint_spread_rms"]
        L.append(
            f"- Seed dispersion about their own mean (secondary view, {seed['n_seeds']} "
            f"seeds x {seed['n_requests']} requests): RMS median {ss['median']:.4f} rad, "
            f"max {seed['joint_spread_max']['max']:.4f}; gripper unanimity broken on "
            f"{100 * seed['gripper_unstable_step_rate']['median']:.1f}% of timesteps "
            "(median). Reported for completeness but **not** used as the reference — it "
            "is a different quantity from a pairwise deviation and mixing the two "
            "inflates ratios."
        )
    if across and noise and not across.get("error"):
        # Both are dispersions about a mean, so this ratio *is* apples-to-apples.
        obs_scale = across["joint_spread_rms"]
        seed_scale = (seed or {}).get("joint_spread_rms", {}).get("median")
        if seed_scale:
            ratio = seed_scale / max(obs_scale, 1e-12)
            L.append("")
            L.append(
                f"> **Seed-driven variation is {100 * ratio:.1f}% of the variation "
                f"driven by the observation** ({seed_scale:.4f} vs {obs_scale:.4f} rad, "
                "both dispersions about a mean). "
                + (
                    "Small: the action tracks what the robot sees far more than which "
                    "future the model happens to imagine — consistent with the "
                    "imagination being weakly load-bearing for the action."
                    if ratio < 0.25
                    else "Large: which future the model imagines materially moves the "
                    "action, so the imagination is load-bearing and cannot simply be "
                    "discarded."
                )
            )
    L.append("")

    # -- per-condition table -------------------------------------------------
    L.append("## Deviation from the 4-step teacher, by condition")
    L.append("")
    L.append(
        "| condition | exp | coverage | joint L2 (rad, median) | worst single joint | "
        "endpoint L2 | gripper disagree | any flip | wall ms (median) |"
    )
    L.append("|---|---|---|---|---|---|---|---|---|")
    for cond, d in sorted(r["per_condition"].items()):
        g = d["gripper_disagreement_rate"]
        L.append(
            f"| `{cond}` | {d.get('experiment') or '—'} | {d['coverage']} | "
            f"{d['joint_l2_mean']['median']:.4f} | {d['joint_linf_max']['max']:.4f} | "
            f"{d['endpoint_joint_l2']['median']:.4f} | {100 * g['median']:.1f}% | "
            f"{d['n_with_any_gripper_flip']}/{d['n_requests']} | "
            f"{d['latency_median_ms']['median']:.0f} |"
        )
    L.append("")
    L.append(
        "`any flip` counts requests where the binarized gripper command differs "
        "anywhere in the chunk. It is called out separately because a single flip can "
        "decide grasp versus no-grasp, and averaging it into a joint-error number "
        "would hide exactly the failure that matters."
    )
    L.append("")
    L.append(
        "⚠ **Do not read the `wall ms` column as latency.** This sweep runs conditions "
        "in back-to-back blocks, which the B3 measurement showed is the one way not to "
        "time anything on this host: identical work spans 3,377–5,974 ms under sustained "
        "load, so a condition measured in a busy block looks slower than the same work "
        "measured in a quiet one. The **deviation** columns are unaffected — they are "
        "deterministic given the seed — but any speedup claim needs a separate "
        "median-of-many measurement with an idle gap and interleaved configs. Where a "
        "latency figure is quoted elsewhere in the project, it comes from that protocol, "
        "not from this table."
    )
    L.append("")

    # -- reading A against E0 -----------------------------------------------
    steps = {c: d for c, d in r["per_condition"].items() if c.startswith("steps_")}
    noise = r.get("sampling_noise_reference")
    if steps and noise:
        L.append("## Experiment A read against the model's own sampling noise")
        L.append("")
        L.append(
            f"The reference is **{noise['value']:.4f} rad** — the median deviation of "
            f"the `seed_*` conditions from the teacher ({noise['n']} seed condition(s)). "
            "This is deliberately the *same* quantity as the table above (pairwise "
            "deviation from the teacher on the same observation), so the ratio is "
            "apples-to-apples. Using the seeds' spread-about-their-own-mean instead "
            "would understate the reference by roughly half and inflate every ratio."
        )
        L.append("")
        for cond, d in sorted(steps.items()):
            dev = d["joint_l2_mean"]["median"]
            ratio = dev / max(noise["value"], 1e-12)
            verdict = (
                f"**within sampling noise** ({ratio:.2f}x)"
                if ratio <= 1.0
                else f"**{ratio:.2f}x sampling noise**"
            )
            L.append(f"- `{cond}`: {dev:.4f} rad — {verdict}")
        L.append("")
        L.append(
            "A condition inside the noise band has not been shown to change the policy "
            "in any way the baseline does not already change itself between runs — a "
            "much stronger statement than 'the deviation looks small'. It is still not "
            "proof of equal task success: only closed-loop episodes settle that, and "
            "this study cannot. It is the evidence that justifies spending episodes."
        )
        L.append("")

    # -- Experiment E: does the action need the imagined future? -------------
    e_conds = {
        c: d
        for c, d in r["per_condition"].items()
        if c.startswith("vision_frames_") or c == "no_imagination_freeze"
    }
    if e_conds and noise:
        ref = noise["value"]
        L.append("## Experiment E — does the action need the imagined future?")
        L.append("")
        L.append(
            "The token census put **85.3% of the 3,188-token sequence** in imagined "
            "future video (8 of 9 latent frames) against 32 action tokens, and the model "
            "is matmul-bound and near-linear in token count. So this is the largest lever "
            "available — if the action tolerates a shorter imagined horizon."
        )
        L.append("")
        L.append("| condition | latent frames | vision tokens | deviation | vs noise | gripper flips |")
        L.append("|---|---|---|---|---|---|")
        # Latent frames for N pixel frames at temporal downsample 4, and 340 tokens each
        # (17x20 spatial grid, measured).
        for cond, d in sorted(
            e_conds.items(),
            key=lambda kv: -(int(kv[0].rsplit("_", 1)[1]) if kv[0][-1].isdigit() else 0),
        ):
            dev = d["joint_l2_mean"]["median"]
            ratio = dev / max(ref, 1e-12)
            if cond.startswith("vision_frames_"):
                n = int(cond.rsplit("_", 1)[1])
                latent = (n - 1) // 4 + 1
                frames_s, tokens_s = str(latent), f"{latent * 340:,}"
            else:
                frames_s, tokens_s = "9 (frozen)", "3,060"
            L.append(
                f"| `{cond}` | {frames_s} | {tokens_s} | {dev:.4f} rad | "
                f"**{ratio:.2f}x** | {d['n_with_any_gripper_flip']}/{d['n_requests']} |"
            )
        L.append(f"| _baseline (reference)_ | 9 | 3,060 | — | — | — |")
        L.append("")

        vf = {c: d for c, d in e_conds.items() if c.startswith("vision_frames_")}
        freeze = e_conds.get("no_imagination_freeze")
        if vf and freeze:
            vf_ratios = [
                d["joint_l2_mean"]["median"] / max(ref, 1e-12) for d in vf.values()
            ]
            fr_ratio = freeze["joint_l2_mean"]["median"] / max(ref, 1e-12)
            fr_flips = freeze["n_with_any_gripper_flip"] / max(freeze["n_requests"], 1)
            vf_flips = max(
                d["n_with_any_gripper_flip"] / max(d["n_requests"], 1) for d in vf.values()
            )
            # The seed conditions' own flip rate is the only fair yardstick for E2b's:
            # the baseline flips the gripper between runs too, and without that number
            # "26% of requests flip" reads as alarming when it may be unremarkable.
            seed_flip_rates = [
                d["n_with_any_gripper_flip"] / max(d["n_requests"], 1)
                for c, d in r["per_condition"].items()
                if c.startswith("seed_")
            ]
            L.append(
                f"> **The imagination is needed, but only a little of it.** Shortening the "
                f"imagined horizon (E2b) costs {min(vf_ratios):.2f}–{max(vf_ratios):.2f}x "
                f"sampling noise, with gripper flips in {100 * vf_flips:.0f}% of requests "
                "at worst — "
                + (
                    f"**inside** the {100 * min(seed_flip_rates):.0f}–"
                    f"{100 * max(seed_flip_rates):.0f}% range the baseline already flips "
                    "between seeds, so its gripper behaviour is not distinguishable from "
                    "sampling noise. "
                    if seed_flip_rates and vf_flips <= max(seed_flip_rates)
                    else f"against {100 * min(seed_flip_rates):.0f}–"
                    f"{100 * max(seed_flip_rates):.0f}% between seeds. "
                    if seed_flip_rates
                    else ""
                )
                + f"*Destroying* the imagination — leaving the tokens in place but freezing "
                f"them at noise (E1) — costs {fr_ratio:.2f}x noise and flips the gripper "
                f"in {100 * fr_flips:.0f}% of requests."
            )
            L.append("")
            L.append(
                "Read together, those two say something more specific than 'vision "
                "matters'. The action does not need a *long* imagined rollout — 2 latent "
                "frames cost little more than 5 — but it does need the imagination to be "
                "**coherent**. Incoherent vision tokens are worse than fewer coherent "
                "ones. That distinction is why E1's large deviation must not be read as "
                "'the imagination is load-bearing': E1 leaves the model out of "
                "distribution (late in denoising it expects nearly-clean vision and gets "
                "noise), whereas E2b keeps every frame coherent and merely asks for fewer."
            )
            L.append("")
            L.append(
                "The compute consequence is what matters for the frontier: E2b removes "
                "real tokens and so is the only one of the two whose latency is a "
                "speedup. E1 keeps the token count identical by construction — it "
                "answers accuracy only."
            )
            L.append("")

    mot = r.get("chunk_motion") or {}
    if mot and TEACHER in mot:
        base = mot[TEACHER]
        L.append("## Does the chunk actually go anywhere? (reference-free)")
        L.append("")
        L.append(
            "Added after a failure the deviation metrics above could not see: "
            "`vision_frames_9` sits at 1.14x sampling noise, and scored **0/9** "
            "closed-loop with **one** contact event across nine episodes (baseline: 96). "
            "Deviation from a teacher cannot detect a chunk that meanders near its start "
            "point — such a chunk sits at a perfectly ordinary L2 distance. Net travel can."
        )
        L.append("")
        L.append("| condition | per-step rate | joint range | vs teacher | net displacement | straightness |")
        L.append("|---|---|---|---|---|---|")
        for cond, m in sorted(mot.items(), key=lambda kv: -kv[1]["joint_range"]):
            rel = m["joint_range"] / base["joint_range"] if base["joint_range"] else 0.0
            L.append(
                f"| `{cond}` | {m['per_step_motion']:.4f} | {m['joint_range']:.4f} | "
                f"{100 * rel:.0f}% | {m['net_displacement']:.4f} | {m['straightness']:.3f} |"
            )
        L.append("")
        L.append(
            "The `vision_frames_*` conditions move at a **normal per-step rate** but sweep "
            "much less joint space — the arm meanders instead of committing to travel, so "
            "over ~29 policy calls it never reaches the object. That is mechanistically "
            "what a shortened imagined horizon should do: the imagination *is* the plan, so "
            "a model that can see only a few latent frames ahead commits to less "
            "displacement."
        )
        L.append("")
        # Quantify the separation instead of asserting it. `joint_range` overlaps
        # (seed_2 at 65% sits between the vision_frames conditions and the rest), but
        # straightness and net displacement do not — and it is worth reporting which
        # statistic actually discriminates rather than a vague "motion is lower".
        vf = {c: m for c, m in mot.items() if c.startswith("vision_frames_")}
        ref_like = {
            c: m
            for c, m in mot.items()
            if c == TEACHER or c.startswith("seed_") or c.startswith("steps_")
        }
        if vf and ref_like:
            for stat, label in (("straightness", "straightness"), ("net_displacement", "net displacement")):
                lo = min(m[stat] for m in ref_like.values())
                hi = max(m[stat] for m in vf.values())
                gap = "separates cleanly" if hi < lo else "overlaps"
                L.append(
                    f"- **{label}** {gap}: every baseline-like condition (teacher, seeds, "
                    f"reduced steps) is ≥ {lo:.3f}; every shortened-horizon condition is "
                    f"≤ {hi:.3f}"
                    + (f" — a {lo / hi:.1f}x gap with no overlap." if hi < lo else ".")
                )
            L.append("")
            L.append(
                "> **This is the most promising offline predictor found so far, and it "
                "ranks the conditions in the same order as the closed-loop outcomes we "
                "have**: `no_imagination_freeze` worst (0.026, and 100% gripper flips), "
                "the shortened-horizon conditions next (0.10–0.13, **0/9** closed-loop), "
                "reduced steps and seed variation clustered with the teacher (0.30–0.39, "
                "2/9 and 5/9). Note that `joint_range` alone does *not* separate them — "
                "`seed_2` sweeps less joint space than `vision_frames_17` yet succeeds — "
                "so the discriminating quantity is whether the chunk **gets somewhere**, "
                "not how much it moves."
            )
            L.append("")
            L.append(
                "It remains a *candidate* predictor on 3 closed-loop methods, not a "
                "validated screen, and it was found only after the closed-loop result "
                "contradicted the deviation-based screen. The durable lesson stands: "
                "**open-loop deviation from a teacher did not predict closed-loop success "
                "here.** It cannot capture error compounding across dozens of policy "
                "calls, and any future screen must be validated against task success "
                "before it is trusted to gate spending."
            )
            L.append("")

    b = r.get("chunk_boundary_discontinuity") or {}
    if b:
        L.append("## Chunk-boundary discontinuity (Experiment B input, spec §11.2)")
        L.append("")
        L.append(
            f"Across **{b['n_boundaries']}** consecutive-request boundaries, the jump "
            f"between the last action of one chunk and the first action of the next has "
            f"median **{b['median_jump_rad']:.4f} rad** (max {b['max_jump_rad']:.4f})."
        )
        if b.get("median_intra_chunk_step_rad"):
            L.append("")
            L.append(
                f"Normal step-to-step motion *within* a chunk is "
                f"{b['median_intra_chunk_step_rad']:.4f} rad, so the boundary jump is "
                f"**{b['jump_vs_intra_ratio']:.1f}x** a normal step."
            )
            L.append("")
            L.append(
                "This is what re-planning buys. "
                + (
                    "A jump comparable to ordinary intra-chunk motion means the policy "
                    "does *not* substantially change its mind at a boundary, so a longer "
                    "horizon is worth testing — the plan was already stable."
                    if (b["jump_vs_intra_ratio"] or 0) < 2.0
                    else "A jump much larger than ordinary motion means re-planning "
                    "materially changes the intention, so extending the horizon would "
                    "execute a stale plan and shortening it may be what buys success. "
                    "It also warns that any keyframe/chunk reuse must be validated "
                    "against contact-phase behaviour, not just averaged deviation."
                )
            )
        L.append("")
        L.append("| task | boundaries | median jump | max jump |")
        L.append("|---|---|---|---|")
        for task, d in sorted(b.get("per_task", {}).items()):
            L.append(
                f"| `{task}` | {d['n_boundaries']} | {d['median_jump']:.4f} | "
                f"{d['max_jump']:.4f} |"
            )
        L.append("")

    L.append("## Limitations")
    L.append("")
    L.append(
        "- **No task success here.** These are action-space deviations on recorded "
        "observations, open-loop. A small deviation can still compound into failure "
        "once the robot acts on it, and a large one can be harmless. Closed-loop "
        "evaluation is the arbiter; this study only decides what is worth evaluating."
    )
    if not r.get("ee_metrics_available"):
        L.append(
            "- **No end-effector-space deviation.** Spec §11.1 asks for EE translation "
            "and rotation error; both need forward kinematics from the Franka model, "
            "which lives in Isaac and not in the Cosmos venv these sweeps run in. Joint "
            "error is reported instead and is *not* a substitute — the same joint error "
            "at the wrist and at the shoulder move the gripper by very different "
            "amounts. Recorded as a gap rather than approximated."
        )
    L.append(
        "- **Deviation is measured open-loop from a fixed observation.** It does not "
        "capture error accumulation across an episode, which is precisely what "
        "closed-loop reuse experiments (B/C) are for."
    )
    L.append("")
    return "\n".join(L) + "\n"


if __name__ == "__main__":
    raise SystemExit(main())
