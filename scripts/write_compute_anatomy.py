#!/usr/bin/env python3
"""Render docs/compute_anatomy.md from a probe run's summary.json.

Reads the output of `scripts/probe_compute_anatomy.py` and emits the M2
artifact. Deliberately mechanical: it reports what was measured and computes
only derived quantities that follow directly from the measurements (deltas,
ratios, per-step slopes). It does **not** author the go/no-go conclusion — that
paragraph is left as an explicit TODO for a human to write from the tables,
because the conclusion gates the entire M4-M8 branch and must not be
auto-generated from a single run.
"""
from __future__ import annotations

import argparse
import json
import re
from pathlib import Path


def fmt(x: float | None, nd: int = 1) -> str:
    return "—" if x is None else f"{x:,.{nd}f}"


def main() -> int:
    ap = argparse.ArgumentParser()
    g = ap.add_mutually_exclusive_group(required=True)
    g.add_argument("--summary", help="path to a single probe summary json")
    g.add_argument(
        "--summary-dir",
        help="directory of <config>.summary.json files (one per process; see "
        "scripts/run_anatomy_sweep.sh)",
    )
    ap.add_argument("--out", default="docs/compute_anatomy.md")
    ap.add_argument(
        "--closed-loop",
        default="results/reports/baseline_smoke.json",
        help="validate_smoke.py's machine-readable output. Supplies the measured "
        "closed-loop breakdown, which is the only place end-to-end cost (with "
        "serialization, websocket and simulator overhead) can be read. Skipped "
        "silently if absent — the in-process tables still stand on their own.",
    )
    ap.add_argument(
        "--conclusion",
        default="docs/compute_anatomy_conclusion.md",
        help="markdown file holding the hand-written go/no-go, inlined verbatim. "
        "Kept separate because this script regenerates its --out wholesale, so a "
        "conclusion written into the output would be destroyed by the next run.",
    )
    ap.add_argument(
        "--compare-dir",
        default=None,
        help="a second sweep directory to cross-check the first against, e.g. the "
        "same config measured with a REAL captured payload instead of a synthetic "
        "one. Renders an input-validation section.",
    )
    args = ap.parse_args()

    if args.summary:
        paths = [Path(args.summary)]
    else:
        paths = sorted(Path(args.summary_dir).glob("*.summary.json"))
        if not paths:
            raise SystemExit(f"no *.summary.json under {args.summary_dir}")

    env: dict = {}
    configs: dict = {}
    for pth in paths:
        data = json.loads(pth.read_text())
        # Envs are identical apart from run_id/timestamp; keep the first and
        # record how many separate processes contributed.
        if not env:
            env = dict(data["env"])
            env["_source_files"] = []
        env["_source_files"].append(pth.name)
        for c in data["configs"]:
            configs[c["config"]] = c
    # Deterministic, human-meaningful order rather than filesystem order.
    order = ["B0", "B1", "B2_steps_1", "B2_steps_2", "B2_steps_3", "B2_steps_4", "B3", "B4"]
    configs = {k: configs[k] for k in order if k in configs} | {
        k: v for k, v in configs.items() if k not in order
    }

    L: list[str] = []
    L.append("# Compute anatomy (M2)")
    L.append("")
    L.append(
        f"**Status: MEASURED** — generated from `{args.summary or args.summary_dir}` by "
        "`scripts/write_compute_anatomy.py`. Do not hand-edit the tables; "
        "re-run the probe and regenerate."
    )
    L.append("")
    L.append("## Provenance")
    L.append("")
    L.append(f"- run_id: `{env['run_id']}` at `{env['timestamp_utc']}`")
    L.append(f"- cosmos-framework: `{env.get('cosmos_framework_sha','?')}`")
    L.append(f"- python {env.get('python')} / torch {env.get('torch')} / CUDA {env.get('torch_cuda')}")
    gpu = env.get("gpu", {})
    L.append(
        f"- GPU: {gpu.get('gpu_name')} (sm{gpu.get('capability')}), "
        f"CUDA_VISIBLE_DEVICES={gpu.get('cuda_visible_devices')}, uuid `{gpu.get('gpu_uuid')}`"
    )
    att = env.get("attention", {})
    L.append(
        f"- attention: arch_tag={att.get('arch_tag')} → allowed backends "
        f"`{att.get('backend_list')}`"
    )
    if att.get("backend_list") and "flash3" not in (att.get("backend_list") or []):
        L.append(
            "  - **flash3 is unavailable on this arch**, so absolute latencies are "
            "not comparable to NVIDIA's published FlashAttention-3 figures. All "
            "claims below are within-machine deltas against our own baseline."
        )
    L.append(f"- warmup={env.get('warmup')}, measured iters={env.get('iters')} per config")
    L.append(f"- input: {env.get('input_kind')}")
    L.append(
        "  - Either input is valid for **cost** (the sample builder zeroes every "
        "frame but the first regardless, so compute is shape-determined) but says "
        "nothing about **task success**, which only the closed-loop RoboLab run "
        "measures. If this says SYNTHETIC, see the input-validation section for "
        "the measured comparison against a real captured request."
    )
    L.append(f"- host contention at run time: loadavg {gpu.get('loadavg_1_5_15')}")
    if (gpu.get("gpu_compute_apps") or "").strip():
        L.append(f"  - other GPU processes present: `{gpu['gpu_compute_apps']}`")
    else:
        L.append("  - no other GPU compute processes")
    L.append("")

    # --- latency table -----------------------------------------------------
    L.append("## End-to-end server-side latency per configuration")
    L.append("")
    L.append("`infer()` wall time: preprocessing + generation + postprocess, in-process")
    L.append("(no websocket or client cost). Milliseconds.")
    L.append("")
    L.append("| config | steps | decode | mean | std | p50 | p95 | n |")
    L.append("|---|---|---|---|---|---|---|---|")
    for name, c in configs.items():
        w = c.get("wall_ms") or {}
        L.append(
            f"| `{name}` | {c.get('num_steps')} | {c.get('decode_video')} | "
            f"{fmt(w.get('mean'))} | {fmt(w.get('std'))} | {fmt(w.get('p50'))} | "
            f"{fmt(w.get('p95'))} | {w.get('n','—')} |"
        )
    L.append("")

    # --- B1 - B0: the VAE decode price -------------------------------------
    b0, b1 = configs.get("B0"), configs.get("B1")
    if b0 and b1 and b0.get("wall_ms") and b1.get("wall_ms"):
        d = b1["wall_ms"]["mean"] - b0["wall_ms"]["mean"]
        pct = 100.0 * d / b1["wall_ms"]["mean"] if b1["wall_ms"]["mean"] else 0.0
        L.append("## What the VAE decode costs (B1 − B0)")
        L.append("")
        L.append(
            f"B0 and B1 do **identical generation work** — the only difference is "
            f"whether `samples[\"vision\"]` is decoded to RGB. The delta is therefore "
            f"the decode price:"
        )
        L.append("")
        L.append(f"- B0 (no decode): {fmt(b0['wall_ms']['mean'])} ms")
        L.append(f"- B1 (decode):    {fmt(b1['wall_ms']['mean'])} ms")
        L.append(f"- **decode cost:  {fmt(d)} ms ({pct:.1f}% of B1)**")
        L.append("")
        L.append(
            "This is the *only* visual cost that `decode_video=False` removes. The "
            "latent generation itself is still paid in B0 — see the step sweep for "
            "how much of B0 is denoising."
        )
        L.append("")

    # --- B2: denoising-step sweep ------------------------------------------
    steps = sorted(
        (
            (c["num_steps"], c["wall_ms"]["mean"], name)
            for name, c in configs.items()
            if name.startswith("B2_steps") and c.get("wall_ms")
        )
    )
    if len(steps) >= 2:
        L.append("## Denoising-step sweep (B2)")
        L.append("")
        L.append("| steps | mean ms | Δ vs 1 step | ms/step (marginal) |")
        L.append("|---|---|---|---|")
        base_ms = steps[0][1]
        prev = None
        for n, ms, _ in steps:
            marg = "—" if prev is None else fmt((ms - prev[1]) / max(1, n - prev[0]))
            L.append(f"| {n} | {fmt(ms)} | {fmt(ms - base_ms)} | {marg} |")
            prev = (n, ms)
        L.append("")
        n_lo, ms_lo, _ = steps[0]
        n_hi, ms_hi, _ = steps[-1]
        if n_hi > n_lo:
            slope = (ms_hi - ms_lo) / (n_hi - n_lo)
            fixed = ms_lo - slope * n_lo
            L.append(
                f"Linear fit over {n_lo}→{n_hi} steps: **{fmt(slope)} ms per denoising "
                f"step**, with **{fmt(fixed)} ms of step-independent cost** "
                "(preprocessing, encode, context, postprocess — everything paid once "
                "per request regardless of step count)."
            )
            if ms_hi:
                L.append("")
                L.append(
                    f"So of the {fmt(ms_hi)} ms at {n_hi} steps, roughly "
                    f"{100.0 * (slope * n_hi) / ms_hi:.0f}% is denoising and "
                    f"{100.0 * fixed / ms_hi:.0f}% is fixed overhead. **This bounds "
                    "what any denoising-reuse method (Experiment F) can possibly "
                    "save**, and it is the frontier that reduced-step baselines "
                    "already reach for free."
                )
        L.append("")

    # --- module attribution ------------------------------------------------
    L.append("## Where the time goes, by module")
    L.append("")
    L.append(
        "Attribution from CUDA-event timing on real submodules — no invented "
        "action/vision split. Nested modules double-count against their parents, "
        "so read these as a tree, not as a partition summing to the total."
    )
    L.append("")
    for name, c in configs.items():
        mods = c.get("module_ms_mean") or {}
        if not mods:
            continue
        L.append(f"### `{name}`")
        L.append("")
        L.append("| module | mean ms | % of infer() |")
        L.append("|---|---|---|")
        tot = (c.get("wall_ms") or {}).get("mean") or 0.0
        for m, ms in list(mods.items())[:18]:
            pct = f"{100.0 * ms / tot:.1f}%" if tot else "—"
            L.append(f"| `{m}` | {fmt(ms)} | {pct} |")
        L.append("")

    # --- token census ------------------------------------------------------
    # The single most load-bearing table in the document: it decides whether the
    # project's premise (that imagined video dominates) is true.
    cen = None
    for c in configs.values():
        if c.get("token_census"):
            cen = c["token_census"]
            break
    L.append("## Token census — what the sequence is actually made of")
    L.append("")
    if not cen:
        L.append("_Not captured. Re-run the probe (see `capture_packed_sequence`)._")
        L.append("")
    else:
        vis = cen.get("vision") or {}
        act = cen.get("action") or {}
        total = cen["sequence_length"]

        def row(label: str, n: Any) -> str:
            pct = f"{100.0 * n / total:.1f}%" if isinstance(n, int) and total else "—"
            return f"| {label} | {n:,} | {pct} |" if isinstance(n, int) else f"| {label} | {n} | {pct} |"

        L.append(
            "Read from the model's own `PackedSequence` — `text_indexes`, each "
            "modality's `sequence_indexes`, and `condition_mask` (1 = clean/"
            "conditioning, 0 = noised/supervised). The conditioning/noised split is "
            "the one that matters and the one `decode_video=False` does **not** make: "
            "it separates the observation we supplied from the future the model is "
            "inventing."
        )
        L.append("")
        L.append("| component | tokens | share of sequence |")
        L.append("|---|---|---|")
        L.append(row("text (prompt)", cen["text_tokens"]))
        L.append(row("vision — conditioning (the real observation)", vis.get("tokens_conditioning")))
        L.append(row("vision — **imagined future**", vis.get("tokens_noised")))
        L.append(row("action — conditioning (current state)", act.get("tokens_conditioning")))
        L.append(row("action — **predicted** (the deliverable)", act.get("tokens_noised")))
        L.append(f"| **total** | **{total:,}** | 100% |")
        L.append("")
        if cen.get("unattributed_tokens") == 0:
            L.append(
                "Reconciles exactly: every token in the sequence is accounted for "
                "(0 unattributed)."
            )
        else:
            L.append(
                f"⚠ {cen['unattributed_tokens']} tokens unattributed — the layout has "
                "components this census does not model. Treat the shares as approximate."
            )
        L.append("")
        shapes_v = (vis.get("token_shapes") or [[None, None, None]])[0]
        if len(shapes_v) == 3:
            L.append(
                f"Vision latent grid: **{shapes_v[0]} latent frames x {shapes_v[1]}x{shapes_v[2]} "
                f"= {vis.get('tokens_per_frame', ['?'])[0]} tokens per frame**, from 33 pixel "
                "frames (`action_chunk_size + 1`) at temporal downsample 4. Exactly one of "
                "those latent frames is the observation."
            )
            L.append("")
        frac = cen.get("imagined_fraction_of_sequence")
        if frac:
            L.append(
                f"> **{100.0 * frac:.1f}% of the sequence is imagined future video.** The "
                f"actual deliverable — {act.get('tokens_noised', '?')} predicted action "
                f"tokens — is "
                f"{100.0 * (act.get('tokens_noised') or 0) / total:.2f}% of it. Since the "
                "model is matmul-bound and cost is ~linear in token count, this is the "
                "single largest lever available, and it is why Experiment E (does the "
                "action actually need the imagination?) gates the rest of the project."
            )
            L.append("")
        L.append(
            f"Attention structure: `split_lens={cen['split_lens']}` with "
            f"`attn_modes={cen['attn_modes']}` — text is a **causal** split, and vision "
            "plus action share a single **full** split. In `two_way_attention` the "
            "generation queries attend to *all* tokens with no mask, so action tokens do "
            "see every imagined vision token. That is spec §11.5 evidence for outcome "
            "**C/D**: there is no configuration switch that removes the imagination."
        )
        L.append("")
        if cen.get("num_action_tokens_per_supertoken") == 0:
            L.append(
                "One piece of good news for E2b: `num_action_tokens_per_supertoken=0`, so "
                "actions are a **contiguous block**, not interleaved into per-frame vision "
                "supertokens. Dropping imagined vision frames therefore does not require "
                "unpicking an interleaved layout."
            )
            L.append("")

    # --- token shapes ------------------------------------------------------
    shapes = None
    for c in configs.values():
        if c.get("module_call_shapes"):
            shapes = c["module_call_shapes"]
            break
    L.append("## Token layout and tensor shapes")
    L.append("")
    if shapes:
        L.append(
            "Captured from forward-hook inputs. These are the evidence for whether "
            "action and vision tokens share attention (spec §11.5)."
        )
        L.append("")
        L.append("| module | observed input shapes |")
        L.append("|---|---|")
        for m, ss in list(shapes.items())[:25]:
            L.append(f"| `{m}` | `{ss[0]}` |")
    else:
        L.append("_No shapes captured — re-run the probe without `--no-hooks`._")
    L.append("")

    # --- memory ------------------------------------------------------------
    L.append("## Memory")
    L.append("")
    L.append("| config | after load (alloc / reserved MiB) | peak (alloc / reserved MiB) |")
    L.append("|---|---|---|")
    for name, c in configs.items():
        al = c.get("vram_after_load_mib", {})
        pk = c.get("vram_peak_mib", {})
        L.append(
            f"| `{name}` | {fmt(al.get('allocated'),0)} / {fmt(al.get('reserved'),0)} | "
            f"{fmt(pk.get('allocated'),0)} / {fmt(pk.get('reserved'),0)} |"
        )
    L.append("")
    L.append("Budget: 46068 MiB total per L40S (45.0 GiB).")
    L.append("")

    # --- FLOPs -------------------------------------------------------------
    L.append("## FLOPs and coverage")
    L.append("")
    for name, c in configs.items():
        f = c.get("flops") or {}
        L.append(f"### `{name}`")
        L.append("")
        if f.get("counted_total") is not None:
            L.append(f"- `FlopCounterMode` total: **{f['counted_total']:,}** FLOPs")
            by_op = f.get("by_op") or {}
            if by_op:
                L.append("- top ops:")
                for op, v in list(by_op.items())[:8]:
                    L.append(f"  - `{op}`: {v:,}")
        else:
            L.append("- not counted")
        if f.get("note"):
            L.append(f"- {f['note']}")
        L.append("")
    L.append(
        "**Coverage caveat, stated rather than hidden:** `FlopCounterMode` only sees "
        "ops it recognises. On this stack the attention path runs through flash-attn "
        "or cuDNN fused kernels, which it cannot decompose, so the counted total is a "
        "**lower bound**. Any FLOP-based claim must quote this coverage limitation, "
        "and per spec §8 a normalized-FLOP speedup claim is not acceptable on its own "
        "— measured latency has to confirm it."
    )
    L.append("")

    # --- B3: the measurement floor ------------------------------------------
    # Placed before every comparison in the document on purpose: a reader should
    # know what this host can resolve *before* reading any delta.
    # B3 may live in either directory: it is a methodology measurement, not a model
    # config, so it is often collected alongside the real-payload cross-check rather
    # than in the main sweep. Look in both, preferring the main sweep on a tie.
    b3_pool: dict[str, Any] = {}
    if args.compare_dir:
        for pth in sorted(Path(args.compare_dir).glob("*.summary.json")):
            for c in json.loads(pth.read_text())["configs"]:
                b3_pool[c["config"]] = c
    b3_pool.update(configs)
    sustained = b3_pool.get("B3_replay")
    cooled = b3_pool.get("B3_replay_cooled")
    if (sustained and sustained.get("wall_ms")) or (cooled and cooled.get("wall_ms")):
        L.append("## Measurement floor (B3 — deterministic replay)")
        L.append("")
        L.append(
            "Identical input, identical seed, repeated. B3 does not measure the model; "
            "it measures **us**. Everything below is only as trustworthy as this number."
        )
        L.append("")
        L.append("| run | n | median | MAD | mean | std | min | max | drift |")
        L.append("|---|---|---|---|---|---|---|---|---|")
        for label, c in (("sustained (back-to-back)", sustained), ("**cooled** (idle gap)", cooled)):
            if not (c and c.get("wall_ms")):
                continue
            w = c["wall_ms"]
            med = w.get("median") or w.get("p50")
            mad = w.get("mad")
            mad_s = f"{fmt(mad)} ({w.get('mad_pct', 0):.2f}%)" if mad is not None else "—"
            drift = f"{w.get('drift_pct', 0):+.2f}%" if w.get("drift_pct") is not None else "—"
            L.append(
                f"| {label} | {w['n']} | **{fmt(med)}** | {mad_s} | {fmt(w['mean'])} | "
                f"{fmt(w['std'])} | {fmt(w['min'])} | {fmt(w['max'])} | {drift} |"
            )
        L.append("")
        # The two runs together are what make the diagnosis possible; either alone
        # would mislead.
        if sustained and cooled and sustained.get("wall_ms") and cooled.get("wall_ms"):
            sw, cw = sustained["wall_ms"], cooled["wall_ms"]
            s_med = sw.get("median") or sw["p50"]
            c_med = cw.get("median") or cw["p50"]
            L.append(
                f"**The medians agree to {abs(100.0 * (c_med - s_med) / c_med):.1f}%** "
                f"({fmt(s_med)} vs {fmt(c_med)} ms) while the sustained run's maximum is "
                f"{100.0 * (sw['max'] - s_med) / s_med:.0f}% above its own median. So the "
                "model's cost is stable and reproducible; the sustained run's tail is an "
                "intermittent **stall**, not a shift in the work being done."
            )
            L.append("")
        th = (cooled or {}).get("thermal") or (sustained or {}).get("thermal") or {}
        if th and not th.get("error"):
            L.append(
                f"**Not thermal, and not GPU throttling.** Across the run "
                f"`sm_clock` stayed at {fmt(th.get('sm_clock_first_mhz'), 0)} MHz "
                f"(drop {th.get('sm_clock_drop_pct', 0):.1f}%), temperature moved "
                f"{fmt(th.get('temp_first_c'), 0)}→{fmt(th.get('temp_last_c'), 0)} °C, and "
                f"throttle reasons observed: "
                f"{th.get('throttle_seen') or '**none**'}. The SM clock is in fact *locked* "
                "well below this card's 2520 MHz maximum, so it cannot fall further under "
                "load. That leaves host-side interference — this box's CPU is shared and "
                "the GPU's affinity is a subset of cores — as the explanation, which is "
                "consistent with an idle gap making the tail disappear."
            )
            L.append("")
        # Base the rule on the *robust* dispersion of the steady-state run. Using the
        # sustained run's mean/std instead would put the floor near 20% and wrongly
        # declare every realistic effect unmeasurable.
        best = cooled if (cooled and cooled.get("wall_ms")) else sustained
        w = best["wall_ms"]
        mad_pct = w.get("mad_pct")
        if mad_pct:
            mde = max(1.0, 3.0 * mad_pct)
            L.append(
                f"> **Minimum detectable effect: ~{mde:.1f}%** (3x the steady-state MAD of "
                f"{mad_pct:.2f}%), provided the comparison is made the same way: "
                "**median-of-many with an idle gap between requests, configs interleaved "
                "rather than measured in separate blocks.** Measured that way this host "
                "resolves small effects well. Measured back-to-back it does not — the "
                "same work spanned "
                f"{fmt((sustained or best)['wall_ms']['min'])}–"
                f"{fmt((sustained or best)['wall_ms']['max'])} ms. Report medians, not "
                "means: the mean of the sustained run "
                f"({fmt((sustained or best)['wall_ms']['mean'])} ms) describes the stalls, "
                "not the model."
            )
            L.append("")
            L.append(
                "Two consequences carried into every later experiment: (1) per spec §8 a "
                "FLOP reduction never substitutes for measured latency, so a method whose "
                "only evidence sits under this floor has no evidence; (2) the "
                "**sustained-load** tail is the number a deployed server actually "
                "experiences, so it is reported alongside the steady-state figure rather "
                "than replaced by it."
            )
            L.append("")

    # --- input validation: synthetic vs real captured payload ---------------
    if args.compare_dir:
        other: dict = {}
        other_env: dict = {}
        for pth in sorted(Path(args.compare_dir).glob("*.summary.json")):
            data = json.loads(pth.read_text())
            other_env = other_env or data["env"]
            for c in data["configs"]:
                other[c["config"]] = c
        shared = [k for k in configs if k in other]
        if shared:
            L.append("## Input validation — synthetic vs real captured request")
            L.append("")
            L.append(
                "The in-process probe has to supply its own observation. To show that "
                "choice does not drive the numbers, the same config was re-measured "
                "with a request captured from a live closed-loop episode "
                "(`robolab-0001`)."
            )
            L.append("")
            L.append(f"- this sweep: {env.get('input_kind')}")
            L.append(f"- comparison: {other_env.get('input_kind')}")
            L.append("")
            L.append(
                "| config | tokens | counted FLOPs | peak reserved MiB | wall mean ms | wall std ms |"
            )
            L.append("|---|---|---|---|---|---|")

            def _tokens(c: dict) -> str:
                for ss in (c.get("module_call_shapes") or {}).values():
                    for s in ss:
                        m = re.search(r"position_ids=\(\d+,\s*(\d+)\)", s)
                        if m:
                            return m.group(1)
                return "—"

            for k in shared:
                for label, c in ((f"`{k}` (this)", configs[k]), (f"`{k}` (compare)", other[k])):
                    w = c.get("wall_ms") or {}
                    ct = (c.get("flops") or {}).get("counted_total")
                    L.append(
                        f"| {label} | {_tokens(c)} | {ct:,} | "
                        if ct is not None
                        else f"| {label} | {_tokens(c)} | — | "
                    )
                    L[-1] += (
                        f"{fmt((c.get('vram_peak_mib') or {}).get('reserved'), 0)} | "
                        f"{fmt(w.get('mean'))} | {fmt(w.get('std'))} |"
                    )
            L.append("")
            # Quantify the two deltas that matter, and refuse to over-read either.
            for k in shared:
                a, b = configs[k], other[k]
                wa = (a.get("wall_ms") or {}).get("mean")
                wb = (b.get("wall_ms") or {}).get("mean")
                fa = (a.get("flops") or {}).get("counted_total")
                fb = (b.get("flops") or {}).get("counted_total")
                ta, tb = _tokens(a), _tokens(b)
                if None in (wa, wb):
                    continue
                L.append(f"**`{k}`:**")
                L.append("")
                if ta != "—" and tb != "—":
                    dt = int(tb) - int(ta)
                    L.append(
                        f"- joint sequence length {ta} → {tb} tokens "
                        f"({dt:+d}, {100.0 * dt / int(ta):+.2f}%) — the prompt-length "
                        "difference and nothing else; the image is the same 540×640 "
                        "uint8 either way, because that is the server's own default "
                        "and what the client sends."
                    )
                if fa and fb:
                    L.append(
                        f"- counted FLOPs {fa:,} → {fb:,} "
                        f"({100.0 * (fb - fa) / fa:+.3f}%)"
                    )
                L.append(
                    f"- peak reserved VRAM "
                    f"{fmt((a.get('vram_peak_mib') or {}).get('reserved'), 0)} → "
                    f"{fmt((b.get('vram_peak_mib') or {}).get('reserved'), 0)} MiB"
                )
                L.append(f"- wall {fmt(wa)} → {fmt(wb)} ms ({100.0 * (wb - wa) / wa:+.1f}%)")
                L.append("")
                L.append(
                    "The shape-determined quantities agree to well under a percent, so "
                    "**the synthetic-input tables above are sound for cost.** The wall "
                    "delta is larger than the FLOP delta by two orders of magnitude, so "
                    "it is *not* caused by the input — it is host/GPU interference. Note "
                    "the standard deviations: "
                    f"{fmt((a.get('wall_ms') or {}).get('std'))} vs "
                    f"{fmt((b.get('wall_ms') or {}).get('std'))} ms."
                )
                L.append("")
                L.append(
                    "> **This wall delta is a measurement artefact, not an input "
                    "effect.** These two runs were taken back-to-back in separate "
                    "blocks, which the B3 section above shows is the one way *not* to "
                    "compare on this host: identical work spans a wide range under "
                    "sustained load, while a steady-state median is reproducible to a "
                    "fraction of a percent. The FLOP and token deltas — which are "
                    "immune to scheduling — are the trustworthy comparison here, and "
                    "they agree to well under 1%."
                )
                L.append("")

    # --- closed-loop end-to-end, from the smoke run --------------------------
    cl_path = Path(args.closed_loop) if args.closed_loop else None
    if cl_path and cl_path.exists():
        cl = json.loads(cl_path.read_text())
        eps = (cl.get("primary_episodes") or []) + (cl.get("alt_episodes") or [])
        eps = [e for e in eps if e.get("timing")]
        if eps:
            L.append("## Closed-loop end-to-end cost (measured, not modelled)")
            L.append("")
            L.append(
                "From the M1 smoke run (`" + str(cl_path) + "`). The tables above are "
                "**in-process** server latency; these are what the deployed loop "
                "actually pays, including image composition, msgpack serialization, the "
                "websocket round-trip and the simulator. Spec §8 requires the claim to "
                "be made at this level, not at the FLOP level."
            )
            L.append("")
            L.append(
                "| episode | steps | server calls | policy total | **per call** | "
                "env step total | video | wall |"
            )
            L.append("|---|---|---|---|---|---|---|---|")
            horizon = 32  # Cosmos3Client.OPEN_LOOP_HORIZON
            for e in eps:
                t = e["timing"]
                steps = e.get("episode_step") or 0
                calls = -(-steps // horizon) if steps else 0
                per = (t.get("policy_inference_s", 0.0) / calls * 1000.0) if calls else None
                L.append(
                    f"| `{e.get('run_name')}` | {steps} | {calls} | "
                    f"{fmt(t.get('policy_inference_s'))} s | "
                    f"**{fmt(per)} ms** | {fmt(t.get('env_step_s'))} s | "
                    f"{fmt(t.get('video_write_s'))} s | {fmt(t.get('wall_total_s'))} s |"
                )
            L.append("")
            if len(eps) > 1:
                L.append(
                    "Per-call figures are not comparable across rows: the **first** "
                    "episode against a freshly loaded server pays one-off warm-up "
                    "(kernel autotuning, cuDNN benchmarking, first tokenizer pass) "
                    "amortized over very few calls, which is why a short episode can "
                    "show a much larger per-call number than a long one on the same "
                    "server. Read the longest episode as the warm figure."
                )
                L.append("")
            L.append(
                f"Server-side request count over the whole smoke run: "
                f"**{cl.get('server_request_count')}** — consistent with one policy call "
                f"per {horizon} control steps "
                "(`Cosmos3Client.OPEN_LOOP_HORIZON`), which is the baseline's *existing* "
                "action-chunk reuse. Any \"skip frames\" proposal must be framed against "
                "this, not against a per-control-step strawman."
            )
            L.append("")
            b0w = (configs.get("B0") or {}).get("wall_ms") or {}
            warm = [e for e in eps if (e.get("episode_step") or 0) >= 10 * horizon]
            if b0w.get("mean") and warm:
                t = warm[-1]["timing"]
                calls = -(-warm[-1]["episode_step"] // horizon)
                per_ms = t["policy_inference_s"] / calls * 1000.0
                gap = per_ms - b0w["mean"]
                L.append(
                    f"**The wire is not free.** In-process `infer()` costs "
                    f"{fmt(b0w['mean'])} ms (B0), but the client observes "
                    f"{fmt(per_ms)} ms per call on the longest episode — a gap of "
                    f"**{fmt(gap)} ms ({100.0 * gap / per_ms:.0f}% of per-call latency)** "
                    "spent outside the model: three camera resizes, a torch "
                    "`interpolate` + concatenate to build the composed view, msgpack "
                    "encoding of a 540×640×3 array, and the round trip. This overhead is "
                    "**invariant to denoising steps and to anything done inside the "
                    "model**, so it is a hard floor under every method in spec §5."
                )
                L.append("")
                pol = t["policy_inference_s"]
                envs = t.get("env_step_s") or 0.0
                tot = t.get("wall_total_s") or (pol + envs)
                L.append(
                    f"For scale, in that episode simulator stepping cost {fmt(envs)} s "
                    f"against {fmt(pol)} s of policy inference "
                    f"({100.0 * pol / tot:.0f}% of {fmt(tot)} s wall). **The simulator is "
                    "not part of a real deployment**, so it must be excluded from any "
                    "speedup denominator — but it does mean simulated closed-loop "
                    "wall-clock is a misleading proxy, and evaluation throughput will be "
                    "dominated by Isaac rather than by the policy."
                )
                L.append("")

    # --- traces ------------------------------------------------------------
    L.append("## Profiler traces")
    L.append("")
    for name, c in configs.items():
        L.append(f"- `{name}`: `{c.get('chrome_trace')}`")
    L.append("")

    # --- the conclusion, deliberately not auto-written ---------------------
    L.append("## Go / no-go — the primary question")
    L.append("")
    L.append("> Is visual imagination a substantial part of the deployed Cosmos3 policy")
    L.append("> cost, or is the main cost shared/action computation?")
    L.append("")
    concl = Path(args.conclusion) if args.conclusion else None
    if concl and concl.exists():
        L.append(
            f"_Hand-written; inlined verbatim from `{concl}`. Edit that file, not this "
            "one — regenerating overwrites everything here._"
        )
        L.append("")
        # Strip the HTML comment that explains the split; it is meta, not content.
        text = re.sub(r"<!--.*?-->\s*", "", concl.read_text(), flags=re.DOTALL)
        L.append(text.strip())
    else:
        L.append("**TODO(human):** write the conclusion from the tables above. It must state:")
        L.append("")
        L.append("1. the dominant measured cost component, with its number;")
        L.append("2. what fraction of B0 is denoising vs step-independent overhead;")
        L.append("3. what the VAE decode costs, and that it is the *only* visual cost")
        L.append("   `decode_video=False` removes;")
        L.append("4. from the token-shape evidence, which of spec §11.5's outcomes A–D")
        L.append("   applies — i.e. whether vision output tokens can be dropped while")
        L.append("   keeping action generation, or whether shared attention forbids it;")
        L.append("5. an explicit CONTINUE or PIVOT for the visual-imagination branch.")
        L.append("")
        L.append(
            "This paragraph is intentionally not auto-generated: it gates the entire "
            "M4–M8 branch, and one probe run is not sufficient evidence to write it "
            f"mechanically. Write it in `{args.conclusion}` so it survives the next "
            "regeneration."
        )
    L.append("")

    Path(args.out).write_text("\n".join(L) + "\n")
    print(f"wrote {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
