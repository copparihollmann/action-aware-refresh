#!/usr/bin/env python3
"""Validate smoke-test logs against spec §7.3 pass criteria and write a report.

Design rule: **never claim a criterion passed from silence, and never claim it
failed from ignorance.** Each check resolves to one of three states:

- ``PASS``    positive evidence found
- ``FAIL``    counter-evidence found (e.g. a NaN complaint, a traceback)
- ``UNKNOWN`` neither; the log does not speak to this criterion

An earlier version of this file scored every criterion with an invented regex
(including a hardcoded ``action_dim 8`` and a dead ``r"(?!...)"`` lookahead that
always matched) and required all of them to be true, so a genuinely successful
run would still have reported FAIL. Every pattern here now carries its
provenance and was read out of the pinned upstream source, then confirmed
against the logs of the first successful closed-loop run (2026-08-03). Overall
status is FAIL if anything FAILs, INCOMPLETE if anything required is UNKNOWN,
else PASS — so an unproven criterion can never masquerade as a pass.

Where a criterion can be settled from structured output, it is: the strong
checks read ``episode_results.jsonl`` (step count, termination, trajectory
metrics) rather than grepping stdout. Two of the original regexes were not just
weak but actively wrong — they matched banner text printed *before* the
simulator stepped, and the word "success" in the Termination-Manager table that
is printed on every run regardless of outcome.

One distinction this file is careful about: **task success is a result, not a
pass criterion.** A smoke test verifies the loop is wired up and honest; a robot
that fails the task while reporting so correctly is a passing smoke test.
"""
from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

PASS, FAIL, UNKNOWN = "PASS", "FAIL", "UNKNOWN"

# ---------------------------------------------------------------------------
# VERIFIED against third_party/cosmos-framework @ a904d2d:
#   cosmos_framework/scripts/action_policy_server_robolab.py
# The server logs a fixed "[robolab-policy-server] ..." prefix; `serve()` emits
# a startup line and a health-check line, and `infer()` logs prompt+seed per
# request. These are real strings from that file, not guesses.
# ---------------------------------------------------------------------------
SERVER_POSITIVE = {
    "server_started": (
        r"\[robolab-policy-server\] starting host=",
        "VERIFIED: serve() startup line",
    ),
    "healthz_advertised": (
        r"\[robolab-policy-server\] Health check: http://",
        "VERIFIED: serve() health-check line",
    ),
    "config_logged": (
        r"action_space=\S+\s+action_dim=(\d+)",
        "VERIFIED: RobolabPolicyService logs the resolved config",
    ),
    "request_served": (
        r"\[robolab-policy-server\] prompt=",
        "VERIFIED: infer() logs prompt+seed per request",
    ),
}

# Counter-evidence. Any hit here is a hard FAIL for the run.
SERVER_NEGATIVE = {
    "no_nans": (
        r"\bnan\b(?!o)|\bNaN\b|\binf\b(?=[^a-z])",
        "numerical failure in the served output",
    ),
    "no_traceback": (r"Traceback \(most recent call last\)", "server exception"),
    "no_cuda_oom": (
        r"CUDA out of memory|torch\.OutOfMemoryError",
        "OOM — expected risk at 45 GiB with ~31 GiB of weights",
    ),
}

# ---------------------------------------------------------------------------
# VERIFIED against third_party/RoboLab @ 0aef241 AND against the logs of the
# first successful closed-loop run (2026-08-03, BananaInBowlTask).
#
# These replace the original HEURISTIC guesses, which were loose enough to be
# meaningless: `sim_advanced` matched "episode 0" in a banner printed before the
# simulator ever stepped, and `episode_terminated` matched the word "success"
# anywhere — including the Termination-Manager table that lists a term *named*
# "success" during setup, on every run, successful or not. Both would have
# reported PASS for a run that produced no motion at all.
#
# The strong checks now come from the structured results file, not from prose;
# see check_episode_results(). What stays here is only what a log line can
# honestly establish.
# ---------------------------------------------------------------------------
CLIENT_POSITIVE = {
    "client_connected": (
        r"\[Cosmos3Client\] Connected to \S+",
        "VERIFIED: policies/cosmos3/client.py __init__ prints this after _connect()",
    ),
    "episode_started": (
        r"\[RoboLab\] Running \S+: ",
        "VERIFIED: robolab/eval/runner.py:251",
    ),
    "output_dir_announced": (
        r"Output\s+:\s+(\S+)",
        "VERIFIED: robolab/core/utils/print_utils.py:30",
    ),
}

CLIENT_NEGATIVE = {
    "no_traceback": (r"Traceback \(most recent call last\)", "client exception"),
    "no_terminated_with_error": (
        r"\[RoboLab\] Terminated with error",
        "VERIFIED: policies/cosmos3/run.py:56 prints this on any exception",
    ),
}

# Isaac Sim emits these on *every* run, including the successful one, during
# normal teardown. They read like failures and are not; matching them as
# counter-evidence would fail every run forever.
BENIGN_TEARDOWN_NOISE = (
    "Could not find category 'Replicator:",
    "USD stage detach not called",
    "Recursive unloadAllPlugins() detected",
    "outstanding SimStageWithHistory",
    "carb.glinterop.plugin",  # no windowing in headless
    "cuDeviceGetUuid",  # warp probe against this driver
)


def read(path: Path) -> str:
    return path.read_text(errors="replace") if path.exists() else ""


def meaningful_tail(text: str, n: int = 30) -> str:
    """Last `n` lines worth reading: no progress-bar redraws, no benign noise.

    tqdm rewrites one line with \\r hundreds of times, so a naive tail is 40
    identical progress frames; Isaac then prints dozens of teardown warnings that
    look alarming and mean nothing. Between them they pushed the real cause of a
    failure off the end of every earlier report.
    """
    keep: list[str] = []
    for raw in text.splitlines():
        # tqdm redraws share one physical line — take only the final frame.
        line = raw.split("\r")[-1].rstrip()
        if not line:
            continue
        if any(s in line for s in BENIGN_TEARDOWN_NOISE):
            continue
        # A bare progress frame carries no information; a frame with a message
        # glued after it does, so only drop the pure ones.
        if re.fullmatch(r"\s*\d+%\|.*?\|\s*\d+/\d+ \[[^\]]*\]\s*", line):
            continue
        keep.append(line)
    return "\n".join(keep[-n:])


def evaluate(
    text: str,
    positive: dict[str, tuple[str, str]],
    negative: dict[str, tuple[str, str]],
) -> dict[str, dict[str, str]]:
    out: dict[str, dict[str, str]] = {}
    if not text.strip():
        for name, (_, prov) in {**positive, **negative}.items():
            out[name] = {"state": UNKNOWN, "note": "log empty or missing", "provenance": prov}
        return out
    for name, (rx, prov) in positive.items():
        m = re.search(rx, text, re.IGNORECASE)
        out[name] = {
            "state": PASS if m else UNKNOWN,
            "note": (m.group(0)[:120] if m else "no matching line"),
            "provenance": prov,
        }
    for name, (rx, why) in negative.items():
        m = re.search(rx, text, re.IGNORECASE)
        out[name] = {
            "state": FAIL if m else PASS,
            "note": (m.group(0)[:120] if m else "none found"),
            "provenance": f"counter-evidence: {why}",
        }
    return out


def find_output_dir(client_text: str) -> Path | None:
    """Recover the run's output directory from the client log.

    VERIFIED: print_utils.py:30 emits ``  Output         : <output_dir>``.
    """
    m = re.search(r"^\s*Output\s+:\s+(\S+)", client_text, re.MULTILINE)
    return Path(m.group(1)) if m else None


def check_episode_results(client_text: str) -> tuple[dict[str, dict[str, str]], list[dict]]:
    """Score the criteria that structured output can settle, not prose.

    ``episode_results.jsonl`` (VERIFIED: robolab/core/logging/results.py:606) is
    written per episode by the runner and is the source of truth for success,
    step count, termination reason and timing. Reading it is strictly better
    than regexing stdout: "the sim advanced" becomes ``episode_step > 0``
    instead of a progress-bar substring, and "termination recorded" becomes an
    actual boolean plus a reason string.

    Returns (checks, episode records).
    """
    prov = "VERIFIED: episode_results.jsonl (robolab/core/logging/results.py:606)"
    out_dir = find_output_dir(client_text)
    if out_dir is None:
        return {
            k: {"state": UNKNOWN, "note": "no `Output :` line in client log", "provenance": prov}
            for k in ("results_written", "sim_advanced", "episode_terminated", "metrics_finite")
        }, []

    results_file = out_dir / "episode_results.jsonl"
    if not results_file.exists():
        return {
            "results_written": {
                "state": FAIL,
                "note": f"{results_file} does not exist",
                "provenance": prov,
            },
            **{
                k: {"state": UNKNOWN, "note": "no results file to read", "provenance": prov}
                for k in ("sim_advanced", "episode_terminated", "metrics_finite")
            },
        }, []

    eps: list[dict] = []
    for line in results_file.read_text().splitlines():
        line = line.strip()
        if line:
            try:
                eps.append(json.loads(line))
            except json.JSONDecodeError as exc:
                return {
                    "results_written": {
                        "state": FAIL,
                        "note": f"{results_file} has unparseable JSON: {exc}",
                        "provenance": prov,
                    },
                    **{
                        k: {"state": UNKNOWN, "note": "results unreadable", "provenance": prov}
                        for k in ("sim_advanced", "episode_terminated", "metrics_finite")
                    },
                }, []

    checks: dict[str, dict[str, str]] = {
        "results_written": {
            "state": PASS if eps else FAIL,
            "note": f"{len(eps)} episode record(s) in {results_file}",
            "provenance": prov,
        }
    }

    # The sim genuinely advanced: a positive step count, from the record itself.
    steps = [e.get("episode_step") for e in eps]
    advanced = [s for s in steps if isinstance(s, int) and s > 0]
    checks["sim_advanced"] = {
        "state": PASS if advanced else (UNKNOWN if not eps else FAIL),
        "note": f"episode_step = {steps}",
        "provenance": prov,
    }

    # Termination recorded: `success` present as a bool AND a reason given.
    # NOTE success=false is still a PASS here — this criterion is "the outcome
    # was recorded", not "the robot did the task". Task success is a *result*,
    # not a smoke-test pass criterion; conflating them would make the smoke test
    # fail whenever the policy merely performed badly.
    terminated = [
        e for e in eps if isinstance(e.get("success"), bool) and e.get("reason") is not None
    ]
    checks["episode_terminated"] = {
        "state": PASS if len(terminated) == len(eps) and eps else UNKNOWN,
        "note": "; ".join(
            f"success={e.get('success')} step={e.get('episode_step')} "
            f"score={e.get('score')} reason={e.get('reason')!r}"
            for e in eps
        )
        or "no records",
        "provenance": prov,
    }

    # No NaNs, checked on the numbers the runner actually computed from the
    # trajectory (spec §7.3 "no NaNs"), rather than by grepping for the word.
    bad: list[str] = []
    for e in eps:
        for key, val in (e.get("metrics") or {}).items():
            if not isinstance(val, (int, float)) or val != val or val in (float("inf"), float("-inf")):
                bad.append(f"{e.get('run_name')}.{key}={val}")
    checks["metrics_finite"] = {
        "state": FAIL if bad else (PASS if any(e.get("metrics") for e in eps) else UNKNOWN),
        "note": ", ".join(bad) if bad else "all trajectory metrics finite",
        "provenance": prov,
    }
    return checks, eps


def timing_table(eps: list[dict]) -> list[str]:
    """Render the runner's own closed-loop timing breakdown.

    This is the first end-to-end evidence we have, and it is the number spec §7
    cares about: total compute with all overhead counted, not GPU time alone.
    """
    rows = [
        "| run | steps | policy inference | env step | video write | wall | it/s |",
        "|---|---|---|---|---|---|---|",
    ]
    any_timing = False
    for e in eps:
        t = e.get("timing") or {}
        if not t:
            continue
        any_timing = True
        rows.append(
            "| `{run}` | {steps} | {pi:.1f} s ({pia:.0f} ms/step) | "
            "{es:.1f} s ({esa:.0f} ms/step) | {vw:.1f} s | {wt:.1f} s | {ips} |".format(
                run=e.get("run_name"),
                steps=e.get("episode_step"),
                pi=t.get("policy_inference_s", float("nan")),
                pia=t.get("policy_inference_avg_ms", float("nan")),
                es=t.get("env_step_s", float("nan")),
                esa=t.get("env_step_avg_ms", float("nan")),
                vw=t.get("video_write_s", float("nan")),
                wt=t.get("wall_total_s", float("nan")),
                ips=t.get("it_per_sec"),
            )
        )
    return rows if any_timing else []


def extract_action_dim(server_text: str) -> str | None:
    """Read the action_dim the server actually reported, rather than asserting 8."""
    m = re.search(r"action_dim=(\d+)", server_text)
    return m.group(1) if m else None


GLYPH = {PASS: "✅", FAIL: "❌", UNKNOWN: "❔"}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--server-log", required=True)
    ap.add_argument("--primary-log", required=True)
    ap.add_argument("--alt-log", required=True)
    ap.add_argument("--primary-task", required=True)
    ap.add_argument("--alt-task", default="(unspecified)")
    ap.add_argument("--alt-status", default="(unspecified)")
    ap.add_argument("--report", required=True)
    ap.add_argument(
        "--expect-action-dim",
        default="8",
        help="Contract value to compare against what the server reports. "
        "Mismatch is a FAIL; absence is UNKNOWN.",
    )
    args = ap.parse_args()

    server_text = read(Path(args.server_log))
    primary_text = read(Path(args.primary_log))
    alt_text = read(Path(args.alt_log))

    server_r = evaluate(server_text, SERVER_POSITIVE, SERVER_NEGATIVE)
    primary_r = evaluate(primary_text, CLIENT_POSITIVE, CLIENT_NEGATIVE)
    alt_r = evaluate(alt_text, CLIENT_POSITIVE, CLIENT_NEGATIVE)

    # Structured checks override anything a regex could say about these.
    primary_struct, primary_eps = check_episode_results(primary_text)
    alt_struct, alt_eps = check_episode_results(alt_text)
    primary_r.update(primary_struct)
    alt_r.update(alt_struct)

    # action_dim is compared, not assumed.
    got_dim = extract_action_dim(server_text)
    if got_dim is None:
        server_r["action_dim_matches_contract"] = {
            "state": UNKNOWN,
            "note": "server did not log action_dim=",
            "provenance": "compared against docs/baseline_contract.md",
        }
    else:
        ok = got_dim == str(args.expect_action_dim)
        server_r["action_dim_matches_contract"] = {
            "state": PASS if ok else FAIL,
            "note": f"server reported action_dim={got_dim}, contract says {args.expect_action_dim}",
            "provenance": "compared against docs/baseline_contract.md",
        }

    required = {**server_r, **primary_r}
    any_fail = any(v["state"] == FAIL for v in required.values())
    any_unknown = any(v["state"] == UNKNOWN for v in required.values())
    status = FAIL if any_fail else ("INCOMPLETE" if any_unknown else PASS)

    def table(results: dict[str, dict[str, str]]) -> list[str]:
        rows = ["| check | state | detail | provenance |", "|---|---|---|---|"]
        for k, v in results.items():
            note = v["note"].replace("|", "\\|").replace("\n", " ")
            rows.append(f"| `{k}` | {GLYPH.get(v['state'], '')} {v['state']} | {note} | {v['provenance']} |")
        return rows

    lines: list[str] = []
    lines.append(f"# Baseline smoke — **{status}**\n")
    lines.append(f"- Primary task: `{args.primary_task}`")
    lines.append(f"- Alt task: `{args.alt_task}` → {args.alt_status}")
    lines.append(
        "\n> `INCOMPLETE` means at least one criterion had no evidence either way. "
        "It is deliberately not `PASS`: spec §7.3 requires positive evidence for "
        "every criterion. Resolve UNKNOWNs by reading the real log and tightening "
        "the pattern in `scripts/validate_smoke.py`.\n"
    )

    lines.append("## Server checks\n")
    lines += table(server_r)
    lines.append("\n## Primary client checks\n")
    lines += table(primary_r)
    lines.append("\n## Alt-task client checks (not required for PASS)\n")
    lines += table(alt_r)

    for label, eps in (("Primary", primary_eps), ("Alt", alt_eps)):
        rows = timing_table(eps)
        if rows:
            lines.append(f"\n## {label} closed-loop timing (runner-reported)\n")
            lines += rows
            lines.append(
                "\n> Reported by RoboLab itself, so these are end-to-end wall times "
                "including serialization and the websocket round-trip — not GPU time. "
                "`policy inference` is the client's total time in the policy step "
                "averaged over **all** control steps, most of which reuse a cached "
                "chunk (`Cosmos3Client.OPEN_LOOP_HORIZON = 32`); divide by the "
                "server-side request count for true per-call latency.\n"
            )

    # Outcome is reported, never scored: see the note in check_episode_results.
    if primary_eps or alt_eps:
        lines.append("\n## Task outcome (a result, not a pass criterion)\n")
        for e in primary_eps + alt_eps:
            lines.append(
                f"- `{e.get('run_name')}` — success=**{e.get('success')}**, "
                f"score={e.get('score')}, {e.get('episode_step')} steps, "
                f"events={e.get('events')}, reason={e.get('reason')!r}"
            )

    lines.append("\n## Log tails\n")
    lines.append(
        "> Progress-bar redraws and known-benign Isaac teardown warnings are "
        "filtered out. Those warnings appear on successful runs too, so leaving "
        "them in buries the actual last words of a failing process — which is "
        "exactly what happened in the earlier FAIL reports.\n"
    )
    for name, p in [
        ("server", Path(args.server_log)),
        ("primary", Path(args.primary_log)),
        ("alt", Path(args.alt_log)),
    ]:
        if p.exists():
            lines.append(f"### {name} (last 30 meaningful lines)\n\n```\n{meaningful_tail(read(p))}\n```\n")

    Path(args.report).write_text("\n".join(lines) + "\n")

    machine = Path(args.report).with_suffix(".json")
    machine.write_text(
        json.dumps(
            {
                "status": status,
                "primary_task": args.primary_task,
                "alt_task": args.alt_task,
                "alt_status": args.alt_status,
                "server": server_r,
                "primary": primary_r,
                "alt": alt_r,
                # Full runner records, so downstream analysis (and the M2 write-up)
                # reads structured data instead of re-parsing this markdown.
                "primary_episodes": primary_eps,
                "alt_episodes": alt_eps,
                "server_request_count": len(
                    re.findall(r"\[robolab-policy-server\] prompt=", server_text)
                ),
            },
            indent=2,
        )
    )

    print(f"wrote {args.report}  (status={status})")
    print(f"wrote {machine}")
    if status == FAIL:
        return 8
    if status == "INCOMPLETE":
        return 9
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
