#!/usr/bin/env python3
"""Validate smoke-test logs against spec §7.3 pass criteria and write report.

We look for signal in both the server log (checkpoint loaded, no NaNs,
action dim printed) and the client log (connection, action received,
episode termination). Absence of positive evidence is a fail — we never
claim success from silence.
"""
from __future__ import annotations

import argparse
import re
from pathlib import Path


def _grep(path: Path, patterns: dict[str, str]) -> dict[str, bool]:
    """For each name → regex, return True iff any line matches."""
    text = path.read_text(errors="replace") if path.exists() else ""
    return {name: re.search(rx, text, re.IGNORECASE) is not None for name, rx in patterns.items()}


SERVER_CHECKS = {
    "checkpoint_loaded": r"cosmos3.*policy.*droid|checkpoint.*loaded|loaded.*checkpoint",
    "listening": r"listen|uvicorn running|server started|running on http",
    "no_nans": r"(?!.*nan detected|.*NaN)",  # inverted below
    "action_dim_8": r"action.?dim.?8|action_dim=8|action shape.*\b8\b",
}

# For "no_nans" we actually invert: fail if we DO see NaN complaints.
NAN_RX = r"NaN detected|found NaN|nan in action|inf in action"

CLIENT_CHECKS = {
    "connected": r"connected|remote-host.*8000|policy client ready",
    "actions_received": r"action received|received .* action|actions:.*shape",
    "sim_advanced": r"step \d+|advanced|env.step|iteration \d+",
    "episode_terminated": r"episode (?:done|terminated|finished)|success=(?:true|false)",
}


def _bool(v: bool) -> str:
    return "✅" if v else "❌"


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--server-log", required=True)
    ap.add_argument("--primary-log", required=True)
    ap.add_argument("--alt-log", required=True)
    ap.add_argument("--primary-task", required=True)
    ap.add_argument("--report", required=True)
    args = ap.parse_args()

    server = Path(args.server_log)
    primary = Path(args.primary_log)
    alt = Path(args.alt_log)

    server_r = _grep(server, SERVER_CHECKS)
    server_r["no_nans"] = not bool(re.search(NAN_RX, server.read_text(errors="replace"), re.IGNORECASE)) if server.exists() else False
    primary_r = _grep(primary, CLIENT_CHECKS)
    alt_r = _grep(alt, CLIENT_CHECKS)

    all_pass = all(server_r.values()) and all(primary_r.values())
    status = "PASS" if all_pass else "FAIL"

    def row(name: str, ok: bool) -> str:
        return f"| {name} | {_bool(ok)} |"

    lines: list[str] = []
    lines.append(f"# Baseline smoke — **{status}**\n")
    lines.append(f"Primary task: `{args.primary_task}`\n")

    lines.append("## Server checks\n")
    lines.append("| check | ok |\n|---|---|")
    for k, v in server_r.items():
        lines.append(row(k, v))

    lines.append("\n## Primary client checks\n")
    lines.append("| check | ok |\n|---|---|")
    for k, v in primary_r.items():
        lines.append(row(k, v))

    lines.append("\n## Alt-task client checks\n")
    lines.append("| check | ok |\n|---|---|")
    for k, v in alt_r.items():
        lines.append(row(k, v))

    lines.append("\n## Log tails\n")
    for name, p in [("server", server), ("primary", primary), ("alt", alt)]:
        if p.exists():
            tail = "\n".join(p.read_text(errors="replace").splitlines()[-30:])
            lines.append(f"### {name} (last 30)\n\n```\n{tail}\n```\n")

    Path(args.report).write_text("\n".join(lines) + "\n")
    print(f"wrote {args.report}  (status={status})")
    return 0 if all_pass else 8


if __name__ == "__main__":
    raise SystemExit(main())
