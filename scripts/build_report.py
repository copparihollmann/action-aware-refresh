#!/usr/bin/env python3
"""Regenerate results/reports/* from raw records.

Aggregates the per-request JSONL into per-episode summaries and produces
the standard set of plots (spec §13). This is a stub during M0–M2; it
runs but only writes an index of what will be filled in later milestones.
"""
from __future__ import annotations

import json
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
RAW = REPO_ROOT / "results" / "raw"
REPORTS = REPO_ROOT / "results" / "reports"


def main() -> int:
    REPORTS.mkdir(parents=True, exist_ok=True)
    runs = sorted(p.name for p in RAW.glob("*") if p.is_dir())
    out = REPORTS / "index.md"
    lines = ["# Results index\n"]
    if not runs:
        lines.append("_(no runs recorded yet — run `make baseline` or `make profile`)_")
    else:
        lines.append("| run_id | kind |")
        lines.append("|---|---|")
        for r in runs:
            kind = "server" if r.startswith("cosmos-server-") else \
                   "profile" if r.startswith("profile-") else \
                   "robolab" if r.startswith("robolab-") else "unknown"
            lines.append(f"| `{r}` | {kind} |")
    out.write_text("\n".join(lines) + "\n")
    print(f"wrote {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
