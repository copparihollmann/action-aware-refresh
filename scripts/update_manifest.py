#!/usr/bin/env python3
"""Refresh reproducibility/source_manifest.json from the cloned repos.

Reads existing manifest, updates commit/branch/dirty/license for every
`primary_source` whose local_path exists on disk. Leaves deferred sources
untouched. Idempotent.
"""
from __future__ import annotations

import datetime as dt
import json
import subprocess
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
MANIFEST = REPO_ROOT / "reproducibility" / "source_manifest.json"


def git(cwd: Path, *args: str) -> str:
    r = subprocess.run(
        ["git", *args],
        cwd=cwd,
        capture_output=True,
        text=True,
        check=False,
    )
    return (r.stdout or "").strip()


def find_license(root: Path) -> str | None:
    for name in ("LICENSE", "LICENSE.txt", "LICENSE.md", "COPYING", "COPYING.md"):
        p = root / name
        if p.is_file():
            return p.name
    return None


def main() -> int:
    data = json.loads(MANIFEST.read_text())
    data["generated_utc"] = dt.datetime.utcnow().replace(microsecond=0).isoformat() + "Z"

    for entry in data.get("primary_sources", []):
        local = REPO_ROOT / entry["local_path"]
        if not (local / ".git").is_dir():
            continue
        entry["branch"] = git(local, "rev-parse", "--abbrev-ref", "HEAD") or None
        entry["commit"] = git(local, "rev-parse", "HEAD") or None
        entry["dirty"] = bool(git(local, "status", "--porcelain"))
        entry["license"] = find_license(local)

    MANIFEST.write_text(json.dumps(data, indent=2))
    print(f"wrote {MANIFEST.relative_to(REPO_ROOT)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
