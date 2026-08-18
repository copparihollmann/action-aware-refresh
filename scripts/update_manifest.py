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
RESEARCH_BRANCH = "research/action-aware-refresh"


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


# Phase and role for a newly discovered clone. Neither is derivable from the repo
# itself, and both belong in the manifest rather than in a hand edit — the file's own
# header says never to edit it by hand, so anything we want recorded has to be
# regenerable from here.
PHASE_HINTS = {
    "cosmos3-efficient-imagination": (
        "M3 (prior art / methods source — NOT a substrate; see private/upstream_fork_audit.md)"
    ),
}

NOTE_HINTS = {
    "cosmos3-efficient-imagination": (
        "Private. Standalone repo that patches cosmos-framework from the outside "
        "(framework_patches/) and monkeypatches its sampler; contains no cosmos_framework "
        "package, so it cannot serve as a substrate. No LICENSE file."
    ),
}


def discover_unregistered(data: dict) -> list[dict]:
    """Register clones present under third_party/ but missing from the manifest.

    `clone_sources.sh` cannot clone a *private* repo — it has no credentials and
    must not sit on a terminal prompt — so `third_party/cosmos3-efficient-imagination`
    arrives by hand. Discovering it here keeps "cloned by hand" and "recorded in the
    manifest" from drifting apart, which is the whole point of the manifest.

    Everything recorded is derived from the clone (origin URL, branch, SHA), so this
    adds no second copy of anything.
    """
    known = {e.get("local_path") for e in data.get("primary_sources", [])}
    added: list[dict] = []
    third_party = REPO_ROOT / "third_party"
    for path in sorted(p for p in third_party.glob("*") if (p / ".git").is_dir()):
        rel = f"third_party/{path.name}"
        if rel in known:
            continue
        url = git(path, "remote", "get-url", "origin") or None
        # Only claim a research branch if one actually exists. A clone we merely read
        # (prior art, a methods source) has none, and asserting one would imply we are
        # carrying local modifications to a repo we are not modifying.
        has_research = bool(
            git(path, "rev-parse", "--verify", "--quiet", f"refs/heads/{RESEARCH_BRANCH}")
        )
        entry = {
            "name": path.name,
            "url": url,
            "phase": PHASE_HINTS.get(path.name),
            "local_path": rel,
            "research_branch": RESEARCH_BRANCH if has_research else None,
        }
        if path.name in NOTE_HINTS:
            entry["note"] = NOTE_HINTS[path.name]
        added.append(entry)
    data.setdefault("primary_sources", []).extend(added)
    return added


def main() -> int:
    data = json.loads(MANIFEST.read_text())
    data["generated_utc"] = (
        dt.datetime.now(dt.UTC).replace(microsecond=0, tzinfo=None).isoformat() + "Z"
    )

    added = discover_unregistered(data)

    for entry in data.get("primary_sources", []):
        local = REPO_ROOT / entry["local_path"]
        if not (local / ".git").is_dir():
            continue
        entry["branch"] = git(local, "rev-parse", "--abbrev-ref", "HEAD") or None
        entry["commit"] = git(local, "rev-parse", "HEAD") or None
        entry["dirty"] = bool(git(local, "status", "--porcelain"))
        entry["license"] = find_license(local)
        # `base_commit` is the pristine upstream point our research branch sits on, and it is
        # what a new machine must clone before applying reproducibility/patches/. Without it
        # the manifest records where we *are* but not where to start, which is the half that
        # matters when rebuilding third_party/ elsewhere.
        entry["base_commit"] = git(local, "rev-parse", "main") or None
        # `tree` is the content hash of the checkout. It is the integrity check bootstrap uses
        # after `git am`, because a replayed commit gets a NEW sha (the committer differs) while
        # the tree must match exactly if the patches reconstructed the same source.
        entry["tree"] = git(local, "rev-parse", "HEAD^{tree}") or None

    MANIFEST.write_text(json.dumps(data, indent=2))
    for entry in added:
        phase = entry["phase"] or "phase UNKNOWN — set PHASE_HINTS in this script"
        print(f"registered new source: {entry['name']} ({phase})")
    print(f"wrote {MANIFEST.relative_to(REPO_ROOT)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
