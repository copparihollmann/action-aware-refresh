#!/usr/bin/env bash
# scripts/bootstrap_new_machine.sh — rebuild third_party/ exactly, on a fresh host.
#
# WHY THIS EXISTS. `third_party/` is 42 GB and is deliberately NOT committed (CLAUDE.md).
# It does not need to be: every local modification we have ever made to an upstream repo
# lives on branch `research/action-aware-refresh` in that clone AND is exported to
# reproducibility/patches/. Verified 2026-08-17: 7 exported patches reproduce all 7 local
# commits byte-for-byte across cosmos-framework (4), RoboLab (2) and the group's efficiency
# repo (1). So the clones are redundant with (pinned SHA + patches), and this script is the
# proof: it turns manifest + patches back into the working tree.
#
# WHAT IT GUARANTEES. After a successful run, each source is checked out on
# `research/action-aware-refresh` whose **tree hash** equals the one recorded in the manifest
# on the origin machine. Tree, not commit: `git am` replays a patch under a new committer, so
# the commit sha legitimately differs while the content must not. A tree mismatch is a hard
# failure here, never a warning — a silently-different upstream invalidates every measurement.
#
# WHAT IT DOES NOT DO. It does not install anything (that is `make setup` /
# scripts/setup_cosmos.sh + setup_robolab.sh), does not download weights, and does not touch
# a GPU. Run it first, then follow docs/runbook_new_machine.md.
#
# Usage:
#   bash scripts/bootstrap_new_machine.sh              # all sources
#   bash scripts/bootstrap_new_machine.sh --dry-run    # say what would happen, touch nothing
#   bash scripts/bootstrap_new_machine.sh --only RoboLab
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO_ROOT"

MANIFEST="reproducibility/source_manifest.json"
PATCH_DIR="reproducibility/patches"
RESEARCH_BRANCH="research/action-aware-refresh"

# Same reason as clone_sources.sh: one source is private, and without this a fetch blocks
# forever on a credential prompt no non-interactive shell will answer.
export GIT_TERMINAL_PROMPT=0

DRY_RUN=0
ONLY=""
while [ $# -gt 0 ]; do
  case "$1" in
    --dry-run) DRY_RUN=1; shift ;;
    --only)    ONLY="${2:?--only needs a source name}"; shift 2 ;;
    -h|--help) sed -n '2,24p' "$0"; exit 0 ;;
    *) echo "unknown argument: $1" >&2; exit 2 ;;
  esac
done

PY="$(command -v python3)"
[ -n "$PY" ] || { echo "FATAL: python3 not found — needed only to read the manifest" >&2; exit 1; }
[ -f "$MANIFEST" ] || { echo "FATAL: $MANIFEST missing — wrong directory?" >&2; exit 1; }

# Disk gate. Full clones of these six are ~42 GB, and RoboLab alone is ~29 GB because it
# carries simulation assets. The shared-machine rule is abort-and-report, never fill.
avail_gb=$(df -BG --output=avail /scratch 2>/dev/null | tail -1 | tr -dc '0-9' || echo 0)
if [ -z "$avail_gb" ]; then avail_gb=0; fi
echo "== disk: ${avail_gb} GB available on /scratch (clones need ~42 GB; installs need ~60 GB more)"
if [ "$avail_gb" -lt 45 ] && [ "$DRY_RUN" -eq 0 ]; then
  echo "FATAL: under 45 GB free. Aborting rather than filling a shared volume." >&2
  echo "       Re-check with: df -h /scratch" >&2
  exit 1
fi

# Emit one shell-quoted record per source: name url local_path base_commit tree private
read_manifest() {
  "$PY" - "$MANIFEST" <<'EOF'
import json, shlex, sys
data = json.load(open(sys.argv[1]))
for e in data.get("primary_sources", []):
    url = e.get("url") or ""
    fields = [
        e.get("name") or "",
        url,
        e.get("local_path") or "",
        e.get("base_commit") or "",
        e.get("tree") or "",
        "private" if url.startswith("git@") else "public",
    ]
    print(" ".join(shlex.quote(f) for f in fields))
EOF
}

# Which patches belong to which source. Kept explicit rather than derived from filenames:
# a wrong guess here applies someone else's diff to a repo, and the failure would look like
# an upstream mismatch rather than a mapping bug.
patches_for() {
  case "$1" in
    cosmos-framework)              ls "$PATCH_DIR"/cosmos-framework-*.patch 2>/dev/null ;;
    RoboLab)                       ls "$PATCH_DIR"/robolab-*.patch 2>/dev/null ;;
    # Their patch lives in private/ (untracked): it quotes their private, unlicensed
    # source, and this repo's origin is public. Absent on a fresh clone by design.
    cosmos3-efficient-imagination) ls private/patches/cei-*.patch 2>/dev/null ;;
    *) : ;;   # cosmos, cosmos-policy, openpi: branch created, never modified
  esac
}

declare -a PASS=() FAIL=() SKIP=()

while read -r name url path base tree visibility; do
  [ -n "$name" ] || continue
  if [ -n "$ONLY" ] && [ "$name" != "$ONLY" ]; then continue; fi

  echo
  echo "==> $name"
  if [ -z "$url" ] || [ -z "$base" ]; then
    echo "    SKIP — manifest has no url/base_commit (a deferred source, not cloned)"
    SKIP+=("$name (deferred)"); continue
  fi
  echo "    url   $url"
  echo "    base  ${base:0:12}   tree ${tree:0:12}"

  mapfile -t pl < <(patches_for "$name")
  echo "    patch ${#pl[@]} to apply"

  if [ "$visibility" = "private" ]; then
    echo "    NOTE: private repo. Needs an SSH key registered with access, or it will fail."
    echo "          It is only required to re-run the split-schedule grid; nothing else uses it."
    if [ "${#pl[@]}" -eq 0 ]; then
      echo "          Its patch is NOT in this repo (private/, untracked — see private/README.md)."
      echo "          Without it the clone is pristine upstream and the tree check below will"
      echo "          MISMATCH by exactly that one commit. Expected on a fresh clone."
    fi
  fi

  if [ "$DRY_RUN" -eq 1 ]; then
    echo "    (dry run — no changes)"; SKIP+=("$name (dry-run)"); continue
  fi

  if [ -d "$path/.git" ]; then
    echo "    exists — fetching"
    git -C "$path" fetch --all --prune --quiet || {
      echo "    FAIL — fetch failed (private repo without access?)" >&2
      FAIL+=("$name (fetch)"); continue; }
  else
    echo "    cloning (this is the slow part; RoboLab is ~29 GB)"
    if ! git clone --quiet "$url" "$path"; then
      echo "    FAIL — clone failed$([ "$visibility" = private ] && echo ' (no access to the private repo?)')" >&2
      FAIL+=("$name (clone)"); continue
    fi
  fi

  # Land exactly on the recorded base, then replay our commits. Branch is recreated from
  # scratch so a re-run is idempotent rather than stacking patches on top of themselves.
  if ! git -C "$path" cat-file -e "${base}^{commit}" 2>/dev/null; then
    echo "    FAIL — base commit $base not present after fetch (upstream force-push?)" >&2
    FAIL+=("$name (missing base)"); continue
  fi
  git -C "$path" checkout --quiet --detach "$base"
  git -C "$path" branch -f "$RESEARCH_BRANCH" "$base" >/dev/null
  git -C "$path" checkout --quiet "$RESEARCH_BRANCH"

  applied=0
  for p in "${pl[@]}"; do
    [ -f "$p" ] || continue
    if git -C "$path" am --quiet "$REPO_ROOT/$p"; then
      applied=$((applied+1))
    else
      git -C "$path" am --abort 2>/dev/null || true
      echo "    FAIL — $p did not apply on $base" >&2
      FAIL+=("$name (patch $(basename "$p"))"); applied=-1; break
    fi
  done
  [ "$applied" -lt 0 ] && continue
  echo "    applied $applied patch(es)"

  got="$(git -C "$path" rev-parse 'HEAD^{tree}')"
  if [ -z "$tree" ]; then
    echo "    tree: no reference recorded — cannot verify"
    SKIP+=("$name (unverifiable)")
  elif [ "$got" = "$tree" ]; then
    echo "    tree: MATCH ${got:0:12} — byte-identical to the origin machine"
    PASS+=("$name")
  else
    echo "    tree: MISMATCH got ${got:0:12}, expected ${tree:0:12}" >&2
    echo "          Do NOT measure against this tree. Diff it before doing anything else." >&2
    FAIL+=("$name (tree mismatch)")
  fi
done < <(read_manifest)

echo
echo "================ summary ================"
printf '  verified : %s\n' "${PASS[*]:-none}"
printf '  skipped  : %s\n' "${SKIP[*]:-none}"
printf '  FAILED   : %s\n' "${FAIL[*]:-none}"
echo
if [ "${#FAIL[@]}" -gt 0 ]; then
  echo "One or more sources are NOT reproduced. Fix before any measurement." >&2
  exit 1
fi
if [ "$DRY_RUN" -eq 1 ]; then
  echo "Dry run only — nothing was cloned, patched or verified."
  exit 0
fi
if [ "${#PASS[@]}" -eq 0 ]; then
  echo "Nothing was verified. Do not treat this as a successful bootstrap." >&2
  exit 1
fi
echo "third_party/ reproduced. Next: docs/runbook_new_machine.md (installs, weights, configs)."
echo "Remember: configs/machine.yaml is gitignored — copy configs/machine.example.yaml and edit."
