#!/usr/bin/env bash
# scripts/clone_sources.sh — clone + pin the primary third-party repos.
#
# Idempotent. Safe to re-run. Only clones what's under `primary_sources`
# in reproducibility/source_manifest.json.
#
# For each repo:
#   1. shallow-clone (--filter=blob:none) if missing, full-fetch after
#   2. create local branch `research/action-aware-refresh` off the default
#   3. record URL, resolved branch, commit SHA, dirty status, license file
#      into the manifest via `scripts/update_manifest.py`
#
# Deferred sources (TeaCache, FasterCache, DeepCache, v2e, ESIM) are only
# recorded in the manifest — NOT cloned here. Clone them in their milestone.

set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO_ROOT"

# Never sit on a credential prompt. One of our sources is private and this host has
# no key or credential helper, so without this a `git fetch` blocks forever waiting
# for a username that no interactive terminal is going to supply.
export GIT_TERMINAL_PROMPT=0

mkdir -p third_party

declare -A REPOS=(
  [cosmos]="https://github.com/NVIDIA/cosmos.git"
  [cosmos-framework]="https://github.com/NVIDIA/cosmos-framework.git"
  [RoboLab]="https://github.com/NVLabs/RoboLab.git"
  [cosmos-policy]="https://github.com/NVlabs/cosmos-policy.git"
  [openpi]="https://github.com/Physical-Intelligence/openpi.git"
)

RESEARCH_BRANCH="research/action-aware-refresh"

for name in "${!REPOS[@]}"; do
  url="${REPOS[$name]}"
  dst="third_party/$name"
  echo "==> $name  ($url)"
  if [ -d "$dst/.git" ]; then
    echo "    exists — fetching"
    git -C "$dst" fetch --all --prune
  else
    echo "    cloning (filter=blob:none)"
    git clone --filter=blob:none "$url" "$dst"
  fi

  # Determine default branch (main or master).
  default_branch="$(git -C "$dst" symbolic-ref --short refs/remotes/origin/HEAD 2>/dev/null | sed 's@^origin/@@' || true)"
  if [ -z "$default_branch" ]; then
    default_branch="$(git -C "$dst" remote show origin | awk '/HEAD branch/ {print $NF}')"
  fi

  # Create/refresh research branch (don't force if already checked out).
  cur="$(git -C "$dst" branch --show-current 2>/dev/null || echo '')"
  if [ "$cur" != "$RESEARCH_BRANCH" ]; then
    if git -C "$dst" show-ref --verify --quiet "refs/heads/$RESEARCH_BRANCH"; then
      git -C "$dst" checkout "$RESEARCH_BRANCH"
    else
      git -C "$dst" checkout -b "$RESEARCH_BRANCH" "origin/$default_branch"
    fi
  fi
done

# ---------------------------------------------------------------------------
# Private sources: present-or-instruct, never cloned here.
#
# chooper1/Cosmos3-Efficient-Imagination is the group's own baseline and is a
# PRIVATE repo. This script cannot clone it: firesim2 has no SSH key and no
# credential helper, and the alternative — accepting a token into this process —
# would put a credential somewhere the repo rules say it must never go. So the
# clone is the operator's, with their own credentials, and this script's job is to
# say so exactly once rather than to fail obscurely later.
#
# `update_manifest.py` discovers it and records URL/branch/SHA once it exists.
# ---------------------------------------------------------------------------
declare -A MANUAL_REPOS=(
  [cosmos3-efficient-imagination]="git@github.com:chooper1/Cosmos3-Efficient-Imagination.git"
)

for name in "${!MANUAL_REPOS[@]}"; do
  dst="third_party/$name"
  if [ -d "$dst/.git" ]; then
    echo "==> $name (private, manual)"
    echo "    present — fetching"
    if ! git -C "$dst" fetch --all --prune; then
      echo "    warning: fetch failed (no credentials on this host?) — using the" >&2
      echo "             checked-out state as-is. The manifest records its SHA." >&2
    fi
  else
    echo "==> $name (private, manual) — MISSING"
    echo "    Clone it yourself; this script will not handle credentials:"
    echo "      git clone ${MANUAL_REPOS[$name]} $dst"
    echo "    (needs an SSH key on this host registered at github.com/settings/keys)"
  fi
done

# Refresh manifest.
python3 "$REPO_ROOT/scripts/update_manifest.py"

echo "clone_sources: done"
echo "→ third_party/ populated; source_manifest.json refreshed"
