#!/usr/bin/env bash
# Thin wrapper — the real audit is in scripts/audit_machine.py so we get
# proper error handling and portable heredocs.
set -euo pipefail
REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO_ROOT"
exec python3 "$REPO_ROOT/scripts/audit_machine.py" "$@"
