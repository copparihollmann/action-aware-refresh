"""The pre-init route that replaces two upstream patches must not fail silently.

These tests are deliberately GPU-free: they cover the argument contract and the *order*
of operations in the launcher. The behavioural claim — that gloo produces bitwise
identical actions — cannot be unit-tested; it is measured by
`scripts/validate_pg_backend.py` and recorded in `results/processed/pg_backend_ab.jsonl`.

The ordering test exists because of a real failure earlier in this project: a patch that
placed a method *after* a `return` compiled, imported and passed its unit tests, then
failed 90 seconds into a closed-loop run. Import-time success is not evidence that a
launcher does what it says, so the order is asserted on the AST.
"""
from __future__ import annotations

import ast
import sys
from pathlib import Path

import pytest

from action_refresh.server.process_group import ensure_single_rank_group

REPO_ROOT = Path(__file__).resolve().parents[2]
ENTRY = REPO_ROOT / "scripts" / "cosmos_server_entry.py"


def test_unknown_backend_is_an_error_not_a_default() -> None:
    with pytest.raises(ValueError, match="backend must be one of"):
        ensure_single_rank_group("mpi")  # type: ignore[arg-type]


def test_none_is_an_explicit_no_op() -> None:
    # "none" must neither touch CUDA nor claim to have done anything, so it stays usable
    # on a machine with no GPU — which is where the rest of the test suite runs.
    what = ensure_single_rank_group("none")
    assert "none" in what
    # Two environments run this file and the available evidence differs between them.
    # Under `make test` (repo venv, no torch at all) the absence of the module is the
    # strongest possible statement that nothing was initialized. Under `make test-model`
    # (a substrate venv) a sibling test has already imported torch by collection time, so
    # asserting on sys.modules there tested import order rather than this function — it
    # failed for that reason, not because "none" did anything. Assert the claim itself.
    if "torch" not in sys.modules:
        return
    import torch.distributed as dist

    assert not dist.is_initialized(), (
        'ensure_single_rank_group("none") must leave the process group untouched'
    )


def _entry_main_body() -> list[ast.stmt]:
    tree = ast.parse(ENTRY.read_text())
    for node in tree.body:
        if isinstance(node, ast.FunctionDef) and node.name == "main":
            return node.body
    raise AssertionError("scripts/cosmos_server_entry.py has no main()")


def test_entry_creates_the_group_before_running_upstream() -> None:
    """A group created *after* upstream's main starts is a group created too late."""
    calls: list[tuple[int, str]] = []
    for stmt in _entry_main_body():
        for sub in ast.walk(stmt):
            if isinstance(sub, ast.Call) and isinstance(sub.func, ast.Name):
                calls.append((stmt.lineno, sub.func.id))
            elif isinstance(sub, ast.Call) and isinstance(sub.func, ast.Attribute):
                calls.append((stmt.lineno, sub.func.attr))
    names = [name for _, name in calls]
    assert "ensure_single_rank_group" in names, names
    assert "run_module" in names, names
    assert names.index("ensure_single_rank_group") < names.index("run_module")


def test_entry_delegates_to_upstream_unmodified() -> None:
    """The wrapper must run upstream's module, not a fork of it."""
    src = ENTRY.read_text()
    assert "cosmos_framework.scripts.action_policy_server_robolab" in src
    # argv is forwarded whole; anything that rewrites flags would make the server we
    # measure differ from the server upstream documents.
    assert "sys.argv[1:]" in src
