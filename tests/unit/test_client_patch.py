"""Structural checks on our patches to RoboLab's Cosmos3 client.

These exist because of a real failure. The `robolab-0002` patch inserted a new method
into `Cosmos3Client` and accidentally landed it *inside* `__init__`, so everything after
the insertion point — including `self.client = self._connect()` — became unreachable code
after a `return`. The module imported fine, compiled fine, and the new method's own unit
tests passed. The break only surfaced as
`AttributeError: 'Cosmos3Client' object has no attribute 'client'` after Isaac Sim had
booted, ~90 seconds into a closed-loop run, wasting the step.

The lesson: testing a patched method in isolation does not test that the patch left the
rest of the class intact. These checks parse the file and construct the object, which is
what would have caught it.

Skipped when the RoboLab clone is absent so the suite still runs on a bare checkout.
"""
from __future__ import annotations

import ast
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[2]
CLIENT = REPO_ROOT / "third_party" / "RoboLab" / "policies" / "cosmos3" / "client.py"

pytestmark = pytest.mark.skipif(
    not CLIENT.is_file(), reason="RoboLab clone not present (run `make sources`)"
)


@pytest.fixture(scope="module")
def client_class() -> ast.ClassDef:
    tree = ast.parse(CLIENT.read_text())
    return next(
        n for n in ast.walk(tree) if isinstance(n, ast.ClassDef) and n.name == "Cosmos3Client"
    )


def methods(cls: ast.ClassDef) -> dict[str, ast.FunctionDef]:
    return {n.name: n for n in cls.body if isinstance(n, ast.FunctionDef)}


def test_init_still_connects(client_class):
    """`self.client` must be assigned in __init__.

    Without it every request raises AttributeError, but only once a request is actually
    made — i.e. after Isaac has booted.
    """
    init = methods(client_class)["__init__"]
    assigned = {
        t.attr
        for stmt in init.body
        if isinstance(stmt, ast.Assign)
        for t in stmt.targets
        if isinstance(t, ast.Attribute)
    }
    assert "client" in assigned, "__init__ must assign self.client (patch truncated it?)"
    assert "open_loop_horizon" in assigned


def test_no_unreachable_code_after_return(client_class):
    """The exact signature of the bug: statements stranded after a `return`."""
    for name, fn in methods(client_class).items():
        for stmt in fn.body[:-1]:
            assert not isinstance(stmt, ast.Return), (
                f"unreachable code after `return` in Cosmos3Client.{name} — a patch "
                "probably landed inside the wrong function body"
            )


def test_required_methods_are_present(client_class):
    """The client contract RoboLab's InferenceClient relies on, plus our additions."""
    required = {
        "__init__",
        "_connect",
        "_infer_with_retry",
        "_extract_observation",
        "_pack_request",
        "_query_server",
        "_unpack_response",
        "_postprocess_chunk",
        "_build_visualization",
        # ours (robolab-0002)
        "_resolve_open_loop_horizon",
    }
    missing = required - set(methods(client_class))
    assert missing == set(), f"Cosmos3Client is missing methods: {sorted(missing)}"


def test_capture_hook_is_still_called(client_class):
    """robolab-0001/0002: the capture must be invoked from _pack_request.

    If the call is lost, the corpus silently stops growing and the offline studies
    quietly run on stale data.
    """
    pack = methods(client_class)["_pack_request"]
    calls = [
        n.func.id
        for n in ast.walk(pack)
        if isinstance(n, ast.Call) and isinstance(n.func, ast.Name)
    ]
    assert "_maybe_capture_request" in calls


def test_client_actually_constructs(monkeypatch):
    """Parsing is not enough — build the object.

    `_connect` is stubbed so no server is needed; everything else runs for real.
    """
    import sys

    robolab = REPO_ROOT / "third_party" / "RoboLab"
    if str(robolab) not in sys.path:
        sys.path.insert(0, str(robolab))
    try:
        from policies.cosmos3 import client as mod
    except ImportError as exc:  # openpi_client / robolab not installed in this venv
        pytest.skip(f"RoboLab client not importable here: {exc}")

    monkeypatch.setattr(mod.Cosmos3Client, "_connect", lambda self: "STUB")
    monkeypatch.delenv("ROBOLAB_OPEN_LOOP_HORIZON", raising=False)
    c = mod.Cosmos3Client(remote_host="h", remote_port=1)
    assert c.client == "STUB"
    assert c.open_loop_horizon == mod.Cosmos3Client.OPEN_LOOP_HORIZON

    monkeypatch.setenv("ROBOLAB_OPEN_LOOP_HORIZON", "8")
    assert mod.Cosmos3Client(remote_host="h", remote_port=1).open_loop_horizon == 8

    # And a bad value must raise rather than silently fall back to the baseline.
    monkeypatch.setenv("ROBOLAB_OPEN_LOOP_HORIZON", "99")
    with pytest.raises(ValueError, match="out of range"):
        mod.Cosmos3Client(remote_host="h", remote_port=1)
