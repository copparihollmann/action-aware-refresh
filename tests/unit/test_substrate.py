"""Substrate resolution must never guess which source tree served a measurement.

Two Cosmos source trees now exist — NVIDIA upstream and the group's efficiency fork
(`chooper1/Cosmos3-Efficient-Imagination`) — and the whole point of comparing them is
that a number carries the identity of the tree that produced it. So every way of *not*
knowing is an error here, not a default: unknown name, unregistered source, missing
clone, missing venv, wrong server module.

These tests are GPU-free and filesystem-only; they build throwaway trees in tmp_path
rather than touching third_party/. The one test that reads the real repo asserts the
shipped config resolves, which is the check that would have caught a typo in
configs/substrates.yaml before a launch.
"""
from __future__ import annotations

import json
import os
import subprocess
from pathlib import Path

import pytest
import yaml

from action_refresh.config import (
    SubstrateError,
    load_substrates,
    resolve_substrate,
)

REPO_ROOT = Path(__file__).resolve().parents[2]


@pytest.fixture(autouse=True)
def _no_inherited_substrate(monkeypatch: pytest.MonkeyPatch) -> None:
    """Ignore any COSMOS_SUBSTRATE from the caller's environment.

    `make test-model` resolves a substrate and exports it before invoking pytest, so
    without this the env var outranks each fixture's own config and every test resolves
    the real repo's substrate instead of its throwaway one. Found exactly that way.
    Tests that care about the variable set it themselves.
    """
    monkeypatch.delenv("COSMOS_SUBSTRATE", raising=False)


def _git_init(root: Path, *, dirty: bool = False) -> str:
    root.mkdir(parents=True, exist_ok=True)
    env = {
        **os.environ,
        "GIT_AUTHOR_NAME": "t",
        "GIT_AUTHOR_EMAIL": "t@t",
        "GIT_COMMITTER_NAME": "t",
        "GIT_COMMITTER_EMAIL": "t@t",
    }
    subprocess.run(["git", "init", "-q"], cwd=root, check=True, env=env)
    (root / "README").write_text("x")
    subprocess.run(["git", "add", "-A"], cwd=root, check=True, env=env)
    subprocess.run(["git", "commit", "-qm", "init"], cwd=root, check=True, env=env)
    head = subprocess.run(
        ["git", "rev-parse", "HEAD"], cwd=root, capture_output=True, text=True, check=True
    ).stdout.strip()
    if dirty:
        (root / "README").write_text("changed")
    return head


def _fake_repo(
    tmp_path: Path,
    *,
    server_module: str = "pkg.server",
    make_venv: bool = True,
    make_server_file: bool = True,
    dirty: bool = False,
    manifest_commit: str | None = None,
) -> tuple[Path, str]:
    """A minimal repo root with one registered substrate, plus its real HEAD sha."""
    src = tmp_path / "third_party" / "thing"
    src.mkdir(parents=True)

    if make_server_file:
        server_file = src / (server_module.replace(".", "/") + ".py")
        server_file.parent.mkdir(parents=True, exist_ok=True)
        server_file.write_text("# server\n")

    # Committed before the venv exists, and .venv ignored — mirroring the real trees,
    # where an installed environment does not make the source tree dirty. Without this
    # every substrate would report dirty=True and the flag would carry no information.
    (src / ".gitignore").write_text(".venv/\n")
    head = _git_init(src, dirty=dirty)

    if make_venv:
        venv_py = src / ".venv" / "bin" / "python"
        venv_py.parent.mkdir(parents=True)
        venv_py.write_text("#!/bin/sh\n")
        venv_py.chmod(0o755)

    (tmp_path / "configs").mkdir()
    (tmp_path / "configs" / "substrates.yaml").write_text(
        yaml.safe_dump(
            {
                "default": "a",
                "substrates": {"a": {"source_name": "thing", "server_module": server_module}},
            }
        )
    )
    (tmp_path / "reproducibility").mkdir()
    (tmp_path / "reproducibility" / "source_manifest.json").write_text(
        json.dumps(
            {
                "primary_sources": [
                    {
                        "name": "thing",
                        "local_path": "third_party/thing",
                        "commit": manifest_commit if manifest_commit is not None else head,
                    }
                ]
            }
        )
    )
    return tmp_path, head


def test_resolves_and_reports_live_commit(tmp_path: Path) -> None:
    root, head = _fake_repo(tmp_path)
    s = resolve_substrate(repo_root=root)
    assert s.name == "a"
    assert s.root == (root / "third_party" / "thing").resolve()
    assert s.commit == head
    assert s.dirty is False
    assert s.manifest_stale is False
    assert s.provenance()["commit"] == head


def test_live_commit_wins_over_a_stale_manifest(tmp_path: Path) -> None:
    """The bug this guards: cosmos-framework's manifest entry was four commits behind.

    A result stamped with the recorded SHA would have named a tree that never ran, and
    looked authoritative doing it.
    """
    root, head = _fake_repo(tmp_path, manifest_commit="0" * 40)
    s = resolve_substrate(repo_root=root)
    assert s.commit == head
    assert s.manifest_stale is True
    prov = s.provenance()
    assert prov["commit"] == head
    assert prov["manifest_commit"] == "0" * 40
    assert prov["manifest_stale"] is True


def test_dirty_tree_is_reported(tmp_path: Path) -> None:
    root, _ = _fake_repo(tmp_path, dirty=True)
    assert resolve_substrate(repo_root=root).dirty is True


def test_env_var_selects_and_explicit_arg_wins(tmp_path: Path, monkeypatch) -> None:
    root, _ = _fake_repo(tmp_path)
    monkeypatch.setenv("COSMOS_SUBSTRATE", "a")
    assert resolve_substrate(repo_root=root).name == "a"

    monkeypatch.setenv("COSMOS_SUBSTRATE", "nonexistent")
    with pytest.raises(SubstrateError, match="unknown substrate"):
        resolve_substrate(repo_root=root)
    # An explicit argument outranks the environment.
    assert resolve_substrate("a", repo_root=root).name == "a"


def test_unregistered_source_points_at_make_sources(tmp_path: Path) -> None:
    root, _ = _fake_repo(tmp_path)
    manifest = root / "reproducibility" / "source_manifest.json"
    manifest.write_text(json.dumps({"primary_sources": []}))
    with pytest.raises(SubstrateError, match="make sources"):
        resolve_substrate(repo_root=root)


def test_missing_clone_is_an_error(tmp_path: Path) -> None:
    """The state we are in right now for the private fork: registered, not yet cloned."""
    root, _ = _fake_repo(tmp_path)
    git_dir = root / "third_party" / "thing" / ".git"
    git_dir.rename(git_dir.with_name(".git-moved"))
    with pytest.raises(SubstrateError, match="not a git clone"):
        resolve_substrate(repo_root=root)


def test_missing_venv_refuses_to_borrow_another(tmp_path: Path) -> None:
    root, _ = _fake_repo(tmp_path, make_venv=False)
    with pytest.raises(SubstrateError, match="setup_cosmos.sh"):
        resolve_substrate(repo_root=root)
    # Opt out only when the caller genuinely does not need to launch anything.
    assert resolve_substrate(repo_root=root, require_venv=False).name == "a"


def test_wrong_server_module_fails_before_launch(tmp_path: Path) -> None:
    """The fork may not expose upstream's entry point; find out here, not at startup."""
    root, _ = _fake_repo(tmp_path, server_module="pkg.server", make_server_file=False)
    with pytest.raises(SubstrateError, match="server_module"):
        resolve_substrate(repo_root=root)


def test_default_must_name_a_defined_substrate(tmp_path: Path) -> None:
    root, _ = _fake_repo(tmp_path)
    cfg = root / "configs" / "substrates.yaml"
    data = yaml.safe_load(cfg.read_text())
    data["default"] = "typo"
    cfg.write_text(yaml.safe_dump(data))
    with pytest.raises(SubstrateError, match="default='typo'"):
        resolve_substrate(repo_root=root)


def test_unknown_key_in_config_is_rejected(tmp_path: Path) -> None:
    root, _ = _fake_repo(tmp_path)
    cfg = root / "configs" / "substrates.yaml"
    data = yaml.safe_load(cfg.read_text())
    data["substrates"]["a"]["venv"] = "third_party/thing/.venv"
    cfg.write_text(yaml.safe_dump(data))
    with pytest.raises(Exception, match="venv"):
        resolve_substrate(repo_root=root)


# ---------------------------------------------------------------------------
# The shipped config, not a fixture.
# ---------------------------------------------------------------------------


def test_shipped_config_is_valid_and_upstream_resolves() -> None:
    cfg = load_substrates(REPO_ROOT / "configs" / "substrates.yaml")
    assert cfg.default == "upstream", "the reference arm must stay the default"

    s = resolve_substrate("upstream", repo_root=REPO_ROOT, require_venv=False)
    assert s.server_file.is_file()
    assert s.commit and len(s.commit) == 40


def test_every_shipped_substrate_names_a_distinct_source() -> None:
    cfg = load_substrates(REPO_ROOT / "configs" / "substrates.yaml")
    sources = [s.source_name for s in cfg.substrates.values()]
    assert len(sources) == len(set(sources)), (
        "two substrates sharing a source tree would make their results "
        "indistinguishable, which defeats the point of naming them"
    )


def test_the_efficiency_repo_is_not_registered_as_a_substrate() -> None:
    """It patches cosmos-framework from the outside; it is not a source tree to serve.

    Asserted rather than merely commented because the mistake is an easy one to repeat:
    the repo sits in third_party/ next to real substrates, and update_manifest.py
    registers it as a source. A substrate entry pointing at it would fail only at launch.
    See private/upstream_fork_audit.md.
    """
    cfg = load_substrates(REPO_ROOT / "configs" / "substrates.yaml")
    named = {s.source_name for s in cfg.substrates.values()}
    assert "cosmos3-efficient-imagination" not in named
    cei = REPO_ROOT / "third_party" / "cosmos3-efficient-imagination"
    if cei.is_dir():
        assert not (cei / "cosmos_framework").exists(), (
            "if this ever grows a cosmos_framework package, revisit the audit's "
            "conclusion that it cannot be a substrate"
        )
