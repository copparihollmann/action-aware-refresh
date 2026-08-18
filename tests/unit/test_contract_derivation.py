"""Regressions for two bugs that produced authoritative-looking wrong values.

Both were the same species: a derivation that guessed instead of reporting UNKNOWN, and a
provenance field trusted from a file that lags reality. Neither failed loudly, which is why
they survived — so they get tests rather than only a fix.
"""
from __future__ import annotations

import importlib.util
import subprocess
import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[2]


def _load_deriver():
    """Import the script by path: scripts/ is not a package."""
    spec = importlib.util.spec_from_file_location(
        "derive_baseline_contract", REPO_ROOT / "scripts" / "derive_baseline_contract.py"
    )
    assert spec and spec.loader
    mod = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = mod
    spec.loader.exec_module(mod)
    return mod


SERVER_SNIPPET = '''
import json
import socket

_DEFAULT_ACTION_CHUNK_SIZE = 32
_DEFAULT_CONDITIONING_FPS = 15.0


class RobolabServerArgs(pydantic.BaseModel):
    port: int = 8000
    num_steps: int = 4
    action_chunk_size: int | None = _DEFAULT_ACTION_CHUNK_SIZE
    conditioning_fps: float | None = _DEFAULT_CONDITIONING_FPS
    decode_video: bool = False
    use_state: bool = True
    vision_frames: int | None = None
    experiment_overrides: list[str] = pydantic.Field(default_factory=list)
    deep: int = _INDIRECT
'''


class TestFieldDefault:
    def test_reads_a_plain_literal(self) -> None:
        assert _load_deriver().field_default(SERVER_SNIPPET, "port") == "8000"

    def test_does_not_span_newlines_to_find_a_digit(self) -> None:
        """The original bug, exactly.

        `port[^=]*=[^)\\d]*(\\d+)` matched the "port" inside `import`, ran past the newline
        to the next `=`, and reported the server's port as 3. A negated character class
        matches newlines; `.` does not. Truth here is 8000.
        """
        got = _load_deriver().field_default(SERVER_SNIPPET, "port")
        assert got == "8000", f"leaked across lines and got {got!r}"

    def test_num_steps_is_four_not_five(self) -> None:
        assert _load_deriver().field_default(SERVER_SNIPPET, "num_steps") == "4"

    def test_resolves_one_constant_hop(self) -> None:
        m = _load_deriver()
        assert m.field_default(SERVER_SNIPPET, "action_chunk_size") == "32"
        assert m.field_default(SERVER_SNIPPET, "conditioning_fps") == "15.0"

    @pytest.mark.parametrize(
        ("field", "expected"),
        [("decode_video", "False"), ("use_state", "True"), ("vision_frames", "None")],
    )
    def test_bool_and_none_literals_are_not_chased_as_constants(
        self, field: str, expected: str
    ) -> None:
        """`False` looks like an identifier; resolving it as one returned UNKNOWN."""
        assert _load_deriver().field_default(SERVER_SNIPPET, field) == expected

    def test_unresolvable_constant_is_unknown_not_a_guess(self) -> None:
        assert _load_deriver().field_default(SERVER_SNIPPET, "deep") is None

    def test_a_factory_default_is_unknown_rather_than_mangled(self) -> None:
        assert _load_deriver().field_default(SERVER_SNIPPET, "experiment_overrides") is None

    def test_absent_field_is_unknown(self) -> None:
        assert _load_deriver().field_default(SERVER_SNIPPET, "not_a_field") is None


class TestClobberGuard:
    def _run(self, *args: str) -> subprocess.CompletedProcess[str]:
        return subprocess.run(
            [str(REPO_ROOT / ".venv" / "bin" / "python"),
             str(REPO_ROOT / "scripts" / "derive_baseline_contract.py"), *args],
            capture_output=True, text=True, cwd=REPO_ROOT, timeout=120,
        )

    def test_refuses_to_overwrite_a_hand_written_file(self, tmp_path: Path) -> None:
        target = tmp_path / "hand_written.md"
        target.write_text("# hand-verified, 239 lines of evidence\n")
        r = self._run("--out", str(target))
        assert r.returncode == 2, r.stderr
        assert "refusing to overwrite" in r.stderr
        assert target.read_text().startswith("# hand-verified")

    def test_the_real_contract_survives_a_default_run(self) -> None:
        """`make contract` must not be able to destroy docs/baseline_contract.md."""
        doc = REPO_ROOT / "docs" / "baseline_contract.md"
        before = doc.read_text()
        assert self._run().returncode == 0
        assert doc.read_text() == before
        assert len(before.splitlines()) > 200, "the hand-verified contract should be long"

    def test_regenerates_its_own_output_without_force(self, tmp_path: Path) -> None:
        target = tmp_path / "derived.md"
        assert self._run("--out", str(target)).returncode == 0
        assert self._run("--out", str(target)).returncode == 0, "not idempotent"

    def test_force_overrides_the_guard(self, tmp_path: Path) -> None:
        target = tmp_path / "hand_written.md"
        target.write_text("# replace me\n")
        assert self._run("--out", str(target), "--force").returncode == 0
        assert "Baseline contract" in target.read_text()


class TestVendorProvenanceStaleness:
    def test_a_stale_manifest_is_reported_not_silently_used(self) -> None:
        """The split-schedule results were stamped with the pre-patch SHA."""
        from action_refresh.server.warm_start_vendor import VendorProvenance

        p = VendorProvenance(
            source_name="x", root=Path("/nonexistent"), sampler_dir=Path("/nonexistent/sampler"),
            commit="8433abd", modules=(), spdx={}, manifest_commit="ecb8de4",
        )
        assert p.manifest_stale
        d = p.as_dict()
        assert d["vendor_commit"] == "8433abd", "the commit that RAN must win"
        assert d["vendor_manifest_commit"] == "ecb8de4"
        assert d["vendor_manifest_stale"] is True

    def test_an_agreeing_manifest_adds_no_noise(self) -> None:
        from action_refresh.server.warm_start_vendor import VendorProvenance

        p = VendorProvenance(
            source_name="x", root=Path("/nonexistent"), sampler_dir=Path("/nonexistent/sampler"),
            commit="8433abd", modules=(), spdx={}, manifest_commit="8433abd",
        )
        assert not p.manifest_stale
        assert "vendor_manifest_stale" not in p.as_dict()
