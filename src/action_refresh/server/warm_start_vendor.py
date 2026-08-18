"""Make the group's inference samplers importable — without vendoring copies of them.

`chooper1/Cosmos3-Efficient-Imagination` ships its samplers as a flat `sampler/` directory
whose modules import each other two ways: as a package (`from warm_start.split_schedule
import ...`, in `patch.py`) and flat (`from ofp_warm_start import ...`, same file). Their
own server entry point imports `warm_start.patch`, so the directory
was evidently on `PYTHONPATH` under the name `warm_start`. This module reconstructs that
without copying anything:

- the directory goes on `sys.path`, satisfying the flat imports;
- a synthetic `warm_start` package is registered whose ``__path__`` is that directory,
  satisfying the package imports.

Why not copy the files in? Three reasons, in order of weight. Their repo has **no LICENSE
file** (individual modules carry `SPDX-License-Identifier: MIT`, but the repo as a whole
grants nothing), so copying is the one form of use that needs a licence we do not clearly
have — importing a clone the operator was granted access to does not. Second, this project's
rule is to preserve upstream and record patches, not to fork. Third, a copy silently goes
stale; an import cannot.

`patch.py` monkeypatches `OmniMoTModel._prepare_inference_data` at the **class** level and
reads `WarmStartConfig.from_env()` **per call**. Two consequences we rely
on: it is server-agnostic, so it applies to RoboLab's service though they only ever ran
another benchmark's; and a whole (V, A) grid can be swept in one process by mutating the environment
between requests, which is the difference between one model load and one per cell.
"""
from __future__ import annotations

import importlib
import json
import sys
from dataclasses import dataclass
from pathlib import Path
from types import ModuleType

#: Name their modules import each other under.
PACKAGE_ALIAS = "warm_start"
#: Manifest entry recording the clone (registered by scripts/update_manifest.py).
SOURCE_NAME = "cosmos3-efficient-imagination"


class VendorError(RuntimeError):
    """Raised when the vendored samplers cannot be located or imported."""


@dataclass(frozen=True)
class VendorProvenance:
    """Where the imported code came from — recorded with every result that uses it."""

    source_name: str
    root: Path
    sampler_dir: Path
    commit: str | None
    modules: tuple[str, ...]
    spdx: dict[str, str]
    dirty: bool = False
    manifest_commit: str | None = None

    @property
    def manifest_stale(self) -> bool:
        """True when the manifest names a different commit than the tree actually holds."""
        return bool(self.manifest_commit) and self.manifest_commit != self.commit

    def as_dict(self) -> dict[str, object]:
        d: dict[str, object] = {
            "vendor_source": self.source_name,
            "vendor_commit": self.commit,
            "vendor_dirty": self.dirty,
            "vendor_sampler_dir": str(self.sampler_dir),
            "vendor_modules": list(self.modules),
            "vendor_spdx": self.spdx,
        }
        # Only carried when it disagrees, so a normal row stays quiet and a drifting one
        # cannot be mistaken for a normal one.
        if self.manifest_stale:
            d["vendor_manifest_commit"] = self.manifest_commit
            d["vendor_manifest_stale"] = True
        return d


def _git_state(root: Path) -> tuple[str | None, bool]:
    """Live HEAD and dirtiness of the clone, or (None, False) if git cannot answer."""
    import subprocess  # noqa: PLC0415 - only needed on this path

    def git(*args: str) -> str | None:
        try:
            r = subprocess.run(
                ["git", *args], cwd=root, capture_output=True, text=True, check=False, timeout=15
            )
        except (OSError, subprocess.SubprocessError):
            return None
        return (r.stdout or "").strip() if r.returncode == 0 else None

    head = git("rev-parse", "HEAD")
    status = git("status", "--porcelain")
    return head, bool(status)


def _manifest_entry(repo_root: Path) -> dict:
    manifest = repo_root / "reproducibility" / "source_manifest.json"
    if not manifest.exists():
        raise VendorError(f"source manifest not found: {manifest} — run `make sources`")
    for entry in json.loads(manifest.read_text()).get("primary_sources", []):
        if entry.get("name") == SOURCE_NAME:
            return entry
    raise VendorError(
        f"{SOURCE_NAME} is not registered in source_manifest.json. It is a PRIVATE repo and "
        "clone_sources.sh will not fetch it; clone it by hand, then run `make sources`."
    )


def locate(repo_root: str | Path = ".") -> VendorProvenance:
    """Find the clone and its sampler directory, or say precisely what is missing."""
    root_dir = Path(repo_root)
    entry = _manifest_entry(root_dir)
    root = (root_dir / entry["local_path"]).resolve()
    sampler = root / "sampler"
    if not sampler.is_dir():
        raise VendorError(
            f"{root} has no sampler/ directory. The audit "
            "(private/upstream_fork_audit.md) describes the layout this expects; if it has "
            "changed, re-read it rather than guessing."
        )

    modules = tuple(sorted(p.stem for p in sampler.glob("*.py")))
    # Record the licence each file actually declares. The repo grants nothing at the top
    # level, so per-file SPDX is the only licence statement that exists, and a result built
    # on these modules should carry it.
    spdx: dict[str, str] = {}
    for p in sorted(sampler.glob("*.py")):
        head = p.read_text(errors="replace")[:400]
        for line in head.splitlines():
            if "SPDX-License-Identifier:" in line:
                spdx[p.name] = line.split("SPDX-License-Identifier:", 1)[1].strip()
                break
        else:
            spdx[p.name] = "UNDECLARED"

    # Live git, not the manifest. `make sources` is run by hand, so the manifest lags every
    # commit made since — and it did: the split-schedule results were stamped with the
    # pre-patch SHA while the code that actually ran was that commit PLUS cei-0001. A result
    # carrying a stale SHA is worse than one carrying none, because it looks authoritative.
    # This mirrors `_git_state` in action_refresh.config, which fixed the same bug for
    # substrates; this path never received it.
    commit, dirty = _git_state(root)
    return VendorProvenance(
        source_name=SOURCE_NAME,
        root=root,
        sampler_dir=sampler,
        commit=commit or entry.get("commit"),
        modules=modules,
        spdx=spdx,
        dirty=dirty,
        manifest_commit=entry.get("commit"),
    )


def install(repo_root: str | Path = ".") -> VendorProvenance:
    """Put the sampler directory on `sys.path` and register the `warm_start` alias.

    Idempotent. Does *not* import `patch`, because importing it applies the monkeypatch —
    that is a separate, explicit step (`apply`).
    """
    prov = locate(repo_root)
    path_str = str(prov.sampler_dir)
    if path_str not in sys.path:
        # Appended, not prepended: their module names are generic enough (`benchmark`,
        # `rollout`) to shadow something of ours if given priority.
        sys.path.append(path_str)

    existing = sys.modules.get(PACKAGE_ALIAS)
    if existing is not None:
        if list(getattr(existing, "__path__", [])) != [path_str]:
            raise VendorError(
                f"a different `{PACKAGE_ALIAS}` package is already imported from "
                f"{list(getattr(existing, '__path__', []))}; refusing to redirect it"
            )
        return prov

    pkg = ModuleType(PACKAGE_ALIAS)
    pkg.__path__ = [path_str]  # type: ignore[attr-defined]
    pkg.__doc__ = (
        f"Synthetic package aliasing {path_str} (see action_refresh.server.warm_start_vendor)."
    )
    sys.modules[PACKAGE_ALIAS] = pkg
    return prov


#: Their `prepare_inject` must pass through our framework's extra `has_noisy_actions`
#: return value; `cei-0001` on their research branch does that. Asserted at apply() time,
#: because without it the failure is a bare `ValueError: not enough values to unpack`
#: three frames deep in upstream code, which says nothing about the cause.
_ARITY_MARKER = "cref, cmask, *rest"


def apply(repo_root: str | Path = ".") -> VendorProvenance:
    """Install the import path and apply their monkeypatch to `OmniMoTModel`.

    Must be called **before** the policy service is constructed for the guardrail bypass
    to matter, though the sampler patch itself is class-level and so takes effect for
    models built either side of it. `patch.py` prints its own confirmation lines; treat
    their absence as evidence the patch did not engage, exactly as their CLAUDE.md warns
    about the training traces.

    Note what their patch does, since it constrains how anything may wrap it: on a warm
    call it installs `prepare_inject` as an **instance** attribute
    and restores the class function afterwards. An
    instance attribute shadows any class-level adapter, so arity has to be fixed inside
    their function rather than around it — hence the recorded patch rather than a wrapper.
    """
    prov = install(repo_root)

    patch_py = prov.sampler_dir / "patch.py"
    if _ARITY_MARKER not in patch_py.read_text(errors="replace"):
        raise VendorError(
            f"{patch_py} still unpacks _prepare_inference_data as a fixed 7-tuple, but our "
            "cosmos-framework returns 8 (upstream added `has_noisy_actions`). Apply "
            "private/patches/cei-0001-prepare-arity.patch on the clone's "
            "research/action-aware-refresh branch. Without it every warm-sampler call dies "
            "inside upstream's generate_samples_from_batch with an unpacking error."
        )

    importlib.import_module(f"{PACKAGE_ALIAS}.patch")
    return prov
