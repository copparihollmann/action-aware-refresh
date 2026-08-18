"""Config loaders for machine.yaml, topology.yaml and substrates.yaml."""
from __future__ import annotations

import json
import os
from dataclasses import dataclass
from pathlib import Path

import yaml
from pydantic import BaseModel, ConfigDict, Field


class GpuAssignment(BaseModel):
    """One role's GPU placement.

    Three separate identifiers, because they are not interchangeable:

    - ``cuda_visible_devices``: the string exported as ``CUDA_VISIBLE_DEVICES``.
      Index form, because Omniverse Kit selects its render device by index.
    - ``uuid``: stable identity, so a run can assert it got the GPU it expected.
      Indices renumber across driver reloads; UUIDs do not.
    - ``nvml_index``: the *physical* index NVML needs for power sampling. NVML
      ignores ``CUDA_VISIBLE_DEVICES``, so sampling the wrong GPU is a silent
      wrong-number bug rather than a crash.
    """

    model_config = ConfigDict(extra="forbid")

    cuda_visible_devices: str
    role: str
    uuid: str | None = None
    nvml_index: int | None = None


class Topology(BaseModel):
    # Forbid extras so a stale or misspelled key fails loudly instead of being
    # silently ignored (CLAUDE.md: no silent fallbacks).
    model_config = ConfigDict(extra="forbid")

    name: str
    host: str
    gpus: dict[str, GpuAssignment]
    cosmos_host: str
    cosmos_port: int
    shared_hf_cache: str
    uv_cache: str
    model_cache_root: str
    output_root: str
    spare_cuda_devices: list[str] = Field(default_factory=list)
    vram_total_gib_per_gpu: float | None = None
    vram_weights_gib_estimate: float | None = None
    ov_cache_root: str | None = None
    ov_data_root: str | None = None
    cpu_affinity_for_gpus: str | None = None


class MachinePaths(BaseModel):
    hf_cache: str
    uv_cache: str
    model_cache: str
    results_root: str


class MachineEnv(BaseModel):
    hf_token_env: str = "HF_TOKEN"


class MachineRuntime(BaseModel):
    container: bool = False
    python: str = "3.11"


class Machine(BaseModel):
    paths: MachinePaths
    env: MachineEnv = Field(default_factory=MachineEnv)
    runtime: MachineRuntime = Field(default_factory=MachineRuntime)


def _load_yaml(path: Path) -> dict:
    if not path.exists():
        raise FileNotFoundError(f"config not found: {path}")
    return yaml.safe_load(path.read_text()) or {}


def load_topology(path: str | Path = "configs/topology.yaml") -> Topology:
    return Topology.model_validate(_load_yaml(Path(path)))


def load_machine(path: str | Path = "configs/machine.yaml") -> Machine:
    return Machine.model_validate(_load_yaml(Path(path)))


# ---------------------------------------------------------------------------
# Substrates — which Cosmos source tree actually serves the policy.
#
# Until now there was exactly one (``third_party/cosmos-framework``) and its path
# was spelled out in four places: two capability greps and a ``cd`` in
# start_cosmos_server.sh, ``UPSTREAM_MODULE`` in cosmos_server_entry.py, and the
# interpreter in the Makefile's ``test-model``. A second substrate (the group's
# efficiency fork) makes that untenable, and not only for tidiness: every number
# we report has to name the tree that produced it, and a path repeated in four
# files is a path that will eventually disagree with itself.
#
# The substrate therefore resolves once, from configs/substrates.yaml, and gets
# its *location* from reproducibility/source_manifest.json rather than carrying a
# second copy of it — which also ties each substrate to a recorded commit SHA.
# ---------------------------------------------------------------------------


class SubstrateError(RuntimeError):
    """Raised when a substrate cannot be resolved to a usable source tree."""


class Substrate(BaseModel):
    model_config = ConfigDict(extra="forbid")

    #: Must match a ``primary_sources[].name`` in reproducibility/source_manifest.json.
    #: The path lives there, so it is recorded alongside the commit we measured.
    source_name: str
    #: Dotted module of the server's ``__main__``, run via runpy so upstream's own
    #: argument parser sees upstream's own argv.
    server_module: str
    notes: str | None = None


class Substrates(BaseModel):
    model_config = ConfigDict(extra="forbid")

    #: Used when neither an explicit argument nor ``COSMOS_SUBSTRATE`` says otherwise.
    #: Changing this changes what every subsequent measurement means, so it is a
    #: deliberate, reviewable edit rather than something inferred at runtime.
    default: str
    substrates: dict[str, Substrate]


@dataclass(frozen=True)
class ResolvedSubstrate:
    """A substrate checked against the filesystem, ready to launch or record."""

    name: str
    source_name: str
    root: Path
    python: Path
    server_module: str
    server_file: Path
    #: Live ``git rev-parse HEAD`` in the source tree — authoritative.
    commit: str | None
    #: Live ``git status --porcelain`` emptiness — authoritative.
    dirty: bool
    #: What the manifest *recorded*, which may be older. Kept so drift is visible
    #: rather than silent: the manifest is regenerated by hand (`make sources`), so
    #: it lags every commit made on a research branch since.
    manifest_commit: str | None

    @property
    def manifest_stale(self) -> bool:
        return bool(self.manifest_commit) and self.manifest_commit != self.commit

    def provenance(self) -> dict[str, object]:
        """The fields every result artifact should carry to name its substrate."""
        prov: dict[str, object] = {
            "substrate": self.name,
            "source_name": self.source_name,
            "commit": self.commit,
            "dirty": self.dirty,
            "server_module": self.server_module,
        }
        if self.manifest_stale:
            prov["manifest_commit"] = self.manifest_commit
            prov["manifest_stale"] = True
        return prov


def load_substrates(path: str | Path = "configs/substrates.yaml") -> Substrates:
    cfg = Substrates.model_validate(_load_yaml(Path(path)))
    if cfg.default not in cfg.substrates:
        raise SubstrateError(
            f"configs/substrates.yaml sets default={cfg.default!r}, which is not one of "
            f"{sorted(cfg.substrates)}"
        )
    return cfg


def _git_state(root: Path) -> tuple[str | None, bool]:
    """Live HEAD and dirtiness of a source tree.

    Read from git, not from the manifest: the manifest is refreshed by hand
    (``make sources``) and so lags every commit made on a research branch since —
    cosmos-framework's manifest entry was four commits stale when this was written.
    A result stamped with a stale SHA is worse than one stamped with none, because
    it looks authoritative.
    """
    import subprocess  # noqa: PLC0415 - only needed on this path

    def git(*args: str) -> str | None:
        try:
            r = subprocess.run(
                ["git", *args], cwd=root, capture_output=True, text=True, check=False, timeout=15
            )
        except (OSError, subprocess.SubprocessError):
            return None
        return r.stdout.strip() if r.returncode == 0 else None

    head = git("rev-parse", "HEAD")
    status = git("status", "--porcelain")
    return head or None, bool(status)


def _manifest_source(name: str, manifest: Path) -> dict:
    if not manifest.exists():
        raise SubstrateError(f"source manifest not found: {manifest} — run `make sources`")
    data = json.loads(manifest.read_text())
    for entry in data.get("primary_sources", []):
        if entry.get("name") == name:
            return entry
    known = sorted(e.get("name", "?") for e in data.get("primary_sources", []))
    raise SubstrateError(
        f"{manifest.name} has no primary source named {name!r} (have: {', '.join(known)}). "
        "Run `make sources` to register it."
    )


def resolve_substrate(
    name: str | None = None,
    *,
    repo_root: str | Path = ".",
    config_path: str | Path | None = None,
    manifest_path: str | Path | None = None,
    require_venv: bool = True,
    require_server_file: bool = True,
) -> ResolvedSubstrate:
    """Resolve a substrate name to concrete paths, or fail loudly saying why.

    Precedence: explicit ``name`` > ``$COSMOS_SUBSTRATE`` > ``default`` in the config.
    There is deliberately no fallback to "whatever tree happens to be on disk": running
    the fork while a result claims upstream (or the reverse) would silently invalidate
    every comparison this project exists to make.
    """
    root_dir = Path(repo_root)
    cfg = load_substrates(config_path or root_dir / "configs" / "substrates.yaml")

    chosen = name or os.environ.get("COSMOS_SUBSTRATE") or cfg.default
    if chosen not in cfg.substrates:
        raise SubstrateError(
            f"unknown substrate {chosen!r}; configs/substrates.yaml defines "
            f"{sorted(cfg.substrates)}"
        )
    spec = cfg.substrates[chosen]

    entry = _manifest_source(
        spec.source_name,
        Path(manifest_path or root_dir / "reproducibility" / "source_manifest.json"),
    )
    src = (root_dir / entry["local_path"]).resolve()
    if not (src / ".git").is_dir():
        raise SubstrateError(
            f"substrate {chosen!r} points at {src}, which is not a git clone. "
            f"Clone it first (see docs/upstream_patches.md for {spec.source_name})."
        )

    python = src / ".venv" / "bin" / "python"
    if require_venv and not os.access(python, os.X_OK):
        raise SubstrateError(
            f"substrate {chosen!r} has no interpreter at {python} — "
            f"run `COSMOS_SRC={entry['local_path']} bash scripts/setup_cosmos.sh`. "
            "Never substitute another substrate's venv: the wheel stacks differ and the "
            "measurement would not describe the tree we think it does."
        )

    server_file = src / (spec.server_module.replace(".", "/") + ".py")
    if require_server_file and not server_file.is_file():
        raise SubstrateError(
            f"substrate {chosen!r} declares server_module={spec.server_module!r}, but "
            f"{server_file} does not exist. Fix `server_module` in "
            "configs/substrates.yaml for this substrate."
        )

    commit, dirty = _git_state(src)
    return ResolvedSubstrate(
        name=chosen,
        source_name=spec.source_name,
        root=src,
        python=python,
        server_module=spec.server_module,
        server_file=server_file,
        commit=commit,
        dirty=dirty,
        manifest_commit=entry.get("commit"),
    )
