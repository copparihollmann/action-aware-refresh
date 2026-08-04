"""Config loaders for machine.yaml and topology.yaml."""
from __future__ import annotations

from pathlib import Path

import yaml
from pydantic import BaseModel, Field


class GpuAssignment(BaseModel):
    cuda_visible_devices: str
    role: str


class Topology(BaseModel):
    name: str
    host: str
    gpus: dict[str, GpuAssignment]
    cosmos_host: str
    cosmos_port: int
    shared_hf_cache: str
    uv_cache: str
    model_cache_root: str
    output_root: str


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
