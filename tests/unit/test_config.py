"""Config loader smoke test — the topology.yaml checked in must be valid."""
from __future__ import annotations

from pathlib import Path

from action_refresh.config import load_topology


def test_topology_loads() -> None:
    t = load_topology(Path(__file__).resolve().parents[2] / "configs" / "topology.yaml")
    assert t.name == "single_host_multi_gpu"
    assert t.cosmos_port == 8000
    assert "cosmos_server" in t.gpus
    assert "robolab_client" in t.gpus
