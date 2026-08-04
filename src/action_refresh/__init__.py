"""Action-aware predictive refresh — research package.

Public API is intentionally small during M0–M2:
- `metrics`   : typed request/episode schemas + JSONL/Parquet writers
- `profiler`  : CUDA/wall/PyTorch-profiler timing layers
- `energy`    : NVML power sampling → integrated energy
- `config`    : machine.yaml + topology.yaml loaders
- `logging`   : structlog setup

Higher-level modules (server, client, gates, cache, oracle, analysis) are
scaffolded but not implemented until later milestones.
"""
__version__ = "0.0.0"
