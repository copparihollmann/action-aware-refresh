# Source map

Where every referenced repo/paper lives and how we use it.

## Primary implementation

| repo | phase | what we take | what we don't |
|---|---|---|---|
| [NVIDIA/cosmos](https://github.com/NVIDIA/cosmos) | M0 reference | high-level overview + shared model configs | not our runtime path |
| [NVIDIA/cosmos-framework](https://github.com/NVIDIA/cosmos-framework) | **M1 runtime** | policy server (`action_policy_server_robolab.py`), inference/loading utilities, dependency groups (`cu130-train`, `policy-server`) | training pipelines (we do NOT retrain) |
| [NVLabs/RoboLab](https://github.com/NVLabs/RoboLab) | **M1 evaluator** | Cosmos3 client (`policies/cosmos3/run.py`), Isaac Sim env registry, task list, evaluation harness | new task authoring during M0–M2 |
| [NVlabs/cosmos-policy](https://github.com/NVlabs/cosmos-policy) | reference | alternative policy loading paths | secondary to cosmos-framework |
| [Physical-Intelligence/openpi](https://github.com/Physical-Intelligence/openpi) | **M9 transfer** | π0.5 loading + inference for the second-architecture check | not until Cosmos3 primary is done |

## Server / client entry points

- Cosmos server (native uv install):
  `third_party/cosmos-framework/cosmos_framework/scripts/action_policy_server_robolab.py`
- RoboLab client:
  `third_party/RoboLab/policies/cosmos3/run.py`
- HF model: `nvidia/Cosmos3-Nano-Policy-DROID` (pin the revision in
  `reproducibility/model_revisions.json` at first pull).

## Research references — see `docs/literature_notes.md` for what we take

- **ImageWAM** — architecture context (world-action model at a higher scale).
- **Foveated Diffusion** — spatial-token relevance masking.
- **DeltaTok / DeltaWorld** — token-delta reuse across frames.
- **Fast-WAM** — action-first, ignore visual imagination challenge.
- **TeaCache / FasterCache / DeepCache / token-wise feature caching** — timestep + block caching baselines for M5.
- **v2e / ESIM** — event-camera simulators for M4 offline robustness.
- **RAFT (torchvision)** — optical flow gate signal, D3.

## Tooling docs (external, no clone)

- uv: https://docs.astral.sh/uv/
- PyTorch Profiler: https://docs.pytorch.org/tutorials/recipes/recipes/profiler_recipe.html
- Nsight Systems: https://docs.nvidia.com/nsight-systems/UserGuide/index.html
- NVIDIA Container Toolkit: https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/latest/install-guide.html
  (not usable on this host — recorded for a future move to a container-capable machine).
