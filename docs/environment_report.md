# Environment report

Generated: `2026-08-03T23:59:38Z` on `bwrc-bwell`  
User: `copparihollmann`

## Host

- OS: **Red Hat Enterprise Linux 9.7 (Plow)**
- Kernel: `Linux-5.14.0-611.47.1.el9_7.x86_64-x86_64-with-glibc2.34`
- CPU: **Intel(R) Xeon(R) w5-3535X** (40 logical cores)
- RAM: **754Gi**
- Repo filesystem: `/dev/nvme0n1p1  7.0T  858G  5.8T  13% /scratch`

## GPUs

Count: **2** — driver **580.82.09** — max CUDA runtime: **580.82.09
13.0**

```
0, NVIDIA RTX PRO 6000 Blackwell Workstation Edition, 00000000:34:00.0, 97887 MiB, 90625 MiB, 6624 MiB, 12.0, 580.82.09, P1
1, NVIDIA RTX PRO 6000 Blackwell Workstation Edition, 00000000:CA:00.0, 97887 MiB, 91478 MiB, 5771 MiB, 12.0, 580.82.09, P1
```

### Current GPU occupants (contention risk for measurement)

```
906103, python, 6060 MiB
906269, python, 550 MiB
906269, python, 5762 MiB
```

### CUDA toolkit on host
`Cuda compilation tools, release 12.6, V12.6.85`

### NVML total-energy counter
`Field "total_energy_consumption" is not a valid field to query.`

→ integrate NVML power over time for energy (counter unsupported here).

## Tools

- `uv_version`: uv 0.11.7 (x86_64-unknown-linux-gnu)
- `git_version`: git version 2.47.3
- `git_lfs_version`: MISSING
- `nsys_version`: MISSING
- `ncu_version`: MISSING

### uv-managed Pythons
```
cpython-3.10.20-linux-x86_64-gnu
cpython-3.11.15-linux-x86_64-gnu
cpython-3.12.13-linux-x86_64-gnu
cpython-3.9.25-linux-x86_64-gnu
```

## Containers

- docker      : MISSING
- podman      : podman version 5.6.0
- nvidia-ctk  : MISSING
- CDI dirs    : none
- subuid line : `` (configured=False)

## Network

- github.com     : HTTP 200
- huggingface.co : HTTP 200

## Blockers identified

- no `nvidia-container-toolkit` — cannot expose GPUs to podman/docker containers; must run natively via `uv`
- no `git-lfs` — Cosmos3 checkpoint weights on HF LFS won't fetch; install user-local (e.g. static binary or `uv tool install git-lfs`)
- no `nsys` (Nsight Systems) — rely on PyTorch profiler; user-local install possible if we need deeper kernel tracing
- no `docker` — using rootless `podman` instead when unavoidable
- `subuid` not configured — podman rootless is in single-mapping mode; some images may misbehave
- 3 process(es) currently on the GPUs — measurement runs must wait for a quiet window; correctness runs can proceed (98 GB VRAM per GPU)

## Selected deployment topology

**`single_host_multi_gpu`** — one Blackwell GPU pinned to Cosmos policy server, the other to RoboLab/Isaac Sim via `CUDA_VISIBLE_DEVICES`. See `configs/topology.yaml`.

