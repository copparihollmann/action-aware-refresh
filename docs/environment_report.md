# Environment report

Generated: `2026-08-04T02:03:14Z` on `firesim2`  
User: `copparihollmann`

## Host

- OS: **Ubuntu 24.04.4 LTS**
- Kernel: `Linux-6.8.0-124-generic-x86_64-with-glibc2.39`
- CPU: **Intel(R) Xeon(R) Gold 6242 CPU @ 2.80GHz** (64 logical cores)
- RAM: **503Gi**
- Repo filesystem: `/dev/md2        3.5T  3.2T  133G  97% /scratch`
- Output filesystem free: **142.5 GB** (96.2% used) — install budget is ~100 GB
- Load average (1/5/15 min): `[47.201171875, 45.55712890625, 38.1181640625]`

### Top CPU consumers (contention risk for host-side timing)

```
chrisdo+  102       00:00 cicc
chrisdo+  100       00:09 cc1plus
chrisdo+ 99.9    05:49:36 nvcc
chrisdo+ 99.9    05:06:10 nvcc
chrisdo+ 99.9    04:10:13 ccache
chrisdo+ 99.9    05:39:01 nvcc
chrisdo+ 99.9    05:20:53 gcc
chrisdo+ 99.9    04:50:52 c++
```

## GPUs

Count: **4** — driver **580.126.18** — max CUDA runtime: **13.0** — compute capability **8.9**

```
0, NVIDIA L40S, 00000000:1B:00.0, 46068 MiB, 45353 MiB, 105 MiB, 8.9, 580.126.18, P0
1, NVIDIA L40S, 00000000:1C:00.0, 46068 MiB, 45365 MiB, 93 MiB, 8.9, 580.126.18, P8
2, NVIDIA L40S, 00000000:3D:00.0, 46068 MiB, 45345 MiB, 113 MiB, 8.9, 580.126.18, P8
3, NVIDIA L40S, 00000000:40:00.0, 46068 MiB, 45365 MiB, 93 MiB, 8.9, 580.126.18, P8
```

### UUIDs (stable identity — indices renumber, UUIDs do not)

```
0, GPU-a2a44f60-c7d1-4450-93df-ead875ea76e8
1, GPU-6a6a9d5a-07da-01f7-4acb-594afc813213
2, GPU-22350958-ee15-b07b-0e89-77c5157f1d3b
3, GPU-019a51ce-7238-4d15-f03d-e9194d1c4e79
```

### PCIe topology (place co-running services on separate switches)

```
[4mGPU0	GPU1	GPU2	GPU3	CPU Affinity	NUMA Affinity	GPU NUMA ID[0m
GPU0	 X 	PIX	NODE	NODE	0-15,32-47	0		N/A
GPU1	PIX	 X 	NODE	NODE	0-15,32-47	0		N/A
GPU2	NODE	NODE	 X 	PIX	0-15,32-47	0		N/A
GPU3	NODE	NODE	PIX	 X 	0-15,32-47	0		N/A
```

### Current GPU occupants (contention risk for measurement)

```
(none)
```

### CUDA toolkit on host
`Cuda compilation tools, release 12.6, V12.6.85`

### NVML total-energy counter
`Field "total_energy_consumption" is not a valid field to query.`

→ integrate NVML power over time for energy (counter unsupported here).

## Tools

- `uv_version`: uv 0.12.1 (x86_64-unknown-linux-gnu)
- `git_version`: git version 2.43.0
- `git_lfs_version`: git-lfs/3.7.0 (GitHub; linux amd64; go 1.24.4; git 92dddf56)
- `nsys_version`: MISSING
- `ncu_version`: MISSING

### uv-managed Pythons
```
cpython-3.11.15-linux-x86_64-gnu
cpython-3.12.3-linux-x86_64-gnu
```

## Containers

- docker      : Docker version 29.5.2, build 79eb04c
- podman      : MISSING
- nvidia-ctk  : NVIDIA Container Toolkit CLI version 1.19.1
- CDI dirs    : /var/run/cdi
- subuid line : `` (configured=False)

## Network

- github.com     : HTTP 200
- huggingface.co : HTTP 200

## Blockers identified

- `nvidia-container-toolkit` present but the docker socket is not usable by this uid (not in the `docker` group, and no rootless prerequisites) — the upstream container path is unavailable. Native `uv` installs need no root, so this is not fatal.
- no `nsys` (Nsight Systems) — rely on PyTorch profiler; user-local install possible if we need deeper kernel tracing
- NVML exposes no total-energy counter — energy must be integrated from sampled power and labelled ESTIMATED
- CPU is contended (1-min loadavg 47 on 64 cores) — host-side timing stages will be perturbed; record loadavg with every timed run and prefer a quiet window for final numbers
- only 142 GB free on the output filesystem against a ~100 GB install budget on a shared volume — check `df` at every install boundary and abort rather than filling it

## Selected deployment topology

**`single_host_multi_gpu`** — 4× NVIDIA L40S (46068 MiB each). One GPU pinned to the Cosmos policy server, a second to RoboLab/Isaac Sim via `CUDA_VISIBLE_DEVICES`; any remainder is spare capacity for later parallel sweeps. Exact index/UUID assignment lives in `configs/topology.yaml` — that file is authoritative, not this text.

