#!/usr/bin/env python3
"""Machine audit — non-destructive inventory.

Writes:
  docs/environment_report.md   (human-readable)
  reproducibility/environment.json  (machine-readable)

No sudo required. Read-only.
"""
from __future__ import annotations

import datetime as dt
import getpass
import json
import os
import platform
import shutil
import socket
import subprocess
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
OUT_MD = REPO_ROOT / "docs" / "environment_report.md"
OUT_JSON = REPO_ROOT / "reproducibility" / "environment.json"


def run(cmd: list[str] | str, timeout: float = 15.0) -> str:
    """Run a command, return stdout (stripped). Never raise."""
    try:
        r = subprocess.run(
            cmd,
            shell=isinstance(cmd, str),
            capture_output=True,
            text=True,
            timeout=timeout,
            check=False,
        )
        return (r.stdout or r.stderr or "").strip()
    except Exception as e:  # noqa: BLE001
        return f"ERROR: {e}"


def have(binary: str) -> bool:
    return shutil.which(binary) is not None


def probe_blockers(env: dict) -> list[str]:
    """Report what actually blocks us on THIS host.

    Deliberately derives every statement from a probe rather than asserting
    host facts: this file previously hardcoded "98 GB VRAM per GPU" and "one
    Blackwell GPU … the other", which silently became false when the project
    moved machines.
    """
    b: list[str] = []
    if not have("nvidia-ctk"):
        b.append(
            "no `nvidia-container-toolkit` — cannot expose GPUs to containers; "
            "must run natively via `uv`"
        )
    elif not env.get("docker_usable"):
        b.append(
            "`nvidia-container-toolkit` present but the docker socket is not "
            "usable by this uid (not in the `docker` group, and no rootless "
            "prerequisites) — the upstream container path is unavailable. "
            "Native `uv` installs need no root, so this is not fatal."
        )
    if not have("git-lfs"):
        b.append(
            "no `git-lfs` — Cosmos3 checkpoint weights on HF LFS won't fetch; "
            "install user-local (e.g. static binary or `uv tool install git-lfs`)"
        )
    if not have("nsys"):
        b.append(
            "no `nsys` (Nsight Systems) — rely on PyTorch profiler; user-local "
            "install possible if we need deeper kernel tracing"
        )
    if not env.get("nvml_total_energy_supported", False):
        b.append(
            "NVML exposes no total-energy counter — energy must be integrated "
            "from sampled power and labelled ESTIMATED"
        )

    procs = env.get("current_gpu_processes") or ""
    if procs.strip():
        n = len([ln for ln in procs.splitlines() if ln.strip()])
        vram = env.get("gpu_memory_total_mib") or "unknown"
        b.append(
            f"{n} process(es) currently on the GPUs — measurement runs must wait "
            f"for a quiet window; correctness runs can proceed "
            f"({vram} MiB VRAM per GPU)"
        )

    # CPU contention distorts every host-side stage (preprocessing,
    # serialization, websocket round-trip, Isaac physics).
    try:
        load1 = os.getloadavg()[0]
        ncpu = os.cpu_count() or 1
        if load1 > 0.5 * ncpu:
            b.append(
                f"CPU is contended (1-min loadavg {load1:.0f} on {ncpu} cores) — "
                "host-side timing stages will be perturbed; record loadavg with "
                "every timed run and prefer a quiet window for final numbers"
            )
    except OSError:
        pass

    # Disk: the install budget is ~100 GB (checkpoint 33 GB + two torch
    # environments + Isaac assets), and /scratch is shared.
    free_gb = env.get("output_fs_free_gb")
    if isinstance(free_gb, (int, float)) and free_gb < 150:
        b.append(
            f"only {free_gb:.0f} GB free on the output filesystem against a "
            "~100 GB install budget on a shared volume — check `df` at every "
            "install boundary and abort rather than filling it"
        )
    return b


def gather() -> dict:
    env: dict = {
        "timestamp_utc": dt.datetime.now(dt.timezone.utc)
        .replace(microsecond=0, tzinfo=None)
        .isoformat()
        + "Z",
        "hostname": socket.gethostname(),
        "user": getpass.getuser(),
        "os": "",
        "kernel": platform.platform(),
        "cpu_model": "",
        "cpu_cores_logical": os.cpu_count() or 0,
        "ram_human": "",
        "repo_fs": "",
    }

    # os-release
    try:
        rel = {}
        with open("/etc/os-release") as f:
            for line in f:
                if "=" in line:
                    k, v = line.rstrip().split("=", 1)
                    rel[k] = v.strip('"')
        env["os"] = rel.get("PRETTY_NAME", "unknown")
    except OSError:
        env["os"] = "unknown"

    env["cpu_model"] = run("lscpu | awk -F': +' '/Model name/ {print $2; exit}'")
    env["ram_human"] = run("free -h | awk '/Mem:/ {print $2}'")
    env["repo_fs"] = run(f"df -h {REPO_ROOT} | tail -1")

    # Disk headroom on the filesystem we will actually write to, and CPU load.
    # Both are contended on a shared host and both gate what we can run.
    try:
        st = os.statvfs(REPO_ROOT)
        env["output_fs_free_gb"] = round(st.f_bavail * st.f_frsize / 1e9, 1)
        env["output_fs_pct_used"] = round(
            100.0 * (1.0 - st.f_bavail / st.f_blocks) if st.f_blocks else 0.0, 1
        )
    except OSError:
        env["output_fs_free_gb"] = None
        env["output_fs_pct_used"] = None
    try:
        env["loadavg_1_5_15"] = list(os.getloadavg())
    except OSError:
        env["loadavg_1_5_15"] = None
    env["top_cpu_processes"] = run(
        "ps -eo user,pcpu,etime,comm --sort=-pcpu --no-headers | head -8"
    )

    # GPUs
    env["gpu_query_csv"] = run(
        "nvidia-smi --query-gpu=index,name,pci.bus_id,memory.total,memory.free,"
        "memory.used,compute_cap,driver_version,pstate --format=csv,noheader"
    )
    env["gpu_count"] = int(run("nvidia-smi -L | wc -l") or 0)
    env["gpu_name"] = run("nvidia-smi --query-gpu=name --format=csv,noheader | head -1")
    env["gpu_memory_total_mib"] = run(
        "nvidia-smi --query-gpu=memory.total --format=csv,noheader,nounits | head -1"
    )
    env["gpu_compute_cap"] = run(
        "nvidia-smi --query-gpu=compute_cap --format=csv,noheader | head -1"
    )
    env["gpu_uuids"] = run("nvidia-smi --query-gpu=index,uuid --format=csv,noheader")
    env["gpu_topo_matrix"] = run("nvidia-smi topo -m 2>/dev/null | head -6")
    env["gpu_driver"] = run("nvidia-smi --query-gpu=driver_version --format=csv,noheader | head -1")
    # The header line reads "Driver Version: X  CUDA Version: Y", so matching on
    # the bare token "Version:" picks up both. Anchor on "CUDA Version:".
    env["cuda_runtime_max"] = run(
        r"nvidia-smi | sed -n 's/.*CUDA Version: *\([0-9.]*\).*/\1/p' | head -1"
    )
    env["current_gpu_processes"] = run(
        "nvidia-smi --query-compute-apps=pid,process_name,used_memory --format=csv,noheader"
    )

    # CUDA toolkit
    if have("nvcc"):
        env["cuda_toolkit_host"] = run("nvcc --version | tail -2 | head -1")
    elif Path("/usr/local/cuda/bin/nvcc").exists():
        env["cuda_toolkit_host"] = run("/usr/local/cuda/bin/nvcc --version | tail -2 | head -1")
    else:
        env["cuda_toolkit_host"] = "not installed"

    # Tools
    env["uv_version"] = run("uv --version") or "MISSING"
    env["git_version"] = run("git --version") or "MISSING"
    env["git_lfs_version"] = run("git-lfs --version") if have("git-lfs") else "MISSING"
    env["nsys_version"] = run("nsys --version | head -1") if have("nsys") else "MISSING"
    env["ncu_version"] = run("ncu --version | head -1") if have("ncu") else "MISSING"
    env["python_uv_installed"] = run(
        "uv python list --only-installed 2>/dev/null | awk '{print $1}' | sort -u"
    )

    # Container. `docker --version` only proves the *client* exists; it says
    # nothing about whether this uid may talk to the daemon socket. Probe that
    # separately, or we repeat the earlier mistake of concluding "docker is
    # available" from a client binary.
    env["docker_version"] = run("docker --version") if have("docker") else "MISSING"
    env["docker_usable"] = bool(
        have("docker") and run("docker info --format '{{.ServerVersion}}' 2>/dev/null")
    )
    env["docker_group_members"] = run("getent group docker | cut -d: -f4")
    env["rootless_prereqs"] = bool(
        Path("/usr/bin/newuidmap").exists() and Path("/usr/bin/newgidmap").exists()
    )
    env["podman_version"] = run("podman --version") if have("podman") else "MISSING"
    env["nvidia_container_toolkit"] = (
        run("nvidia-ctk --version | head -1") if have("nvidia-ctk") else "MISSING"
    )
    cdi = []
    for p in ("/etc/cdi", "/var/run/cdi"):
        if Path(p).is_dir():
            cdi.append(p)
    env["cdi_present"] = ",".join(cdi) if cdi else "none"

    subuid = run(f"awk -F: '$1==\"{env['user']}\"' /etc/subuid")
    env["subuid_configured"] = bool(subuid)
    env["subuid_line"] = subuid

    # Network
    env["network_github_http"] = run(
        "curl -sS -o /dev/null -w '%{http_code}' --max-time 10 https://github.com"
    )
    env["network_hf_http"] = run(
        "curl -sS -o /dev/null -w '%{http_code}' --max-time 10 https://huggingface.co"
    )

    # NVML total-energy counter probe
    e = run("nvidia-smi --query-gpu=total_energy_consumption --format=csv,noheader")
    env["nvml_total_energy_probe"] = e
    env["nvml_total_energy_supported"] = "not a valid field" not in e.lower()

    return env


def render_md(env: dict, blockers: list[str]) -> str:
    lines: list[str] = []
    lines.append(f"# Environment report\n")
    lines.append(f"Generated: `{env['timestamp_utc']}` on `{env['hostname']}`  ")
    lines.append(f"User: `{env['user']}`\n")

    lines.append("## Host\n")
    lines.append(f"- OS: **{env['os']}**")
    lines.append(f"- Kernel: `{env['kernel']}`")
    lines.append(f"- CPU: **{env['cpu_model']}** ({env['cpu_cores_logical']} logical cores)")
    lines.append(f"- RAM: **{env['ram_human']}**")
    lines.append(f"- Repo filesystem: `{env['repo_fs']}`")
    lines.append(
        f"- Output filesystem free: **{env.get('output_fs_free_gb')} GB** "
        f"({env.get('output_fs_pct_used')}% used) — install budget is ~100 GB"
    )
    lines.append(f"- Load average (1/5/15 min): `{env.get('loadavg_1_5_15')}`\n")

    lines.append("### Top CPU consumers (contention risk for host-side timing)\n")
    lines.append("```")
    lines.append(env.get("top_cpu_processes") or "(none)")
    lines.append("```\n")

    lines.append("## GPUs\n")
    lines.append(
        f"Count: **{env['gpu_count']}** — driver **{env['gpu_driver']}** — "
        f"max CUDA runtime: **{env['cuda_runtime_max']}** — "
        f"compute capability **{env.get('gpu_compute_cap')}**\n"
    )
    lines.append("```")
    lines.append(env["gpu_query_csv"] or "(nvidia-smi unavailable)")
    lines.append("```\n")

    lines.append("### UUIDs (stable identity — indices renumber, UUIDs do not)\n")
    lines.append("```")
    lines.append(env.get("gpu_uuids") or "(unavailable)")
    lines.append("```\n")

    lines.append("### PCIe topology (place co-running services on separate switches)\n")
    lines.append("```")
    lines.append(env.get("gpu_topo_matrix") or "(unavailable)")
    lines.append("```\n")

    lines.append("### Current GPU occupants (contention risk for measurement)\n")
    lines.append("```")
    lines.append(env["current_gpu_processes"] or "(none)")
    lines.append("```\n")

    lines.append(f"### CUDA toolkit on host\n`{env['cuda_toolkit_host']}`\n")
    lines.append(f"### NVML total-energy counter\n`{env['nvml_total_energy_probe']}`\n")
    lines.append(
        "→ integrate NVML power over time for energy (counter unsupported here).\n"
        if not env["nvml_total_energy_supported"]
        else "→ total-energy counter is available.\n"
    )

    lines.append("## Tools\n")
    for k in (
        "uv_version",
        "git_version",
        "git_lfs_version",
        "nsys_version",
        "ncu_version",
    ):
        lines.append(f"- `{k}`: {env[k]}")
    lines.append("\n### uv-managed Pythons\n```")
    lines.append(env["python_uv_installed"])
    lines.append("```\n")

    lines.append("## Containers\n")
    lines.append(f"- docker      : {env['docker_version']}")
    lines.append(
        f"- docker usable by this uid : **{env.get('docker_usable')}** "
        f"(group members: `{env.get('docker_group_members') or '(none)'}`, "
        f"rootless prereqs: {env.get('rootless_prereqs')})"
    )
    lines.append(f"- podman      : {env['podman_version']}")
    lines.append(f"- nvidia-ctk  : {env['nvidia_container_toolkit']}")
    lines.append(f"- CDI dirs    : {env['cdi_present']}")
    lines.append(f"- subuid line : `{env['subuid_line']}` (configured={env['subuid_configured']})\n")

    lines.append("## Network\n")
    lines.append(f"- github.com     : HTTP {env['network_github_http']}")
    lines.append(f"- huggingface.co : HTTP {env['network_hf_http']}\n")

    lines.append("## Blockers identified\n")
    if not blockers:
        lines.append("_(none)_\n")
    else:
        for b in blockers:
            lines.append(f"- {b}")
        lines.append("")

    lines.append("## Selected deployment topology\n")
    n = env.get("gpu_count") or 0
    name = env.get("gpu_name") or "GPU"
    vram = env.get("gpu_memory_total_mib") or "?"
    if n >= 2:
        lines.append(
            f"**`single_host_multi_gpu`** — {n}× {name} ({vram} MiB each). One GPU "
            "pinned to the Cosmos policy server, a second to RoboLab/Isaac Sim via "
            "`CUDA_VISIBLE_DEVICES`; any remainder is spare capacity for later "
            "parallel sweeps. Exact index/UUID assignment lives in "
            "`configs/topology.yaml` — that file is authoritative, not this text.\n"
        )
    elif n == 1:
        lines.append(
            f"**`single_host_shared_gpu`** — only one {name} ({vram} MiB) is "
            "present, so Cosmos and Isaac Sim must share it. Per spec §5 this is "
            "only valid once memory measurements show both fit safely; measure "
            "before assuming. See `configs/topology.yaml`.\n"
        )
    else:
        lines.append(
            "**BLOCKED** — no NVIDIA GPU detected. RoboLab requires one; finish "
            "source checkout and non-GPU setup, then report the blocker.\n"
        )
    return "\n".join(lines) + "\n"


def main() -> int:
    OUT_MD.parent.mkdir(parents=True, exist_ok=True)
    OUT_JSON.parent.mkdir(parents=True, exist_ok=True)

    env = gather()
    blockers = probe_blockers(env)
    env["blockers"] = blockers

    OUT_MD.write_text(render_md(env, blockers))
    OUT_JSON.write_text(json.dumps(env, indent=2, sort_keys=True))

    print(f"wrote {OUT_MD.relative_to(REPO_ROOT)}")
    print(f"wrote {OUT_JSON.relative_to(REPO_ROOT)}")
    if blockers:
        print(f"blockers: {len(blockers)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
