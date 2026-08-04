# Troubleshooting

Living record of issues and fixes. New entries at bottom.

---

## `podman info` complains about `subuid` on rootless

`subuid` is not configured for `copparihollmann`. Podman falls back to
single-mapping mode. Fine for read-only images and Cosmos wouldn't be
using containers anyway. If we ever need multi-user containers, ask
sysadmin to add lines to `/etc/subuid` and `/etc/subgid`.

## `git-lfs` missing → Cosmos3 model weights won't fetch

Install user-local:

```bash
# Option 1: statically linked binary → ~/.local/bin
mkdir -p ~/.local/bin && cd /tmp
curl -sL https://github.com/git-lfs/git-lfs/releases/download/v3.5.1/git-lfs-linux-amd64-v3.5.1.tar.gz \
  | tar xz && install -m 0755 git-lfs-*/git-lfs ~/.local/bin/
git lfs install
```

Verify with `which git-lfs && git-lfs --version`.

## NVML `total_energy_consumption` returns "not a valid field"

Not supported on this GPU/driver. Use `EnergyMeter` (10 Hz power
integration) from `src/action_refresh/energy.py`.

## `nvidia-smi` shows other users' processes on our target GPU

Check `nvidia-smi --query-compute-apps=pid,process_name,used_memory
--format=csv,noheader` before measurement runs. If other processes are
using >5 GB VRAM or >10% util, either wait or fall back to the other GPU
via `CUDA_VISIBLE_DEVICES`.
