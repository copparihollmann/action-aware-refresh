#!/usr/bin/env bash
# scripts/lib/common.sh — shared shell helpers. Source, don't execute:
#
#   source "$REPO_ROOT/scripts/lib/common.sh"
#
# Expects $REPO_ROOT to be set by the caller.

# ---------------------------------------------------------------------------
# REPO_PY — the interpreter that can import *this repo's* dependencies.
#
# Bare `python3` is the system 3.12: it happens to have PyYAML but NOT pydantic,
# so `from action_refresh.config import load_topology` dies with
# ModuleNotFoundError. That is exactly how the first closed-loop smoke attempt
# failed — the topology read exploded, $TOPO_DEVICES came back unbound, and
# `set -u` aborted the server two seconds in.
#
# Resolve the repo venv explicitly and fail loudly if it is missing. Do NOT add
# a system-python fallback: a fallback here would silently skip the GPU-UUID
# assertion, which is the one guard standing between us and attributing timings
# to the wrong device.
#
# Note this is the *repo* venv (pydantic, yaml, structlog for our own tooling),
# which is a different environment from third_party/*/.venv (torch, isaacsim).
# Never conflate them.
# ---------------------------------------------------------------------------
resolve_repo_py() {
  local py="$REPO_ROOT/.venv/bin/python"
  if [ ! -x "$py" ]; then
    echo "error: repo venv interpreter not found at $py" >&2
    echo "→ run 'make venv' (uv sync --extra dev) and retry." >&2
    return 3
  fi
  # Assert the imports we actually rely on, so the failure names the cause
  # rather than surfacing as an unbound variable three lines later.
  if ! "$py" -c 'import pydantic, yaml' 2>/dev/null; then
    echo "error: $py cannot import pydantic/yaml — the repo venv is incomplete." >&2
    echo "→ run 'make venv' (uv sync --extra dev) and retry." >&2
    return 3
  fi
  REPO_PY="$py"
}

# ---------------------------------------------------------------------------
# gpu_pin_from_topology <role> — echo "DEVICES UUID" for a topology role and
# assert the live GPU at that index really has the recorded UUID.
#
# Indices renumber across driver reloads; a silent swap would attribute every
# measurement to the wrong device. Pinning by index and *verifying* by UUID is
# the only combination that is both usable by CUDA and safe.
# ---------------------------------------------------------------------------
topology_devices() {
  local role="$1"
  "$REPO_PY" - "$role" <<'PY'
import sys

sys.path.insert(0, "src")
from action_refresh.config import load_topology

role = sys.argv[1]
t = load_topology("configs/topology.yaml")
if role not in t.gpus:
    sys.stderr.write(
        f"error: configs/topology.yaml has no gpus.{role} "
        f"(have: {', '.join(sorted(t.gpus))})\n"
    )
    raise SystemExit(4)
g = t.gpus[role]
print(f"TOPO_DEVICES={g.cuda_visible_devices}")
print(f"TOPO_UUID={g.uuid or ''}")
print(f"TOPO_NVML={g.nvml_index if g.nvml_index is not None else ''}")
PY
}

# ---------------------------------------------------------------------------
# export_cache_env — point HF_HOME / UV_CACHE_DIR at the /scratch paths from
# configs/machine.yaml.
#
# setup_cosmos.sh already did this at *install* time, but the launchers did not,
# so the server resolved HF_HOME to its default $HOME/.cache/huggingface and
# re-downloaded the entire 34 GB checkpoint onto a 60 GB NFS home volume — then
# mmap-paged it back in at 20 MiB/s, turning a ~40 s model load into ~25 min.
# Same failure class as the `uv run` re-resolve: the RUN path diverged from the
# INSTALL path, so the thing we measured was not the thing we installed.
#
# No fallback: a wrong default here writes tens of GB to an arbitrary volume.
# ---------------------------------------------------------------------------
export_cache_env() {
  if [ ! -f configs/machine.yaml ]; then
    echo "error: configs/machine.yaml not found." >&2
    echo "→ cp configs/machine.example.yaml configs/machine.yaml and edit the paths" >&2
    echo "  (or export HF_HOME and UV_CACHE_DIR explicitly)." >&2
    return 2
  fi
  eval "$("$REPO_PY" <<'PY'
import yaml

paths = yaml.safe_load(open("configs/machine.yaml"))["paths"]
print("HF_HOME_DEFAULT=" + paths["hf_cache"])
print("UV_CACHE_DEFAULT=" + paths["uv_cache"])
PY
)"
  export HF_HOME="${HF_HOME:-$HF_HOME_DEFAULT}"
  export UV_CACHE_DIR="${UV_CACHE_DIR:-$UV_CACHE_DEFAULT}"
  mkdir -p "$HF_HOME" "$UV_CACHE_DIR"
}

# ---------------------------------------------------------------------------
# export_omni_cache_env — redirect Omniverse/Kit caches to /scratch.
#
# Isaac Sim defaults OMNI_CACHE_ROOT / OMNI_DATA_ROOT into $HOME, which on this
# host is a 60 GB NFS volume; shader and asset caches there are both a quota
# risk and slow. setup_robolab.sh redirected them at install time; the client
# launcher did not. Same run-vs-install divergence as export_cache_env.
# ---------------------------------------------------------------------------
export_omni_cache_env() {
  eval "$("$REPO_PY" <<'PY'
import sys

sys.path.insert(0, "src")
from action_refresh.config import load_topology

t = load_topology("configs/topology.yaml")
print(f"OV_CACHE={t.ov_cache_root or ''}")
print(f"OV_DATA={t.ov_data_root or ''}")
PY
)"
  if [ -n "${OV_CACHE:-}" ]; then
    export OMNI_CACHE_ROOT="${OMNI_CACHE_ROOT:-$OV_CACHE}"
    export OMNI_DATA_ROOT="${OMNI_DATA_ROOT:-${OV_DATA:-$OV_CACHE/data}}"
    mkdir -p "$OMNI_CACHE_ROOT" "$OMNI_DATA_ROOT"
  fi
}

# ---------------------------------------------------------------------------
# resolve_substrate [name] [require_venv] — decide which Cosmos source tree
# serves the policy, and export it.
#
# Two substrates now exist (NVIDIA upstream and the group's efficiency fork), so
# the path can no longer be a literal in each launcher: a result that does not
# name its substrate is uninterpretable, and four copies of a path eventually
# disagree. configs/substrates.yaml holds the choice, the manifest holds the
# location and SHA, and `action_refresh.config.resolve_substrate` does the checks.
#
# Exports COSMOS_SUBSTRATE, COSMOS_SRC, COSMOS_PY, COSMOS_SERVER_MODULE,
# COSMOS_SERVER_FILE, COSMOS_SUBSTRATE_COMMIT, COSMOS_SUBSTRATE_DIRTY.
#
# Requires resolve_repo_py first. Fails loudly — there is no "use whatever is on
# disk" path, because running the fork under a result labelled upstream (or the
# reverse) would silently invalidate every comparison.
# ---------------------------------------------------------------------------
resolve_substrate() {
  local name="${1:-${COSMOS_SUBSTRATE:-}}"
  local require_venv="${2:-true}"
  local out
  # Capture first, then eval: `eval "$(cmd)"` swallows cmd's exit status, so a
  # failed resolution would silently leave the variables unset and `set -u` would
  # blame a line three steps downstream.
  if ! out="$("$REPO_PY" - "$REPO_ROOT" "$name" "$require_venv" <<'PY'
import sys

repo_root, name, require_venv = sys.argv[1], sys.argv[2], sys.argv[3]
sys.path.insert(0, f"{repo_root}/src")
from action_refresh.config import SubstrateError, resolve_substrate

try:
    s = resolve_substrate(
        name or None,
        repo_root=repo_root,
        require_venv=require_venv.lower() == "true",
    )
except SubstrateError as exc:
    sys.stderr.write(f"error: {exc}\n")
    raise SystemExit(12)

if s.manifest_stale:
    sys.stderr.write(
        f"warning: source_manifest.json records {(s.manifest_commit or '')[:12]} for "
        f"{s.source_name}, but HEAD is {(s.commit or '')[:12]}. The run will be stamped "
        "with the live SHA; run `make sources` to refresh the record.\n"
    )

print(f"COSMOS_SUBSTRATE={s.name}")
print(f"COSMOS_SRC={s.root}")
print(f"COSMOS_PY={s.python}")
print(f"COSMOS_SERVER_MODULE={s.server_module}")
print(f"COSMOS_SERVER_FILE={s.server_file}")
print(f"COSMOS_SUBSTRATE_COMMIT={s.commit or ''}")
print(f"COSMOS_SUBSTRATE_DIRTY={'true' if s.dirty else 'false'}")
PY
  )"; then
    echo "→ check configs/substrates.yaml and \$COSMOS_SUBSTRATE." >&2
    return 12
  fi
  eval "$out"
  export COSMOS_SUBSTRATE COSMOS_SRC COSMOS_PY COSMOS_SERVER_MODULE COSMOS_SERVER_FILE
  export COSMOS_SUBSTRATE_COMMIT COSMOS_SUBSTRATE_DIRTY
}

assert_gpu_uuid() {
  local role="$1" devices="$2" expected_uuid="$3"
  [ -z "$expected_uuid" ] && return 0
  local actual
  actual="$(nvidia-smi --query-gpu=uuid --format=csv,noheader --id="$devices" | tr -d '[:space:]')"
  if [ "$actual" != "$expected_uuid" ]; then
    echo "error: GPU index $devices has uuid $actual," >&2
    echo "       but configs/topology.yaml records $expected_uuid for $role." >&2
    echo "→ the driver renumbered, or topology.yaml is stale. Re-run 'make audit'." >&2
    return 8
  fi
}
