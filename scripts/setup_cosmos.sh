#!/usr/bin/env bash
# scripts/setup_cosmos.sh — install Cosmos Framework natively via `uv`.
#
# Upstream documents a `docker build` path, but the docker socket is not usable
# by this uid (not in the `docker` group; no rootless prerequisites) — see the
# Containers section of docs/environment_report.md. It is also unnecessary: this
# host is already Ubuntu 24.04, the same base as the upstream image, and `uv`
# needs no root. So we install into `third_party/cosmos-framework/.venv` using
# the repo's own pyproject/uv config.
#
# Requires: cloned third_party/cosmos-framework (see scripts/clone_sources.sh).
#
# Env:
#   COSMOS_SRC      source tree to install, relative to the repo root.
#                   Default third_party/cosmos-framework. For the group's efficiency
#                   fork: COSMOS_SRC=third_party/cosmos3-efficient-imagination.
#                   Each tree gets its OWN .venv — the forks may pin different wheels,
#                   and sharing an environment would make "which stack produced this
#                   number" unanswerable. This script takes a path rather than a
#                   substrate name so it stays runnable before the repo venv exists
#                   (configs/substrates.yaml is resolved by the launchers, which need
#                   pydantic; bootstrap must not).
#   HF_TOKEN        export before pulling a *gated* model (never written to disk).
#                   Cosmos3-Nano-Policy-DROID is NOT gated, so it is optional.
#   HF_HOME         set below → persistent cache on /scratch
#   UV_CACHE_DIR    set below → persistent uv cache
#   COSMOS_GROUP    override the cuXXX dependency group (default: from driver)
#   COSMOS_EXTRAS   e.g. "--all-extras" to match upstream's documented command
#                   exactly. Omitted by default because the `train` extra is
#                   large and M1/M2 only serve inference; add it if an import
#                   fails, and record the change in docs/decision_log.md.

set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO_ROOT"

: "${COSMOS_SRC:=third_party/cosmos-framework}"
COSMOS_SRC_NAME="$(basename "$COSMOS_SRC")"

if [ ! -d "$COSMOS_SRC/.git" ]; then
  echo "error: $COSMOS_SRC is not a git clone." >&2
  echo "→ run scripts/clone_sources.sh (or, for the private fork, clone it yourself" >&2
  echo "  — clone_sources.sh prints the command and will not handle credentials)." >&2
  exit 2
fi
echo "installing substrate source: $COSMOS_SRC"

# ---- caches ---------------------------------------------------------------
# Read machine.yaml if it exists, otherwise use defaults from topology.yaml.
if [ -f configs/machine.yaml ]; then
  # Heredoc, not `python3 -c '...'`: backslash-escaped quotes inside a
  # single-quoted shell string are a SyntaxError in an f-string expression.
  eval "$(python3 <<'PY'
import yaml

paths = yaml.safe_load(open("configs/machine.yaml"))["paths"]
print("HF_HOME_DEFAULT=" + paths["hf_cache"])
print("UV_CACHE_DEFAULT=" + paths["uv_cache"])
PY
)"
else
  # No silent fallback to someone else's paths: a wrong default here dumps
  # ~100 GB into an arbitrary directory on a shared, nearly-full volume.
  echo "error: configs/machine.yaml not found." >&2
  echo "→ cp configs/machine.example.yaml configs/machine.yaml and edit the paths" >&2
  echo "  (or export HF_HOME and UV_CACHE_DIR explicitly)." >&2
  exit 2
fi
export HF_HOME="${HF_HOME:-$HF_HOME_DEFAULT}"
export UV_CACHE_DIR="${UV_CACHE_DIR:-$UV_CACHE_DEFAULT}"
mkdir -p "$HF_HOME" "$UV_CACHE_DIR"

echo "HF_HOME     = $HF_HOME"
echo "UV_CACHE_DIR= $UV_CACHE_DIR"

# ---- dep group selection --------------------------------------------------
# Per the Cosmos3 cookbook, the cuXXX group must match the CUDA version the
# *driver* supports (not the host toolkit): 13.x → cu130, 12.x → cu128.
cd "$COSMOS_SRC"

DRIVER_CUDA="$(nvidia-smi | sed -n 's/.*CUDA Version: *\([0-9]*\)\..*/\1/p' | head -1)"
echo "driver reports CUDA major = ${DRIVER_CUDA:-unknown}"

GROUP="${COSMOS_GROUP:-}"
if [ -z "$GROUP" ]; then
  case "$DRIVER_CUDA" in
    13) PREFER="cu130-train"; ALT="cu128-train" ;;
    12) PREFER="cu128-train"; ALT="cu130-train" ;;
    *)
      echo "error: could not determine the driver's CUDA major version." >&2
      echo "→ set COSMOS_GROUP explicitly (cu130-train or cu128-train)." >&2
      exit 3
      ;;
  esac
  if grep -q "^${PREFER} = \|^${PREFER}=" pyproject.toml 2>/dev/null || grep -q "$PREFER" pyproject.toml 2>/dev/null; then
    GROUP="$PREFER"
  elif grep -q "$ALT" pyproject.toml 2>/dev/null; then
    GROUP="$ALT"
    echo "warning: ${PREFER} not present upstream; falling back to ${ALT}." >&2
    echo "         Record this in docs/decision_log.md — it changes the wheel stack." >&2
  else
    echo "error: neither cu130-train nor cu128-train group found in pyproject.toml" >&2
    echo "→ inspect and set COSMOS_GROUP env var explicitly." >&2
    exit 3
  fi
fi

EXTRAS="${COSMOS_EXTRAS:-}"
echo "cosmos-framework dependency group = $GROUP"
echo "→ uv sync ${EXTRAS} --group=$GROUP --group=policy-server"

# ---- install --------------------------------------------------------------
# This downloads several GB of PyTorch + CUDA wheels the first time, onto a
# shared volume. Refuse to start if there is not enough headroom.
FREE_GB="$(df -BG --output=avail "$UV_CACHE_DIR" | tail -1 | tr -dc '0-9')"
echo "free space on the uv cache filesystem: ${FREE_GB} GB"
if [ "${FREE_GB:-0}" -lt 40 ]; then
  echo "error: only ${FREE_GB} GB free — the Cosmos env alone needs ~25 GB and" >&2
  echo "the checkpoint another ~33 GB. Free space or relocate the caches." >&2
  exit 6
fi

# shellcheck disable=SC2086  # EXTRAS is an intentional word-split flag list
uv sync ${EXTRAS} --group="$GROUP" --group=policy-server

# Keep the shared volume from silently filling with wheel archives.
uv cache prune || true
df -h "$UV_CACHE_DIR" | tail -1

# ---- sanity ---------------------------------------------------------------
# CRITICAL: invoke the venv interpreter directly, NOT `uv run`.
# `uv run` re-syncs the project to its *default* dependency set, which does not
# include the cuXXX group we just selected. Observed on 2026-08-03: a bare
# `uv run python -c "import torch"` silently uninstalled torch 2.10.0+cu130 and
# installed 2.13.0+cu130, leaving flash-attn/natten/transformer-engine (all
# built against the torch 2.10 ABI, `+cu130.torch210`) importing with an
# undefined-symbol ImportError. Upstream's own Docker flow puts `.venv/bin` on
# PATH and never uses `uv run` either. If you must, use `uv run --no-sync`.
PYBIN="$(pwd)/.venv/bin/python"
"$PYBIN" -c "import torch; print('torch', torch.__version__, 'cuda?', torch.cuda.is_available())"
"$PYBIN" -c "import cosmos_framework; print('cosmos_framework OK')" 2>&1 | tail -1

# Assert the environment matches the group we asked for. A mismatch here means
# something re-resolved the venv behind our back — fail loudly rather than
# profiling a stack that is not the one we intended to measure.
EXPECTED_TORCH="$(python3 - "$GROUP" <<'PY'
import re
import sys

group = sys.argv[1]
text = open("pyproject.toml").read()
m = re.search(rf'^{re.escape(group)} = \[(.*?)^\]', text, re.S | re.M)
block = m.group(1) if m else text
# The group may `include-group` its non-train sibling, which holds the pin.
if 'torch==' not in block and group.endswith('-train'):
    base = group[: -len('-train')]
    m2 = re.search(rf'^{re.escape(base)} = \[(.*?)^\]', text, re.S | re.M)
    block = m2.group(1) if m2 else block
m3 = re.search(r'"torch==([^"]+)"', block)
print(m3.group(1) if m3 else "")
PY
)"
ACTUAL_TORCH="$("$PYBIN" -c 'import torch; print(torch.__version__)')"
echo "torch: expected '${EXPECTED_TORCH:-<unpinned>}' / actual '$ACTUAL_TORCH'"
if [ -n "$EXPECTED_TORCH" ] && [ "$EXPECTED_TORCH" != "$ACTUAL_TORCH" ]; then
  echo "error: torch is $ACTUAL_TORCH but group '$GROUP' pins $EXPECTED_TORCH." >&2
  echo "→ something re-resolved the venv (most likely a bare 'uv run')." >&2
  echo "  Re-run this script, and use .venv/bin/python or 'uv run --no-sync'." >&2
  exit 7
fi

# The ABI-sensitive extensions must actually import, not merely be installed.
"$PYBIN" - <<'PY'
import importlib
import sys

failed = []
for mod in ("flash_attn", "natten", "transformer_engine"):
    try:
        importlib.import_module(mod)
        print(f"  import {mod:20s} OK")
    except Exception as exc:  # noqa: BLE001 - we want the reason, whatever it is
        print(f"  import {mod:20s} FAIL: {type(exc).__name__}: {str(exc)[:160]}")
        failed.append(mod)
if failed:
    print(
        "\nerror: these are prebuilt against a specific torch ABI and did not "
        f"import: {failed}. The attention backend selection in "
        "cosmos_framework/model/attention/backends.py depends on them; a broken "
        "flash2 leaves only cudnn/natten and changes what we measure.",
        file=sys.stderr,
    )
    raise SystemExit(9)
PY

# ---- record the resolved group + versions --------------------------------
python3 - "$GROUP" "$REPO_ROOT" "$COSMOS_SRC_NAME" "$COSMOS_SRC" <<'PY'
import datetime as dt
import json
import subprocess
import sys

group, repo_root, src_name, src_path = sys.argv[1], sys.argv[2], sys.argv[3], sys.argv[4]
pybin = ".venv/bin/python"


def ver(pkg: str) -> str | None:
    try:
        return subprocess.check_output(
            [pybin, "-c", f"import importlib.metadata as m; print(m.version({pkg!r}))"],
            text=True,
        ).strip()
    except Exception:
        return None


info = {
    "source_path": src_path,
    "cosmos_group": group,
    "installed_utc": dt.datetime.now(dt.timezone.utc)
    .replace(microsecond=0, tzinfo=None)
    .isoformat()
    + "Z",
    "python_version": subprocess.check_output(
        [pybin, "-c", "import sys; print('.'.join(map(str, sys.version_info[:3])))"], text=True
    ).strip(),
    "packages": {
        p: ver(p)
        for p in (
            "torch",
            "torchvision",
            "torchcodec",
            "triton",
            "flash-attn",
            "flash-attn-3-nv",
            "natten",
            "transformer-engine",
            "torchao",
            "numpy",
        )
    },
    "gpu": subprocess.check_output(
        [
            pybin,
            "-c",
            "import torch; print(f'{torch.cuda.get_device_name(0)} sm{\"\".join(map(str, torch.cuda.get_device_capability(0)))}')",
        ],
        text=True,
    ).strip(),
    "note": (
        "flash3 is Hopper-only (arch_tag==90); on SM 8.9 get_backend_list() "
        "returns ['flash2','cudnn','natten'], so absolute latencies are not "
        "comparable to NVIDIA's published FA3 numbers."
    ),
}

# Keyed by source tree, not flat. Two substrates are installed side by side and their
# wheel stacks may differ; a single flat record would be overwritten by whichever was
# installed last, silently relabelling one substrate's environment as the other's.
#
# An existing flat file is migrated under "cosmos-framework" rather than discarded: it
# can only have come from that tree, since it was the only substrate when it was written.
out_path = f"{repo_root}/reproducibility/cosmos_install.json"
try:
    existing = json.load(open(out_path))
except (OSError, ValueError):
    existing = {}
if existing and "installs" not in existing:
    existing = {"installs": {"cosmos-framework": existing}}
doc = {"installs": existing.get("installs", {})}
doc["installs"][src_name] = info
open(out_path, "w").write(json.dumps(doc, indent=2))
print(f"wrote reproducibility/cosmos_install.json (installs.{src_name})")
PY

echo "setup_cosmos: done"
