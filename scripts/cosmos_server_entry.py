#!/usr/bin/env python3
"""Launch the **unmodified** upstream RoboLab policy server.

This is a wrapper, not a fork: it creates the one-rank process group that
`maybe_init_distributed()` would otherwise create with NCCL (which core-dumps on this
stack — see `action_refresh.server.process_group`), then hands control to upstream's own
`__main__` with argv untouched. Every CLI flag, default and behaviour is upstream's.

    scripts/cosmos_server_entry.py --port 8000 [any upstream flag]

`COSMOS_PG_BACKEND` selects the backend (`gloo` default, `nccl`, or `none` to let
upstream initialize its own group — which needs patches 0002/0003 applied). It is read
here rather than added as a CLI flag precisely so upstream's argument parser sees only
upstream's arguments: `RobolabServerArgs` sets `extra="forbid"`, so an unknown flag is a
hard error, and that strictness is worth keeping.

`COSMOS_SERVER_MODULE` names the module to run. It is exported by
`resolve_substrate` in scripts/lib/common.sh, because there is now more than one
candidate source tree (NVIDIA upstream and the group's efficiency fork) and the fork
need not expose the same entry point. The default matches upstream, and the module that
actually gets imported is printed with its file so the log names the tree that ran.
"""
from __future__ import annotations

import os
import runpy
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

DEFAULT_SERVER_MODULE = "cosmos_framework.scripts.action_policy_server_robolab"


def main() -> int:
    from action_refresh.server.process_group import ensure_single_rank_group

    server_module = os.environ.get("COSMOS_SERVER_MODULE") or DEFAULT_SERVER_MODULE

    backend = os.environ.get("COSMOS_PG_BACKEND", "gloo")
    what = ensure_single_rank_group(backend)  # type: ignore[arg-type]
    print(f"[cosmos-server-entry] process group: {what}", flush=True)

    # Import the module's top-level package and report where it came from. With two
    # substrates installed, "which tree served this request" is answered by the
    # interpreter's resolution, not by the launcher's intent — so print the resolved
    # path rather than the one we asked for.
    root_pkg = __import__(server_module.split(".", 1)[0])
    src = Path(root_pkg.__file__).resolve().parents[1]
    print(
        f"[cosmos-server-entry] substrate={os.environ.get('COSMOS_SUBSTRATE', '<unset>')} "
        f"{root_pkg.__name__} from {src}",
        flush=True,
    )
    print(f"[cosmos-server-entry] exec {server_module} argv={sys.argv[1:]}", flush=True)

    # argv[0] must look like the module's own entry point: tyro reports usage with it.
    sys.argv = [server_module.rsplit(".", 1)[-1], *sys.argv[1:]]
    runpy.run_module(server_module, run_name="__main__")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
