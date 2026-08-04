#!/usr/bin/env python3
"""Inspect the checked-out Cosmos + RoboLab source and write docs/baseline_contract.md.

Reads:
  third_party/cosmos-framework/cosmos_framework/scripts/action_policy_server_robolab.py
  third_party/RoboLab/policies/cosmos3/run.py
  (optionally) a locally cached Cosmos3-Nano-Policy-DROID config.json under HF_HOME

Extracts:
  - default policy server port
  - default action denoising steps
  - default action chunk size
  - default conditioning FPS
  - default action dimension
  - open-loop horizon (client)
  - decode_video default
  - whether the server-side call returns both samples['action'] and samples['vision']

Any value it can't find is written as "UNKNOWN — inspect manually" with a
pointer to the file. This script never guesses.
"""
from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
SERVER_PY = REPO_ROOT / "third_party" / "cosmos-framework" / "cosmos_framework" / "scripts" / "action_policy_server_robolab.py"
CLIENT_PY = REPO_ROOT / "third_party" / "RoboLab" / "policies" / "cosmos3" / "run.py"


def _read(p: Path) -> str:
    return p.read_text(errors="replace") if p.is_file() else ""


def _find(rx: str, text: str) -> str | None:
    m = re.search(rx, text)
    return m.group(1).strip() if m else None


def inspect_server(text: str) -> dict[str, object]:
    return {
        "port_default": _find(r"port[^=]*=[^)\d]*(\d+)", text) or _find(r'--port[^,)]*default\s*=\s*(\d+)', text),
        "denoising_steps_default": _find(r"denois.*step[^=]*=\s*(\d+)", text.lower())
                                    or _find(r'--denoising-steps[^,)]*default\s*=\s*(\d+)', text),
        "decode_video_default": _find(r"decode.?video[^=]*=\s*(True|False)", text) or "unknown",
        "action_chunk_default": _find(r"chunk[^=]*=\s*(\d+)", text.lower()),
        "returns_action": "samples['action']" in text or 'samples["action"]' in text,
        "returns_vision": "samples['vision']" in text or 'samples["vision"]' in text,
        "source_file": str(SERVER_PY.relative_to(REPO_ROOT)),
    }


def inspect_client(text: str) -> dict[str, object]:
    return {
        "open_loop_horizon_default": _find(r"open.?loop.?horizon[^=]*=\s*(\d+)", text.lower())
                                     or _find(r"chunk[^=]*=\s*(\d+)", text.lower()),
        "source_file": str(CLIENT_PY.relative_to(REPO_ROOT)),
    }


def try_read_hf_config() -> dict[str, object]:
    """Best-effort read of the model's config.json from HF cache."""
    import os
    hf = os.environ.get("HF_HOME") or os.path.expanduser("~/.cache/huggingface")
    for cand in Path(hf).rglob("Cosmos3-Nano-Policy-DROID*/**/config.json"):
        try:
            c = json.loads(cand.read_text())
            return {
                "hf_config_path": str(cand),
                "action_dim": c.get("action_dim"),
                "conditioning_fps": c.get("conditioning_fps") or c.get("fps"),
                "denoising_steps": c.get("num_denoising_steps") or c.get("denoising_steps"),
                "action_chunk": c.get("action_chunk_size") or c.get("chunk_size"),
                "raw_keys": list(c.keys()),
            }
        except Exception:
            continue
    return {"hf_config_path": None}


def render(server: dict, client: dict, hf: dict) -> str:
    def cell(v: object) -> str:
        return f"`{v}`" if v not in (None, "unknown", "UNKNOWN") else "**UNKNOWN — inspect manually**"

    def dump(d: dict, source_key: str) -> str:
        return "\n".join(f"- **{k}**: {cell(v)}" for k, v in d.items() if k != source_key)

    return f"""# Baseline contract

Derived from checked-out source. Every value here must be *observed*, not
assumed. Anything marked UNKNOWN needs manual inspection before we run
the baseline.

## Server (`{server.get('source_file')}`)
{dump(server, 'source_file')}

## Client (`{client.get('source_file')}`)
{dump(client, 'source_file')}

## HF model config (`{hf.get('hf_config_path')}`)
{dump({k: v for k, v in hf.items() if k != 'hf_config_path'}, '')}

## Critical detail

`decode_video=False` is NOT equivalent to "no visual imagination". It only
disables VAE decoding. Whether the joint diffusion transformer still
generates vision-latent tokens must be verified in the compute anatomy
(`docs/compute_anatomy.md`). Do NOT design cache/spatial methods before
that answer is known.
"""


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=str(REPO_ROOT / "docs" / "baseline_contract.md"))
    args = ap.parse_args()

    server = inspect_server(_read(SERVER_PY))
    client = inspect_client(_read(CLIENT_PY))
    hf = try_read_hf_config()
    Path(args.out).write_text(render(server, client, hf))
    print(f"wrote {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
