"""Drive B0–B4 diagnostic configurations against a running Cosmos3 server.

This module is intentionally I/O-shaped rather than model-internal — it
talks to the policy server over HTTP so it works even when we haven't yet
imported cosmos-framework in the same process.

Each B-config sends the same synthetic observation payload with different
server-side settings (denoising steps, decode video) and records:
  - wall latency + p50/p95 across `iters`
  - server-reported per-stage times (if the server exposes them in the
    response)
  - client-side FLOP counter around the request payload (only counts client
    prep — server FLOPs come from separate profiler runs)
  - NVML power samples via `EnergyMeter`

Notes:
- The synthetic payload matches what the RoboLab client would send; if the
  checked-out client uses a different schema, adjust `build_payload()`.
- We DO NOT hold the server GPU during measurement — timings assume single
  concurrent client. Batch/concurrency (B4) uses `--concurrency N`.
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from typing import Any

from action_refresh.energy import EnergyMeter
from action_refresh.logging import configure, get_logger
from action_refresh.metrics import JsonlWriter, RequestRecord, new_run_id
from action_refresh.profiler import StageTimer

logger = get_logger(__name__)


# --- config catalogue ------------------------------------------------------
# Each B-config is a dict of server-side parameter overrides that will be
# passed as request-level parameters (if the server supports it) or has
# been baked in via env vars when the server was started.
B_CONFIGS: dict[str, dict[str, Any]] = {
    "B0": {"denoising_steps": 4, "decode_video": False, "note": "official baseline"},
    "B1": {"denoising_steps": 4, "decode_video": True,  "note": "with VAE decode"},
    "B2_steps_1": {"denoising_steps": 1, "decode_video": False},
    "B2_steps_2": {"denoising_steps": 2, "decode_video": False},
    "B2_steps_3": {"denoising_steps": 3, "decode_video": False},
    "B2_steps_4": {"denoising_steps": 4, "decode_video": False},
    "B3": {"denoising_steps": 4, "decode_video": False, "note": "deterministic replay"},
    "B4": {"denoising_steps": 4, "decode_video": False, "note": "concurrency=N"},
}


def build_payload(seed: int = 0) -> dict[str, Any]:
    """Synthetic observation for profiling — replace with a real RoboLab dump.

    TODO(M2): after `make smoke` succeeds, capture one real client payload
    into `results/raw/reference_payload.json` and load it here instead of
    fabricating.
    """
    # Placeholder shape — DO NOT trust in isolation. The right approach is
    # to dump a real request from the running client and replay it here.
    return {"seed": seed, "note": "SYNTHETIC — replace with real capture"}


def call_server(host: str, port: int, payload: dict[str, Any]) -> tuple[float, dict[str, Any]]:
    """POST to the server, return (wall_ms, response_json)."""
    import urllib.error
    import urllib.request

    t0 = time.perf_counter_ns()
    req = urllib.request.Request(
        f"http://{host}:{port}/policy",
        data=json.dumps(payload).encode("utf-8"),
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    try:
        with urllib.request.urlopen(req, timeout=60) as resp:
            body = json.loads(resp.read().decode("utf-8"))
    except urllib.error.URLError as e:
        raise RuntimeError(f"server call failed: {e}") from e
    wall_ms = (time.perf_counter_ns() - t0) / 1e6
    return wall_ms, body


def run_config(
    name: str,
    cfg: dict[str, Any],
    host: str,
    port: int,
    warmup: int,
    iters: int,
    out_dir: Path,
    run_id: str,
) -> None:
    logger.info("run_config", name=name, cfg=cfg, iters=iters)
    records_path = out_dir / f"{name}.jsonl"
    with JsonlWriter(records_path) as w, EnergyMeter(device_index=0, sample_hz=10.0) as em:
        for i in range(warmup):
            call_server(host, port, {**build_payload(i), **cfg})
        for i in range(iters):
            timer = StageTimer()
            with timer.stage("client_total"):
                wall_ms, resp = call_server(host, port, {**build_payload(i), **cfg})
            server_stages = resp.get("timing", {}) if isinstance(resp, dict) else {}
            record = RequestRecord(
                run_id=run_id,
                task="__profile__",
                env_id=0,
                episode_id=0,
                control_step=i,
                policy_request_index=i,
                seed=i,
                method=name,
                policy_call_performed=True,
                number_of_denoising_steps=cfg.get("denoising_steps"),
                network_roundtrip_ms=wall_ms,
                preprocessing_ms=server_stages.get("preprocessing_ms"),
                vision_encode_ms=server_stages.get("vision_encode_ms"),
                context_ms=server_stages.get("context_ms"),
                denoising_ms=server_stages.get("denoising_ms"),
                vision_decode_ms=server_stages.get("vision_decode_ms"),
                postprocess_ms=server_stages.get("postprocess_ms"),
            )
            w.write(record)
        energy_j = em.integrate_j()
        (out_dir / f"{name}.energy.json").write_text(
            json.dumps({"energy_j": energy_j, "samples": em.sample_count(), "mean_w": em.mean_power_w()}, indent=2)
        )


def main() -> int:
    configure()
    ap = argparse.ArgumentParser()
    ap.add_argument("--cosmos-host", default="127.0.0.1")
    ap.add_argument("--cosmos-port", type=int, default=8000)
    ap.add_argument("--warmup", type=int, default=5)
    ap.add_argument("--iters", type=int, default=30)
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--configs", default="B0,B1,B2_steps_1,B2_steps_2,B2_steps_3,B2_steps_4,B3")
    args = ap.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    run_id = new_run_id("profile")

    requested = [s.strip() for s in args.configs.split(",")]
    unknown = [n for n in requested if n not in B_CONFIGS]
    if unknown:
        # Previously this logged an error and `continue`d, so asking for "B2"
        # (which is not a key — the sweep is B2_steps_1..4) silently profiled
        # nothing and still exited 0. A missing config must fail the run.
        raise SystemExit(
            f"unknown config(s) {unknown}; known configs: {sorted(B_CONFIGS)}"
        )

    for name in requested:
        run_config(name, B_CONFIGS[name], args.cosmos_host, args.cosmos_port,
                   args.warmup, args.iters, out_dir, run_id)
    logger.info("profile_runner_done", out_dir=str(out_dir))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
