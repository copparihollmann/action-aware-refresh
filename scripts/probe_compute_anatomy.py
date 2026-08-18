#!/usr/bin/env python3
"""M2 compute anatomy — in-process probe of the Cosmos3 policy path.

MUST be run with the cosmos-framework venv interpreter, because it imports
cosmos_framework:

    cd third_party/cosmos-framework
    ./.venv/bin/python ../../scripts/probe_compute_anatomy.py --help

Why in-process rather than over the websocket:

1. The server exposes no per-stage timings, so driving it over the wire can only
   ever measure one number (round-trip). Stage attribution needs to be inside.
2. `num_steps` and `decode_video` are server-*start* arguments, not per-request
   fields, so a wire-level B0-B4 sweep would need a server restart per config.

We instantiate upstream's own `RobolabPolicyService` and call its real `infer()`,
so preprocessing, sample construction, generation and postprocessing are exactly
the deployed code path — nothing is reimplemented here.

Honesty constraints this script holds itself to:

- It does **not** invent an action-vs-vision time split. It reports what the
  module structure actually exposes, plus captured token shapes, and leaves the
  separability question to be answered from that evidence.
- FLOPs from `FlopCounterMode` exclude fused attention kernels (flash-attn is
  opaque to it). We report the counted total, the op types seen, an *analytic*
  attention estimate from captured shapes, and a coverage ratio — never a single
  number presented as exact.
- The input is preferably a REAL request captured from a live closed-loop episode
  (`--request-npz`, produced by research patch robolab-0001 during
  `scripts/smoke_test.sh`). Without it we fall back to a synthetic observation,
  which is defensible for *cost* — compute here is shape-determined, not
  content-determined, since the sample builder zeroes every frame but the first
  regardless — but it puts the composed-view geometry, dtypes and prompt length
  in our hands rather than the deployment's, and those set the token count.
  Either way, neither input measures task success; only the closed-loop RoboLab
  run does. Which input was used is recorded in every record and summary.
"""
from __future__ import annotations

import argparse
import contextlib
import datetime as dt
import json
import os
import platform
import re
import subprocess
import sys
import time
from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np
import torch

# --- deterministic synthetic observation -----------------------------------
# Fixed seed so every config sees byte-identical input; B3 replay depends on it.
OBS_SEED = 12345
DEFAULT_PROMPT = "Put the banana in the bowl"


def build_observation(image_h: int, image_w: int, seed: int = OBS_SEED) -> dict[str, Any]:
    """Minimal valid observation for `action_space="joint_pos"`.

    Required keys, read from `_build_sample` / `_extract_observation_image`:
      prompt, observation/image, observation/joint_position (width 7),
      observation/gripper_position.
    With use_state=True and history_length=1, num_history_rows == 0, so no
    history rows are needed.
    """
    rng = np.random.default_rng(seed)
    return {
        "prompt": DEFAULT_PROMPT,
        "observation/image": rng.integers(0, 256, (image_h, image_w, 3), dtype=np.uint8),
        "observation/joint_position": rng.uniform(-1.0, 1.0, (7,)).astype(np.float32),
        "observation/gripper_position": np.float32(0.0),
    }


def load_captured_observation(path: Path) -> tuple[dict[str, Any], str]:
    """Load a request captured from the live closed loop.

    Produced by `Cosmos3Client._pack_request` under
    ACTION_REFRESH_CAPTURE_REQUEST (research patch robolab-0001), which
    `scripts/smoke_test.sh` sets for the primary task. Preferred over
    `build_observation`: it carries the composed-view geometry, dtypes and real
    instruction text the deployed loop actually sends, so the captured token
    shapes below describe the real workload rather than one we invented.

    The prompt comes from the sidecar .json — npz cannot hold a str.
    """
    npz = np.load(path)
    obs: dict[str, Any] = {k: npz[k] for k in npz.files}
    meta_path = path.with_suffix(".json")
    if not meta_path.exists():
        raise SystemExit(
            f"error: {path} has no sidecar {meta_path.name}; the prompt is stored there.\n"
            "Re-capture with scripts/smoke_test.sh rather than hand-assembling the npz."
        )
    meta = json.loads(meta_path.read_text())
    prompt = (meta.get("strings") or {}).get("prompt")
    if not prompt:
        raise SystemExit(f"error: {meta_path} has no strings.prompt — cannot rebuild the request.")
    obs["prompt"] = prompt

    missing = {
        "observation/image",
        "observation/joint_position",
        "observation/gripper_position",
    } - set(obs)
    if missing:
        raise SystemExit(f"error: captured request {path} is missing keys: {sorted(missing)}")

    img = obs["observation/image"]
    kind = (
        f"REAL captured request from {path.name} "
        f"(image {'x'.join(map(str, img.shape))}:{img.dtype}, prompt {prompt!r})"
    )
    return obs, kind


# --- per-iteration GPU state ------------------------------------------------
# The first B3 run made this mandatory. Identical work, no competing GPU process,
# and yet wall time ramped monotonically from 3,377 ms to 5,974 ms (+76%) across 40
# consecutive requests — the eight slowest were the last eight in time order. That
# is not contention noise, it is *drift*, and a mean/std over such a run describes
# the drift rather than the model.
#
# So sample clocks/temperature/throttle reasons alongside every timed iteration.
# Without them, a config measured late in a sweep looks slower than one measured
# early, and the sweep silently rewards whichever config ran first.
class GpuState:
    """Cheap per-iteration NVML read: clocks, temperature, power, throttle flags."""

    def __init__(self) -> None:
        self.handle = None
        self.error: str | None = None
        try:
            import pynvml  # noqa: PLC0415

            self.pynvml = pynvml
            pynvml.nvmlInit()
            # CUDA_VISIBLE_DEVICES remaps indexes, so resolve by the UUID torch
            # reports rather than trusting index 0 to be our card.
            want = None
            if torch.cuda.is_available():
                props = torch.cuda.get_device_properties(0)
                want = getattr(props, "uuid", None)
            idx = 0
            if want is not None:
                target = str(want).replace("-", "").lower()
                for i in range(pynvml.nvmlDeviceGetCount()):
                    got = pynvml.nvmlDeviceGetUUID(pynvml.nvmlDeviceGetHandleByIndex(i))
                    got_s = got.decode() if isinstance(got, bytes) else got
                    if target in got_s.replace("-", "").lower():
                        idx = i
                        break
            self.handle = pynvml.nvmlDeviceGetHandleByIndex(idx)
            self.nvml_index = idx
        except Exception as exc:  # noqa: BLE001
            self.error = f"{type(exc).__name__}: {exc}"

    def read(self) -> dict[str, Any]:
        if self.handle is None:
            return {"error": self.error}
        p = self.pynvml
        out: dict[str, Any] = {}
        for key, fn in (
            ("sm_clock_mhz", lambda: p.nvmlDeviceGetClockInfo(self.handle, p.NVML_CLOCK_SM)),
            ("mem_clock_mhz", lambda: p.nvmlDeviceGetClockInfo(self.handle, p.NVML_CLOCK_MEM)),
            ("temp_c", lambda: p.nvmlDeviceGetTemperature(self.handle, p.NVML_TEMPERATURE_GPU)),
            ("power_w", lambda: p.nvmlDeviceGetPowerUsage(self.handle) / 1000.0),
            ("throttle_bits", lambda: p.nvmlDeviceGetCurrentClocksThrottleReasons(self.handle)),
        ):
            try:
                out[key] = fn()
            except Exception:  # noqa: BLE001, PERF203
                out[key] = None
        bits = out.get("throttle_bits")
        if isinstance(bits, int):
            # Decode only the reasons that would invalidate a timing comparison.
            out["throttle"] = [
                name
                for name, mask in (
                    ("sw_power_cap", 0x4),
                    ("hw_slowdown", 0x8),
                    ("sw_thermal", 0x20),
                    ("hw_thermal", 0x40),
                    ("hw_power_brake", 0x80),
                )
                if bits & mask
            ]
        return out

    def close(self) -> None:
        if self.handle is not None:
            with contextlib.suppress(Exception):
                self.pynvml.nvmlShutdown()


def thermal_summary(records_path: Path) -> dict[str, Any]:
    """Clock/thermal state across a run, to rule GPU-side causes in or out.

    Measured on this host: `clocks.sm` sits at **1065 MHz regardless of load** (71 W
    idle, 260 W under inference) with throttle reasons `0x0` throughout, against a
    2520 MHz max — i.e. the SM clock is *locked*, not throttling. So when a run drifts
    (the first B3 ramped +76%), this table is what lets us say the GPU was not the
    cause. Compare it against the wall-vs-CUDA split, which localises the drift.

    A non-empty `throttle_seen`, or a real `sm_clock_drop_pct`, would mean
    cross-iteration timings in that run are not comparable.
    """
    temps: list[float] = []
    clocks: list[float] = []
    throttles: set[str] = set()
    try:
        for line in records_path.read_text().splitlines():
            if not line.strip():
                continue
            rec = json.loads(line)
            for slot in ("gpu_before", "gpu_after"):
                st = rec.get(slot) or {}
                if isinstance(st.get("temp_c"), (int, float)):
                    temps.append(float(st["temp_c"]))
                if isinstance(st.get("sm_clock_mhz"), (int, float)):
                    clocks.append(float(st["sm_clock_mhz"]))
                throttles.update(st.get("throttle") or [])
    except (OSError, json.JSONDecodeError):
        return {"error": "could not read per-iteration GPU state"}
    if not temps and not clocks:
        return {"error": "no GPU state captured (pynvml unavailable?)"}
    out: dict[str, Any] = {"throttle_seen": sorted(throttles)}
    if temps:
        out |= {"temp_first_c": temps[0], "temp_last_c": temps[-1], "temp_max_c": max(temps)}
    if clocks:
        first = clocks[0] or 1.0
        out |= {
            "sm_clock_first_mhz": clocks[0],
            "sm_clock_last_mhz": clocks[-1],
            "sm_clock_min_mhz": min(clocks),
            "sm_clock_drop_pct": 100.0 * (first - min(clocks)) / first,
        }
    return out


def robust_stats(xs: list[float]) -> dict[str, float] | None:
    """Median/MAD alongside mean/std.

    Reported because the mean is not robust to the thermal ramp described above: in
    the first B3 run the mean was 3,775 ms while the median was 3,396 ms and the MAD
    was 12.6 ms. The median describes the model; the mean describes the cooling.
    """
    if not xs:
        return None
    a = np.asarray(xs, dtype=float)
    med = float(np.median(a))
    mad = float(np.median(np.abs(a - med)))
    return {
        "n": int(a.size),
        "mean": float(a.mean()),
        "std": float(a.std(ddof=1)) if a.size > 1 else 0.0,
        "median": med,
        "mad": mad,
        "mad_pct": (100.0 * mad / med) if med else 0.0,
        "min": float(a.min()),
        "p50": med,
        "p95": float(np.percentile(a, 95)),
        "max": float(a.max()),
        # Drift: does the second half differ from the first? A large value means the
        # run was not in steady state and the comparison is unsafe.
        "drift_pct": (
            100.0 * (float(np.median(a[a.size // 2 :])) - float(np.median(a[: a.size // 2]))) / med
            if a.size >= 4 and med
            else 0.0
        ),
    }


# --- token census -----------------------------------------------------------
# The whole project rests on the claim that most of the sequence is *imagined
# future video*. M2 could only infer that from `position_ids=(3, 3184)`, which
# gives the total and nothing else. So read the real layout instead.
#
# The authoritative source is the PackedSequence that the model builds:
#   - `text_indexes`                     -> text token positions
#   - `<modality>.sequence_indexes`      -> that modality's token positions
#   - `<modality>.condition_mask`        -> per-payload, 1 = clean/conditioning,
#                                          0 = noised/supervised  ("imagined")
#   - `<modality>.token_shapes`          -> (T,H,W) for vision, (T,) for action
#   - `split_lens` / `attn_modes`        -> attention structure
#
# The condition_mask is what separates "the observation we were given" from "the
# future the model is inventing", which is exactly the distinction the research
# question turns on and the one `decode_video=False` does NOT make.
def capture_packed_sequence(model: Any) -> Any:
    """Context manager yielding a one-slot list that receives the PackedSequence.

    `_pack_input_sequence` is called once per denoising step per CFG branch, so we
    keep only the first (they share a layout — `_can_reuse_inference_pack_templates`
    exists precisely because the layout is stable across steps).

    Wraps the *instance* attribute and deletes it afterwards, leaving the class
    untouched: patching the class would leak into any other model in the process.
    """

    class _Cap(contextlib.ContextDecorator):
        def __init__(self):
            self.slot: list[Any] = []

        def __enter__(self):
            original = model._pack_input_sequence

            def wrapper(*a, **kw):
                out = original(*a, **kw)
                if not self.slot:
                    self.slot.append(out)
                return out

            model._pack_input_sequence = wrapper  # instance attribute shadows the bound method
            self._original = original
            return self.slot

        def __exit__(self, *exc):
            try:
                del model._pack_input_sequence
            except AttributeError:  # pragma: no cover - defensive
                model._pack_input_sequence = self._original
            return False

    return _Cap()


def _modality_census(mod: Any) -> dict[str, Any] | None:
    """Split one modality's tokens into conditioning vs noised, per payload."""
    if mod is None:
        return None
    total = int(mod.sequence_indexes.numel())
    clean = noised = 0
    for cm in mod.condition_mask:
        # condition_mask is per *frame/step*, not per token, so weight each frame
        # by its token count before summing. Vision payloads are (T,H,W): H*W
        # tokens per latent frame. Action payloads are (T,): 1 token per step.
        n = int(cm.numel())
        if n == 0:
            continue
        c = int(cm.sum().item())
        clean += c
        noised += n - c
    out: dict[str, Any] = {
        "tokens_total": total,
        "frames_conditioning": clean,
        "frames_noised": noised,
        "token_shapes": [list(s) for s in mod.token_shapes],
        "mse_supervised_tokens": int(mod.mse_loss_indexes.numel()),
    }
    # Derive tokens-per-frame from the payload shape so the frame->token weighting
    # is stated rather than assumed.
    per_frame = []
    for s in mod.token_shapes:
        if len(s) == 3:  # (T,H,W) vision
            per_frame.append(int(s[1]) * int(s[2]))
        elif len(s) == 1:  # (T,) action
            per_frame.append(1)
    if per_frame:
        out["tokens_per_frame"] = per_frame
        out["tokens_conditioning"] = clean * per_frame[0]
        out["tokens_noised"] = noised * per_frame[0]
    return out


def token_census(packed: Any) -> dict[str, Any]:
    """Full token accounting for one packed sequence, with a reconciliation check."""
    census: dict[str, Any] = {
        "sequence_length": int(packed.sequence_length),
        "split_lens": list(packed.split_lens),
        "attn_modes": list(packed.attn_modes),
        "sample_lens": list(packed.sample_lens),
        "text_tokens": int(packed.text_indexes.numel()),
        "num_action_tokens_per_supertoken": int(packed.num_action_tokens_per_supertoken),
        "null_action_supertokens": bool(packed.null_action_supertokens),
        "vision": _modality_census(packed.vision),
        "action": _modality_census(packed.action),
        "sound": _modality_census(packed.sound),
    }
    # Reconcile: text + vision + action + sound should account for the sequence.
    # A mismatch means the layout has extra tokens we have not attributed (special
    # tokens, padding), so report the residual instead of quietly asserting.
    attributed = census["text_tokens"] + sum(
        (census[m] or {}).get("tokens_total", 0) for m in ("vision", "action", "sound")
    )
    census["attributed_tokens"] = attributed
    census["unattributed_tokens"] = census["sequence_length"] - attributed
    vis = census["vision"] or {}
    if vis.get("tokens_noised") and census["sequence_length"]:
        census["imagined_fraction_of_sequence"] = vis["tokens_noised"] / census["sequence_length"]
    return census


# --- module-level timing ----------------------------------------------------
class ModuleProfiler:
    """CUDA-event timing + input-shape capture per named submodule.

    Attribution comes from the real module tree, so we never assert a category
    ("vision", "action") the implementation does not expose. Synchronization
    happens once, in `finalize()`.
    """

    def __init__(self, model: torch.nn.Module, names: list[str], capture_shapes: bool = True):
        self.model = model
        self.names = names
        self.capture_shapes = capture_shapes
        self._handles: list[Any] = []
        self._pending: list[tuple[str, torch.cuda.Event, torch.cuda.Event]] = []
        self._open: dict[str, torch.cuda.Event] = {}
        self.calls: dict[str, int] = defaultdict(int)
        self.shapes: dict[str, list[str]] = defaultdict(list)

    def _pre(self, name: str):
        def hook(_mod, args, kwargs=None):  # noqa: ANN001
            ev = torch.cuda.Event(enable_timing=True)
            ev.record()
            self._open[name] = ev
            self.calls[name] += 1
            if self.capture_shapes and len(self.shapes[name]) < 3:
                desc = []
                for a in args:
                    if isinstance(a, torch.Tensor):
                        desc.append(f"{tuple(a.shape)}:{str(a.dtype).replace('torch.', '')}")
                if kwargs:
                    for k, v in kwargs.items():
                        if isinstance(v, torch.Tensor):
                            desc.append(f"{k}={tuple(v.shape)}")
                if desc:
                    self.shapes[name].append(", ".join(desc))
            return None

        return hook

    def _post(self, name: str):
        def hook(_mod, _args, _out):  # noqa: ANN001
            start = self._open.pop(name, None)
            if start is None:
                return None
            end = torch.cuda.Event(enable_timing=True)
            end.record()
            self._pending.append((name, start, end))
            return None

        return hook

    def __enter__(self) -> "ModuleProfiler":
        by_name = dict(self.model.named_modules())
        for name in self.names:
            mod = by_name.get(name)
            if mod is None:
                continue
            try:
                self._handles.append(
                    mod.register_forward_pre_hook(self._pre(name), with_kwargs=True)
                )
            except TypeError:  # older torch without with_kwargs
                self._handles.append(mod.register_forward_pre_hook(self._pre(name)))
            self._handles.append(mod.register_forward_hook(self._post(name)))
        return self

    def __exit__(self, *exc: object) -> None:
        for h in self._handles:
            h.remove()
        self._handles.clear()

    def finalize(self) -> dict[str, float]:
        if not self._pending:
            return {}
        torch.cuda.synchronize()
        out: dict[str, float] = defaultdict(float)
        for name, s, e in self._pending:
            out[name] += s.elapsed_time(e)
        self._pending.clear()
        return dict(out)


def pick_modules(
    model: torch.nn.Module,
    max_depth: int = 2,
    max_modules: int = 40,
    pattern: str | None = None,
) -> list[str]:
    """Modules to hook: enough to attribute cost, few enough to stay cheap.

    Hooking every leaf would add measurable overhead and swamp the report, so the
    default stays shallow. `pattern` (a regex on the module name) opts into a
    deeper slice — Experiment F needs per-transformer-block timing and residuals,
    e.g. `--hook-pattern 'language_model\\.layers\\.\\d+$'`.

    IMPORTANT — what depth does *not* buy: hooking deeper cannot separate
    vision-token cost from action-token cost. Vision and action tokens are packed
    into one sequence and flow through the *same* modules (`two_way_attention`
    runs one full-attention call over the whole generation split), so no module
    boundary corresponds to a modality boundary. The modality split has to come
    from the token census plus the measured near-linearity of cost in token count
    (99.4% of counted FLOPs are `aten.mm`) — and Experiment E2b tests that
    attribution directly by removing tokens and measuring what actually falls off.
    Reporting a per-module "vision time" would be inventing a number the
    implementation does not expose.
    """
    rx = re.compile(pattern) if pattern else None
    names = []
    for name, _ in model.named_modules():
        if not name:
            continue
        if rx is not None:
            if rx.search(name):
                names.append(name)
        elif name.count(".") < max_depth:
            names.append(name)
    if len(names) > max_modules:
        # Truncating silently would understate cost and look like a complete
        # attribution, so say what was dropped.
        print(
            f"  note: {len(names)} modules matched, hooking the first {max_modules} "
            "(raise --hook-max-modules to widen)",
            flush=True,
        )
    return names[:max_modules]


# --- environment / provenance ----------------------------------------------
def sh(cmd: str) -> str:
    try:
        return subprocess.run(
            cmd, shell=True, capture_output=True, text=True, timeout=20
        ).stdout.strip()
    except Exception:
        return ""


def gpu_snapshot() -> dict[str, Any]:
    try:
        load = list(os.getloadavg())
    except OSError:
        load = None
    return {
        "gpu_name": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
        "capability": (
            "".join(map(str, torch.cuda.get_device_capability(0)))
            if torch.cuda.is_available()
            else None
        ),
        "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
        "gpu_uuid": sh(
            "nvidia-smi --query-gpu=uuid --format=csv,noheader --id="
            + (os.environ.get("CUDA_VISIBLE_DEVICES", "0").split(",")[0])
        ),
        "loadavg_1_5_15": load,
        "gpu_compute_apps": sh(
            "nvidia-smi --query-compute-apps=pid,process_name,used_memory --format=csv,noheader"
        ),
        "top_cpu": sh("ps -eo user,pcpu,comm --sort=-pcpu --no-headers | head -5"),
    }


def attention_backend_probe() -> dict[str, Any]:
    """Record which attention backends this arch is even allowed to use.

    SM 8.9 excludes flash3 (Hopper-only), so our absolute latencies are not
    comparable to NVIDIA's published FA3 figures. This must travel with results.
    """
    info: dict[str, Any] = {}
    try:
        from cosmos_framework.model.attention.backends import get_backend_list
        from cosmos_framework.model.attention.utils import get_arch_tag

        tag = get_arch_tag()
        info["arch_tag"] = tag
        info["backend_list"] = get_backend_list(tag)
    except Exception as exc:  # noqa: BLE001
        info["error"] = f"{type(exc).__name__}: {exc}"
    return info


# --- one configuration ------------------------------------------------------
def run_config(
    name: str,
    overrides: dict[str, Any],
    warmup: int,
    iters: int,
    out_dir: Path,
    run_id: str,
    profile_trace: bool,
    hook_modules: bool,
    request_npz: Path | None = None,
    hook_depth: int = 2,
    hook_max: int = 40,
    hook_pattern: str | None = None,
    cooldown_s: float = 0.0,
) -> dict[str, Any]:
    from cosmos_framework.scripts.action_policy_server_robolab import (  # noqa: PLC0415
        RobolabServerArgs,
        RobolabPolicyService,
    )

    args = RobolabServerArgs(**overrides)
    t_load0 = time.perf_counter()
    # Create the one-rank process group ourselves, with gloo, so upstream's
    # maybe_init_distributed() finds one already there and its collectives run
    # unmodified. Without this the NCCL group upstream builds core-dumps on this stack.
    # See src/action_refresh/server/process_group.py.
    from action_refresh.server.process_group import ensure_single_rank_group  # noqa: PLC0415

    print(f"[pg] {ensure_single_rank_group(os.environ.get('COSMOS_PG_BACKEND', 'gloo'))}", flush=True)

    service = RobolabPolicyService(args)
    load_s = time.perf_counter() - t_load0

    if request_npz is not None:
        obs, input_kind = load_captured_observation(request_npz)
    else:
        obs = build_observation(args.image_height, args.image_width)
        input_kind = (
            "SYNTHETIC deterministic observation (seed %d) — valid for cost, "
            "NOT for task success" % OBS_SEED
        )

    torch.cuda.reset_peak_memory_stats()
    after_load_alloc = torch.cuda.memory_allocated() / 2**20
    after_load_reserved = torch.cuda.memory_reserved() / 2**20

    # --- token census -------------------------------------------------------
    # Done on a throwaway forward before the timing loop so the hook wrapper is
    # never in place while we measure. The layout is stable across denoising steps
    # and configs, so one capture is enough.
    census: dict[str, Any] | None = None
    census_error: str | None = None
    try:
        with capture_packed_sequence(service.model) as slot:
            service.infer(obs)
        if slot:
            census = token_census(slot[0])
        else:
            census_error = "_pack_input_sequence was never called — layout unknown"
    except Exception as exc:  # noqa: BLE001
        # A census failure must not lose the timings; record why and continue.
        census_error = f"{type(exc).__name__}: {exc}"
    if census_error:
        print(f"  token census FAILED: {census_error}", flush=True)
    elif census:
        v = census.get("vision") or {}
        print(
            f"  tokens: {census['sequence_length']} total = "
            f"{census['text_tokens']} text + {v.get('tokens_conditioning', '?')} vision-cond "
            f"+ {v.get('tokens_noised', '?')} vision-imagined + "
            f"{(census.get('action') or {}).get('tokens_total', '?')} action "
            f"(unattributed {census['unattributed_tokens']})",
            flush=True,
        )

    module_names = (
        pick_modules(service.model, max_depth=hook_depth, max_modules=hook_max, pattern=hook_pattern)
        if hook_modules
        else []
    )
    stage_totals: dict[str, list[float]] = defaultdict(list)
    wall_ms: list[float] = []
    cuda_ms: list[float] = []
    shapes_seen: dict[str, list[str]] = {}

    gpu_state = GpuState()
    records_path = out_dir / f"{name}.jsonl"
    with records_path.open("w") as fh:
        for i in range(warmup + iters):
            is_warm = i >= warmup
            state_before = gpu_state.read()
            # Wall AND CUDA time per iteration. These separate host-side interference
            # from real GPU work, which is the difference between "another user's
            # compile stole our cores" and "this config is slower". The first B3 run
            # ramped +76% in wall time while sm_clock stayed pinned at 1065 MHz with
            # zero throttle flags, so the tail had to be host-side — but wall time
            # alone could not prove it. Recording both settles it per iteration.
            ev_start = torch.cuda.Event(enable_timing=True)
            ev_end = torch.cuda.Event(enable_timing=True)
            torch.cuda.synchronize()
            ev_start.record()
            t0 = time.perf_counter_ns()
            if is_warm and hook_modules:
                with ModuleProfiler(service.model, module_names) as mp:
                    out = service.infer(obs)
                per_module = mp.finalize()
                if not shapes_seen:
                    shapes_seen = {k: v for k, v in mp.shapes.items() if v}
            else:
                out = service.infer(obs)
                per_module = {}
            dt_ms = (time.perf_counter_ns() - t0) / 1e6
            ev_end.record()
            torch.cuda.synchronize()
            cuda_dt_ms = ev_start.elapsed_time(ev_end)

            if not is_warm:
                continue

            wall_ms.append(dt_ms)
            cuda_ms.append(cuda_dt_ms)
            for k, v in per_module.items():
                stage_totals[k].append(v)

            action = out["action"]
            rec = {
                "run_id": run_id,
                "method": name,
                "iter": i - warmup,
                "input_kind": input_kind,
                "wall_ms": dt_ms,
                "cuda_ms": cuda_dt_ms,
                "host_overhead_ms": dt_ms - cuda_dt_ms,
                "action_shape": list(np.shape(action)),
                "action_finite": bool(np.isfinite(np.asarray(action)).all()),
                "video_returned": "video" in out,
                "video_shape": list(np.shape(out["video"])) if "video" in out else None,
                "num_steps": args.num_steps,
                "decode_video": args.decode_video,
                "peak_allocated_mib": torch.cuda.max_memory_allocated() / 2**20,
                "peak_reserved_mib": torch.cuda.max_memory_reserved() / 2**20,
                "module_ms": per_module,
                # Thermal/clock state bracketing this iteration. `gpu_before` is the
                # state the request started from; `gpu_after` shows where it ended.
                # A rising temp with a falling sm_clock across a run is throttling,
                # and makes cross-iteration timing comparisons invalid.
                "gpu_before": state_before,
                "gpu_after": gpu_state.read(),
            }
            fh.write(json.dumps(rec) + "\n")
            fh.flush()  # so a killed run still leaves the iterations it finished

            # Optional idle gap between requests. Back-to-back inference on a 350 W
            # passively-cooled L40S heats it into a lower clock state, which shows up
            # as a monotonic latency ramp (measured: +76% over 40 requests). A
            # cooldown trades wall-clock for a steady-state measurement, which is
            # what a *comparison* needs. Closed-loop numbers should NOT use it —
            # there the sustained-load behaviour is the deployed behaviour.
            if cooldown_s > 0:
                time.sleep(cooldown_s)

    # --- FLOPs, on one extra request ---------------------------------------
    flops: dict[str, Any] = {"counted_total": None, "note": None}
    try:
        from torch.utils.flop_counter import FlopCounterMode

        ctr = FlopCounterMode(display=False)
        with ctr:
            service.infer(obs)
        total = ctr.get_total_flops()
        by_op = {
            str(k): int(v)
            for k, v in sorted(
                ctr.get_flop_counts().get("Global", {}).items(),
                key=lambda kv: -kv[1],
            )
        }
        flops = {
            "counted_total": int(total),
            "by_op": by_op,
            "note": (
                "FlopCounterMode counts only ops it recognises. Fused attention "
                "(flash-attn / cudnn fmha) is opaque to it, so this is a LOWER "
                "BOUND, not the true total. Compare against the analytic "
                "attention estimate and report coverage."
            ),
        }
    except Exception as exc:  # noqa: BLE001
        flops["note"] = f"FLOP counting failed: {type(exc).__name__}: {exc}"

    # --- one chrome trace ---------------------------------------------------
    trace_path = None
    if profile_trace:
        try:
            from torch.profiler import ProfilerActivity, profile

            trace_path = str(out_dir / f"{name}.chrome_trace.json")
            with profile(
                activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA],
                record_shapes=True,
                with_flops=True,
            ) as prof:
                service.infer(obs)
            prof.export_chrome_trace(trace_path)
        except Exception as exc:  # noqa: BLE001
            trace_path = f"FAILED: {type(exc).__name__}: {exc}"

    # `robust_stats` supersedes the old mean/std-only helper: it keeps mean and std
    # for continuity but adds median/MAD and a drift indicator, because the thermal
    # ramp observed in B3 makes the mean a statement about cooling rather than compute.
    stats = robust_stats
    gpu_state.close()

    summary = {
        "config": name,
        "overrides": {k: (str(v) if isinstance(v, Path) else v) for k, v in overrides.items()},
        "num_steps": args.num_steps,
        "decode_video": args.decode_video,
        "model_load_s": load_s,
        "vram_after_load_mib": {
            "allocated": after_load_alloc,
            "reserved": after_load_reserved,
        },
        "vram_peak_mib": {
            "allocated": torch.cuda.max_memory_allocated() / 2**20,
            "reserved": torch.cuda.max_memory_reserved() / 2**20,
        },
        "input_kind": input_kind,
        "token_census": census,
        "token_census_error": census_error,
        "thermal": thermal_summary(records_path),
        "cooldown_s": cooldown_s,
        "wall_ms": stats(wall_ms),
        # GPU-only time, and the host-side residual. If cuda_ms is stable while
        # wall_ms is noisy, the noise is ours (CPU contention), not the model's —
        # and per spec §8 the *end-to-end* number is still what a speedup claim
        # must use, so both are reported rather than picking the flattering one.
        "cuda_ms": stats(cuda_ms),
        "host_overhead_ms": stats([w - c for w, c in zip(wall_ms, cuda_ms)]),
        "module_ms_mean": {
            k: float(np.mean(v)) for k, v in sorted(stage_totals.items(), key=lambda kv: -np.mean(kv[1]))
        },
        "module_call_shapes": shapes_seen,
        "flops": flops,
        "chrome_trace": trace_path,
        "records": str(records_path),
    }

    del service
    torch.cuda.empty_cache()
    return summary


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--warmup", type=int, default=3)
    ap.add_argument("--iters", type=int, default=20)
    ap.add_argument(
        "--configs",
        default="B0,B1,B2_steps_1,B2_steps_2,B2_steps_3",
        help="comma-separated subset of the B-config catalogue",
    )
    ap.add_argument("--checkpoint-path", default="nvidia/Cosmos3-Nano-Policy-DROID")
    ap.add_argument("--hf-revision", default="6706d7680581c255ff61e0f3bb49d90eac55c79e")
    ap.add_argument(
        "--request-npz",
        type=Path,
        default=None,
        help="profile a REAL request captured from the live closed loop "
        "(results/raw/captured_request_<task>.npz, written by scripts/smoke_test.sh "
        "via research patch robolab-0001). Strongly preferred: the composed-view "
        "geometry, dtypes and instruction text determine the token count, and "
        "therefore the cost attribution. Falls back to a synthetic observation "
        "when omitted; either way the choice is recorded in every output record.",
    )
    ap.add_argument("--no-trace", action="store_true", help="skip chrome traces")
    ap.add_argument("--no-hooks", action="store_true", help="skip per-module timing")
    ap.add_argument(
        "--hook-depth",
        type=int,
        default=2,
        help="module-name dot-depth to hook (default 2: top-level blocks only)",
    )
    ap.add_argument("--hook-max-modules", type=int, default=40)
    ap.add_argument(
        "--cooldown-s",
        type=float,
        default=0.0,
        help="idle seconds between timed requests. Back-to-back inference throttles "
        "this card (measured +76%% latency ramp over 40 requests), so a steady-state "
        "comparison needs a gap. Leave at 0 for deployment-representative numbers.",
    )
    ap.add_argument(
        "--hook-pattern",
        default=None,
        help="regex on module names, overriding --hook-depth. Experiment F wants "
        r"per-block timing: --hook-pattern 'language_model\.layers\.\d+$'. Note this "
        "gives a depth profile, NOT a vision/action split — those tokens share every "
        "module (see pick_modules).",
    )
    ap.add_argument(
        "--no-guardrails",
        action="store_true",
        help="disable Cosmos guardrail runners (requires the research patch). "
        "Needed when nvidia/Cosmos-Guardrail1 access has not been granted. "
        "Recorded as a deviation in every output record.",
    )
    args = ap.parse_args()

    base = {
        "checkpoint_path": args.checkpoint_path,
        "hf_revision": args.hf_revision,
        "deterministic_seed": True,  # B3-style replay: identical input AND seed
        # DEVIATION, recorded: the guardrail runners require the GATED repo
        # nvidia/Cosmos-Guardrail1, which this account has no approved access to.
        # They are a text/video safety filter, not part of the world-action model
        # we are measuring, but they ARE real deployed cost -- so every number
        # produced with guardrails=False understates the official baseline by
        # whatever the guardrails cost. Re-measure once access is granted.
        "guardrails": not args.no_guardrails,
    }

    catalogue: dict[str, dict[str, Any]] = {
        # B0 official baseline: 4 steps, vision latent generated, no VAE decode.
        "B0": {**base, "num_steps": 4, "decode_video": False},
        # B1 prices the VAE decode separately (same generation work as B0).
        "B1": {**base, "num_steps": 4, "decode_video": True},
        # B2 denoising-step sweep; nothing else altered.
        "B2_steps_1": {**base, "num_steps": 1, "decode_video": False},
        "B2_steps_2": {**base, "num_steps": 2, "decode_video": False},
        "B2_steps_3": {**base, "num_steps": 3, "decode_video": False},
        "B2_steps_4": {**base, "num_steps": 4, "decode_video": False},
        # B3 is byte-identical to B0 by construction. Its purpose is not to measure
        # the model but to measure *us*: with identical input and seed, the spread
        # of wall times is the noise floor of this host. M2 saw the same work vary
        # +5.8% between runs while another user shared the GPU, so without this
        # number no later speedup claim is interpretable. Run it with a high
        # --iters, in a quiet window.
        "B3_replay": {**base, "num_steps": 4, "decode_video": False},
        # Same settings as B3_replay; the difference is that it is meant to be run
        # with --cooldown-s so the card stays in steady state. Two entries rather
        # than one because the sustained and steady-state floors are *both* real
        # numbers we need: the steady-state one bounds what a fair A/B comparison
        # can resolve, and the sustained one is what a deployed server actually
        # experiences under continuous load.
        "B3_replay_cooled": {**base, "num_steps": 4, "decode_video": False},
    }

    requested = [s.strip() for s in args.configs.split(",") if s.strip()]
    unknown = [c for c in requested if c not in catalogue]
    if unknown:
        raise SystemExit(f"unknown config(s) {unknown}; known: {sorted(catalogue)}")

    if len(requested) > 1 and not args.allow_multi:
        raise SystemExit(
            f"refusing to run {len(requested)} configs in one process: the model is "
            "~31 GiB resident and the previous one is not reliably freed, so the "
            "second load OOMs. Run one --configs per process (see "
            "scripts/run_anatomy_sweep.sh), or pass --allow-multi if you know the "
            "model fits twice."
        )

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    run_id = "anatomy-" + dt.datetime.now(dt.timezone.utc).strftime("%Y%m%dT%H%M%SZ")

    env = {
        "run_id": run_id,
        "timestamp_utc": dt.datetime.now(dt.timezone.utc)
        .replace(microsecond=0, tzinfo=None)
        .isoformat()
        + "Z",
        "python": platform.python_version(),
        "torch": torch.__version__,
        "torch_cuda": torch.version.cuda,
        "gpu": gpu_snapshot(),
        "attention": attention_backend_probe(),
        "cosmos_framework_sha": sh("git rev-parse HEAD"),
        "warmup": args.warmup,
        "iters": args.iters,
        "input_kind": (
            f"REAL captured request ({args.request_npz})"
            if args.request_npz
            else "SYNTHETIC deterministic observation (seed %d)" % OBS_SEED
        ),
        "guardrails_enabled": not args.no_guardrails,
        "deviations": ([] if not args.no_guardrails else ["guardrails disabled (gated Cosmos-Guardrail1 unavailable)"]),
    }
    print(json.dumps(env, indent=2))

    summaries = []
    for name in requested:
        print(f"\n=== {name} ===", flush=True)
        s = run_config(
            name,
            catalogue[name],
            args.warmup,
            args.iters,
            out_dir,
            run_id,
            profile_trace=not args.no_trace,
            hook_modules=not args.no_hooks,
            request_npz=args.request_npz,
            hook_depth=args.hook_depth,
            hook_max=args.hook_max_modules,
            hook_pattern=args.hook_pattern,
            cooldown_s=args.cooldown_s,
        )
        summaries.append(s)
        w = s["wall_ms"]
        if w:
            print(
                f"  wall {w['mean']:.1f} +/- {w['std']:.1f} ms "
                f"(p50 {w['p50']:.1f}, p95 {w['p95']:.1f}, n={w['n']})",
                flush=True,
            )
        print(f"  peak VRAM {s['vram_peak_mib']['reserved']:.0f} MiB reserved", flush=True)

    # One file per config. The model is ~31 GiB resident, so two configs cannot
    # coexist in one process (observed: B1 OOMs after B0 because references held
    # by the profiler/hooks keep the first model alive despite del+empty_cache).
    # Run one config per process and let these files compose.
    for s in summaries:
        (out_dir / f"{s['config']}.summary.json").write_text(
            json.dumps({"env": env, "configs": [s]}, indent=2)
        )
        print(f"wrote {out_dir / (s['config'] + '.summary.json')}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
