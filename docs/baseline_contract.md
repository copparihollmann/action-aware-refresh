# Baseline contract

**Status: VERIFIED against checked-out source on 2026-08-03** (host `firesim2`).
Every value below was read out of the pinned clones, out of the captured
`--help`, or probed on this host. Nothing here is a spec expectation carried
forward. Where a quantity must be *measured* rather than read it is marked
**[M2]** and left empty — an unmeasured number is never filled in with a
plausible one.

## Pinned revisions

| source | commit |
|---|---|
| `NVIDIA/cosmos-framework` | `a904d2d36b774a51dd06ff9ff906816b1a04f579` |
| `NVLabs/RoboLab` | `0aef241fb088ca21bb4ebd24448940ed56620d17` |
| `NVIDIA/cosmos` | `404b9bf2144640834c63ae7d9e7269e0f4ea02cb` |
| `NVlabs/cosmos-policy` | `18a2accadf4e7a3531e56754102af5a24d2316da` |
| `Physical-Intelligence/openpi` | `15a9616a00943ada6c20a0f158e3adb39df2ccac` |
| `nvidia/Cosmos3-Nano-Policy-DROID` | `6706d7680581c255ff61e0f3bb49d90eac55c79e` |

The model repo reports `gated: False`, `license: other`. **Upstream's default
`--hf-revision` is the moving ref `main`** — we pin the SHA above in
`reproducibility/model_revisions.json`, which is what makes a run reproducible.

## Server defaults

Source: `cosmos_framework/scripts/action_policy_server_robolab.py`,
`class RobolabServerArgs(pydantic.BaseModel)` with `extra="forbid"`, CLI via
`tyro` (`snake_case` → `--kebab-case`; an unknown flag is a hard error). Every
row below was cross-checked against `docs/generated/cosmos_server_help.txt`.

| property | value | flag |
|---|---|---|
| checkpoint | `nvidia/Cosmos3-Nano-Policy-DROID` | `--checkpoint-path` |
| HF revision | `main` (upstream default; we override) | `--hf-revision` |
| port | **8000** | `--port` |
| host | `0.0.0.0` (we bind `127.0.0.1`) | `--host` |
| denoising steps | **4** | `--num-steps` |
| decoded video | **disabled** | `--decode-video` / `--no-decode-video` |
| action chunk size | **32** | `--action-chunk-size` |
| conditioning FPS | **15.0** | `--conditioning-fps` |
| action dim | **8** (`joint_pos`; 10 for `midtrain`) | `--action-dim` |
| image height × width | 540 × 640 | `--image-height` / `--image-width` |
| transform resolution | `480` | `--resolution` |
| action space | `joint_pos` | `--action-space` |
| use_state | `True` | `--use-state` |
| history length | 1 | `--history-length` |
| sampler | `unipc` (or `edm`) | `--sampler` |
| guidance | 3.0 | `--guidance` |
| shift | 5.0 | `--shift` |
| seed / deterministic | 0 / `False` | `--seed` / `--deterministic-seed` |
| domain name | `droid_lerobot` | `--domain-name` |

All eight spec §3 expectations are **confirmed** (port 8000, 4 steps, chunk 32,
fps 15, action dim 8, `decode_video=False`, client horizon 32, and both
`samples["action"]` and `samples["vision"]` produced).

Transport is **OpenPI's `WebsocketPolicyServer`** (msgpack + NumPy), not plain
HTTP: on connect it sends an empty metadata dict, each client message is an
observation dict, each response is a dict with `action` and — only when enabled —
`video`. An HTTP health endpoint is advertised alongside:
`http://<host>:<port>/healthz`.

`--deterministic-seed` is what makes the **B3** deterministic-replay
configuration possible; without it the server advances a NumPy RNG per request.

> Note for the profiler: denoising steps and `decode_video` are **server-start
> arguments, not per-request parameters**. The B0–B4 sweep therefore requires
> restarting the server per configuration (or driving the model in-process); it
> cannot be done by varying fields in a request payload.

## The `decode_video` question — answered

This is spec §3's critical point, and the answer is unambiguous. From `infer()`:

```python
samples = self.model.generate_samples_from_batch(
    data_batch, guidance=..., seed=[seed], num_steps=self.cfg.num_steps, shift=...
)
action = samples["action"][0][:, : self.cfg.action_dim]   # ALWAYS
...
if self.cfg.decode_video:
    pred_vision_latent = samples["vision"][0]             # only read here
    video = self.model.decode(pred_vision_latent)         # VAE decode
```

**`decode_video=False` disables only the VAE decode.** `generate_samples_from_batch`
returns `samples["vision"]` unconditionally; the flag merely decides whether that
latent is decoded to RGB. "No decoded video" is *not* "no visual imagination",
exactly as the spec warned.

`_build_sample()` shows how much imagination is being requested:

```python
t_frames = self.cfg.action_chunk_size + 1                     # 33
video = torch.zeros((3, t_frames, image_h, image_w), dtype=torch.uint8)
video[:, 0] = <the single real observation frame>              # frames 1..32 are zeros
```

One real frame conditions the generation of **32 future frames**. So the vision
branch is doing substantial work *by construction*. Whether that work is
*separable* from action generation is a different question, and the model config
says the two are entangled: `joint_attn_implementation: "two_way"`,
`action_gen: true`, `position_embedding_type: "unified_3d_mrope"` with
`unified_3d_mrope_temporal_modality_margin: 15000`, `patch_spatial: 2`,
`latent_downsample_factor: 16`, `max_vae_latent_side_after_patchify: 20`.

**[M2]** What fraction of server time the vision tokens actually cost, and
whether they can be omitted, is the go/no-go measurement. **Not yet answered —
do not pre-judge it from the shapes above.**

## Action tensor shapes and the open-loop horizon

Server side (`_build_sample` → `infer`):

- `use_state_rows = 1` (because `use_state=True`), so the action tensor is
  `(action_chunk_size + 1, action_dim)` = **(33, 8)**.
- returned actions are trimmed by `history_length=1` → `action[1:]` = **(32, 8)**.
- the gripper channel is inverted server-side:
  `action_np[:, -1] = 1.0 - action_np[:, -1]`.

Client side (`RoboLab/policies/cosmos3/client.py`, `robolab/eval/base_client.py`):

- `Cosmos3Client.OPEN_LOOP_HORIZON = 32` (base class default is 1).
- `_needs_new_chunk(env_id)` is
  `env_id not in self._chunks or self._counters[env_id] >= self.open_loop_horizon`.
- the gripper is re-thresholded client-side:
  `chunk[..., -1] = (chunk[..., -1] > 0.5)`.

**Consequence for Experiment B (spec §11.2):** the server returns exactly 32
actions and the client consumes exactly 32 before requesting again. The baseline
already calls the policy **once per 32 control steps**, with no overlap and no
early replanning. Experiment B must therefore be framed as *changing that
cadence*. It must **not** be described as "skipping control frames the baseline
would otherwise have computed" — the baseline does not compute them.

Client entry point: `policies/cosmos3/run.py` declares `--remote-host` /
`--remote-port`, then adds `robolab.eval.runner.add_common_eval_args` (supplying
`--task`, `nargs="+"`) and isaaclab `AppLauncher` args (supplying `--headless`).
It forces `enable_cameras = True` and registers the `WRIST_LEFT_RIGHT_HEAD`
three-camera preset. Note the server composes its own single conditioning image
from the observation (`_extract_observation_image` / `_compose_roboarena_views`).

## Benchmark surface

- **120** benchmark task classes in `robolab/tasks/benchmark/` = RoboLab-120.
- Difficulty: 64 `simple`, 39 `moderate`, 17 `complex` (`difficulty_score` 1–11).
- Competency attributes: semantics 60, spatial 29, color 26, affordance 12,
  sorting 12, conjunction 8, counting 7, vague 7, size 6, stacking 6,
  reorientation 6.
- `num_sequential_stages`: 113 single-stage, 5 two-stage, 2 three-stage.
  Per-task `episode_s` ranges 20–300 s.
- Task sets are generated from this metadata by `scripts/select_task_sets.py`.
  Both smoke tasks (`BananaInBowlTask`, `RubiksCubeAndBananaTask`) were confirmed
  present in the registry.
- **Cost:** upstream's README quotes ~30 GPU-hours per 100 tasks → ≈36
  GPU-hours for RoboLab-120 **per configuration**.

## Runtime environment (this host)

| property | value |
|---|---|
| host | `firesim2`, Ubuntu 24.04.4, glibc 2.39, 64× Xeon Gold 6242, 503 GB RAM |
| GPU (Cosmos) | NVIDIA L40S, 46068 MiB (45.0 GiB), **SM 8.9**, index 0, UUID `GPU-a2a44f60-…` |
| GPU (RoboLab) | NVIDIA L40S, index 2, UUID `GPU-22350958-…` (separate PCIe switch) |
| driver / CUDA | 580.126.18 / 13.0 |
| Cosmos env | Python 3.13.14; `uv sync --all-extras --group=cu130-train --group=policy-server` |
| torch | **2.10.0+cu130** (asserted against the group's pin at install time) |
| torchvision / torchcodec | 0.25.0+cu130 / 0.10.0+cu130 |
| flash-attn / flash-attn-3-nv | 2.7.4.post1+cu130.torch210 / 1.0.3+cu130.torch210 |
| natten / transformer-engine | 0.21.6.dev6+cu130.torch210.gb300 / 2.12+cu130.torch210 |
| triton / numpy | 3.6.0 / 2.2.6 |

Recorded in `reproducibility/cosmos_install.json`.

`--all-extras` is required, not optional: without it `iopath` is missing and the
server module fails to import. This matches upstream's documented command.

**Never invoke the server with a bare `uv run`.** `uv run` re-resolves the
project to its default dependency set, which excludes the `cuXXX` group. Observed
on this host: it replaced torch 2.10.0+cu130 with 2.13.0+cu130 and left
`flash_attn` failing to import with an ABI error, silently changing the stack
under measurement. Use `.venv/bin/python` (as `scripts/start_cosmos_server.sh`
now does) or `uv run --no-sync`.

### Attention backend — a recorded deviation

`cosmos_framework/model/attention/backends.py::get_backend_list(arch_tag)`
returns, for `arch_tag >= 80` (our SM 8.9 → 89):

```
["flash2", "cudnn", "natten"]
```

`flash3` is gated to `arch_tag == 90` (Hopper). **We run flash2/cudnn where
NVIDIA's published figures use FlashAttention-3.** Absolute latencies here are
therefore not comparable to upstream's numbers. Every claim in this project is a
within-machine delta against our own measured baseline, which is sound — but the
selected backend must be recorded in each result.
**[M2]** log the backend `choose_backend()` selects at runtime.

### Weights and VRAM

- Checkpoint total 32.9 GB: transformer 7 shards ≈ 30.35 GB bf16 (≈15.2 B
  params), VAE 1.409 GB, vision encoder 1.153 GB.
- Against 45.0 GiB usable that leaves ≈14 GiB for activations and workspace.
- **[M2]** measured `peak_allocated` / `peak_reserved` at batch 1, and the
  largest `--num-envs` that fits.
- If it OOMs: reduce concurrency, or shard across GPUs 0+1 and record that as
  *the* baseline topology, applied consistently. **Not** quantization — spec §5
  requires that be a separate experiment, not a way to make the baseline fit.

### Energy

`nvidia-smi --query-gpu=total_energy_consumption` → *"not a valid field to
query"* on these GPUs. Energy is therefore a 10 Hz NVML power integral and every
such field is labelled ESTIMATED. The idle floor is **not** uniform across
devices: GPU 0 idles at ≈67 W (P0) while GPUs 1–3 idle at ≈34 W (P8), so the
floor must be measured on the same GPU as the workload. Report gross and
idle-adjusted. NVML indices ignore `CUDA_VISIBLE_DEVICES`; resolve through the
UUID (`action_refresh.energy.nvml_index_for_uuid`).

### Measurement hazard: CPU, not GPU

The GPUs were idle on arrival. The CPU is not: 1-min loadavg ≈46 on 64 cores from
another user's parallel compile, and the GPUs' CPU affinity is cores
`0-15,32-47`. Host-side stages (preprocessing, msgpack serialization, websocket
round-trip, Isaac physics) will be perturbed. `loadavg` plus a process snapshot
is recorded with every run by `scripts/start_cosmos_server.sh`; final anatomy
numbers should be taken in a quieter window, with the contention noted.

## Still to be measured (M2)

- Stage-level split: preprocessing, vision encode, context, joint diffusion
  transformer, VAE decode, postprocess, serialization, communication.
- Action-token vs vision-token counts and tensor shapes; whether the two share
  attention in a way that makes action-only inference impossible without
  adaptation (spec §11.5 outcomes A–D).
- FLOPs with an explicit coverage fraction (fused attention kernels will be
  missed by `FlopCounterMode`).
- Per-request variance under deterministic replay (B3).
- VAE decode cost, priced separately via B1.
- Peak VRAM, and the concurrency ceiling for B4.
