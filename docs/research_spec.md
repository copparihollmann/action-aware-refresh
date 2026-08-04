# Research spec — Action-Aware Predictive Refresh for Efficient World-Action Models

Original prompt used to bootstrap this project (2026-08-03). This file is
the durable source of truth. Any deviation from the spec must be recorded
in `docs/decision_log.md` with rationale.

---

You are the lead research engineer responsible for building a reproducible
end-to-end research prototype called:

    Action-Aware Predictive Refresh for Efficient World-Action Models

Work autonomously, but do not hide failures, fabricate results, or silently
change the research question. Build the project from an empty directory
into a reproducible experimental framework.

## 1. Research mission

Primary platform:

    Cosmos3-Nano-Policy-DROID + RoboLab

Primary research question:

    Can we reduce the end-to-end inference computation of a pretrained
    world-action model by reusing keyframes, visual-imagination latents,
    action chunks, denoising intermediates, and spatial representations,
    while selectively refreshing only action-relevant changes and
    preserving closed-loop robot task success?

The intended application is π0.7/ImageWAM-style robot manipulation, where
future images or visual subgoals are intermediate representations rather
than the final product.

The final product is the robot action.

Images are diagnostic only.

The main result must be a task-success-versus-compute Pareto frontier.

### Hard constraints

1. Use Cosmos3 + RoboLab as the primary baseline and benchmark.
2. Do not retrain Cosmos3 from scratch.
3. Lightweight fine-tuning is allowed:
   - LoRA
   - adapters
   - residual-correction modules
   - small cache-validity or refresh gates
4. Do not optimize primarily for image quality.
5. Do not assume that visual imagination is necessary.
6. Include an action-only or no-imagination baseline.
7. Count all auxiliary overhead:
   - optical flow
   - event generation
   - gate inference
   - cache management
   - serialization and communication
8. Every claimed speedup must report:
   - normalized operations or FLOPs
   - measured GPU time
   - end-to-end latency
   - task success
9. Do not claim that fewer tokens or fewer theoretical FLOPs imply a real
   speedup unless measured latency confirms it.
10. Preserve upstream repositories. Use local research branches, wrappers,
    or clearly recorded patches.
11. Never commit API tokens, Hugging Face tokens, credentials, model
    weights, generated videos, or large experiment outputs.
12. Do not start any run estimated to require more than four GPU-hours
    without first writing the estimated cost and asking for approval.
13. Do not accept licenses or EULAs on my behalf. Stop and request explicit
    confirmation when license acceptance is required.
14. Prefer rapid falsification over elaborate engineering.

## 2. Required sources

Read the current versions of these sources before implementing anything.
Do not rely only on this prompt for commands because repositories may have
changed. Record the date, branch, and exact commit SHA used for every
source.

### Primary implementation sources

- Cosmos overview: https://github.com/NVIDIA/cosmos
- Cosmos Framework: https://github.com/NVIDIA/cosmos-framework
- Cosmos3-Nano-Policy-DROID model card: https://huggingface.co/nvidia/Cosmos3-Nano-Policy-DROID
- Cosmos3-Nano-Policy-DROID model configuration:
  https://huggingface.co/nvidia/Cosmos3-Nano-Policy-DROID/blob/main/config.json
- Current Cosmos3 RoboLab policy server:
  https://github.com/NVIDIA/cosmos-framework/blob/main/cosmos_framework/scripts/action_policy_server_robolab.py
- Current server documentation:
  https://github.com/NVIDIA/cosmos-framework/blob/main/docs/action_policy_droid_server.md
- RoboLab: https://github.com/NVLabs/RoboLab
- RoboLab Cosmos3 client: https://github.com/NVLabs/RoboLab/tree/main/policies/cosmos3
- RoboLab project page: https://research.nvidia.com/labs/srl/projects/robolab/
- RoboLab leaderboard: https://research.nvidia.com/labs/srl/projects/robolab/leaderboard.html

### Auxiliary implementation and transfer sources

- Cosmos Policy: https://github.com/NVlabs/cosmos-policy
- OpenPI: https://github.com/Physical-Intelligence/openpi
- π0.7: https://www.pi.website/blog/pi07

### Core research references

- ImageWAM: https://arxiv.org/pdf/2606.19531
- Foveated Diffusion: https://bchao1.github.io/foveated-diffusion/
- DeltaTok / DeltaWorld: https://deltatok.github.io/ · https://arxiv.org/abs/2604.04913
- Fast-WAM: https://arxiv.org/abs/2603.16666 · https://yuantianyuan01.github.io/FastWAM/

### Diffusion caching references

- TeaCache: https://github.com/ali-vilab/TeaCache · https://arxiv.org/abs/2411.19108
- FasterCache: https://github.com/Vchitect/FasterCache · https://arxiv.org/abs/2410.19355
- DeepCache: https://github.com/horseee/DeepCache · https://arxiv.org/abs/2312.00858
- Token-wise feature caching: https://arxiv.org/abs/2410.05317

### Event and flow references

- v2e: https://github.com/SensorsINI/v2e · https://arxiv.org/abs/2006.07722
- ESIM: https://github.com/uzh-rpg/rpg_esim
- TorchVision RAFT: https://pytorch.org/vision/stable/models/raft.html

### Tooling documentation

- uv: https://docs.astral.sh/uv/
- NVIDIA Container Toolkit: https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/latest/install-guide.html
- PyTorch Profiler: https://docs.pytorch.org/tutorials/recipes/recipes/profiler_recipe.html
- Nsight Systems: https://docs.nvidia.com/nsight-systems/UserGuide/index.html

Before coding, create:

    docs/literature_notes.md
    docs/source_map.md
    reproducibility/source_manifest.json

For each paper or project, summarize only what is relevant to this project:

- reusable state
- unit of computation
- inference path
- training requirements
- available code
- computational claim
- evaluation metric
- limitation
- what we can realistically reuse

Do not reproduce long paper summaries. The purpose is implementation guidance.

## 3. Expected current baseline — verify, do not blindly assume

The current source is expected to have approximately these properties:

- Model: nvidia/Cosmos3-Nano-Policy-DROID
- Policy server port: 8000
- Default action denoising steps: 4
- Default action chunk size: 32
- Default conditioning FPS: 15
- Default action dimension: 8
- Cosmos3 RoboLab client open-loop horizon: 32
- Video decoding disabled by default
- The model still generates a vision latent even when decoding is disabled
- The policy server calls generate_samples_from_batch and receives both:
  - samples["action"]
  - samples["vision"]

Verify every property directly against the checked-out source and write the
verified values to:

    docs/baseline_contract.md

This detail is critical:

    "decode_video=False" is not equivalent to "no visual imagination."

It only appears to skip VAE decoding. Determine whether the diffusion
transformer still spends significant computation generating vision tokens.

Do not proceed to visual-cache engineering until this is profiled.

## 4. Create the workspace

Create this structure:

    action-aware-refresh/
    ├── README.md
    ├── CLAUDE.md
    ├── pyproject.toml
    ├── uv.lock
    ├── .gitignore
    ├── Makefile
    ├── configs/
    │   ├── machine.example.yaml
    │   ├── topology.example.yaml
    │   ├── tasks/
    │   ├── methods/
    │   └── sweeps/
    ├── docs/
    │   ├── literature_notes.md
    │   ├── source_map.md
    │   ├── baseline_contract.md
    │   ├── environment_report.md
    │   ├── compute_anatomy.md
    │   ├── experiment_protocol.md
    │   ├── decision_log.md
    │   └── troubleshooting.md
    ├── reproducibility/
    │   ├── source_manifest.json
    │   ├── environment.json
    │   ├── commands.jsonl
    │   ├── model_revisions.json
    │   └── patches/
    ├── third_party/
    │   ├── cosmos/
    │   ├── cosmos-framework/
    │   ├── RoboLab/
    │   ├── cosmos-policy/
    │   └── openpi/
    ├── src/action_refresh/
    │   ├── __init__.py
    │   ├── config.py
    │   ├── logging.py
    │   ├── metrics.py
    │   ├── profiler.py
    │   ├── energy.py
    │   ├── server/
    │   ├── client/
    │   ├── gates/
    │   ├── flow/
    │   ├── events/
    │   ├── cache/
    │   ├── oracle/
    │   └── analysis/
    ├── scripts/
    │   ├── audit_machine.sh
    │   ├── clone_sources.sh
    │   ├── setup_cosmos.sh
    │   ├── setup_robolab.sh
    │   ├── start_cosmos_server.sh
    │   ├── run_robolab.sh
    │   ├── smoke_test.sh
    │   ├── profile_baseline.sh
    │   ├── run_sweep.py
    │   └── build_report.py
    ├── tests/
    │   ├── unit/
    │   ├── integration/
    │   └── smoke/
    ├── experiments/
    │   ├── registry.yaml
    │   └── task_sets.yaml
    └── results/
        ├── raw/
        ├── processed/
        ├── profiles/
        ├── plots/
        └── reports/

Initialize the top-level repository.

For each third-party repository:

1. Clone it.
2. Record:
   - URL
   - branch
   - commit SHA
   - dirty status
   - license
3. Create a local branch named:
       research/action-aware-refresh
4. Do not push anywhere.
5. Keep modifications as small commits.
6. Export all third-party diffs into:
       reproducibility/patches/

Do not clone every optional caching or event repository immediately.
Initially record their URLs in source_manifest.json. Clone them only when
their phase is reached.

## 5. Phase 0 — Machine and deployment audit

First inspect the machine without changing it.

Collect: `uname -a`, `/etc/os-release`, CPU model + core count, total RAM,
free disk by filesystem, NVIDIA GPU names, GPU count, VRAM per GPU, NVIDIA
driver, CUDA version reported by nvidia-smi, nvcc version if installed,
Docker version, Docker NVIDIA runtime availability, git + git-lfs + uv +
Python versions, Nsight Systems availability, nvidia-smi power/energy
counter support, network connectivity to GitHub and Hugging Face.

Write:

    docs/environment_report.md
    reproducibility/environment.json

### Hard platform checks

- Native Linux is required.
- Prefer Ubuntu 22.04 or newer.
- RoboLab requires an NVIDIA GPU.
- Determine whether Cosmos and RoboLab can fit concurrently.

Choose one deployment topology:

A. **single_host_multi_gpu** — one GPU or GPU set for Cosmos, another GPU
   for RoboLab/Isaac Sim.
B. **single_host_shared_gpu** — only when memory measurements demonstrate
   that both fit safely.
C. **two_host** — Cosmos policy server on a model machine; RoboLab client
   on a simulation machine.

Create `configs/topology.yaml` with cosmos_host, cosmos_port,
cosmos_visible_devices, robolab_visible_devices, shared_hf_cache,
output_root, model_cache_root, expected network path, expected clock
synchronization assumptions.

Do not silently attempt quantization to make the official baseline fit.
Quantization changes the baseline and must be a separate experiment.

If the machine is insufficient:

1. Finish source checkout and non-GPU setup.
2. Produce a precise blocker report.
3. State minimum missing capabilities.
4. Do not pretend the baseline ran.

## 6. Legal and access checkpoint

Check whether the following require manual acceptance:

- NVIDIA Cosmos model license
- Hugging Face gated repository access
- NVIDIA Omniverse/Isaac Sim EULA

Do not accept these automatically.

Ask me to complete or confirm them only when required.

Never write HF_TOKEN into a file.

Use one of:

    uvx hf@latest auth login

or an environment variable passed at runtime.

Record only whether authentication succeeded, never the token.

## 7. Phase 1 — Install and verify the official baseline

Use official instructions from the current source checkout rather than
relying only on this prompt.

### 7.1 Cosmos Framework setup

Preferred reproducible path:

    cd third_party/cosmos-framework
    docker build -t action-refresh-cosmos-framework:<short_sha> .

Select the dependency group from the actual driver:

- CUDA 13.x-compatible driver: `--group=cu130-train`
- CUDA 12.x-compatible driver: `--group=cu128-train`

The policy server also needs `--group=policy-server`.

Verify the current pyproject and README before selecting a group.

Use a persistent Hugging Face cache volume and persistent uv cache. Do not
mount an anonymous `/workspace/.venv` that disappears unexpectedly.

Create `scripts/setup_cosmos.sh` and `scripts/start_cosmos_server.sh`. The
start script must expose the selected GPUs explicitly, bind port 8000 by
default, persist logs, record the checkpoint revision, record the command,
expose a health check, support additional server arguments, support
deterministic and nondeterministic seeds, default to no decoded video, and
never put tokens in command history files.

Before starting, run:

    python -m cosmos_framework.scripts.action_policy_server_robolab --help

Store the output in `docs/generated/cosmos_server_help.txt`. Use exact
current option spelling.

Expected baseline semantics to verify:

    checkpoint: nvidia/Cosmos3-Nano-Policy-DROID
    denoising steps: 4
    decoded video: disabled
    action chunk: 32

Pin the Hugging Face revision to the resolved model commit when possible.

### 7.2 RoboLab setup

Read the current root README and `policies/cosmos3/README.md`.

The repository may support both uv and Docker paths. Choose the path that
is currently recommended and compatible with the machine. Prefer an
isolated environment from Cosmos because their dependency stacks may
conflict.

For a native uv setup, expected shape is approximately:

    sudo apt install ffmpeg
    uv venv --python 3.11
    source .venv/bin/activate
    uv sync --extra isaac50

or:

    uv sync --extra isaac51

Never install Isaac 5.0 and 5.1 into the same environment.

Use only one simulator stack for all comparisons.

Record: Isaac Sim version, Isaac Lab version, PhysX version when available,
RoboLab commit.

Run the official verification suite before policy evaluation:

    uv run pytest tests/

Also run a no-policy smoke test and a gripper sanity test.

Do not set `OMNI_KIT_ACCEPT_EULA=Y` until I have confirmed acceptance.

Create `scripts/setup_robolab.sh` and `scripts/run_robolab.sh`.

### 7.3 Baseline server-client smoke test

Start the Cosmos server. Confirm `curl http://<host>:8000/healthz`.

Then run one environment for a short task:

    python policies/cosmos3/run.py \
      --remote-host <host> \
      --remote-port 8000 \
      --task BananaInBowlTask \
      --num-envs 1 \
      --headless

Use the exact current command supported by the checked-out repository.

Then run a second sanity task with different semantics, such as the
repository's recommended RubiksCubeAndBananaTask if still available.

A smoke test passes only when:

- server loads the correct checkpoint
- client connects
- actions are returned
- simulator advances
- results are written
- no NaNs are present
- action dimensions match
- episode termination is recorded
- videos or diagnostic frames are produced only when requested

Create `results/reports/baseline_smoke.md`. Do not proceed if the baseline
is not reproducible.

## 8. Phase 2 — Build the measurement system before optimization

Every run must have: run_id, timestamp, git SHAs, model revision, simulator
version, GPU model, method, full method configuration, task list, seeds,
number of environments, warmup count, episode count, command, output paths.

Store raw request-level metrics as JSONL or Parquet.

### Required request-level fields

run_id, task, env_id, episode_id, control_step, policy_request_index, seed,
method, cache_mode, cache_age, gate_decision, gate_features,
policy_call_performed, full_refresh, partial_refresh, keyframe_refresh,
number_of_denoising_steps, number_of_transformer_forwards, blocks_recomputed,
visual_tokens_processed, action_tokens_processed, tokens_recomputed,
cache_hits, cache_misses, preprocessing_ms, serialization_ms,
network_roundtrip_ms, server_queue_ms, model_wall_ms, model_cuda_ms,
vision_encode_ms, context_ms, denoising_ms, vision_decode_ms,
postprocess_ms, peak_allocated_vram_mb, peak_reserved_vram_mb,
gpu_power_w_mean, gpu_energy_j, estimated_flops, measured_flop_coverage,
observation_hash, action_chunk_hash, contact_state when available,
subtask_progress when available, failure reason when available.

### Required episode-level fields

success, continuous progress score, steps, duration, policy calls,
keyframe refreshes, full refreshes, partial refreshes, total denoising
steps, total visual tokens, total action tokens, total estimated FLOPs,
total GPU time, total wall time, total auxiliary overhead, total energy,
mean action latency, p50/p90/p95/p99 action latency, mean cache age,
maximum cache age, missed refresh count, unnecessary refresh count,
task competency, task difficulty.

### Three timing layers

1. End-to-end client timing
2. Server stage timing with CUDA events
3. PyTorch Profiler or Nsight Systems traces for selected requests

Use `torch.cuda.Event` for GPU timing and synchronize only at controlled
measurement boundaries. Do not synchronize every operation in normal runs.

### Energy

Use NVML sampling when total-energy counters are unavailable. Integrate
power over time. Sample at a documented frequency. Measure idle power
separately. Report both gross and idle-adjusted energy. Clearly label
estimates.

### FLOPs

Use PyTorch profiler `with_flops` when supported. Also try
`torch.utils.flop_counter.FlopCounterMode`. Identify unsupported custom
attention or fused kernels. Add analytic counts from actual tensor shapes
where necessary. Report FLOP coverage rather than presenting incomplete
counts as exact.

Create `docs/experiment_protocol.md` and `docs/compute_anatomy.md`.

## 9. Establish the true compute anatomy

Instrument one deterministic inference request and several warmed requests.

Break total server time into: preprocessing, vision encoding,
language/context processing, joint diffusion transformer, action-related
computation, vision-related computation, scheduler overhead, VAE decoding,
postprocessing, serialization, communication.

The current policy path may jointly denoise visual and action tokens. Do
not invent separate action and vision timing if the implementation does
not expose it. Instead:

1. map the model modules and token layouts;
2. record tensor shapes;
3. determine whether action and vision share attention;
4. identify which computations can actually be disabled independently;
5. distinguish shared cost from modality-specific cost.

Run these diagnostic configurations:

- **B0** Official baseline: four denoising steps, action + vision latent generation, no visual decoding.
- **B1** Diagnostic decode: same as B0 + enable decoded rollout video; measure decoding overhead separately.
- **B2** Denoising-step sweep: {1, 2, 3, 4} steps; do not alter other settings.
- **B3** Deterministic replay: identical input and seed, repeated requests to estimate measurement variance.
- **B4** Batch/concurrency: single request, supported RoboLab parallel-environment configurations, separate throughput and latency reporting.

Produce: stage-level latency breakdown, GPU kernel time, memory breakdown,
estimated FLOPs, action/vision token counts, a flame graph or profiler
trace, a written conclusion identifying the dominant cost.

**Primary go/no-go question:** Is visual imagination a substantial part of
the deployed Cosmos3 policy cost, or is the main cost shared/action
computation?

Do not begin keyframe residual refinement until this is answered.

## 10. Baseline task set and statistical protocol

Do not immediately run all 120 tasks.

- **SMOKE** — 2 tasks × 1–2 episodes.
- **PILOT** — 12–18 tasks representing: pick-and-place, grasp/transport,
  relational multi-object, articulated manipulation, insertion/alignment,
  contact-rich, procedural/multi-stage, visual distractors. Include tasks
  where baseline success is neither always zero nor always one.
- **FULL** — all RoboLab-120 tasks.

Task selection must be based on a short official-baseline screening run.
Do not cherry-pick only successful examples.

Episode counts:
- initial screening: 5–10 paired episodes per task
- promoted configurations: at least 20–30 paired episodes per task
- final frontier configurations: at least 50 where computationally feasible
- final full benchmark: use the benchmark's standard evaluation protocol

Use the same initial states and seeds across methods whenever RoboLab
supports it.

Report: macro-average across tasks, micro-average across episodes, 95% CI,
task-level paired differences, stratification by competency, stratification
by difficulty.

Non-inferiority margins: 0, 1, 2, and 5 absolute success percentage points.

Main compute measure:

    normalized_total_compute =
        method_total_compute_including_overhead /
        official_baseline_total_compute

Main figure: task success versus normalized total compute.

Also compute:

- minimum compute at ≤ 1 point success loss
- minimum compute at ≤ 2 points success loss
- minimum compute at ≤ 5 points success loss
- area under the success-compute Pareto frontier

## 11. Rapid prototype ladder

Implement in the order below. Do not jump directly to spatial diffusion or
LoRA.

### 11.1 Experiment A — Reduced denoising steps

Hypothesis: the four-step baseline contains some action-generation redundancy.

Configurations: `num_steps ∈ {1, 2, 3, 4}`.

First run offline on recorded observations. Measure: action L2 deviation
from four-step teacher, end-effector translation deviation, rotation
deviation, gripper-state disagreement, action endpoint deviation, latency,
GPU time, FLOPs, vision latent deviation as diagnostic only.

Then run closed-loop PILOT evaluation.

This is the strongest trivial baseline and must remain in every later plot.

Stop condition: if 2 or 3 steps already provide almost all available
speedup, later caching must beat this frontier, not merely beat the
four-step baseline.

### 11.2 Experiment B — Action-chunk and policy-call reuse

First inspect the actual RoboLab InferenceClient behaviour. The current
Cosmos3 client is expected to use a 32-step open-loop horizon. Verify
whether all 32 returned actions are executed before the next policy call.

Do not describe this as "skip every control frame" if the baseline already
calls the policy only once per 32 control steps.

Test:
- shorter horizons: 4, 8, 16
- official horizon: 32
- longer reuse policies: 48 or 64 only when a principled suffix/repetition
  or extrapolation rule exists
- early refresh before chunk exhaustion
- late refresh after chunk exhaustion

Separate:
1. changing how many existing actions are executed;
2. reusing a stale visual/keyframe representation;
3. skipping an entire new model invocation.

Measure: policy calls per episode, success, progress, contact failures,
latency, compute, action discontinuity at chunk boundaries, failure after
extending a stale chunk.

Mandatory baseline: fixed chunk age / fixed refresh interval.

An adaptive gate is valuable only when it beats the fixed schedule at
equal compute.

### 11.3 Experiment C — Oracle temporal refresh

Before optical flow or event cameras, use simulator privileged state to
estimate the upper bound.

Construct an oracle refresh signal from available RoboLab state: contact
onset, contact loss, task-object pose change, target pose change,
unexpected object motion, gripper/object relative motion mismatch,
semantic subtask transition, failed expected motion, maximum cache age.

Do not assume every signal exists. Inspect RoboLab APIs and document which
are available.

Oracle decision: `reuse`, `action refresh`, `full refresh`.

Compare against fixed intervals at matched policy-call counts.

Continue only if oracle temporal scheduling provides at least
approximately 20–25% total-compute reduction with no more than 2 absolute
success points lost. If it does not beat fixed scheduling, record the
negative result and deprioritize learned temporal gates.

### 11.4 Experiment D — Cheap event and optical-flow gate

Implement signals in increasing complexity.

- **D0** Frame difference — downsampled grayscale absolute difference.
- **D1** Cheap event proxy — log-intensity threshold crossings, positive
  and negative event counts, event density by region, no expensive
  interpolation.
- **D2** Classical low-resolution optical flow — OpenCV DIS or Farneback;
  include its measured overhead.
- **D3** RAFT-small — only after D1/D2 show signal; TorchVision impl, low
  resolution, include GPU cost and memory.
- **D4** Feature residual — current vs prior observation features; use
  already available Cosmos features when accessible; otherwise a
  lightweight frozen encoder; do not add a large new encoder without
  proving benefit.

Signals: global event density, robot-region event density, task-object
event density, background event density, flow magnitude, flow
inconsistency, ego-motion-compensated residual, predicted-vs-observed
object motion, proprioceptive residual, contact transition, cache age,
previous action magnitude.

Initial gate: deterministic thresholds; tune only on a calibration split;
keep a separate evaluation split. Then train a small gate (logistic
regression, gradient-boosted tree, MLP, or similarly small) with target
labels from counterfactual teacher evaluation; asymmetric loss with larger
penalty for missed critical refreshes; no backbone retraining.

Always report: refresh precision, refresh recall, recall on
contact-critical timesteps, false refreshes caused by camera/background
motion, detection delay, gate overhead, percentage of oracle savings
recovered.

Do not use full v2e online initially. Use v2e later for offline
robustness testing of event threshold, noise, bandwidth, and latency.

### 11.5 Experiment E — No-imagination / action-only baseline

This is mandatory.

Inspect the Cosmos3 generation implementation and determine whether visual
output tokens can be disabled while retaining action generation.

Do not confuse:

    disable decoded video

with:

    disable visual latent generation

Implement the least invasive correct action-only path available.

Possible outcomes:
- A. Configuration already supports action-only generation.
- B. Token packing can omit vision output tokens.
- C. Model code requires a small patch.
- D. Shared attention makes true action-only inference impossible without
     adaptation.

Document the actual outcome.

Compare: official joint vision+action latent generation, no VAE decode,
action-only generation, reduced-step action-only generation.

Measure task success and total compute.

This is the Fast-WAM challenge to the project.

Key decision: if action-only inference is equally accurate and cheaper
than cached imagination, visual residual refinement is not the main
contribution.

Do not force a positive result for visual imagination.

### 11.6 Experiment F — Cross-denoising-step caching

First implement offline analysis. For every denoising step and transformer
block, record: residual output, attention output where accessible,
hidden-state norm, inter-step difference norm, action-token difference,
vision-token difference.

Compute:

    d[s,l] = ||R[s,l] - R[s-1,l]|| / (||R[s-1,l]|| + epsilon)

Test in this order:

- **F0** Uniform whole-block reuse
- **F1** Timestep-dependent block reuse
- **F2** Residual-threshold reuse
- **F3** Separate action-token and vision-token thresholds
- **F4** Token-wise reuse
- **F5** Learned linear residual correction
- **F6** Request-level budget selected by physical-surprise score

Use TeaCache, FasterCache, DeepCache, and token-wise caching as baselines
or design references. Do not copy methods without respecting their
licenses.

The novel condition should be: denoising reuse budget is conditioned on
action-relevant physical surprise.

Physical signals do not need to change inside one denoising trajectory.
They can select the cache budget for the entire request, while denoising
residuals select specific blocks or tokens.

Measure: action deviation, closed-loop success, actual block executions,
actual CUDA time, cache memory, cache lookup/copy overhead,
contact-sensitive failure rate.

Continue when: at least approximately 1.3× measured action-inference
speedup beyond the strongest reduced-step baseline, no more than 2
absolute success points lost.

### 11.7 Experiment G — Oracle spatial refresh

Do not begin with mixed-resolution token deletion. First use the original
token grid and oracle masks.

Construct the action-relevance mask as the union of available regions:
robot/gripper, manipulated object, target/receptacle, contact
neighbourhood, unexpected-motion region, occlusion/disocclusion region.

Possible mask sources: simulator semantic segmentation, projected object
bounds, task metadata, contact geometry, optical-flow residual.

Start with: copy cached block outputs for inactive tokens, recompute
active tokens where technically valid, always preserve selected global
tokens, dilate mask boundaries, periodically force a global refresh.

Compare: random mask with same density, saliency mask, motion-only mask,
oracle task mask, oracle task + unexpected-change mask, full computation.

Measure: fraction of tokens recomputed, actual kernel time, memory copies,
action deviation, task success, mask density, contact-region recall,
boundary artefacts in latent diagnostics.

Continue only if the oracle mask yields at least approximately 15–20%
additional model-compute reduction. If the oracle fails, do not invest in
a learned foveation mask.

### 11.8 Experiment H — Keyframe and latent reuse

This experiment should directly test the original keyframe idea.

Compare:
- **H0** Full visual rollout latent as in baseline
- **H1** Decode visual rollout for diagnostics only
- **H2** Use only a selected endpoint/keyframe latent
- **H3** Reuse previous endpoint latent across policy calls
- **H4** Warp the cached endpoint using motion
- **H5** Residually refine the endpoint
- **H6** Refine only invalidated spatial regions
- **H7** Use internal caches without decoding RGB

First establish whether actions depend on visual imagination.

Offline test: hold current observation fixed, vary or remove visual
latent, reuse latent from earlier calls, measure action change.

Closed-loop test: reuse keyframe across multiple 32-step action chunks,
refresh on fixed age, refresh on oracle physical surprise, refresh on
learned event/flow gate.

Perturbation tests: successful grasp, failed grasp, object slip, target
movement, object occlusion, irrelevant background motion, camera motion,
subtask transition.

A good gate should refresh for the first 5 or 6 task-relevant cases and
avoid unnecessary refresh for irrelevant background changes.

Decoded images are permitted only for debugging: visualise stale-vs-
refreshed keyframe, visualise mask, inspect catastrophic latent drift.

Do not use FID, LPIPS, or VBench as primary metrics.

### 11.9 Experiment I — Lightweight adaptation

Only proceed when oracle or training-free experiments establish a real
upper bound.

Allowed trainable components: refresh gate, low-rank adapters,
residual-correction network, small action-conditioning adapter, spatial
mask predictor.

Freeze the main Cosmos3 backbone initially.

Generate training data from official full-compute rollouts.

Create counterfactual labels: reuse previous action chunk, reuse visual
latent, use fewer denoising steps, reuse selected blocks, partially
refresh selected tokens, full refresh.

A timestep is refresh-critical when reuse causes: material action
deviation, loss of task progress, contact failure, eventual episode
failure in paired rollout.

Suggested loss:

    action_distillation_loss
      + missed_refresh_penalty
      + compute_penalty
      + optional task-progress loss

Missed refresh must be penalized more than unnecessary refresh.

For LoRA: use the current Cosmos Framework training support, verify
target modules from the model configuration, start with small ranks, do
not launch full-scale multi-node training, use a tiny calibration run
first, retain a completely frozen-backbone baseline.

Randomize during adaptation: denoising-step count, cache age, cached
blocks, spatial mask, visual-latent quality, event/flow noise,
stale-keyframe duration.

The goal is robustness to partial computation, not image realism.

### 11.10 Experiment J — Transfer validation

Primary work must finish on Cosmos3 + RoboLab first.

Then validate the architecture-independent component on one second model.

Preferred order:
1. OpenPI π0.5 with RoboLab
2. Cosmos Policy on LIBERO or RoboCasa
3. π0.7 only if official implementation/checkpoint access exists

Do not claim π0.7 implementation if only the paper/blog is public.

Transfer only the general mechanism first: action-chunk reuse, event/flow
cache invalidation, adaptive denoising-step budget, fixed vs learned
refresh.

The model-specific spatial cache can remain Cosmos-specific.

A valid architecture-agnostic claim requires: same high-level decision
interface, comparable success-compute improvement, two distinct model
families.

Suggested common interface: `REUSE`, `ACTION_REFRESH`,
`PARTIAL_WORLD_REFRESH`, `FULL_REFRESH`.

## 12. Experiment registry

Create `experiments/registry.yaml` with at least:

    baseline_full
    baseline_decode_video
    baseline_steps_1
    baseline_steps_2
    baseline_steps_3
    baseline_steps_4
    baseline_action_only
    baseline_fixed_horizon_8
    baseline_fixed_horizon_16
    baseline_fixed_horizon_32
    oracle_temporal
    event_threshold
    flow_threshold
    event_flow_fused
    learned_gate
    cache_uniform
    cache_teacache_style
    cache_physical_aware
    spatial_random
    spatial_oracle
    keyframe_fixed_refresh
    keyframe_oracle_refresh
    keyframe_event_refresh
    combined_best

Every entry must define: parent baseline, hypothesis, code revision,
server arguments, client arguments, task set, seeds, episode count,
expected runtime, metrics, stop criterion.

Create a runner that: validates configs, refuses duplicate run IDs,
records commands, resumes interrupted experiments safely, detects server
failure, marks incomplete runs, never mixes simulator versions, never
mixes baseline revisions without explicit labels.

## 13. Required analysis and plots

Generate scripts, not manually edited charts.

Required outputs:

1. Success vs normalized total compute
2. Success vs measured latency
3. Success vs energy
4. Compute breakdown by subsystem
5. Latency breakdown by subsystem
6. Policy calls per episode
7. Denoising steps per episode
8. Visual tokens processed per episode
9. Cache-hit rate
10. Refresh precision / recall
11. Success by task competency
12. Success by difficulty
13. Savings by task phase
14. Action deviation vs cache age
15. Failure rate around contact events
16. Oracle vs learned gate
17. Fixed cadence vs adaptive gate
18. Action-only vs imagined-keyframe methods
19. Ablation waterfall for: temporal reuse, denoising reuse, spatial
    reuse, combined method.

Do not set a single arbitrary accuracy metric called "accuracy." Use: task
success, task progress, action agreement diagnostics.

Report absolute success-point differences, not only relative percentages.

## 14. Decision criteria

Use these as project gates, not assumed results.

- **Temporal reuse** — continue when: 20–25% total compute reduction,
  ≤ 2 abs success-pt loss, adaptive method beats fixed schedule at equal
  compute.
- **Cross-step caching** — continue when: ≥ 1.3× measured inference
  speedup beyond reduced-step baseline, ≤ 2 pt success loss, cache
  overhead is small.
- **Oracle spatial refresh** — continue when: ≥ 15–20% additional model
  compute reduction, actual latency improves, contact success remains
  stable.
- **Learned mask/gate** — continue when: recovers a substantial fraction
  of oracle benefit, flow/event overhead does not erase savings,
  refresh-critical recall is high.
- **Keyframe reuse** — continue when: keyframe-conditioned inference
  beats action-only inference at equal compute, stale-keyframe failures
  can be detected, reuse improves the Pareto frontier.
- **Combined method — strong result:** 30–50% end-to-end compute savings,
  ≤ 2 abs success-pt loss, measured latency improvement, gains across
  several task classes.
- **Combined method — major result:** > 50% savings, statistically
  non-inferior success, transfer to a second architecture.

**Stop or pivot the visual-imagination branch when:** visual imagination
is a small fraction of total cost, action-only inference matches it,
oracle spatial masks provide little benefit, decoding is the only visual
cost, cache memory traffic cancels compute savings, or gains disappear on
contact-rich tasks.

A negative visual result may still support a strong project on:

    action-aware denoising and policy-invocation scheduling

## 15. Required deliverables

At the end of each phase, update `docs/decision_log.md`.

Required milestones:

- **M0** — source checkout, environment audit, reproducibility manifest
- **M1** — official Cosmos3 + RoboLab smoke test, exact commands,
  baseline result, server/client logs
- **M2** — compute anatomy, profiler trace, latency/FLOP/energy schema
- **M3** — reduced-step and horizon baselines, first Pareto plot
- **M4** — oracle temporal refresh, flow/event feasibility, go/no-go result
- **M5** — generic and physical-aware denoising caching, measured speedup
- **M6** — oracle spatial reuse, go/no-go result
- **M7** — keyframe/latent reuse, action-only comparison
- **M8** — optional lightweight fine-tuning, combined method
- **M9** — second-architecture transfer

Final report at `results/reports/final_report.md` must contain: exact
research question, architecture diagram in text or Mermaid, source
revisions, machine configuration, baseline contract, compute anatomy, all
baselines, all failed ideas, Pareto frontier, statistical protocol,
ablations, failure analysis, novelty assessment, limitations, recommended
next step, and a direct answer to:

    "Does this make a significant dent?"

Also generate `results/reports/reproduction.md` with exact clean-machine
commands.

## 16. Engineering quality

Use: typed Python, dataclasses or validated config models, structured
logging, unit tests for gate logic, integration tests for client/server
protocol, deterministic offline tests, clear exception handling, no hidden
global state, no hardcoded personal paths, no silent fallbacks, no broad
catch-and-ignore blocks.

Provide one-command targets where practical:

    make audit
    make setup
    make test
    make smoke
    make baseline
    make profile
    make pilot
    make report

The baseline command must remain independently runnable after research
patches.

## 17. How to operate

Begin now.

Do not ask me questions that can be answered by inspecting the machine or
source.

Ask only when blocked by: license acceptance, authentication, unavailable
hardware, a decision to spend more than 4 GPU-hours, ambiguity that
materially changes the research result.

### First actions

1. Audit the machine.
2. Create the workspace.
3. Clone and pin the primary sources.
4. Read current setup documentation.
5. Write `baseline_contract.md`.
6. Report any legal/access blockers.
7. Install the official baseline.
8. Run the smallest possible smoke test.
9. Instrument one deterministic request.
10. Produce the initial compute-anatomy report.

Do not start LoRA training, full RoboLab-120 evaluation, RAFT installation,
v2e simulation, or spatial token modifications before the baseline and
compute anatomy are complete.

### End-of-first-session report

At the end of the first execution session, report exactly:

- what succeeded
- what failed
- current repository SHAs
- selected deployment topology
- exact baseline command
- baseline smoke result
- dominant measured compute component
- next three experiments
- estimated GPU-hours for those experiments
- blockers requiring my action
