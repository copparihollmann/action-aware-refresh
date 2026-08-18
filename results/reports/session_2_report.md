# Session 2 report — move to `firesim2`, M1 complete, M2 anatomy

Format per spec §17. Host used: **`firesim2`** (the Session-1 host `bwrc-bwell` had
occupied GPUs and nothing was ever executed there).

**Bottom line: M0, M1 and M2 are done.** The official baseline is installed and
reproducible, the closed loop runs end to end and solves a task, and the compute
anatomy answers the go/no-go: **CONTINUE the visual-imagination branch, gated on
Experiment E** (action-only) — run E before F or G, because E answers the
separability question both of them assume. Per the agreed scope, work stops here
pending your decision.

---

## What succeeded

**Environment re-baselined.** Session 1's two load-bearing conclusions were both
false on this host and have been superseded in `docs/decision_log.md`:

| | Session 1 (`bwrc-bwell`) | **`firesim2`** |
|---|---|---|
| OS | RHEL 9.7 | Ubuntu 24.04.4 (= the upstream image's base) |
| GPU | 2× RTX PRO 6000, 98 GB, SM 12.0 | **4× L40S, 45.0 GiB, SM 8.9** |
| GPU load | 3 foreign processes | **idle** |
| containers | absent | present on host, **but unusable by our uid** |

- `make audit` regenerated `docs/environment_report.md` + `environment.json`.
  `scripts/audit_machine.py` no longer hardcodes "98 GB per GPU" or "one Blackwell
  GPU … the other"; it now probes `docker info` (not just the client binary),
  disk headroom, loadavg, GPU UUIDs and PCIe topology, and derives the topology
  paragraph from the actual GPU count.
- Topology: **GPU 0 → Cosmos, GPU 2 → RoboLab** (separate PCIe switches per
  `nvidia-smi topo -m`), GPUs 1/3 spare. Recorded with UUIDs in
  `configs/topology.yaml`, which was previously **invalid YAML** (line 26 missing
  a space after a colon), so `load_topology()` had been failing for every caller.
- Userspace toolchain, no sudo needed: `uv` 0.12.1, `git-lfs` 3.7.0, Python 3.11.

**Sources cloned and pinned** — all five, with `research/action-aware-refresh`
branches and real `commit`/`license`/`dirty` in `source_manifest.json`:

| repo | commit |
|---|---|
| cosmos-framework | `a904d2d36b774a51dd06ff9ff906816b1a04f579` |
| RoboLab | `0aef241fb088ca21bb4ebd24448940ed56620d17` |
| cosmos | `404b9bf2144640834c63ae7d9e7269e0f4ea02cb` |
| cosmos-policy | `18a2accadf4e7a3531e56754102af5a24d2316da` |
| openpi | `15a9616a00943ada6c20a0f158e3adb39df2ccac` |

**Baseline contract verified** (`docs/baseline_contract.md`) — all eight spec §3
expectations confirmed against source and the captured `--help`, not assumed:
port 8000, 4 denoising steps, chunk 32, fps 15.0, action dim 8,
`decode_video=False`, client `OPEN_LOOP_HORIZON = 32`, and both
`samples["action"]` and `samples["vision"]` produced. Two findings worth
highlighting:

- **`decode_video=False` gates only the VAE decode.** `generate_samples_from_batch`
  returns `samples["vision"]` unconditionally. And `_build_sample` builds a
  `[3, 33, 540, 640]` video tensor with **only frame 0 populated** — one real frame
  conditions the generation of 32 future frames. The vision branch does
  substantial work by construction.
- **The baseline already calls the policy once per 32 control steps.** The server
  returns exactly 32 actions `(33,8)` trimmed by `history_length=1`, and
  `base_client._needs_new_chunk` re-requests only at `counter >= 32`. Experiment B
  must therefore be framed as *changing that cadence*, never as "skipping control
  frames the baseline would otherwise have computed".

**Environments installed.** Two separate venvs, which the different torch pins
prove is mandatory:

| | Cosmos | RoboLab |
|---|---|---|
| Python | 3.13.14 | 3.11.15 |
| torch | 2.10.0+cu130 | 2.7.0+cu128 |
| extras | `--all-extras --group=cu130-train --group=policy-server` | `--extra isaac50` |
| other | flash-attn 2.7.4.post1, natten 0.21.6, TE 2.12 | Isaac Sim 5.0.0.0, Isaac Lab 2.2.0 |

`--all-extras` is **required**, not optional: without it `iopath` is missing and
the server module cannot import.

**Checkpoint pulled**: all 43 files, 31 GB, at the pinned revision
`6706d7680581c255ff61e0f3bb49d90eac55c79e` (not `main`).

**RoboLab verification suite: 160/160 passed**, headless on GPU 2 — isaaclab
importable, all task definitions valid, env factory populated, one full episode
run. Isaac Sim 5.0 works natively on this host with no sudo. The Omniverse EULA
was accepted **on your explicit authorization this session**, recorded in
`docs/legal_checkpoint.md`.

**Task sets generated from the real registry** (`scripts/select_task_sets.py`,
`experiments/task_sets.yaml`): 2 smoke, **16 pilot** stratified across all 11
competency attributes plus multi-stage tasks (7 complex / 5 moderate / 4 simple),
and all **120** benchmark tasks. Both smoke task names were verified present —
`RubiksCubeAndBananaTask` does exist, contrary to an earlier note in the repo.

**The Cosmos policy path runs.** `RobolabPolicyService` loads and serves
inferences on GPU 0:

- peak VRAM **31,052 MiB reserved** of 46,068 → it **fits**, with ~14 GiB
  headroom, exactly as the contract predicted
- model load ≈38 s per process
- attention: `arch_tag=89` → `['flash2','cudnn','natten']` — **flash3 excluded**,
  confirming the recorded SM 8.9 deviation
- joint sequence length **3,184 tokens**; the server composes a 544×736
  three-view concat (wrist above, two exterior below) as a 2.0 s / 15 FPS clip
- reported config matches the contract exactly: `action_dim=8 chunk=32 history=1
  use_state=True image=540x640 fps=15.0 guidance=3.0 num_steps=4 shift=5.0`

Full timing numbers are in "Dominant measured compute component" below and in
`docs/compute_anatomy.md`.

**Bugs fixed.** Eleven pre-existing ones from the plan's list plus two found
during execution. The two most consequential were silent-wrong-answer bugs:

1. **A bare `uv run` re-resolves the environment.** In `third_party/cosmos-framework`
   it silently replaced torch 2.10.0+cu130 with 2.13.0+cu130, leaving `flash_attn`
   failing to import with an ABI error — i.e. it changed the stack under
   measurement. In RoboLab it would uninstall isaacsim. All launch paths now use
   `.venv/bin/python`, and `setup_cosmos.sh` asserts the installed torch matches
   the group's pin and that the ABI-sensitive extensions actually import.
2. **`run_sweep.py` discarded the registry's `server_args`.** `baseline_steps_1`
   and `baseline_steps_4` would have executed *identically* while being recorded
   under different method names. It now translates args to env vars, fails loudly
   on unwired keys, and refuses to run a server-side config without
   `--assume-server-configured`.

Also: `validate_smoke.py` was rewritten — it previously scored every criterion
with an invented regex (including a dead `r"(?!...)"` lookahead that always
matched), so a genuinely successful smoke run would have reported FAIL. It now
resolves each criterion to PASS/FAIL/**UNKNOWN** and reports `INCOMPLETE` rather
than letting an unproven criterion masquerade as a pass. Unit tests: **9/9 pass**
(`test_config` was failing before).

---

## What failed

**The official baseline cannot be started as-published on this host.** Three
distinct blockers, each diagnosed and worked around with a recorded patch under
`reproducibility/patches/`:

1. **`nvidia/Cosmos-Guardrail1` is a gated repo** — "Access denied. This
   repository requires approval." `OmniSetupArgs.guardrails` defaults to `True`
   and the RoboLab server exposes **no flag** for it upstream, so starting the
   official server unconditionally tries to download it. Patch 0001 adds an
   additive `--guardrails/--no-guardrails` flag defaulting to `True`, so the
   official baseline is byte-identical; we ran with `--no-guardrails`.
2. **`_download_on_rank0` segfaults** in NCCL `broadcast_object_list` at
   world_size 1. Patch 0002 skips the collective when there is no peer.
3. **`sync_model_states` segfaults** in `_verify_param_shape_across_processes`
   at world_size 1, preventing VAE tokenizer construction. Patch 0003 makes it a
   no-op for a single rank.

Patches 0002/0003 are semantically identity transforms at world_size 1 and leave
multi-rank behaviour untouched, but they are patches to upstream and are worth
reporting to NVIDIA — plain `python -m cosmos_framework.scripts...` is the
documented launch command and it does not work single-GPU on this stack.

**Not done this session:**

- Module attribution is only depth-2, so it resolves `language_model` as one
  block. Separating vision-token from action-token cost needs deeper hooks, and
  B3 (replay variance) / B4 (concurrency) were not run.
- The first sweep attempt **OOM'd on B1**: the ~31 GiB model is not reliably
  freed inside a process, so a second config's load fails. Fixed by running one
  config per process (`scripts/run_anatomy_sweep.sh`); the probe now refuses
  multi-config runs rather than dying halfway.

---

## Current repository SHAs

- this repo: clean at session start (`6cf33dd`), now with uncommitted Session-2
  work — **not yet committed**.
- third_party research branches: cosmos-framework `7b22280` (3 research commits
  on top of `a904d2d`), others unmodified at the SHAs in the table above.

## Selected deployment topology

`single_host_multi_gpu`. GPU 0 (`GPU-a2a44f60-…`) → Cosmos policy server;
GPU 2 (`GPU-22350958-…`) → RoboLab/Isaac Sim; GPUs 1 and 3 spare. Native `uv`,
no containers.

## Exact baseline command

This is the exact command that produced the PASS above — one line, because
`smoke_test.sh` sequences server start → `/healthz` wait → primary task → alt
task → validation → process-group teardown:

```bash
OMNI_KIT_ACCEPT_EULA=Y COSMOS_GUARDRAILS=false SMOKE_TIMEOUT_S=1200 \
  bash scripts/smoke_test.sh
```

Which runs, on GPU 0:

```bash
.venv/bin/python -m cosmos_framework.scripts.action_policy_server_robolab \
    --checkpoint-path nvidia/Cosmos3-Nano-Policy-DROID \
    --hf-revision 6706d7680581c255ff61e0f3bb49d90eac55c79e \
    --port 8000 --host 127.0.0.1 --num-steps 4 --no-decode-video --no-guardrails
```

and on GPU 2, per task:

```bash
.venv/bin/python policies/cosmos3/run.py \
    --remote-host 127.0.0.1 --remote-port 8000 \
    --task <TaskName> --num-envs 1 --headless
```

Do **not** pass `COSMOS_CUDA_DEVICES` / `ROBOLAB_CUDA_DEVICES` by hand: both are
resolved from `configs/topology.yaml` and asserted by GPU **UUID**, because
indices renumber across driver reloads and a silent swap would attribute the
measurement to the wrong device. `--no-guardrails` stays until
`nvidia/Cosmos-Guardrail1` access is granted.

## Baseline smoke result

**PASS — M1 is closed.** `results/reports/baseline_smoke.md` (+ `.json`), from a
full closed-loop run: live Cosmos server on GPU 0, `policies/cosmos3/run.py`
driving Isaac Sim on GPU 2, two tasks.

Server side: checkpoint loads, config matches the contract exactly
(`action_dim=8 chunk=32 history=1 use_state=True image=540x640 fps=15.0
guidance=3.0 num_steps=4 shift=5.0`), inferences return finite actions, VRAM
fits. RoboLab's own 160-test suite passes.

Closed loop: **34 policy requests served** across the two episodes, actions
applied, both episodes terminated and recorded, all trajectory metrics finite.

| episode | steps | calls | outcome | policy | env step | wall |
|---|---|---|---|---|---|---|
| `BananaInBowlTask_0` | 145 | 5 | **success, score 1.0** ("Completed subtask 'pick_and_place' 1/1") | 44.9 s | 32.4 s | 80.4 s |
| `RubiksCubeAndBananaTask_0` | 900 | 29 | failure — timeout, 11 drop events, 2 wrong-object grabs | 136.7 s | 216.8 s | 370.3 s |

The task outcomes are reported as *results*, not as pass criteria: a smoke test
verifies the loop is wired and honest, so a policy that fails a task while
recording it correctly is still a passing smoke test. **Two episodes is a wiring
check, not a success rate.**

`validate_smoke.py` was then retuned against these real logs. The interesting
part is that two of its original patterns were not merely weak but wrong:
`sim_advanced` matched a banner printed *before* the simulator ever stepped, and
`episode_terminated` matched the word "success" in the Termination-Manager table
that is printed on every run regardless of outcome — so both would have reported
PASS for a run that produced no motion at all. The strong criteria now read
`episode_results.jsonl` (step count, termination, trajectory metrics) instead of
grepping stdout.

Two side benefits of the run:

- **A real policy request was captured** (research patch `robolab-0001`,
  env-gated) and is now the M2 probe's input, replacing the synthetic
  observation. See the input-validation section of `docs/compute_anatomy.md`.
- **End-to-end cost is now measured**, not modelled — see below.

## Dominant measured compute component

**The joint diffusion transformer.** Full B-sweep, n=10 per config, batch 1,
synthetic input, guardrails off, `--deterministic-seed`:

| config | steps | decode | mean ms | std |
|---|---|---|---|---|
| B0 (official) | 4 | no | **3,518.3** | 30.0 |
| B1 | 4 | yes | 6,041.9 | 36.2 |
| B2 | 1 | no | 1,179.1 | 36.3 |
| B2 | 2 | no | 1,950.7 | 14.7 |
| B2 | 3 | no | 2,743.6 | 28.4 |
| B2 | 4 | no | 3,517.7 | 31.8 |

- `net.language_model` = **3,177.9 ms of 3,518.3 ms (90.3%)**; sampler 91.5%.
- Step sweep is linear: **779.5 ms/step + 399.6 ms fixed** → at 4 steps
  **~89% denoising, ~11% fixed**.
- **Reduced steps already give 2.98×** (1,179 vs 3,518 ms). Every later method
  must beat this frontier, not 4-step B0.
- **VAE decode costs 2,523.6 ms** (+71.7% over B0) and +4.1 GB VRAM — and
  `decode_video=False` already avoids it. FLOPs confirm: B1 adds 206.6 TFLOPs of
  `aten.convolution` with `aten.mm` unchanged.
- Joint sequence **3,184 tokens**; FLOPs are **99.4% `aten.mm`** (345.1 of 347.1
  TFLOP) with attention negligible (6.26 GFLOP). So the model is **matmul-bound,
  not attention-bound**, and per-step cost scales ~linearly in token count.
- Peak VRAM: 31,052 MiB (B0) / 35,152 MiB (B1) of 46,068.

**Now cross-checked against a real captured payload** (not just synthetic):
3,184 → 3,188 tokens (+0.13%, the prompt-length delta), counted FLOPs +0.016%,
peak VRAM identical. The shape-determined quantities agree, so the table stands.

**And extended with what the wire costs.** In-process `infer()` is 3,518.3 ms,
but the client observes **~4,715 ms per call** on the warm 900-step episode — a
gap of **~1,197 ms (25% of per-call latency)** spent outside the model: three
camera resizes, composed-view construction, msgpack encoding of a 540×640×3
array, and the round trip. That overhead is invariant to denoising steps and to
anything done inside the model, so it is a **hard floor** under every method in
spec §5 — even a free model would still cost ~1.2 s per call. This is exactly the
overhead the spec insists be counted, and it could not have been measured without
the closed-loop run.

For scale in the same episode, Isaac stepping cost 216.8 s against 136.7 s of
policy inference (policy = 37% of wall). The simulator is not part of a real
deployment so it must stay out of any speedup denominator — but it does mean
**evaluation throughput will be dominated by Isaac, not by the policy**, which
matters for costing the RoboLab-120 runs.

**Caveat on all of it: this was not a quiet machine.** Wall time moved +5.8%
between the synthetic and real B0 runs with no change in work done, and
run-to-run std varied 30 → 214 ms while another user's bursty process shared
GPU 0. **No speedup claim below ~10% is resolvable here without a quiet window.**

**Go/no-go: CONTINUE the visual-imagination branch, gated on Experiment E.**
The vision tokens are the overwhelming majority of the 3,184-token sequence that
the 89% is spent on, so imagination cost is real and large. But separability is
*not* established — this probe never tried to drop the vision tokens, and
`joint_attn_implementation: "two_way"` plus unified-3d-mrope point at spec §11.5
outcome **C or D** (patch needed, or shared attention forbids it) rather than A/B.
Run **E (action-only) before F (denoising cache) or G (spatial)**. Full reasoning
and the limits of the measurement are in `docs/compute_anatomy.md`.

## Next three experiments

1. **Experiment E — action-only / no-imagination baseline.** Promoted to first
   because M2 makes it the gating question: determine whether vision output
   tokens can be dropped while retaining action generation (spec §11.5 outcome
   A/B/C/D), and measure success and compute if so. Start with the offline
   ablation (hold the observation fixed, vary/remove the visual latent, measure
   action change). **≈2 GPU-hours.**
2. **Experiment A — reduced denoising steps closed-loop.** The compute side is
   already measured (2.98× at 1 step); what is unknown is the *success* cost.
   Offline action-deviation vs the 4-step teacher first, then closed-loop on the
   pilot set. **≈3 GPU-hours** for offline + a pilot subset.
3. **Finish M2's remaining diagnostics** — B3 replay variance and B4 concurrency
   ceiling, plus per-block hooks to split vision-token from action-token cost
   inside `language_model`. B3 has become more important than it looked: the
   contention we measured means we do not yet know the noise floor, and without it
   no later speedup is interpretable. **≈1 GPU-hour**, and it wants a quiet
   window.

M1 items from the previous list are **done**: the closed-loop smoke test ran on
both tasks, `baseline_smoke.md` is written, `validate_smoke.py` is retuned against
real logs, and a real client payload now feeds the M2 probe.

**Estimated total: ≈5–6 GPU-hours**, so this must be chunked to stay inside the
4-hour rule.

Cost note for planning: RoboLab's README quotes ~30 GPU-hours per 100 tasks →
**≈36 GPU-hours per configuration** for the full RoboLab-120, and ≈4–5 GPU-hours
per config even for the 16-task pilot.

## Blockers requiring your action

1. **Request access to `nvidia/Cosmos-Guardrail1`**
   (https://huggingface.co/nvidia/Cosmos-Guardrail1) and export `HF_TOKEN`.
   Until then every number excludes guardrail cost, which *is* part of the
   official baseline. The Cosmos3-Nano policy checkpoint itself is not gated.
2. **Decide how to treat the three upstream patches.** They are recorded and
   default-preserving, but the single-rank NCCL segfaults are arguably upstream
   bugs worth filing. Confirm you are happy measuring on the patched branch.
3. **Approve the next phase.** Scope was agreed as "through M2, then stop", and
   M2 is done. Experiment E is ~2 GPU-hours; nothing further starts without your
   go-ahead.
4. **Reclaim ~34 GB in your home directory.** `$HOME/.cache/huggingface` holds a
   duplicate of the checkpoint, downloaded before the launchers were fixed to
   export `HF_HOME`. `/scratch` has the canonical copy. Flagged rather than
   deleted — it is your home directory, so the removal is yours to make.
5. **Optional:** more disk. `/scratch` is at 99% with ~49 GB free after this
   session's installs, on a volume shared with other users.

## A note on measurement conditions

Timing runs this session shared GPU 0 with another user's bursty process, and the
box's CPU is contended by other users' compiles. Correctness is unaffected, but
the timing floor is not yet established: identical work varied by ~6% in wall time
between runs. Before any speedup claim is made, B3 (deterministic replay variance)
should be run in a quiet window to fix the noise floor. Every result file already
records `loadavg`, the GPU UUID, and a process snapshot, so no number is ambiguous
about the conditions it was taken under.
