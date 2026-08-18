# Decision log

Append-only. Newest at bottom. Every non-obvious choice made during the
project — with the reasoning — lives here.

---

## 2026-08-03 — Session 1 kickoff

**Deployment topology: single-host multi-GPU, native `uv`, no containers.**

- Reason: host has no `nvidia-container-toolkit`, no CDI, no docker, no
  sudo. Rootless podman 5.6 alone cannot expose GPUs to containers on this
  system. Trying to work around that would eat all the session time.
- Alternative rejected: request sysadmin install `nvidia-container-toolkit`.
  Kept as a future ask if the native uv path proves painful for Cosmos or
  RoboLab (e.g. Isaac Sim system deps).
- Impact: the spec's preferred `docker build` path for Cosmos Framework is
  replaced with a native `uv sync` inside `third_party/cosmos-framework/`.
  Same dependency group (`cu130-train` + `policy-server`) applies.

**GPU assignment: GPU 0 → Cosmos server, GPU 1 → RoboLab client.**

- Both are 98 GB Blackwell — plenty of headroom for either workload alone.
- The machine is currently shared with user `ken_ho` (small memory, ~90–95%
  util spikes). Correctness runs are fine. Timing measurements will wait
  for a quiet window and record `nvidia-smi` snapshots at run start.

**Python: 3.11 via uv.**

- Cosmos + RoboLab both officially target 3.11. Available via uv (system
  Python is 3.9 which is too old).

**Energy: NVML power-integration, not total-energy counter.**

- `nvidia-smi --query-gpu=total_energy_consumption` is unsupported on this
  GPU/driver. `energy.py` samples power at 10 Hz and trapezoid-integrates.
  Every energy field will be labeled ESTIMATED and record the sampling
  cadence in the run metadata.

---

## 2026-08-03 — Session 2: moved to `firesim2`

The project moved hosts because `bwrc-bwell`'s GPUs were occupied and nothing
had been executed there. **Session-1 decisions 1 and 2 are superseded.** Their
factual premises no longer hold; re-derived below from a fresh `make audit`.

**Superseded — "no containers because the toolkit is absent".**

- The new host *does* have `docker` 29.5.2 and `nvidia-container-toolkit`
  1.19.1. But `docker info` fails for our uid: we are not in the `docker`
  group (members: six other accounts; names withheld — shared host)
  and rootless prerequisites are absent (`newuidmap`/`newgidmap` missing, no
  `/etc/subuid` entry). No podman/apptainer/singularity either.
- **Same conclusion, different reason**: native `uv`. The user was offered the
  option of requesting `docker` group access and declined it as unnecessary.
- It *is* unnecessary: the upstream Dockerfile's only additions over this host
  are `curl/ffmpeg/git/git-lfs/tree/wget` — all present or installable in
  user space — and its base image is `ubuntu24.04`, which this host already
  runs. Nothing in either install needs sudo.
- Lesson recorded in `scripts/audit_machine.py`: `docker --version` proves only
  that a *client binary* exists. The audit now probes `docker info` separately
  (`docker_usable`), because the earlier report concluded "docker MISSING →
  using rootless podman" when podman was not even installed.

**Superseded — "GPU 0 → Cosmos, GPU 1 → RoboLab, both 98 GB".**

- Now **4× L40S at 45.0 GiB / SM 8.9**, not 2× Blackwell at 98 GB / SM 12.0.
- New assignment: **GPU 0 → Cosmos, GPU 2 → RoboLab/Isaac**, GPUs 1 and 3
  spare. Reason: `nvidia-smi topo -m` shows PCIe-switch pairs (0,1) and (2,3),
  so 0 and 2 sit behind *different* switches and their host traffic does not
  share a link. Assignment is recorded with UUIDs in `configs/topology.yaml`
  so a run can assert it got the GPU it expected.
- **VRAM is now the binding model constraint**: 32.9 GB of bf16 weights against
  45.0 GiB leaves ~14 GiB. Batched multi-env requests may not fit. If they
  OOM: reduce concurrency, or shard across GPUs 0+1 and record that as *the*
  baseline topology. Not quantization — spec §5 makes that a separate
  experiment, not a way to make the baseline fit.

**New — attention backend differs on SM 8.9, and that is a recorded deviation.**

- `get_backend_list()` in `cosmos_framework/model/attention/backends.py` maps
  `arch_tag >= 80` → `["flash2", "cudnn", "natten"]`. `flash3` is gated to
  `arch_tag == 90` (Hopper). So we run flash2/cudnn where NVIDIA's published
  numbers use FA3.
- Impact: **absolute** latencies are not comparable to upstream's figures. All
  our claims are within-machine deltas against our own measured baseline, so
  this is sound — but the selected backend must be logged in every result and
  in `docs/baseline_contract.md`.

**New — the measurement hazard is CPU and disk, not GPU.**

- GPUs were idle on arrival (0% util, no compute apps). GPU 1 briefly showed
  631 MiB during the audit and returned to idle, so other users do touch these
  devices: keep taking a contention snapshot per run.
- CPU is heavily contended (1-min loadavg ~46 on 64 cores, another user's
  parallel compile). The GPUs' CPU affinity is cores `0-15,32-47`. This
  perturbs preprocessing, serialization, websocket round-trip, and Isaac
  physics — every host-side stage. `loadavg` + a process snapshot are now
  recorded by the audit and must accompany every timed run.
- `/scratch` was 96% full (≈134–142 GB free) against a ~100 GB install budget,
  on a volume shared with other users. Both setup scripts now refuse to start
  below a headroom threshold and `uv cache prune` after syncing, and the audit
  reports free space as a blocker. The user chose to proceed on `/scratch` with
  monitoring rather than wait for `/nscratch` space.
- GPU 0 idles at ~67 W (P0) versus ~34 W (P8) for GPUs 1–3, so the idle floor
  is not uniform across devices. Report idle-adjusted energy alongside gross,
  and measure the floor on the *same* GPU as the workload.

**New — NVML index is not the CUDA-visible index.**

- `EnergyMeter(device_index=0)` was an NVML *physical* index while the server
  is pinned with `CUDA_VISIBLE_DEVICES`. Pinning to a non-zero physical GPU
  would have silently reported a different GPU's power — wrong numbers, no
  error. `energy.py` now takes a `uuid`, verifies the index against it, and
  corrects or raises. `nvml_index_for_uuid()` is the supported route.

**Unchanged from Session 1.**

- Python 3.11 via `uv` (host has only 3.12; RoboLab requires >=3.11).
- Energy by 10 Hz power integration, always labelled ESTIMATED —
  `total_energy_consumption` is unsupported here too, re-probed and confirmed.

**Access: one blocker dissolved, one authorized.**

- `nvidia/Cosmos3-Nano-Policy-DROID` reports `gated: False` (HF API, revision
  `6706d7680581c255ff61e0f3bb49d90eac55c79e`). There is no click-through to
  accept and **no `HF_TOKEN` is needed** to download it. License is `other`
  (NVIDIA Open Model License) — recorded, nothing to accept.
- Isaac Sim / Omniverse EULA: **the user explicitly authorized acceptance in
  this session.** See `docs/legal_checkpoint.md`. Note that RoboLab's own
  install-verification suite (`uv run pytest tests/`) auto-accepts it, which
  is precisely why authorization had to be obtained before running it.

**Scope for this session: through M2, then stop.**

- Install + smoke + full compute anatomy (B0–B4), ≈2–3 GPU-hours, inside the
  4-GPU-hour rule. No method work until the go/no-go is answered.
- Noted for M3 planning: RoboLab's README quotes **30 GPU-hours per 100
  tasks**, so full RoboLab-120 is ≈36 GPU-hours *per configuration* and even a
  12–18 task PILOT is ≈4–5 GPU-hours per method config. Requires explicit
  approval and chunking.

---

## 2026-08-04 — closing M1, and what the closed loop taught us

**`openpi-client` is installed explicitly, from `setup_robolab.sh`, pinned to the
server's version.**

`policies/cosmos3/client.py` imports `openpi_client`, but it is *not* a declared
RoboLab dependency: upstream deliberately keeps it out so non-openpi backends do
not pull it in, and vendors `image_tools` into
`robolab/core/utils/image_utils.py` for the same reason. The consequence is nasty
— `uv sync` succeeds, RoboLab's 160-test suite passes, and the closed-loop client
then dies with `ModuleNotFoundError` *after* Isaac Sim has finished booting, which
is where this session's first two smoke attempts ended.

Upstream's README says to clone the whole openpi repo and `uv pip install -e
../openpi/packages/openpi-client`. We install the published wheel instead, pinned
to **0.1.2** — the version cosmos-framework's `policy-server` group resolved —
because the two ends must agree on the msgpack wire format. `uv pip install`, not
`uv add`: upstream's `pyproject.toml` and `uv.lock` stay clean. Verified with
`--dry-run` first: it adds only `openpi-client` and `dm-tree`, with no numpy or
pillow churn and `isaacsim` untouched.

**One research patch to RoboLab: capture a real request.**

`robolab-0001` dumps the first genuine request built by
`Cosmos3Client._pack_request` during a live episode, gated on
`ACTION_REFRESH_CAPTURE_REQUEST` and off by default. The M2 probe drives the model
in-process and so must supply its own observation; hand-authoring one risks
profiling a payload the deployment never sends, and since the model is
matmul-bound and roughly linear in token count, a prompt of a different length is
a different measurement. Writes once per process and swallows its own exceptions —
a debug artifact must never be able to fail an eval run. See
`docs/upstream_patches.md`.

**The synthetic observation was already sound — now demonstrated, not asserted.**

Re-measuring B0 with the captured payload: 3,184 → **3,188** tokens (+0.13%,
entirely the prompt-length delta), counted FLOPs +0.016%, peak VRAM identical.
The server's own default image size is 540×640, which is exactly what the client
sends, so the geometry never differed. The probe now defaults to the real capture
anyway, because the cheap guarantee is worth more than the argument.

**We do not yet have a usable measurement floor.**

Wall time moved **+5.8%** between those two runs with no change in work done, and
run-to-run std went 30 → 214 ms. Both runs shared GPU 0 with another user's bursty
process. So: **no speedup claim below roughly 10% is resolvable on this host
without a quiet window**, and B3 (deterministic replay variance) is promoted from
"nice to have" to a prerequisite for interpreting any later result.

**End-to-end overhead is a hard floor, and it is large.**

From the closed-loop run: in-process `infer()` is 3,518 ms, but the client
observes ~4,715 ms per call warm — **~1,197 ms (25%)** spent outside the model on
camera resizes, composed-view construction, msgpack encoding of a 540×640×3 array,
and the round trip. It is invariant to denoising steps and to anything done inside
the model. Spec §8 demands this be counted; it could not have been measured
without closing M1.

**Two script-hygiene fixes worth recording, because both were silent-wrong-answer
class.**

- `run_anatomy_sweep.sh` hardcoded a personal `HF_HOME` and set no
  `CUDA_VISIBLE_DEVICES`, so it profiled whatever the driver called device 0 with
  no UUID assertion. Both now come from `configs/machine.yaml` /
  `configs/topology.yaml` via `scripts/lib/common.sh`, asserted.
- The same script re-rendered `docs/compute_anatomy.md` unconditionally, so an
  exploratory one-config run into a scratch directory silently overwrote the M2
  deliverable with a one-config stub. (It did, once. The doc is fully derived from
  the summary JSONs, so it was regenerated intact.) Rendering is now gated on the
  canonical output directory, and the hand-written go/no-go lives in
  `docs/compute_anatomy_conclusion.md` so regeneration cannot destroy it.

**`validate_smoke.py`: two of its patterns were actively wrong, not just weak.**

`sim_advanced` matched a banner printed *before* the simulator stepped, and
`episode_terminated` matched the word "success" in the Termination-Manager table
that prints on every run regardless of outcome — so both would have reported PASS
for a run that produced no motion at all. The strong criteria now read
`episode_results.jsonl` (step count, termination, trajectory metrics) rather than
grepping stdout. Also recorded there: **task success is a result, not a pass
criterion.** A policy that fails a task while reporting it correctly is a passing
smoke test; conflating the two would make the smoke test fail whenever the policy
merely performed badly.

---

## 2026-08-04 — Phase 0 of M3/M4: what the measurements actually say

**The project's premise is confirmed exactly, not approximately.** The token census
now reads the model's own `PackedSequence` instead of inferring from `position_ids`:

| component | tokens | share |
|---|---|---|
| text (prompt) | 95 | 3.0% |
| vision — conditioning (the real observation) | 340 | 10.7% |
| vision — **imagined future** | **2,720** | **85.3%** |
| action — conditioning (current state) | 1 | 0.03% |
| action — **predicted** (the deliverable) | 32 | 1.0% |
| total | 3,188 | 100% |

Reconciles to zero unattributed tokens. Nine latent frames of 17x20=340 tokens, from
33 pixel frames at temporal downsample 4, of which **exactly one** is the observation.
So the model spends 85% of its sequence imagining video in order to emit 32 action
vectors, in a model that is matmul-bound and near-linear in token count. That is the
lever, and it is why Experiment E gates everything.

Two structural facts fell out of the same read. `split_lens=[95, 3093]` with
`attn_modes=['causal','full']`: text is causal, and vision+action share **one** full
attention split — in `two_way_attention` the generation queries attend to all tokens
with no mask at all. So there is no configuration switch that removes the imagination
(spec §11.5 outcome **C/D**, as predicted). But `num_action_tokens_per_supertoken=0`,
so actions are a **contiguous block** rather than interleaved into per-frame vision
supertokens — which makes the E2b deletion surgery tractable rather than hopeless.

**I was wrong about the measurement floor, and the correction matters.**

The first B3 run (40 back-to-back replays of identical work) gave mean 3,775 ms, std
749 ms — 19.8% — and I initially read that as "no speedup below ~10% is resolvable
here". That conclusion was an artefact of using the mean on a contaminated sample. The
eight slowest requests were the last eight *in time order*, ramping monotonically to
+76%.

My first hypothesis was thermal throttling. **It was wrong, and the data says so
plainly:** `clocks.sm` sits at 1065 MHz whether the card is idle at 71 W or loaded at
260 W, with throttle reasons `0x0` throughout, against a 2520 MHz maximum. The SM
clock is *locked*, so it cannot fall under load. Re-running with an 8-second idle gap
between requests:

- median **3,430.9 ms**, MAD **11.5 ms = 0.34%**, drift **+0.06%**, spread 2.5%
- the two runs' medians agree to 1.0%

So the model's cost is stable and reproducible; the sustained run's tail is an
intermittent **host-side stall** (this box's CPU is shared and the GPU's affinity is a
subset of cores), not a change in the work done. **Minimum detectable effect is ~1%,
not ~10%** — small effects *are* measurable here, provided the comparison is
median-of-many with an idle gap and interleaved configs. Report medians: the sustained
mean describes the stalls, not the model.

Consequences now baked into the tooling: every timed iteration records `cuda_ms`
alongside `wall_ms` (so host-side interference can be localised rather than guessed
at), plus clocks/temperature/throttle flags; `robust_stats` reports median/MAD and a
drift indicator next to mean/std; and the sustained-load tail is reported separately
because that is what a deployed server actually experiences.

**A closed-loop reproducibility limit worth stating before it bites.** The server's
seed *sequence* is reproducible (`default_rng(cfg.seed)`), but closed-loop outcomes are
not: the same task at the same settings ended at 145 steps once and ran to the
750-step timeout another time. The sampler and simulator are not bitwise deterministic.
Pairing can therefore fix initial conditions and the seed sequence, but not the
trajectory — so single-episode comparisons decide nothing, and the Pareto table says so
instead of quoting an interval from n=1.

**Infrastructure built for this, and why.** An 18-GPU-hour chain on a shared box will
be interrupted, so `src/action_refresh/ledger.py` makes every unit of work
content-addressed and crash-safe: `result.json` is written to a temp file and atomically
renamed, so a unit is either absent or complete — never torn and mistaken for a result.
Failures are recorded distinctly from "never attempted" and are not retried by default
(silently re-running a deterministic failure burns GPU-hours in a loop); a killed unit
leaves a `started.json` and reads as interrupted, not as success. Falling out of a claim
without setting a result is an error, so nothing can be marked done by accident. Twelve
tests cover exactly these cases, including a simulated mid-sweep kill.

**Two bugs found and fixed on the way.** `run_anatomy_sweep.sh` re-rendered
`docs/compute_anatomy.md` unconditionally, so an exploratory one-config run overwrote
the M2 deliverable with a stub (recoverable — the doc is derived from the summary JSONs
— and now gated on the canonical output directory, with the hand-written conclusion moved
to `docs/compute_anatomy_conclusion.md` so regeneration cannot destroy it). And
`experiments/registry.yaml` had `keyframe_oracle_refresh:{ parent: ... }` — a missing
space made the key `"keyframe_oracle_refresh:{ parent"`, so the entry existed under a
name no runner would match and `keyframe_oracle_refresh` simply did not exist. Nothing
failed; it would just have been unrunnable at M7. `tests/unit/test_registry.py` now
checks key wellformedness, parent resolution, spec §12 coverage, and that
`baseline_steps_N` actually declares N steps.

**Also corrected an analysis error of my own.** The offline study first compared a
*pairwise* deviation (steps_1 vs teacher) against a *spread-about-the-mean* (the E0 seed
dispersion). For two samples the latter is half the former, so every ratio was inflated
~2x and `steps_1` was reported as 1.6x sampling noise when the like-for-like comparison
puts it at 0.80x — i.e. *within* it. The reference is now explicitly the same quantity as
the table it is compared against.

---

## 2026-08-04 — Experiment C's compute premise is falsified, offline, for free

Before building a client-side gate patch and buying ~3 GPU-h of closed-loop oracle runs,
two of the three quantities spec §11.3's gate depends on turned out to be computable from
episodes already recorded. Over 6 episodes (`docs/oracle_temporal.md`):

| | |
|---|---|
| contact-critical steps (median) | **8.9%** of an episode |
| baseline refresh rate | **3.2%** (one call per 32 control steps) |
| threshold oracle's calls vs baseline | **5.25x** (median) |
| episodes where a matched budget covers every critical moment | **1 of 6** |

**An oracle cannot save compute by refreshing only when necessary.** Contact-critical
moments are nearly 3x *more* frequent than the baseline's fixed cadence, so covering
them all costs 5.25x more calls, not fewer. This is structural: the baseline already
reuses each action chunk for 32 control steps, and no gate cleverness changes that. So
§11.3's "≥20–25% total compute reduction from temporal scheduling" is **not reachable by
refreshing less often**, and the negative result is recorded here as §11.3 instructs.

Worse for the remaining hope: at a *matched* call budget the oracle can be fresh at
every critical moment in only 1 of 6 episodes. So even the "same compute, better
placement" framing is budget-constrained in 5 of 6 cases — a negative closed-loop result
there would say more about the budget than about oracle scheduling.

**Consequences.** Experiment C's closed-loop run and Experiment D (learned event/flow
gates) are **deprioritized**: both were predicated on temporal scheduling having compute
headroom, and it does not. That frees roughly 3 GPU-h plus all of D's flow/RAFT/v2e work,
and it redirects the project to where M2 said the cost actually is — *inside* the
transformer, over tokens, not in call scheduling. The live directions remain **E**
(action-only: 85.3% of the sequence is imagined future) and **A** (reduced steps: already
2.98x, with offline deviation inside the model's own sampling noise).

The horizon sweep (Experiment B, horizons 8/16) stays in the chain, but its question has
changed with it: it is no longer "can we refresh less?" but "**is the baseline's horizon
of 32 already too long — does refreshing more often buy success?**" That is a real
Pareto-frontier question at *higher* compute, and worth the measurement.

**Caveats kept attached to the result.** The contact signal is a *proxy*: there is no
per-step contact-force array in the recordings, only a step-indexed event log
(`OBJECT_GRABBED_SUCCESS`, `TARGET_OBJECT_DROPPED`, `GRIPPER_HIT_OBJECT`, …), used as
such and labelled as such rather than fabricating contact from object velocities. And
these are oracles, not gates: they read privileged state from completed episodes and are
charged zero feature-extraction overhead, so they bound any learned gate **from above** —
a learned gate would have to pay for flow/event computation out of a saving that is
already absent.

One implementation note worth recording because it would have biased results in our
favour. The gate interface originally made the *caller* track `cache_age` and reset it on
refresh. But a refresh consumes an action immediately, so the age afterwards is 1, not 0
— getting that wrong yields a refresh period of `horizon + 1`, i.e. ~3% fewer calls than
the method actually makes, silently flattering it. The gate now owns the counter
(`Gate.decide` updates it in one place), and a test asserts the refresh period is exactly
`horizon` and that the fixed gate reproduces the measured 5 + 29 = 34 calls of the smoke
run.

---

## 2026-08-04 — closed-loop cost structure, and a budget correction

The baseline's closed-loop pass (15 of 16 pilot tasks, 1 episode each) cost **5.4 GPU-h**
and succeeded on **5/15 (33%)**. The cost breakdown is the important part:

| component | time | share |
|---|---|---|
| simulator stepping | 4.21 h | **78%** |
| policy inference | 0.91 h | **16.8%** |
| video write | ~0 (disabled) | — |

**Evaluation cost is almost independent of policy cost.** A 3× cheaper policy saves ~11%
of closed-loop wall time, not 3×. So closed-loop episodes cannot double as a compute
comparison — they measure *success*, and compute must be measured in-process. This
justifies the split the project already had (offline screens carry compute, closed-loop
carries success) but it is worth stating as a measured fact rather than an assumption.

**Failing is expensive, which makes budgets fail in the unsafe direction.** A failed task
runs to its full timeout (10–71 min here); a solved one terminates early (2–6.5 min). The
10 failures cost 4.7 of the 5.4 GPU-h. So a *worse* method costs *more* to evaluate, and
any estimate built from mean episode length underestimates a method that regresses. My own
per-method estimate was **3× low** for exactly this reason: I budgeted from `episode_s` and
a real-time factor measured on a cheap task (6.2×), when the expensive multi-object scenes
run at ~16×.

**Consequence, decided with the user.** Four methods over the full pilot would have cost
~32 GPU-h against ~18 authorised. Rather than cut methods arbitrarily, the remaining three
run on a **screened 9-task subset derived from the measured baseline** — which is what spec
§10 asks for anyway ("tasks where baseline success is neither always zero nor always one"),
and which could only be identified *after* a baseline pass:

- the **5 tasks the baseline solved** — the only tasks that can reveal *degradation*, which
  is the main risk for a cheaper policy. A task the baseline already fails at 0/1 cannot
  show degradation at all.
- the **4 cheapest of its failures** — retaining the ability to detect *improvement*
  without paying for the six most expensive saturated tasks.

1.40 GPU-h per method instead of 7.3, so all three remaining methods fit. The excluded
tasks and the one unmeasured task (`CleanUpToysTask`) are listed explicitly in
`experiments/task_sets.yaml` with the reasoning, so no reader has to guess why the
denominator changed.

The honest limitation travels with the set: comparisons on it are macro-averaged over 9
tasks, not 16, and it is biased toward what the baseline finds easy or cheap. It is a
screening set for deciding what deserves a full-pilot run, not a substitute for one.

**One thing this validated.** `baseline_steps_1`'s server came up with `num_steps=1`,
confirmed in its own log. That is the plumbing whose absence previously made
`baseline_steps_1` and `baseline_steps_4` execute *identically* while being recorded under
different names, and it is now checked at the point of use rather than trusted.

Also worth recording: re-running the `baseline_full` step against the screened set took
**seconds** — the ledger found all 9 tasks already measured and did nothing, saving 1.4
GPU-h of recomputation. Content-addressed work units meant changing the task set did not
invalidate the results already in hand.

---

## 2026-08-04 — the offline screen did not predict closed-loop success

This is the session's most important result, and it is negative about a tool I built.

Closed-loop outcomes on the 9 screened tasks, 1 episode each:

| method | success | offline deviation vs sampling noise | offline verdict I gave |
|---|---|---|---|
| `baseline_full` | **5/9** | — | reference |
| `baseline_steps_1` | **2/9** | 0.97x | "within sampling noise" |
| `baseline_vision_frames_9` | **0/9** | 1.14x | "1.14x noise, gripper flips inside baseline range" |

`steps_1` deviated *no more from the teacher than re-running the baseline with a different
seed does*, and still lost 3 of the 5 tasks the baseline solved. `vision_frames_9` lost all
five. **Open-loop action deviation on a fixed observation did not predict closed-loop
success.** It cannot: it ignores compounding, and each policy call re-anchors on an
observation that the previous slightly-wrong actions produced. Over 29–57 calls that
diverges.

The contact-event counts show the two failures are not the same kind:

| method | contact events (screened tasks) |
|---|---|
| `baseline_full` | 96 — 34 wrong-object grabs, 24 bumps, 17 drops |
| `baseline_steps_1` | 150 — 44 drops, 23 gripper-fully-closed |
| `baseline_vision_frames_9` | **1** |

`steps_1` manipulates *more* clumsily (44 drops vs 17) — degraded but engaged.
`vision_frames_9` barely touches anything at all. That is not mis-manipulation; it is
failure to engage.

**A reference-free metric explains it, and was missing from my screen.** Deviation from a
teacher cannot see a chunk that meanders near its start point — such a chunk sits at a
perfectly ordinary L2 distance. Net travel can:

| condition | per-step rate | joint range | net displacement | **straightness** | closed-loop |
|---|---|---|---|---|---|
| `teacher_steps4` | 0.042 | 0.655 | 0.536 | **0.385** | 5/9 |
| `steps_3` | 0.044 | 0.632 | 0.547 | 0.378 | — |
| `steps_1` | 0.062 | 0.628 | 0.536 | 0.299 | 2/9 |
| seeds | 0.035–0.039 | 0.424–0.582 | 0.365–0.462 | 0.307–0.347 | (= baseline) |
| `vision_frames_17/9/5` | 0.049–0.060 | 0.365–0.495 | 0.170–0.225 | **0.102–0.130** | 0/9 |
| `no_imagination_freeze` | 0.408 | 1.278 | 0.336 | **0.026** | not run |

**Straightness** (net displacement / path length) separates cleanly — every baseline-like
condition ≥ 0.299, every shortened-horizon condition ≤ 0.130, a 2.3× gap with no overlap —
and it ranks the conditions in the same order as the closed-loop outcomes. `joint_range`
does *not* separate them (`seed_2` sweeps less joint space than `vision_frames_17` and
still succeeds), so the discriminating quantity is whether the chunk **gets somewhere**,
not how much it moves.

Mechanistically this is what a shortened imagined horizon should do: **the imagination *is*
the plan.** A model that can see only 3 latent frames ahead commits to less displacement,
so the arm meanders and over ~29 calls never reaches the object. Which sharpens the
Experiment E answer considerably. Offline, shortening the horizon looked nearly free; in
closed loop it is fatal. The imagination is not decoration whose resolution can be traded
away — it is the lookahead the action chunk is derived from.

**What this changes.**

- `vision_frames_5` is *not* worth running: it is strictly more aggressive than the config
  that already scored 0/9, so the outcome is not in doubt and 1.4 GPU-h is better spent
  elsewhere. Recorded rather than measured.
- `steps_2` **is** worth running and is now in flight: `steps_1` degraded but survived on
  2 tasks, and `steps_2`/`steps_3` sit at teacher-like straightness (0.325 / 0.378), so a
  milder step reduction is the best remaining candidate for a real Pareto point (1.8× and
  1.28× cheaper respectively).
- `straightness` is added to `src/action_refresh/deviation.py` with tests, but is labelled
  a **candidate** predictor on 3 methods — not a validated screen. It was found *after* the
  closed-loop result contradicted the deviation screen, which is exactly the order in which
  one should distrust it.

**The durable methodological lesson.** I used an offline screen to decide what deserved
GPU-hours, and stated at the time that it "only decides what is worth evaluating, not
whether it works". That caveat turned out to be load-bearing: the screen's ranking was
wrong in the one direction that matters. Any future offline screen must be validated
against closed-loop success on at least one known-good and one known-bad configuration
before it is trusted to gate spending.

---

## 2026-08-04 — Experiment F measured: nothing to cache across denoising steps

**First measurement of our own mechanism rather than a mandated baseline.** Per §11.6,
`d[s,l] = ||R[s,l] - R[s-1,l]|| / ||R[s-1,l]||`, computed per transformer block and split
by modality using the model's *actual* packed-sequence indices (36 blocks, 4 steps, one
real captured request). `docs/denoising_residuals.md`.

| modality | median d | below 5% | below 10% |
|---|---|---|---|
| vision | **0.315** | 0% | 0% |
| action | **0.179** | 0% | 0.9% |
| text | **0.000** | 100% | 100% |

**The asymmetry we predicted is real: action residuals are 0.57× vision residuals.** The
action stream genuinely is the more stable of the two, which is the direction the
action-aware framing assumed — now measured.

**But nothing is cacheable, so the asymmetry is not exploitable.** Reuse needs
near-identical block outputs; 18–32% change is a different tensor. No block helps either —
vision spans 0.22 (early blocks) to 0.36 (late), action 0.139–0.216, so there is no cheap
subset to skip.

The mechanism is clean: TeaCache / DeepCache / token-wise caching all target **20–50-step**
schedules where consecutive steps barely differ. Cosmos3-Nano runs **4**. The short schedule
has already squeezed out the inter-step redundancy those methods harvest. F3's separate
action/vision thresholds are therefore correct in principle and moot in practice here.

**One exact, free saving does exist.** Text-token residuals are **identically zero** across
every block and step — the understanding tower's output does not depend on the noised
latents, so it can be computed once per request and reused for all steps with *bit-identical*
results. Ceiling is small (95 of 3,188 tokens, ~3%) and it must first be checked whether the
implementation already avoids recomputing it.

**Two bugs found in my own analysis before reporting it**, both of which would have produced
an over-favourable result:

1. **Block outputs are not tensors.** They are Cosmos `SequencePack` dicts carrying
   `causal_seq` [95, 4096] (text) and `full_only_seq` [3093, 4096] (generation), plus
   `_full_indices` mapping generation rows to global positions. My `isinstance(h, Tensor)`
   guard silently rejected every capture, and the first two runs recorded *nothing* while
   reporting only "no residuals captured". Fixed by reading the structure and inverting
   `_full_indices` to slice vision from action — with an assertion, because a wrong mapping
   would compare vision against action and the modality split is the entire point.
2. **Consecutive calls alternate CFG branches.** Guidance is 3.0, so the net runs twice per
   step, and the *unconditional* branch skips text tokens but keeps the same generation
   length — so it passed my length filter. Comparing each call to the previous one measured
   **cond-vs-uncond**, a guidance difference, and presented it as a step-to-step residual.
   The symptom was an alternating low/high pattern across steps that no denoising schedule
   would produce. With each branch now compared against itself, action residuals went
   0.062 → **0.179** and the "42% of action samples below 5%" reading collapsed to **0%**.
   The buggy version would have justified building a cache that could not work.

I also corrected the pre-registered interpretation logic, which fired on the action/vision
*ratio* while ignoring absolute magnitude — a favourable ratio between two large residuals
buys nothing. It now requires absolute cacheability before claiming exploitability.

**Where this leaves the project.** Four levers have now been measured and none delivers
compute at preserved success:

| lever | verdict |
|---|---|
| A — reduced denoising steps | 2/9 closed-loop vs baseline 5/9 |
| C/D — temporal refresh scheduling | no compute headroom (critical steps 2.8× more frequent than baseline cadence) |
| E2b — shorten the imagined horizon | **0/9**; the imagination *is* the plan |
| F — cross-denoising-step caching | nothing to reuse at 4 steps |

The consistent picture is that **Cosmos3-Nano-Policy-DROID is already tightly provisioned**:
4 denoising steps, a 32-step action chunk and a 9-frame imagined horizon are each near the
minimum that works, and the model is matmul-bound and linear in token count. The remaining
untested idea is **G (spatial masking)** — cutting tokens *within* a frame rather than
removing frames or steps. It is the last lever that could reduce token count while leaving
the plan's temporal structure intact, and it is now the highest-value next experiment.

---

## 2026-08-04 — corrections to earlier numbers, with more data

Three figures quoted earlier were computed on too little data or on a contaminated pool.
Recording the corrections rather than leaving the originals to be cited.

**Experiment C, oracle feasibility.** Originally computed on 6 episodes; now on the
**baseline's 15** episodes, and restricted to them *on purpose*. Pooling every method's
episodes is a confound: a degraded method barely engages (`vision_frames_9` produced one
contact event in nine episodes), so its episodes contribute almost no "critical" steps and
drag the rate down — pooling all 59 gave a misleadingly low 5.3%. `analyze_oracle_temporal.py`
now resolves a method's episodes through its closed-loop ledger, since the output directories
are timestamps and carry no method label.

| quantity | 6 episodes (as first reported) | baseline's 15 episodes |
|---|---|---|
| median critical-step rate | 8.9% | **6.6%** |
| baseline refresh rate | 3.2% | 3.2% |
| threshold oracle's calls vs baseline | 5.25× | **4.27×** |
| episodes where a matched budget covers all critical moments | 1 of 6 | **2 of 15** |

**The conclusion is unchanged and now better supported**: critical moments remain ~2× more
frequent than the baseline's cadence, so an oracle still cannot save compute by refreshing
only when necessary, and a matched budget still cannot cover the critical moments in 13 of 15
episodes. The magnitudes moved; the direction did not.

**Latency contention warning was a false positive.** `docs/latency.md` initially warned that
all 90 samples were taken with another process on the GPU. That process was the benchmark
itself — `nvidia-smi --query-compute-apps` lists the measuring PID. Filtering our own PID
clears it, and the check now records `gpu_compute_apps_including_self` separately so the
distinction is visible rather than lost. An unfiltered check would have discredited perfectly
good data with a warning about itself.

**Authoritative latency, now measured properly** (interleaved, cooled, median-of-many, MAD
0.32–2.73%, no competing process):

| condition | CUDA median | speedup |
|---|---|---|
| baseline (4 steps, 33 frames) | 3,465 ms | — |
| `steps_2` | 1,886 ms | 1.84× |
| `vision_frames_9` | 1,516 ms | 2.29× |
| `steps_1` | 1,100 ms | 3.15× |
| **`steps_1 × vision_frames_9`** | **505 ms** | **6.86×** |
| **`steps_1 × vision_frames_5`** | **386 ms** | **8.97×** |

**The two levers compose almost multiplicatively — measured, not assumed.** 3.15 × 2.29
predicts 7.21×; measured 6.86× is 95% of that. Both act on independent factors of a
matmul-bound, token-linear cost. Spec §8 forbids inferring this without measurement, and the
measurement agrees with the inference — which is worth knowing precisely because it so often
would not.

Peak VRAM is **identical (30,996 MiB) across every configuration**: weights dominate and
activations are negligible beside them, so none of these levers relieves memory pressure.
That is relevant to "does it fit" independently of speed.

**The uncomfortable summary.** There is an 8.97× speedup available and precisely measured, and
every configuration reaching it scores at or near zero closed-loop. The project's difficulty
is not finding compute reductions in this model; it is that the model appears to need what it
is using.

---

## 2026-08-04 — CORRECTION: I over-read the Experiment C result

Re-examining the oracle conclusion under a sensitivity sweep on the one parameter I chose
freely — the dilation applied around each contact/subtask event — shows my reported finding
was partly an artefact of that choice.

| dilation | critical-step rate | vs baseline refresh (3.19%) | matched budget covers all critical |
|---|---|---|---|
| **0** | **2.01%** | **rarer than baseline** | **18/23** |
| 1 | 4.72% | more frequent | 6/23 |
| 2 *(as reported)* | 6.64% | more frequent | 4/23 |
| 4 | 10.72% | more frequent | 0/23 |

**What I got wrong.** I reported "critical moments are ~2× more frequent than the baseline
cadence, so an oracle cannot save compute", and used it to deprioritize Experiments C
(closed-loop) and D (learned event/flow gates). At dilate=0 critical moments are *rarer* than
the baseline's refresh rate, and a matched budget covers them in **78%** of episodes rather
than 17%. The ±2-step dilation is defensible — events are logged when they *complete*, so a
refresh needs lead time to help — but it drove the headline number, and I presented a
parameter-dependent result as settled. **The "same compute, better placement" version of C is
NOT falsified, and D's motivation is correspondingly more intact than I claimed.**

**What survives, and is stronger than the argument I originally made.** At dilate=0, critical
moments are rarer than the baseline's refresh rate and the threshold oracle *still* makes
**2.96×** the calls. That has nothing to do with event frequency; it is the
`max_cache_age = 32` cap. And that cap is not a tuning knob — **it is the chunk length**. The
server returns exactly 32 actions; once they are executed there is nothing left to execute, so
the client must call again.

So the baseline does not refresh once per 32 steps because it judged that often enough — **it
refreshes because the chunk is exhausted.** It sits on a hard structural floor, and
"refresh less often" is not a scheduling decision available to us at all. Reaching below
1-per-32 requires a *new mechanism*: extrapolating, repeating, or re-planning a suffix past
chunk exhaustion. Spec §11.2 anticipates exactly this, permitting horizons of 48/64 "only when
a principled suffix/repetition or extrapolation rule exists."

**We never built that rule, so the compute-saving version of the temporal idea was never
tested.** The client patch clamps the horizon to [1, OPEN_LOOP_HORIZON] and raises otherwise —
a guard I wrote deliberately — so the horizon sweep only ever went *shorter* (8, 16, 32). What
we actually falsified is "refreshing **more** often helps": it does not (5.98× compute, −50
points, strictly dominated).

**Is this a setup problem?** No misconfiguration: `OPEN_LOOP_HORIZON = 32` is verified in
upstream source and confirmed empirically (34 requests for 1,045 control steps). At 15 Hz that
is **2.13 s of open-loop execution per policy call**, which is aggressive amortization by any
standard. It is a **framing** mismatch instead. The adaptive-refresh and diffusion-caching
literature targets models that recompute per-frame or on 20–50-step schedules; Cosmos3-Nano
already amortizes over 32 control steps *and* runs only 4 denoising steps. Both redundancies
those methods harvest were already spent by the upstream design — which is the same reason
Experiment F found nothing to reuse. The idea would work against a baseline that recomputes
often. This baseline does not.

**Consequences for the plan.**

1. **Chunk extrapolation is now the highest-value untested idea** for temporal reuse: execute
   the 32 actions, then extend by a principled suffix rule to reach effective horizons of
   48/64, rather than calling the policy. This is the only version that can reduce call count.
   Prototypable offline against the existing 88-request corpus.
2. **Matched-budget oracle placement is reinstated as open**, at low dilation, and is a
   *success* question at equal compute. It is what would justify D.
3. **Dilation must be reported as a sensitivity**, not a fixed choice, wherever the oracle
   result appears. `analyze_oracle_temporal.py` takes `--dilate`; the sweep above should
   accompany any citation of the number.

---

## 2026-08-04 — the paired 2-episode result, and what the evaluation can actually resolve

Second episode (seed 1) of `baseline_full` and `baseline_steps_2` on the screened 9, run in
parallel on disjoint GPUs. This was the measurement flagged as the one n=1 could not settle.

| task | baseline e1/e2 | steps_2 e1/e2 |
|---|---|---|
| BananaThenRubiksCube | 1 / 1 | 1 / 1 |
| BowlStackingLeftOnRight | 1 / 1 | 0 / 0 |
| ReorientJug | 1 / **0** | 0 / 0 |
| RubiksCubeAndBanana | 1 / **0** | 1 / 1 |
| Stack3RubiksCube | 1 / **0** | 0 / 0 |
| BlockStackingOrderAgnostic | 0 / 0 | **1** / 0 |
| BlockStackingSpecifiedOrder, BigPumpkinInBin, BlackItemsInBin | 0 / 0 | 0 / 0 |
| **total** | **7/18 (38.9%)** | **5/18 (27.8%)** |

**The deficit halved with one more episode: −22.2 → −11.1 points.** And it is not significant
by any reading:

- **The baseline disagrees with itself on 3 of 9 tasks** between the two seeds (ReorientJug,
  RubiksCubeAndBanana, Stack3RubiksCube each 1/0). `steps_2` disagrees on 1 of 9.
- Task-level: baseline better on 3, `steps_2` better on 2, tied on 4 — **sign test p = 1.00**.
- One-sided binomial against the baseline's 38.9% rate:

| method | score | P(≤ observed) | verdict |
|---|---|---|---|
| `vision_frames_9` | 0/9 | **0.012** | **significant** |
| `baseline_fixed_horizon_8` | 1/8 | 0.118 | not significant on success |
| `baseline_steps_1` | 2/9 | 0.253 | not significant |
| `baseline_steps_2` | 5/18 | 0.237 | not significant |

**So exactly one success claim survives**: `vision_frames_9` genuinely breaks the policy
(p = 0.012, and independently corroborated by 1 contact event versus 96 and the straightness
collapse). `horizon_8` remains ruled out, but on **compute** grounds alone — it uses 5.98× the
compute and is at best no better, so it is dominated without needing a success test. The
reduced-step deficits are **not established**.

**The cost of resolving them, which is the number that should drive planning.** Per-episode
success is 0.389 vs 0.278, a 0.111 difference. For 80% power at α=0.05:

> **~31 episodes per task per arm ≈ 43 GPU-h per arm ≈ 87 GPU-h for the pair.**

That is ~5× the entire budget spent on this session, to resolve *one* comparison. Our n=2 is
roughly 15× too small.

**This is itself a result, and arguably the most useful one for planning.** RoboLab closed-loop
evaluation on this policy has enough intrinsic variance that differences below ~20–30 absolute
success points are not resolvable at feasible cost — and the project's own §14 gate is **2
points**. Detecting a 2-point effect would need thousands of episodes. Any future claim of
"non-inferior success" on this benchmark has to either buy that many episodes, or move to a
lower-variance signal (task progress score, subtask completion counts) rather than binary
success.

Combined with the 78%-simulator cost structure, the practical conclusion is that **compute
should be measured in-process and success measured on a progress-style metric**, because binary
success at feasible episode counts cannot support the comparisons this project needs.

---

## 2026-08-04 — SETUP BUG: we served the wrong prompt format for three sessions

Prompted by the question "shouldn't these baselines work out of the box?", I finally
investigated a warning that appears in **every server log this project has ever produced** and
that I had read repeatedly and dismissed as metadata noise:

```
could not load training config for transforms: Missing key _type
no training action dataset config found; using default
ActionTransformPipeline (format_prompt_as_json=False)
```

**What it means.** The server decides the prompt format by loading the checkpoint's *training*
config, which requires a local `checkpoint.json`. We pass `--checkpoint-path
nvidia/Cosmos3-Nano-Policy-DROID` — an HF repo id — so `_load_checkpoint_metadata` finds no such
file, `setup_args.load_config()` then fails with "Missing key _type", and the server falls back
to `format_prompt_as_json=False`, i.e. **plain-text prompts**.

But the training recipe for this exact checkpoint —
`cosmos_framework/configs/base/experiment/action/posttrain_config/action_policy_droid_nano.py:219`,
docstring "Cosmos3-Nano DROID action policy SFT recipe", `joint_pos` 8D + `use_state`,
640×360 — sets **`format_prompt_as_json=True`**. The model was trained on structured prompts:

```json
{"cinematography": {"framing": "..."},
 "actions": [{"time": "0:00-0:00", "description": "...", "idle_frame": "..."}],
 "duration": "...", "fps": ..., "resolution": ..., "aspect_ratio": ...}
```

and we were sending a flat sentence. The formats are **mutually exclusive**: with JSON enabled
the viewpoint and duration/FPS text augmentors are disabled, because that metadata moves into
the structure. So **every request in three sessions had off-distribution conditioning — the
baseline included.** Verified fixed: the server now logs `format_prompt_as_json=True` and serves
`prompt='{"cinematography": {"framing": ...'`.

**Why it evaded three sessions of scrutiny.** The fallback warns but is silent *in effect* — it
reads like routine metadata chatter, not "your conditioning is wrong". And it only triggers on
the **HF-repo-id path**: anyone serving from a local checkpoint directory containing
`checkpoint.json` never sees it. The documented quick-start path is precisely the one that
misconfigures the model. Worth reporting upstream.

**Wiring.** `COSMOS_PROMPT_JSON` now threads through `start_cosmos_server.sh` and
`run_closed_loop.py`'s `SERVER_ARG_ENV`. Spelling note: the field is `bool | None`, so tyro wants
a **value** (`--format-prompt-as-json True`), not a `--flag/--no-flag` pair — the pair form
aborts with "Expected one of ('None', 'True', 'False')", which cost one startup rather than a
silently mis-served run.

**What is now suspect.** Anything whose value depends on baseline *success*:

- baseline 5/15 and 7/18 — likely an underestimate of the model's real ability
- the `steps_1` (−16.7 pts) and `steps_2` (−11.1 pts) deficits — measured against a handicapped
  reference
- `vision_frames_9`'s 0/9 (p = 0.012), the only statistically significant result — being
  re-measured now under the corrected config, though a policy yielding *one* contact event in
  nine episodes is unlikely to be explained by prompt format alone
- **the screened 9-task set itself**, which was derived from which tasks the handicapped
  baseline solved

**What is unaffected**, because it is arithmetic rather than success-dependent: the token census
(3,188 = 95 + 340 + 2,720 + 33; prompt format does not change the 95 text tokens materially),
the chunk-exhaustion floor behind the temporal result, the cross-step residual measurement, the
measurement floor (B3), and the latency table.

**The lesson, which is the same one as the offline-screen failure.** I treated a warning as
noise because it appeared on every run and nothing visibly broke. "Appears always" is not
evidence of harmlessness — it is equally consistent with a defect that has been active the whole
time. Every warning in a measured pipeline should be either explained or silenced; leaving it in
place trained me to ignore it.

---

## 2026-08-04 — the baseline WAS working: we reproduce the published number

The calibration I should have fetched in session 1. Published RoboLab-120 leaderboard:

| policy | success rate | score |
|---|---|---|
| **Cosmos3-Nano-Policy (WAM)** | **36.8%** (441/1200) | 51.9 |
| π0.5 (VLA) | 28.0% | 43.4 |
| DreamZero (WAM) | 25.7% | 39.8 |
| Cosmos3-Edge-Policy (WAM) | 22.9% | — |
| π0-FAST | 15.5% | 26.9 |
| GR00T N1.6 | 7.2% | 17.1 |
| π0 | 5.0% | 12.2 |
| paligemma-binning | 3.4% | 9.9 |

Ours: **33.3%** (pilot, 15 tasks), **38.9%** (screened 9, 2 episodes), **33.3%** (screened 9,
corrected JSON prompts). **We reproduce the published 36.8% within our noise.** The baseline is
not broken, the setup is not misconfigured, and ~37% is state of the art on this benchmark —
Cosmos3-Nano-Policy is the top-ranked entry and the runner-up is 8.8 points behind.

**Which resolves the prompt-format question too.** The bug was real — we did serve plain text to
a model trained on structured JSON — but its measured effect on success is **undetectable**:
3/9 with JSON sits inside the baseline's own 2/9–5/9 seed-to-seed range. So it is worth reporting
upstream and worth fixing, but it does **not** explain the low absolute success and it does
**not** invalidate the earlier comparisons. Prior results stand.

**A second thing this settles, and it is more consequential than the first.** The official
protocol is **1200 episodes over 120 tasks = 10 episodes per task**. Two facts follow:

1. Our 1–2 episodes per task is 5–10× below the benchmark's own protocol, which corroborates the
   power analysis (~31 episodes/task/arm for 80% power on an 11-point effect) from a completely
   independent direction.
2. **The project's §14 gate of ≤2 absolute success points is below the resolution of the
   benchmark's official protocol.** The leaderboard's top-two gap is 8.8 points and is considered
   a meaningful ranking difference; at 10 episodes/task, a 2-point difference is not separable.
   Asking for non-inferiority within 2 points is asking for roughly 4× finer resolution than the
   benchmark was designed to deliver.

**So the §14 success criterion should be revised, and the benchmark already provides the fix.**
NVIDIA reports a second metric alongside success rate — **Score** (51.9 for Cosmos3-Nano), a
partial-credit/progress measure. That is precisely the lower-variance signal the power analysis
called for, and using it is not a deviation from the benchmark but using it as intended. It also
matches what RoboLab records per step (`subtask/score`, `subtask/completed`), which we already
parse. Recommendation: **re-express the non-inferiority gate in terms of Score, and report binary
success only as a secondary headline.**

Note the difficulty breakdown also validates our task-set construction: Simple 40.6%, Moderate
35.4%, Complex 25.3%. Our stratified pilot landing at 33% is consistent with a mix weighted
toward moderate/complex.

---

## 2026-08-04 — E2b re-tested under the corrected config: result holds, mechanism narrative was partly wrong

Re-ran `vision_frames_9` with the prompt format the checkpoint was trained with, against a
baseline run the same way.

| | success (screened 9) | contact events |
|---|---|---|
| baseline, JSON prompts | **3/9** | **147** |
| `vision_frames_9`, JSON prompts | **0/9** | **22** |
| `vision_frames_9`, plain-text prompts *(as first reported)* | 0/9 | **1** |

**The conclusion survives.** 0/9 against a 33.3% baseline rate gives P(0 successes) = **0.026**,
still significant. Shortening the imagined horizon from 9 to 3 latent frames does break the
policy, and that is now established under the configuration the model was trained for.

**But the mechanism I described was partly a prompt-format artefact, and I need to withdraw part
of it.** I characterised the failure as "the robot barely engages — one contact event across nine
episodes, versus 96 for the baseline". Under correct prompts it produces **22** events. So it
*does* manipulate; it engages roughly 7× less than the baseline and never completes a task, which
is a materially different and less dramatic failure mode than near-catatonia. The single-event
figure was measuring the plain-text handicap compounded with the shortened horizon, not the
shortened horizon alone.

**This also puts a caveat on the straightness finding.** The offline study — including the
straightness measurement (0.10–0.13 for the `vision_frames_*` conditions against 0.30–0.39 for
teacher/seeds/steps) — was computed by replaying captured requests through a server running the
**default plain-text** format. So those numbers describe the same compounded condition. The
*direction* is likely robust, since the mechanism (a shorter imagined horizon means less lookahead
and so less committed displacement) does not depend on prompt format, and E2b still fails. But the
magnitudes should be re-measured under JSON prompts before straightness is cited as a predictor,
and the "2.3× gap with no overlap" claim in particular should not be quoted until then.

**Cost to close this out:** the offline study is ~0.6 GPU-h to re-run for the E-conditions under
JSON prompts, which would also re-establish the A-condition deviations on the correct
configuration. That is the cheapest remaining item and should precede any further closed-loop
spending.

**Wider lesson, third instance of the same pattern this session.** A measurement taken under a
subtly wrong configuration can still yield the *right* verdict for the *wrong* reason — and the
wrong reason is what gets written down as the mechanism. The verdict here (E2b fails) was robust;
the explanation (near-total disengagement) was not. Verdicts and mechanisms need separate
confirmation.

---

## 2026-08-04 — CORRECTION: plain-text prompts are the DOCUMENTED path; and we are not running the closest baseline

Read `third_party/RoboLab/policies/cosmos3/README.md` — the recipe upstream publishes for
reproducing the leaderboard entry. Two things follow, one a correction and one a real gap.

**Correction: the prompt-format "bug we shipped" is the documented default.** The official server
command is bare:

```
python -m cosmos_framework.scripts.action_policy_server_robolab --port 8000
```

No `--checkpoint-path`, so it resolves the HF repo id — which means the *official* recipe hits
exactly the same training-config fallback and also serves **plain text**. So our original
`baseline_full` runs match the documented serving path, and the published 36.8% was almost
certainly produced that way too (consistent with our plain-text runs reproducing it).

I over-claimed by calling this "a setup bug we shipped for three sessions". The accurate framing
is an **upstream train/serve inconsistency**: training uses `format_prompt_as_json=True`
(`action_policy_droid_nano.py:219`) while the documented serving path yields `False`. That is
genuinely worth reporting to NVIDIA — a model served differently from how it was trained — but our
configuration was correct *per the documentation*, nothing was invalidated, and
`baseline_full` (plain text) is the closer match to the official baseline, not
`baseline_full_promptjson`. Both score the same within noise anyway (38.9% vs 33.3%, P = 0.51).

**The real gap: `--num-envs`.** The official client command is

```
python policies/cosmos3/run.py --task <T> --num-envs 10 --headless
```

and `runner.py` states outright: *"Total episodes = num_runs * num_envs. **Prefer increasing
--num-envs for more episodes.** Only increase --num-runs if you run out of GPU memory."* With the
leaderboard at 1200 episodes / 120 tasks = **10 episodes per task**, `--num-envs 10` is
near-certainly how the published number was produced.

**We ran `--num-envs 1` throughout.** That is our largest remaining deviation, and closing it is
*cheaper per episode*, not more expensive: the 10 envs share one Isaac instance and one scene
load, and physics steps them together. Policy requests are serial per env
(`base_client.py:79` — a dict comprehension over `env_id`, not a batched payload), so compute
scales with envs while startup and physics do not. Ten episodes in one invocation instead of ten
invocations.

**RoboLab also ships the fix for our variance problem.** `--num-episodes-adaptive MAX_N` keeps
adding batches of `num_envs` episodes *until the 95% Beta credible interval on success rate is
within `--ci-pp-width`*, default **0.14 (14 percentage points)**, with recommended MAX_N=200
matching the TRI LBM sim protocol. Two implications:

1. We never had to hand-roll episode budgeting or the power analysis — the benchmark has a
   principled stopping rule built in, and we should use it.
2. The default target CI width is **14 points**. Our §14 non-inferiority gate is **2**. So the
   benchmark's own default tolerance is 7× looser than the criterion this project set itself —
   a third independent confirmation that the 2-point gate is unusable, after the power analysis
   (~31 episodes/task/arm) and the leaderboard's 8.8-point top-two gap.

**So: the closest baseline we can run is `baseline_full` (plain text, guardrails ON) with
`--num-envs 10`, ideally under `--num-episodes-adaptive`.** Remaining deviations and whether they
are closable:

| deviation | closable? |
|---|---|
| `--num-envs 1` instead of 10 | **yes, and cheaper per episode** — being tested now |
| episode budgeting by hand instead of `--num-episodes-adaptive` | **yes**, use the built-in rule |
| guardrails disabled | **no** — needs approved access to gated `nvidia/Cosmos-Guardrail1` (user action). Affects latency, not plausibly success |
| 9–16 tasks instead of 120 | yes in principle; ~19 GPU-h per single-episode pass over 120 |
| flash2/cuDNN instead of flash3 | **no** — flash3 is Hopper-only, we are on L40S |
| NCCL single-rank patches (0002/0003) | **no** — required for the server to start at all |
| Docker path | no, and immaterial — the host already matches the image base |

---

## 2026-08-04 — "Are you sure all of those are not closable?": three of them were

The table immediately above said `no` four times. Two of those `no`s were wrong, and one
was answering the wrong question. Re-derived from source rather than from our own notes.

### NCCL single-rank patches (0002/0003): **closable, and closed**

`maybe_init_distributed()` builds a one-rank **NCCL** group on the standalone-server path
and *returns early if a group already exists*. Pre-creating the group with **gloo** makes
upstream's collectives run unmodified instead of being skipped by our patches.

Verified in three escalating steps: a model-free 40-line probe (NCCL dumps core, gloo
completes all three collectives); four timed arms against a `PYTHONPATH`-shadowed git
worktree of genuinely unpatched source (no resolvable latency difference — 2–4% apparent,
against 4.6–6.9% between-instance spread); and a full-array digest showing the produced
actions are **bitwise identical** (`5c01880496f1f666…`), which is what makes the switch
free of re-runs.

Adopted via `scripts/cosmos_server_entry.py` (a wrapper that runs upstream's `__main__`
with argv untouched) plus `ensure_single_rank_group()` in the four in-process probes.
Validated end to end: server up on a spare GPU, `healthz=200`.

The residual deviation is the backend *name* — upstream uses NCCL on hardware where NCCL
works. We now carry one line of our own setup instead of two edits inside upstream
functions, which is the trade the "run the vanilla version" instruction asks for.

### Guardrails: **the deviation is immaterial, and we had been overstating it**

`_run_text_guardrail` / `_run_video_guardrail` are called only from
`OmniInference.generate_batch` and `_generate_reasoner_batch`. `RobolabPolicyService.infer`
calls `model.generate_samples_from_batch` directly and never enters either; `grep -rn
guardrail cosmos_framework/model/` returns zero hits; and the video guardrail could only
act on a decoded rollout, which `decode_video=False` never produces.

So the sentence this project shipped for three sessions — *"every latency understates the
official baseline"* — was **false**. Enabling guardrails costs a startup download and
resident VRAM, and changes no measured quantity. Corrected in
`docs/upstream_patches.md` and in the per-unit `deviations_from_official` record.

Also: the repo is `gated: "auto"` — auto-approved click-through, not a pending review. It
remains a user action (accepting a licence), but a 30-second one, and now a cosmetic one.

### Not deviations at all

`--no-decode-video` is the upstream **default** (`decode_video: bool = False`), and the
pinned `--hf-revision 6706d768…` is exactly what the cached `main` ref resolves to. Both
had been carried in the deviation list; neither belongs there.

### Still genuinely open

- **`flash3`**: `get_backend_list` gates it on `arch_tag == 90`. This is sm89 silicon. Not
  closable by any amount of work.
- **9 tasks vs the published 120 × 10 episodes**: closable, but for **~79–115 GPU-h** —
  see the 2026-08-05 entry, which supersedes the "~41 GPU-h" figure written here from a
  7-task partial. A budget decision, not a technical one — and the only remaining way to
  compare against 36.8% on its own terms.

Client-side, `scripts/run_robolab.sh` already invokes the official
`policies/cosmos3/run.py` with official flags, so nothing there needed changing.

---

## 2026-08-05 — `baseline_official` completed: 31.1% at the official 10-episode protocol

The first run at the benchmark's own protocol (`--num-envs 10` → 10 episodes/task, official
server defaults) finished all 9 screened tasks. `results/reports/baseline_official.json`.

| task | success | mean score | policy calls | control steps | wall (min) |
|---|---|---|---|---|---|
| RubiksCubeAndBananaTask | 10/10 | 1.00 | 145 | 4,569 | 21.0 |
| BananaThenRubiksCubeTask | 7/10 | 0.95 | 191 | 5,900 | 24.9 |
| BowlStackingLeftOnRightTask | 6/10 | 0.60 | 76 | 2,264 | 9.2 |
| ReorientJugTask | 4/10 | 0.50 | 241 | 7,488 | 31.7 |
| Stack3RubiksCubeTask | 1/10 | 0.50 | 288 | 8,934 | 38.3 |
| BlockStackingOrderAgnosticTask | 0/10 | 0.50 | 430 | 13,500 | 57.3 |
| BlockStackingSpecifiedOrderTask | 0/10 | 0.03 | 430 | 13,500 | 56.9 |
| BigPumpkinInBinTask | 0/10 | 0.00 | 290 | 9,000 | 40.9 |
| BlackItemsInBinTask | 0/10 | 0.44 | 570 | 18,000 | 76.8 |
| **total** | **28/90 = 31.1%** | **0.502** | **2,661** | **83,155** | **5.95 GPU-h** |

Sole deviation recorded per unit: guardrails disabled — and per the 2026-08-04 audit above,
that is immaterial on this code path.

This is the reference all methods are compared against from here. It replaces the 1–2
episode/task pilots for that role.

### Correction: single-episode screening is not just noisy, it is biased upward here

Comparing each task's 1-env estimate (2 seeds, `baseline_full`) against the 10-env truth:

| task | 10-env truth | 1-env estimate | |
|---|---|---|---|
| BananaThenRubiksCube | 70% | 100% | overstates |
| BowlStackingLeftOnRight | 60% | 100% | overstates |
| Stack3RubiksCube | **10%** | **50%** | overstates |
| RubiksCubeAndBanana | 100% | 50% | understates |
| ReorientJug | 40% | 50% | agrees |
| 4 × zero-success tasks | 0% | 0% | agrees |

Four of nine agree only because they are floored at zero. Among the five tasks with any
successes, three overstate and one understates. A previous entry in this log described the
error as symmetric seed noise; on this evidence it is not — early-terminating successes are
over-represented in a 1–2 episode sample. **Every success rate in this project measured at
1–2 episodes/task should be read as an upper bound, not an estimate.**

### Correction: the per-episode saving from `--num-envs 10` is 2.35×, not 4.5×

Measured across the same 9 tasks: 3.97 min/episode at `--num-envs 10` versus 9.31
min/episode at `--num-envs 1`. Per-task ratios span 1.14× (ReorientJug) to 4.36×
(BigPumpkinInBin) — the saving is largest exactly where episodes run to the step cap, because
parallel envs amortize the policy call across 10 environments and a failing episode makes the
most calls. The earlier "4.5×" was extrapolated from one cheap task and is withdrawn.

Consequence: the powered-pair estimate becomes **~37 GPU-h** (87 GPU-h at `--num-envs 1` ÷
2.35), not the ~19 GPU-h previously quoted.

### Correction: a full 120 × 10 reproduction costs ~79–115 GPU-h, not 41

Now measured rather than extrapolated, at exactly the target protocol: **0.661 GPU-h per
task-of-10**.

- Flat extrapolation over 120 tasks → **79 GPU-h**.
- Length-normalized: our 9 tasks average 68.9 s of metadata `episode_s` against 91.2 s for
  all 120, i.e. this set is **0.76× the average task length** → **105 GPU-h**.
- Scaling the metadata budget by the measured ratio (median 3.79× the nominal 10-episode
  budget) → **115 GPU-h**.

Pulling the other way: this set is failure-enriched (31.1% vs the published 36.8%), and
failures run to the step cap, so a representative set terminates earlier per episode. Honest
range: **79–115 GPU-h**, best estimate ≈100. The earlier "~41 GPU-h" came from a 7-task
partial and is withdrawn. This is now firmly a budget decision, and a large one.

### Compute anatomy at the closed-loop level

Summed over parallel envs: 42.48 h of policy inference against 15.44 h of simulator stepping
→ **73.3% of closed-loop wall is policy inference.** This is the first measurement of the
Amdahl ceiling *at the protocol level* rather than per-request, and it bounds every method in
spec §5: even an infinitely fast policy would only cut total benchmark wall by 73%.

---

## 2026-08-10 — the group's efficiency repo reshapes the plan; direction set to inference-only on RoboLab/DROID

`chooper1/Cosmos3-Efficient-Imagination` was cloned and audited
(`private/upstream_fork_audit.md`, commit `ecb8de4`). It is **not** a substrate — a standalone
repo that patches cosmos-framework from the outside — and it is on a **different benchmark**
(their own simulator benchmark, Cosmos3-Nano + LoRA) than ours (RoboLab, `Cosmos3-Nano-Policy-DROID`).
Their headline number and our 31.1% are not comparable.

**Direction chosen by the user (2026-08-10): stay on RoboLab/DROID, and do no retraining.**
This is also the only self-consistent option: every cell in their table sits on a
trained LoRA checkpoint, so without training their benchmark is unreachable, while our DROID
checkpoint is pretrained.

### What that retires

| our experiment | status | reason |
|---|---|---|
| E2a/E2b — attention masking | **retired** | needs LoRA retraining. They measured every train/serve mismatch at 0.000, including serving a mask to an unmasked-trained model |
| G — spatial masking of the video block | **retired as designed** | their whole-block mask = 0.000 across 6 cells: the serving batch has no `pixel_values`, so the video latent is the policy's *only* channel to the world. Masking it blinds the robot rather than removing imagination |
| σ_a floor widening | **retired** | training-side (their action-sigma floor knob) |
| anything from their training-priors patch | **blocked** | imports a correlated-noise helper, vendored from a private training repo we lack access to, which the user does not have access to |

### What survives, and the risk it carries

Survives, all inference-only: **asymmetric video/action denoising** (their
their three warm-start sampler modules), **frozen-video KV
caching** — the ~99% FLOP saving their own report flags as *not yet implemented*
(recorded in their own write-up) — and **cross-call latent reuse**, which their prior-drift
gate was written for but which has no eval cells in their recorded eval cells.

**The risk, established from source rather than assumed.** Asymmetric scheduling needs the
action to carry its own flow-time. The seam exists in stock upstream
(`omni_mot_model.py` — public NVIDIA source), so no framework patch is required. But the *checkpoint*:

- `nano_model_config.py:90` sets `independent_action_schedule=False`, and
  `action_policy_droid_nano.py` never overrides it (its only model edit is `loss_scale=10.0`
  at line 247). **Our checkpoint was trained with video and action on one shared σ.**
- `shift_action=None` → inherits `shift`, which for this recipe is
  `{"256": 3, "480": 5, "720": 10}`. Inference config ships `shift=1`
  (`nano_model_config.py`, `rectified_flow_inference_config`) — a train/serve shift mismatch
  in its own right, and they measured that serving at the training shift made their benchmark *worse*.
- Action σ is sampled `logitnormal`, which by their measurement puts a negligible fraction
  of training mass at low σ. Their own floor bound therefore predicts
  that large action-step counts visit σ the stock checkpoint never saw — and we cannot widen
  the floor without training.

So V≠A is **out of distribution for our checkpoint**, exactly the condition their
their sampler's documented caveat warns about ("the validation question is whether it degrades
*gracefully* vs the hard 0% collapse"). This is a real possibility of a null result, and it is
cheap to falsify offline before any closed-loop spend: replay the 89-request corpus
(`results/raw/corpus/`) over a V×A grid and measure action deviation against the 4-step joint
teacher. No simulator, no training.

Recorded now so that a null is a finding rather than a surprise.

### Also noted

Their repo has **no LICENSE file**, though individual sampler modules carry
`SPDX-License-Identifier: MIT`. Provenance must be recorded per file if any of it is ported.
`sampler/patch.py` imports a `warm_start.*` package that does not exist in this layout (the
modules sit flat in `sampler/`), so porting needs a packaging alias, not just a copy.

---

## 2026-08-10 — NEGATIVE RESULT: asymmetric video/action denoising collapses on our stock checkpoint

Their headline method — fewer *video* denoising steps than *action* steps — was ported to our
stack and measured offline on the captured request corpus. **It does not transfer to
`Cosmos3-Nano-Policy-DROID` without retraining.** This closes the direction chosen earlier
today, for ~0.3 GPU-h and before any closed-loop episodes were spent.

Artifacts: `results/processed/split_schedule{,_stratified}.jsonl`,
`results/reports/split_schedule{,_stratified}.json`,
`scripts/offline_split_schedule.py`, `scripts/analyze_split_schedule.py`.

### The measurement

Their `KNFrozenVideoSampler` in NO-PRIOR mode (`KNFV_NO_PRIOR=1`,
`KNFV_VIDEO_ENTRY_SIGMA=1.0`, rolling prior off): the video denoises cold in K steps and then
freezes for the remaining N−K action steps. Deviation is against the 4-step joint teacher —
the sampler that produced `baseline_official` (28/90 = 31.1%). Stratified, 4 requests from
each of 4 tasks, all 16 cells engaged (verified via their `[knfv]` trace, which is gated
behind `KNFV_DEBUG=1`):

| cell | V | A | joint L2 (rad) | linf | gripper | **cosine** | CUDA ms |
|---|---|---|---|---|---|---|---|
| teacher repeat | 4 | 4 | **0.0** | 0.0 | 0.000 | **1.000** | 7,402 |
| knfv V4/A4 | 4 | 4 | 0.052 | 0.078 | 0.000 | 0.460 | 7,721 |
| knfv V2/A4 | 2 | 4 | 1.460 | 1.46 | 0.031 | −0.028 | 7,650 |
| knfv V1/A4 | 1 | 4 | 1.695 | 1.60 | 0.031 | −0.060 | 7,335 |
| knfv V1/A8 | 1 | 8 | 1.804 | 1.74 | 0.000 | 0.055 | 15,686 |

Uniform across tasks — V1/A4 cosine per task: −0.060 / −0.028 / −0.034 / −0.099. Not a
task-specific artifact. A first 16-request run happened to draw a single task
(`--limit` samples the sorted corpus, which groups by task); `--per-task` was added and the
stratified run above supersedes it. Both agree.

### Why this is collapse and not degradation

Cosine ≈ 0 means the emitted chunk is **unrelated** to the teacher's, not a noisier version
of it. That is the outcome their own their sampler's documented caveat predicts for a
checkpoint trained with `independent_action_schedule=False` — ours
(`nano_model_config.py:90`, not overridden in `action_policy_droid_nano.py:247`). The σ_a
floor argument predicted the same thing: action σ is `logitnormal`, a negligible fraction of training mass
below σ=0.1, and widening that floor is training.

### Two findings worth more than the verdict

**1. The pipeline is bit-deterministic.** The stock-repeat control measured *exactly* 0.0
deviation, cosine 1.000, on all 16 requests. `deterministic_seed=True` holds for the whole
inference path, not merely the seed. So this is the **B3 noise floor** the plan has owed
since M2 — it is zero, no minimum-detectable-effect threshold is needed for offline
comparisons, and any nonzero offline deviation is signal. Obtained free, as a control.

**2. There is no speedup on this axis even in principle, today.** Cost tracks A, not V:
V1/A16 measured 30.4 s against the teacher's 7.3 s (4× *slower*), and V1/A4 measured
7,335 ms against 7,402 ms — indistinguishable. This is their unimplemented-KV-cache caveat
(recorded in their own write-up) appearing as measured latency, and it is exactly the claim our rules
forbid making from FLOPs alone.

**Their mechanics reproduce correctly**, so the collapse is the checkpoint's response and not
a wiring artifact: V4/A4 (their K=N "mechanics proof") measures 0.052 here against the value they document, same order, against a 0.0 floor.

### Integration recorded

- `src/action_refresh/server/warm_start_vendor.py` imports their `sampler/` directory as the
  `warm_start` package their code expects, without copying it — their repo has no top-level
  LICENSE (per-file SPDX is MIT for 8 of 9 modules), so importing a granted clone is the use
  that needs no grant. Per-file SPDX is recorded in every result row.
- `private/patches/cei-0001-prepare-arity.patch` (branch
  `research/action-aware-refresh` on their clone) makes `prepare_inject` arity-agnostic: our
  framework returns an 8th value (`has_noisy_actions`) their base lacked. A wrapper cannot fix
  this — on a warm call they install `prepare_inject` as an *instance* attribute
  as an instance attribute, which shadows any class-level adapter.

### What remains, still inference-only

Symmetric step reduction (`cold-N`) stays in-distribution here and remains our real lever;
per their decomposition our M2 "2.98× at 1 vs 4 steps" should be labelled `cold-N`, not read
as two independent axes. Their cross-call latent-reuse gate
(their prior-drift probe, with no eval cells behind it) is untested and is closest to this
project's original thesis. Note the caution this entry supplies for it: reusing a *previous*
imagination is a strictly larger perturbation than freezing the current one, and freezing
alone already collapsed the action.

### Follow-up (same day): blocked on an external dependency, by choice

Retraining was priced rather than assumed. a private training repo we lack access to is **not** the
blocker — it appears in exactly two hunks of their training-priors patch, both importing
that helper for an env-gated correlated-noise mode; the frozen-regime prior, their video-ahead knob,
their V=1 rollout knob and their no-imagination knob are self-contained. The real gates:

- **No training data on this host.** The 34 GB HF cache holds models only
  (Cosmos3-Nano-Policy-DROID, Cosmos-Guardrail1, Wan2.2-TI2V-5B); no LeRobot/DROID/their benchmark
  dataset root exists.
- **hundreds of GPU-h per training mode** (derived from their internal throughput sweep; derivation in `private/`). Against ~18 approved and ~30
  spent, for one mode, where their result is a comparison of modes.
- **many tens of GB per checkpoint**, never auto-pruned, against 328 GB free.

**Decision: ask the group for a trained checkpoint instead of training one.** Evaluating
theirs costs ~0 training GPU-h and yields the V=1 arm we cannot otherwise obtain. Request
drafted (checkpoints their frozen-regime LoRA checkpoint / `_v1rollout` / `_baseline`,
the `--experiment` name, the serving env block, their framework base commit, and
that private training repo access). Not sent by me — the user sends it.

Note for whoever picks this up: evaluating a trained for their benchmark checkpoint requires the
**their benchmark's eval stack** eval stack, which this host does not have (we have RoboLab/Isaac).
That install is a prerequisite for the checkpoint arm and is not yet scoped. Their
checkpoint cannot be evaluated on RoboLab/DROID instead — it is LoRA-tuned on
their benchmark suite and would not control a DROID robot.

---

## 2026-08-17 — handover written; and the host changed underneath us

No experiments this interval. Two documents were written — `docs/handover.md` (state,
blockers, sanctioned next steps, accumulated gotchas) and
`results/reports/session_4_report.md` (session 4 per spec §17) — and the drafted checkpoint
request was moved out of session scratch to `private/outbound_checkpoint_request.md` so it
survives. Still unsent; sending it is the user's.

**The reason this entry exists rather than just the two documents: the machine's state moved,
and both changes invalidate work that would otherwise look runnable.**

- **No GPU can hold the checkpoint.** All four L40S carry ~30.7 GB from a single other
  process (pid 2953978, another user — the author of the efficiency repo), leaving
  ~14.2–14.8 GB free against 32.9 GB of bf16 weights. This is *capacity*, not contention: any
  server start today OOMs. Do not preempt or crowd that process.
- **`/scratch` fell from 328 GB free to 49 GB (99% full).** Not ours — our footprint is
  unchanged at ~80 GB (repo 42, HF cache 34 but on `~`, not `/scratch`); `du` cannot read
  other users' directories, so the remaining ~3.2 TB is external and invisible. The practical
  effect is that the their benchmark's eval stack install the checkpoint arm would need (~15–25 GB) now
  has no safe margin — `setup_cosmos.sh`'s ≥40 GB precheck would pass by 9 GB, which is not a
  margin. Per the shared-machine rule: check, and abort rather than fill.

`loadavg ≈ 33` unchanged. Verification unchanged: 117 passed / 2 skipped; ruff reports 71
findings, all in pre-M3 files (style rules), none in anything written in sessions 3–4.

**One defect recorded while writing the handover, not yet fixed.**
`warm_start_vendor.locate()` reads the vendor commit from the manifest rather than from live
git, so `results/reports/split_schedule*.json` record `vendor_commit=ecb8de4` while the code
that actually executed was `8433abd` (= `ecb8de4` + the arity patch `cei-0001`). This is the
same staleness class already fixed for substrates by preferring live git — the vendor path
never received the fix. It should be fixed before any further result cites their commit,
because as it stands the provenance understates what ran.

**Also flagged, unchanged from session 4:** nothing has been committed since `6cf33dd`.
2,294 files are staged (+111,023/−525) — the baseline, the anatomy, the Pareto machinery, the
substrate abstraction, and the negative result exist only in one machine's git index. The
largest staged file is 480 KB and `.gitignore` correctly excludes npz/traces/videos, so the
tree is safe to commit; it needs a review pass, not a cleanup.

### Same day — what the public repo may and may not carry

`origin` is a **public** GitHub repo (checked, not assumed: the unauthenticated GitHub API
returns 200). The staged tree contained the group's unpublished work — a patch quoting their
unlicensed `sampler/patch.py`, an audit of their private repo, their measured cells, and a
personal message to a named person. Pushing that would have published a colleague's
unreleased research; GitHub caches and indexes even what is later deleted.

**Decision (user, 2026-08-17): register their repo as a re-clonable third-party source, and
push nothing that lives in it.** Implemented as:

- `private/` — untracked (`.gitignore`), holding their patch, the audit, the outbound request
  and the unscrubbed measurement rows. `private/README.md` says how to move it out of band.
- Their repo stays in `reproducibility/source_manifest.json` with URL, `base_commit` and
  `tree`, so anyone with access re-clones it via `scripts/bootstrap_new_machine.sh`.
- Redacted from tracked files: their measured cell values, their per-iteration training
  throughput, their σ_a floor-bound formula, their internal file/line citations, their
  training-side env knobs, their checkpoint names, their benchmark by name, and quotations of
  their wording. **Our own** numbers are untouched and complete.
- `vendor_modules` / `vendor_spdx` / `vendor_sampler_dir` stripped from the pushed result
  rows: the first two enumerate their repo's file inventory, and the third is an absolute
  personal path that CLAUDE.md bans independently. Originals kept in `private/`.
- Also scrubbed, unrelated to them: `reproducibility/environment.json` listed six other
  accounts on this shared host by name.

**One judgement call, stated so it can be overruled.** Interface names their code requires —
the `warm_start` package, the `KNFV_*` / `COSMOS_WARMSTART_*` env contract,
`KNFrozenVideoSampler` — remain in the public tree. Our own scripts cannot call their sampler
without them, and an API surface is not the same as their content. If that line is wrong, the
files to change are `scripts/offline_split_schedule.py`,
`scripts/analyze_split_schedule.py`, `src/action_refresh/server/warm_start_vendor.py` and
`docs/handover.md`.

**Also settled today: `third_party/` needs no committing.** All 7 local commits across
cosmos-framework (4), RoboLab (2) and their repo (1) are exported to
`reproducibility/patches/` byte-for-byte, and replaying them onto the recorded `base_commit`
reproduces the recorded `tree` exactly — tested end-to-end on cosmos-framework. The manifest
now records `base_commit` and `tree` per source, and `scripts/bootstrap_new_machine.sh`
rebuilds and verifies. The corpus (88 MB) stays out of git by the user's choice; the handover
records that re-capturing it costs ~0.2 GPU-h and currently needs a GPU we cannot get.

### Same day — two silent-wrong-value bugs closed (no GPU)

Both were the same species: a value that looked derived and was in fact guessed, failing
silently. Neither needed a GPU, which is why they were done while all four are occupied.

**1. `make contract` (task #18, open since session 3).** It wrote
`docs/baseline_contract.md` — 243 hand-verified lines carrying transport details, tensor
shapes and the JSON-prompt finding — with a ~25-line regex derivation. Two independent
faults:

- *The clobber.* Default output is now `docs/generated/baseline_contract_derived.md`, and the
  script refuses to overwrite any file that lacks its own generated-marker unless `--force`.
  `--check` prints instead. Same guard class as the anatomy-overwrite fix in
  `run_anatomy_sweep.sh`.
- *The values were wrong, not merely coarse.* `port[^=]*=[^)\d]*(\d+)` uses a negated
  character class, which **matches newlines** — so it caught the "port" inside `import`, ran
  to the next `=` several lines below, and reported the server's default port as **3** against
  a true **8000**. `num_steps` came out **5** against a true **4** the same way. The docstring
  claimed "this script never guesses"; it guessed. Extraction is now anchored to the
  declaration form (`^name: type = default`), resolves exactly one constant hop, and reports
  UNKNOWN — counted on stderr, never silent — for anything it cannot resolve.
- The client horizon was UNKNOWN because it is a **class attribute** in
  `policies/cosmos3/client.py`, not an argparse default in `run.py`, which is the file the
  script read.

All 23 derived values now agree with the hand-verified contract (port 8000, 4 steps, chunk 32,
fps 15.0, horizon 32, `decode_video=False`, both return paths present), with zero UNKNOWN. The
derived file is now useful for what it can actually do: detect staleness. If it disagrees with
the hand-verified document, one of the two is out of date and the source is the tie-breaker.

**2. Vendor provenance drift.** `warm_start_vendor.locate()` took the commit from the
manifest, which `make sources` refreshes by hand and therefore lags. Consequence, recorded
when found: `results/reports/split_schedule*.json` were stamped with the pre-patch SHA while
the code that actually ran was that commit **plus** `cei-0001`. It now reads live git and
carries the manifest's value only when the two disagree (`vendor_manifest_stale`), mirroring
`_git_state` in `action_refresh.config` — the same fix that was applied to substrates in this
session's earlier work, in the one path that never received it. Also records `vendor_dirty`.

A stale SHA is worse than no SHA, because it looks authoritative. That is the general rule
both of these violate, and it is why each got regression tests rather than only a fix: 16 new
tests, including one that asserts the real 243-line contract survives a default `make
contract`. **133 passed, 2 skipped.**

Not re-stamped: the existing split-schedule result rows still carry the manifest SHA they were
written with. Rewriting a recorded result to look better is not a fix, and the correction is
recorded here and in the session-4 report instead.
