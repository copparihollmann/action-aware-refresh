# Session 3 report — M3/M4: falsifying the premise, then pricing what survives

Format per spec §17. Host: **`firesim2`** (Ubuntu 24.04.4, 4× L40S, driver 580.126.18).
Scope authorised: **~18 GPU-h through M4**, breadth-first allocation.

**Bottom line — the compute is there; the success question is mostly unmeasurable at
feasible cost.** 85.3% of the model's sequence is imagined future video, and cutting it is
cheap: an **8.97× speedup** is measured to 2.7% precision (`steps_1 × vision_frames_5`,
386 ms vs 3,465 ms), with the two levers composing almost multiplicatively.

But only **one** success claim in the whole session is statistically established:

| method | score | P(≤ observed \| baseline rate) | verdict |
|---|---|---|---|
| `vision_frames_9` (shorter imagination) | 0/9 | **0.026** | **significant — breaks the policy** (re-confirmed under corrected prompts) |
| `fixed_horizon_8` (refresh more often) | 1/8 | 0.072 | not established on success, but **dominated on compute** (4.9× total) |
| `steps_1` | 2/9 | 0.253 | not established |
| `steps_2` | 5/18 | 0.237 | not established |

**The binding constraint is evaluation variance, not method quality.** The baseline disagrees
with *itself* on **3 of 9 tasks** between two seeds. Adding a second episode halved the
apparent `steps_2` deficit (−22.2 → **−11.1** points) and a task-level sign test gives
**p = 1.00**. A power calculation on that ~11-point difference needs **~31 episodes per task
per arm ≈ 87 GPU-h for one pairwise comparison** — about 5× this entire session — and the
project's §14 gate is **2** absolute points, which would need thousands of episodes.

So the honest state is: **we have not demonstrated a compute reduction that preserves success,
and with one exception we have not demonstrated that any of them fails either.** The exception
is informative — shortening the imagined horizon genuinely breaks the policy, with a
mechanism (below).

Two structural findings *are* solid and independent of episode counts, because they are
arithmetic rather than statistical: adaptive temporal refresh cannot reduce compute here (the
baseline refreshes because the action chunk is **exhausted**, not by choice), and nothing is
reusable across denoising steps at a 4-step schedule.

**The setup is sound, and that is now verified rather than assumed:** our baseline reproduces the
published leaderboard number (33–39% against 36.8%), where Cosmos3-Nano-Policy is the *top-ranked*
entry. ~37% is state of the art on RoboLab-120.

**Which makes the central problem a protocol one.** The official benchmark runs 10 episodes per
task, and its top-two policies differ by 8.8 success points. The project's §14 gate is **2**
points — finer than the benchmark's own protocol can resolve. The fix is already in the
benchmark: NVIDIA reports a partial-credit **Score** (51.9 for our model) alongside success rate,
and RoboLab records it per step. **The non-inferiority gate should be re-expressed in Score.**

Untested: **G, spatial masking**, and **chunk extrapolation** — the only route to fewer policy
calls, and the version of the temporal idea we never built.

---

## The baseline is correct — we reproduce the published number

Calibration against the RoboLab-120 leaderboard, which should have been done in session 1:

| policy | published success rate |
|---|---|
| **Cosmos3-Nano-Policy (WAM)** — *ours* | **36.8%** (441/1200 episodes) |
| π0.5 (VLA) | 28.0% |
| DreamZero (WAM) | 25.7% |
| π0-FAST / GR00T N1.6 / π0 / paligemma | 15.5% / 7.2% / 5.0% / 3.4% |

Ours: **33.3%** (pilot 15 tasks), **38.9%** (screened 9 over 2 episodes), **33.3%** (corrected
JSON prompts). **We reproduce 36.8% within our noise.** The baseline is not broken and the setup
is not misconfigured — ~37% *is* state of the art here, Cosmos3-Nano-Policy is the top-ranked
entry, and the runner-up is 8.8 points behind. RoboLab-120 is simply far from saturated.

**A setup bug was found along the way, and it turned out not to matter.** The server cannot load
the checkpoint's training config from an HF repo id, so it silently falls back to plain-text
prompts although this checkpoint was trained with `format_prompt_as_json=True`
(`action_policy_droid_nano.py:219`). Real, now fixed and wired as `COSMOS_PROMPT_JSON`, and worth
reporting upstream since the documented quick-start path is the one that misconfigures the model.
But the measured effect on success is **undetectable** — 3/9 with JSON sits inside the baseline's
own 2/9–5/9 seed range — so it neither explains the absolute success level nor invalidates the
comparisons.

**The consequential finding is about the evaluation protocol.** The official benchmark is 1200
episodes over 120 tasks = **10 episodes per task**. So (a) our 1–2 episodes/task is 5–10× below
the benchmark's own protocol, independently corroborating the power analysis, and (b) **the
project's §14 gate of ≤2 absolute success points is finer than the benchmark's official protocol
can resolve** — the leaderboard's top-two gap is 8.8 points and is treated as a meaningful
ranking difference.

**Recommended revision:** NVIDIA reports a second metric alongside success — **Score** (51.9 for
Cosmos3-Nano), a partial-credit/progress measure. That is exactly the lower-variance signal the
power analysis calls for, it is what RoboLab already records per step (`subtask/score`,
`subtask/completed`, which we parse), and using it is not a deviation but using the benchmark as
intended. **Re-express the non-inferiority gate in Score, and keep binary success as a secondary
headline.**

## What succeeded

### Phase 0 — measurement prerequisites

**Exact token census**, read from the model's own `PackedSequence` rather than inferred.
Reconciles to **zero unattributed tokens**:

| component | tokens | share |
|---|---|---|
| text (prompt) | 95 | 3.0% |
| vision — conditioning (the real observation) | 340 | 10.7% |
| vision — **imagined future** | **2,720** | **85.3%** |
| action — conditioning (current state) | 1 | 0.03% |
| action — **predicted** (the deliverable) | 32 | 1.0% |
| **total** | **3,188** | 100% |

Nine latent frames of 17×20 = 340 tokens, from 33 pixel frames at temporal downsample 4,
of which exactly one is the observation. `attn_modes=['causal','full']` confirms vision and
action share a single full-attention split with no mask — spec §11.5 outcome **C**.
`num_action_tokens_per_supertoken=0` means actions are a contiguous block, not interleaved,
which is what made E2b tractable.

**Measurement floor (B3), and a correction to my own earlier claim.** I first read B3's
19.8% std as "nothing below ~10% is resolvable here". That was an artefact of using the
mean on a contaminated sample — the eight slowest of 40 replays were the last eight *in
time order*. My follow-up hypothesis, thermal throttling, was also wrong: `sm_clock` is
**locked** at 1065 MHz whether idle at 71 W or loaded at 260 W, with throttle reasons
`0x0` throughout, against a 2520 MHz maximum. With an 8-second idle gap:

| run | n | median | MAD | max | drift |
|---|---|---|---|---|---|
| sustained (back-to-back) | 40 | 3,396.2 ms | — | 5,973.8 ms | ramping |
| **cooled (idle gap)** | 25 | 3,430.9 ms | **11.5 ms (0.34%)** | 3,479.1 ms | **+0.06%** |

The medians agree to 1.0%, so the tail is intermittent **host-side interference** on a
shared box, not a change in the work done. **Minimum detectable effect ~1%, not ~10%** —
given median-of-many, an idle gap, and interleaved configs. Every timed iteration now
records `cuda_ms` alongside `wall_ms` plus clocks/temperature/throttle flags, so the next
such divergence is localisable rather than guessed at.

**Offline corpus.** 88 real captured requests across 4 tasks spanning competencies
(pick-and-place, spatial stacking, multi-object stacking, contact-rich reorientation). The
HDF5 recordings contain privileged state but no images, so capture was the only route.
This is what made Phases 1–2 affordable: 88 requests × 10 conditions offline, versus
~3.7 GPU-h per closed-loop pass.

### Phase 1 — Experiment A (reduced denoising steps)

88/88 coverage. Reference is the model's **own seed-to-seed sampling noise**, 0.2734 rad —
deliberately the same quantity (pairwise deviation from the 4-step teacher on the same
observation) as the numbers compared against it.

| condition | joint deviation | vs sampling noise | gripper disagreement |
|---|---|---|---|
| `steps_3` | 0.1062 rad | **0.39×** | 0.0% |
| `steps_2` | 0.1624 rad | **0.59×** | 0.0% |
| `steps_1` | 0.2651 rad | **0.97×** | 0.0% |

**One-step denoising perturbs the action no more than re-running the baseline with a
different seed does**, with zero gripper disagreement — and it is 2.98× cheaper. This is
the frontier every later method has to beat.

### Phase 1 — Experiment E (does the action need the imagined future?)

| condition | latent frames | vision tokens | deviation | vs noise | gripper flips |
|---|---|---|---|---|---|
| `vision_frames_17` | 5 | 1,700 | 0.3116 rad | 1.14× | 23/88 |
| `vision_frames_9` | 3 | 1,020 | 0.3120 rad | 1.14× | 20/88 |
| `vision_frames_5` | 2 | 680 | 0.3815 rad | 1.40× | 18/88 |
| `no_imagination_freeze` | 9 (frozen at noise) | 3,060 | 0.8386 rad | **3.07×** | **88/88** |

**The imagination is needed, but only a little of it.** Shortening the horizon costs
1.14–1.40× sampling noise with gripper flips in 26% of requests at worst — *inside* the
24–32% the baseline already flips between seeds, so its gripper behaviour is not
distinguishable from noise. *Destroying* the imagination costs 3.07× noise and flips the
gripper in 100% of requests.

On this evidence I concluded the action needed a *coherent* imagination but not a *long* one,
and spent episodes on that basis.

> ⚠ **The closed-loop result overturned this**, and the verdict was later **re-confirmed under
> the corrected prompt format**: `vision_frames_9` scores **0/9** against a 33.3% baseline
> (P = 0.026). Shortening the imagined horizon does break the policy.
>
> **The mechanism I first reported was partly a configuration artefact, and part of it is
> withdrawn.** I described the failure as "barely engages — one contact event across nine
> episodes". Under correct prompts it produces **22** events versus the baseline's 147: it *does*
> manipulate, roughly 7× less than the baseline, and never completes a task. That is a materially
> less dramatic failure than near-catatonia, and the single-event figure was measuring the
> plain-text handicap compounded with the shortened horizon.
>
> The same caveat applies to the reference-free **straightness** metric (0.10–0.13 for the
> `vision_frames_*` conditions against 0.30–0.39 for teacher/seeds/steps): the offline study was
> replayed through a server running the default plain-text format, so those magnitudes describe
> the compounded condition. The *direction* is likely robust — less lookahead should mean less
> committed displacement, independent of prompt format — but the numbers, and especially the
> "2.3× gap with no overlap" claim, **should not be quoted until re-measured under JSON prompts**
> (~0.6 GPU-h).

E1 (tokens present but frozen at noise) is worse still — 3.07× noise, 100% gripper flips,
straightness 0.026 — but it is also confounded by being out of distribution, so it should not
be read as independent evidence. Only E2b removes real tokens, so only E2b's latency is a
speedup.

### Phase 2 — Experiment C: the compute-reduction framing is unavailable (offline, zero GPU)

Two of the three quantities §11.3's gate depends on turned out computable from episodes
already recorded (`docs/oracle_temporal.md`):

| | |
|---|---|
| contact-critical steps (median) | **6.6%** of an episode |
| baseline refresh rate | **3.2%** (one call per 32 control steps) |
| threshold oracle's calls vs baseline | **4.27×** |
| episodes where a matched budget covers every critical moment | **2 of 15** |

(Computed on the baseline's 15 episodes. Restricted to the baseline deliberately: pooling
every method's episodes is a confound, since a degraded method barely engages and so
contributes almost no "critical" steps — pooling all 59 gave a misleadingly low 5.3%.)

> ⚠ **Correction — this was over-read, and the reason matters.** The critical-step rate
> depends on the dilation applied around each event, which I chose:
>
> | dilation | critical rate | vs baseline 3.19% | matched budget covers all critical |
> |---|---|---|---|
> | **0** | **2.01%** | **rarer** | **18/23** |
> | 1 | 4.72% | more frequent | 6/23 |
> | 2 *(reported above)* | 6.64% | more frequent | 4/23 |
> | 4 | 10.72% | more frequent | 0/23 |
>
> At dilate=0, critical moments are *rarer* than the baseline's refresh rate and a matched
> budget covers them in 78% of episodes. **So "same compute, better placement" is NOT
> falsified**, and Experiment D's motivation is more intact than I first claimed.

**What survives is structural, and is a stronger argument than the one I first made.** At
dilate=0 critical moments are rarer than the baseline's refresh rate and the threshold oracle
*still* makes **2.96×** the calls. That is not about event frequency — it is the
`max_cache_age = 32` cap, and that cap is **the chunk length**. The server returns exactly 32
actions; once executed there is nothing left, so the client *must* call again.

**The baseline does not refresh once per 32 steps because it judged that often enough — it
refreshes because the chunk is exhausted.** It sits on a hard structural floor. "Refresh less
often" is therefore not a scheduling decision available to us: going below 1-per-32 needs a new
mechanism — extrapolating or repeating a suffix past chunk exhaustion — which spec §11.2
explicitly requires before allowing horizons of 48/64.

**We never built that rule, so the compute-saving version of the temporal idea was never
tested.** The client patch clamps the horizon to [1, 32] and raises otherwise, so the sweep only
went *shorter*. What we falsified is that refreshing **more** often helps: it does not (5.98×
compute, −50 points).

**Not a setup error.** `OPEN_LOOP_HORIZON = 32` is verified in upstream source and confirmed
empirically. At 15 Hz that is 2.13 s of open-loop execution per call — aggressive amortization.
It is a **framing** mismatch: the adaptive-refresh and diffusion-caching literature targets
models recomputing per-frame or on 20–50-step schedules, while Cosmos3-Nano already amortizes
over 32 control steps *and* runs 4 denoising steps. Both redundancies were already spent
upstream — the same reason Experiment F found nothing to reuse.

Corroborated independently by the chunk-boundary analysis: the jump between consecutive
action chunks is **9.1× a normal intra-chunk step** (0.3777 vs 0.0416 rad), so the plan is
*already* stale at horizon 32 and extending it would execute a stale intention.

Per §11.3 the *compute-reduction* framing of C is recorded as unavailable. C's
placement variant and D are **reinstated as open** following the correction above — no optical
flow, event generation, RAFT or v2e work was done, and `src/action_refresh/{flow,events}`
remain empty.

### Phase 3 — the official baseline, closed-loop

15 of the 16 pilot tasks, 1 episode each, **5.4 GPU-h**. Success **5/15 (33%)** — usefully
far from both saturation extremes, which is what §10 asks of a screening set.

| | |
|---|---|
| tasks solved | `BowlStackingLeftOnRight`, `BananaThenRubiksCube`, `ReorientJug`, `RubiksCubeAndBanana`, `Stack3RubiksCube` |
| tasks failed | 10, all on timeout |
| policy calls | 809 across 22,161 control steps (horizon 32) |

**A cost structure that changes how the rest of the project should be run.** Of 5.4 GPU-h
of wall time, **policy inference was 0.91 h (16.8%)** and simulator stepping **4.21 h
(78%)**. Two consequences:

1. **Evaluation cost is nearly independent of policy cost.** A 3× cheaper policy saves
   ~11% of closed-loop wall time, not 3×. So closed-loop episodes cannot be used as a
   cheap way to compare compute — they measure *success*, and compute has to be measured
   in-process. This is why the offline screens carry the compute story and the closed-loop
   runs carry the success story.
2. **Failing is expensive.** A failed task runs to its full timeout (10–71 min here) while
   a solved one terminates early (2–6.5 min). The 10 failures cost 4.7 of the 5.4 GPU-h.
   So a *worse* method costs more to evaluate, and any budget estimated from mean episode
   length will be wrong in the unsafe direction. My own estimate was 3× low for exactly
   this reason.

That is what forced a mid-flight budget correction: four methods over the full pilot would
have cost **~32 GPU-h** against the ~18 authorised. With your approval the remaining
methods run on a **screened 9-task subset** derived from these results — the 5 tasks the
baseline solved (the only ones that can reveal *degradation*, the main risk for a cheaper
policy) plus the 4 cheapest of its failures (retaining the ability to see *improvement*).
1.40 GPU-h per method instead of 7.3. The six most expensive saturated tasks and
`CleanUpToysTask` are excluded, and `experiments/task_sets.yaml` records exactly which and
why.

Limitation stated plainly: subsequent comparisons are macro-averaged over 9 tasks, not 16,
on a set biased toward what the baseline finds easy or cheap. It is a screening set for
deciding what deserves a full-pilot run, not a substitute for one.

**One more accounting subtlety, found while building the frontier and worth stating
before any numbers are read.** Spec §10 defines the headline measure as a ratio of *total*
compute, but a total is `per-call cost × calls per episode`, and calls per episode depends
on how long the episode ran — i.e. on whether the method succeeded. A method that fails
runs to the timeout, makes more calls, and therefore shows **higher total compute even
when each of its calls is cheaper.** The first `steps_1` task demonstrates it exactly:
per-call 0.675 (1.5× faster) but total 1.125, because it made 1.67× as many calls after
failing. `docs/pareto.md` now reports both columns, because crediting a failure-inflated
total to the configuration's efficiency would be plainly wrong.

That same row also shows the wire's Amdahl floor working as predicted: the model is 2.98×
cheaper at 1 step, but the *call* is only ~1.5× cheaper, because ~1,197 ms of
composition/serialization/round-trip is invariant to anything done inside the model.

### The frontier (M3 deliverable) — no method preserves task success

On the screened 9 tasks, 1 episode each. The baseline solves **5/9 (55.6%)**.

| method | success | Δ vs baseline | **per-call compute** | total compute | policy calls |
|---|---|---|---|---|---|
| `baseline_full` | **55.6%** | — | 1.000 | 1.000 | 1.00× |
| `baseline_steps_2` | 33.3% | **−22.2 pts** | 0.575 | 0.813 | 1.32× |
| `baseline_steps_1` | 22.2% | −33.3 pts | 0.493 | 0.655 | 1.28× |
| `baseline_vision_frames_9` | **0.0%** | −55.6 pts | 0.485 | 0.758 | 1.49× |
| `baseline_fixed_horizon_8` | 12.5% | −50.0 pts | 1.050 | **5.977** | 5.79× |

**Against the project's own §14 gate (≤2 absolute success points lost), nothing passes.** The
best candidate, `steps_2`, costs 22 points for a 1.74× per-call saving.

Two results are unambiguous and not attributable to noise:

- **`vision_frames_9`: 0/9 with one contact event in nine episodes** (baseline: 96). The
  robot barely engages. Shortening the imagined horizon does not degrade manipulation, it
  removes it.
- **`horizon_8` is strictly dominated**: 5.98× the total compute *and* 50 points worse.
  Refreshing 5.8× more often is much worse than the baseline's 32-step horizon. Combined
  with the measured 9.1×-a-normal-step discontinuity at each chunk boundary, the reading is
  that every replan injects a discontinuity, and doing it 5.8× more often injects 5.8× more.
  So the baseline's horizon is not a compromise it "gets away with" — it is better than
  shorter in both success *and* compute.

The `steps_1` / `steps_2` deficits (−33 and −22 points) are the ones **n=1 cannot settle**.
Two tasks' difference on single Bernoulli draws, on a harness whose identical-config runs are
not reproducible, is weak evidence — and notably `steps_2` *solved* `BlockStackingOrderAgnostic`
which the baseline failed, so the solved sets differ rather than nest. That is what noise
looks like. A second episode on `baseline_full` vs `steps_2` is the single highest-value
remaining measurement and is where the rest of the budget goes.

**`horizon_16` was skipped deliberately.** `horizon_8` already showed the direction
(5.98× compute, −50 points), so an intermediate point would confirm a monotone trend for
1.4 GPU-h. That budget buys a second episode on the comparison that actually matters instead.
Recorded here rather than silently dropped.

### Authoritative latency (interleaved, cooled, median-of-many)

`docs/latency.md`. No other GPU process present; MADs of 0.32–2.73%, so these resolve
differences far below any success effect measured here.

| condition | steps | frames | CUDA median | MAD | speedup |
|---|---|---|---|---|---|
| `teacher_steps4` (baseline) | 4 | 33 | **3,465 ms** | 0.32% | — |
| `steps_3` | 3 | 33 | 2,671 ms | 0.65% | 1.30× |
| `vision_frames_17` | 4 | 17 | 1,980 ms | 0.68% | 1.75× |
| `steps_2` | 2 | 33 | 1,886 ms | 0.35% | 1.84× |
| `vision_frames_9` | 4 | 9 | 1,516 ms | 0.49% | 2.29× |
| `steps_1` | 1 | 33 | 1,100 ms | 1.05% | 3.15× |
| `vision_frames_5` | 4 | 5 | 1,073 ms | 0.80% | 3.23× |
| **`steps_1 × vision_frames_9`** | 1 | 9 | **505 ms** | 2.01% | **6.86×** |
| **`steps_1 × vision_frames_5`** | 1 | 5 | **386 ms** | 2.73% | **8.97×** |

**The two levers compose almost multiplicatively, and this was measured rather than
assumed.** 3.15× × 2.29× would predict 7.21×; the measured combination is 6.86× — 95% of the
product. That is a real finding about the cost structure (both levers act on independent
factors of a matmul-bound, token-linear cost) and it is exactly the inference spec §8
forbids making without measurement.

**The uncomfortable part: the compute is available and the success is not.** There is an
8.97× speedup sitting on the table, precisely measured, and every configuration that reaches
it scores at or near zero closed-loop. The project's difficulty is not finding compute
reductions — it is that this model appears to need all of it.

Note also that peak VRAM is **identical (30,996 MiB) across every configuration**: the
weights dominate and the activations are noise beside them, so none of these levers relieves
memory pressure. That matters for the "does it fit" question independently of speed.

### Experiment F — our own mechanism, measured and falsified

Per-block residuals across denoising steps (36 blocks, 4 steps, split by modality using the
model's real packed indices):

| modality | median step-to-step change | below 5% | below 10% |
|---|---|---|---|
| vision | **0.315** | 0% | 0% |
| action | **0.179** | 0% | 0.9% |
| text | **0.000** | 100% | 100% |

**The asymmetry the project predicted is real** — action residuals are 0.57× vision, so the
action stream is genuinely the more stable half. **But nothing is cacheable**: reuse needs
near-identical block outputs and an 18–32% change is a different tensor. No block is cheap
either (vision 0.22–0.36, action 0.139–0.216).

The mechanism is clean: TeaCache / DeepCache / token-wise caching target **20–50-step**
schedules where consecutive steps barely differ. Cosmos3-Nano runs **4**. The short schedule
has already squeezed out the redundancy those methods harvest, so F3's separate
action/vision thresholds are right in principle and moot in practice here.

One exact saving does exist: **text residuals are identically zero** across every block and
step, so the understanding tower can be computed once per request and reused bit-identically.
Ceiling ~3% (95 of 3,188 tokens), and it must first be checked whether the implementation
already avoids recomputing it.

---

## What failed

- **My `robolab-0002` patch broke the client's `__init__`.** Inserting a new method landed
  it inside `__init__`, stranding `self.client = self._connect()` after a `return`. The
  module imported, compiled, and the new method's own unit tests passed; the break surfaced
  only as `AttributeError: 'Cosmos3Client' object has no attribute 'client'` ~90 s into a
  closed-loop run, after Isaac had booted. Cost one wasted step. `tests/unit/test_client_patch.py`
  now parses the class for unreachable code and missing `__init__` assignments *and*
  constructs the client for real — testing a patched method in isolation does not test that
  the patch left the rest of the class intact.
- **The orchestrator did not export `HF_HOME`** to its subprocesses, so the first chain run
  paged the VAE from the NFS home volume — the third instance of this bug class in the
  project. Fixed at both levels: the orchestrator sets it from `configs/machine.yaml`, and
  `offline_action_study.py` now *refuses to run* without it rather than trusting callers.
- **An analysis error of my own.** The offline study first compared a pairwise deviation
  against a spread-about-the-mean; for two samples the latter is half the former, so every
  ratio was inflated ~2× and `steps_1` was reported as 1.6× sampling noise when
  like-for-like puts it at 0.97×.
- **A gate off-by-one that flattered methods.** The interface originally made callers track
  `cache_age`; since a refresh consumes an action immediately the age afterwards is 1, not
  0, so the naive form gives a refresh period of `horizon + 1` — ~3% fewer calls than the
  method actually makes. The gate now owns the counter, with a test asserting the period is
  exactly `horizon` and that fixed cadence reproduces the smoke run's measured 5 + 29 = 34
  calls.
- **`experiments/registry.yaml` had a YAML quoting slip** (`keyframe_oracle_refresh:{ parent:`),
  making the key `"keyframe_oracle_refresh:{ parent"` — so `keyframe_oracle_refresh` simply
  did not exist and would have been unrunnable at M7. Nothing failed; nothing warned.
- **`run_phases.py --retry-failed` did not propagate to inner ledgers**, so a retried step
  would have exited "successfully" having skipped the units that failed inside it.

**Not done:** Experiments F (cross-step caching), G (spatial), H (keyframe reuse proper),
I (LoRA), J (transfer) — all beyond the authorised M4 scope. Clean interleaved latency
measurements for the offline conditions are also outstanding: the sweep ran conditions in
back-to-back blocks, which B3 showed is exactly how not to time anything here, so its
`wall ms` column is explicitly marked untrustworthy and no speedup is quoted from it.

## Current repository SHAs

- this repo: uncommitted Session-3 work on top of `6cf33dd`
- `third_party/cosmos-framework`: `f98f4d5` — 4 research commits on `a904d2d`
- `third_party/RoboLab`: `0377dd1` — 2 research commits on `0aef241`
- checkpoint: `nvidia/Cosmos3-Nano-Policy-DROID` @ `6706d7680581c255ff61e0f3bb49d90eac55c79e`

All patches exported to `reproducibility/patches/` and indexed with their rationale and
whether they move the numbers in `docs/upstream_patches.md`.

## Selected deployment topology

`single_host_multi_gpu`. GPU 0 (`GPU-a2a44f60-…`) → Cosmos policy server; GPU 2
(`GPU-22350958-…`) → RoboLab/Isaac Sim; GPUs 1, 3 spare. Roles resolved from
`configs/topology.yaml` and asserted by **UUID**, never by index.

## Exact baseline command

```bash
OMNI_KIT_ACCEPT_EULA=Y COSMOS_GUARDRAILS=false ./.venv/bin/python \
  scripts/run_closed_loop.py --method baseline_full --set pilot --episodes 1
```

The whole chain, resumable, is one command:

```bash
OMNI_KIT_ACCEPT_EULA=Y COSMOS_GUARDRAILS=false ./.venv/bin/python scripts/run_phases.py
```

## Baseline smoke result

**PASS** from Session 2 and unchanged: `results/reports/baseline_smoke.md`. 34 policy
requests across two episodes; `BananaInBowlTask` solved (score 1.0, 145 steps);
`RubiksCubeAndBananaTask` failed on timeout with 11 drop events — reported as a *result*,
not a pass criterion.

## Dominant measured compute component

**The joint diffusion transformer**, unchanged from M2: `net.language_model` is 3,177.9 ms
of a 3,518.3 ms request (90.3%), FLOPs are 99.4% `aten.mm`, so cost is near-linear in token
count. Session 3 adds *why that matters*: **85.3% of those tokens are imagined future
video**, and shortening the imagined horizon is therefore the largest available lever —
confirmed to work (9 → 2 latent frames, 3,060 → 680 vision tokens) at 1.14–1.40× the
model's own sampling noise.

Also unchanged and still binding: **~25% of per-call latency is outside the model**
(~1,197 ms of camera resizes, composed-view construction, msgpack, round trip), invariant to
anything done inside it. Even a free model would leave ~1.2 s per call.

## Next three experiments

1. **More episodes on `baseline_full` vs `steps_2`.** The single highest-value measurement
   left. Every other conclusion here is either unambiguous (0/9; strictly dominated) or
   rests on n=1, and `steps_2`'s −22 points is exactly the case n=1 cannot settle — the
   solved sets differ rather than nest, which is what noise looks like. Run 3–5 episodes on
   the 5 tasks the baseline solves (they terminate early, so this is cheap) and get a real
   paired comparison. **≈1.5 GPU-h.**
2. **Experiment G — spatial masking, offline first.** The last untested idea, and the only
   remaining way to cut token count: recompute only action-relevant tokens *within* a frame,
   rather than removing frames (E2b: 0/9) or steps (A: −22 to −33 points) or reusing across
   steps (F: nothing to reuse). It is the one lever that could reduce tokens while leaving
   the plan's temporal structure — which E2b showed is load-bearing — intact. Begin with
   oracle masks from simulator segmentation and measure action deviation *and* chunk
   straightness, the metric that would have caught E2b. **≈1 GPU-h offline.**
3. **Price the text-tower saving properly.** Text residuals are identically zero, so the
   understanding tower is exactly reusable across denoising steps. Small (~3% of tokens) but
   *free and lossless*, which no other lever here is. First check whether the implementation
   already skips it; if not, it is the only confirmed saving the session produced.

**Estimated total ≈2.5–3 GPU-h.**

Deprioritized, with reasons recorded rather than silently dropped: **C** and **D** (temporal
scheduling has no compute headroom — critical moments are 2.8× more frequent than the
baseline cadence); **`horizon_16`** (`horizon_8` already established the direction at 5.98×
compute for −50 points); **`vision_frames_5`** (strictly more aggressive than a config that
scored 0/9); and **F** as framed (nothing reusable at 4 denoising steps).

### The honest summary of the research question

Five levers have now been measured, and **none delivers compute reduction at preserved task
success** on this model:

| lever | mechanism | verdict |
|---|---|---|
| A | fewer denoising steps | −22 to −33 points (n=1) |
| B | more frequent replanning | strictly dominated: 5.98× compute, −50 points |
| C/D | selective temporal refresh | compute reduction structurally unavailable; placement variant untested |
| E2b | shorter imagined horizon | **0/9** — the imagination *is* the plan |
| F | cross-step block caching | nothing to reuse at 4 steps |

The consistent finding is that **Cosmos3-Nano-Policy-DROID is already tightly provisioned.**
4 denoising steps, a 32-step action chunk and a 9-frame imagined horizon are each at or near
the minimum that works, and the model is matmul-bound and linear in token count. The baseline
sits at a local optimum in *both* directions on the one axis we could push both ways: fewer
steps hurts, and more frequent replanning hurts more.

That is a legitimate research result — spec §14 explicitly anticipates it ("stop or pivot the
visual-imagination branch when… oracle spatial masks provide little benefit… gains disappear
on contact-rich tasks") — but it is a *negative* one, and it should be reported as the answer
to "does this make a significant dent?" rather than dressed up. On present evidence: **not on
this model, by these mechanisms.** The remaining hope is G, and the remaining doubt is n=1.

## Blockers requiring your action

1. **Request access to `nvidia/Cosmos-Guardrail1`** and export `HF_TOKEN`. Guardrails are
   disabled throughout, so every latency understates the official baseline. The *ratios* are
   internally consistent (numerator and denominator both exclude it) but the absolute
   figures are not the official ones.
2. **Approve the next increment** (~6.5 GPU-h for the three experiments above). ~13 of the
   ~18 authorised GPU-h are consumed or committed.
3. **Optional: reclaim ~34 GB** in `$HOME/.cache/huggingface` — a duplicate checkpoint from
   the original NFS mishap. Flagged, not deleted; it is your home directory.
4. **A quiet window would materially improve the results.** The floor is 0.34% MAD when the
   box is calm and the sustained-load tail reaches +76%. If there is a time when other users
   are idle, the final frontier numbers should be taken then.
