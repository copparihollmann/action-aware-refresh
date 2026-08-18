# Compute anatomy (M2)

**Status: MEASURED** — generated from `results/profiles/m2-anatomy` by `scripts/write_compute_anatomy.py`. Do not hand-edit the tables; re-run the probe and regenerate.

## Provenance

- run_id: `anatomy-20260804T024233Z` at `2026-08-04T02:42:33Z`
- cosmos-framework: `7b2228018e80416959915aa8443bb17584450388`
- python 3.13.14 / torch 2.10.0+cu130 / CUDA 13.0
- GPU: NVIDIA L40S (sm89), CUDA_VISIBLE_DEVICES=None, uuid `GPU-a2a44f60-c7d1-4450-93df-ead875ea76e8`
- attention: arch_tag=89 → allowed backends `['flash2', 'cudnn', 'natten']`
  - **flash3 is unavailable on this arch**, so absolute latencies are not comparable to NVIDIA's published FlashAttention-3 figures. All claims below are within-machine deltas against our own baseline.
- warmup=3, measured iters=10 per config
- input: SYNTHETIC deterministic observation (seed 12345)
  - Either input is valid for **cost** (the sample builder zeroes every frame but the first regardless, so compute is shape-determined) but says nothing about **task success**, which only the closed-loop RoboLab run measures. If this says SYNTHETIC, see the input-validation section for the measured comparison against a real captured request.
- host contention at run time: loadavg [51.578125, 55.0498046875, 53.7998046875]
  - other GPU processes present: `943516, /scratch/kris/KernelBlasterRelease/third_party/stitchcuda/.venv/bin/python, 532 MiB
943718, /scratch/kris/KernelBlasterRelease/third_party/stitchcuda/.venv/bin/python, 500 MiB`

## End-to-end server-side latency per configuration

`infer()` wall time: preprocessing + generation + postprocess, in-process
(no websocket or client cost). Milliseconds.

| config | steps | decode | mean | std | p50 | p95 | n |
|---|---|---|---|---|---|---|---|
| `B0` | 4 | False | 3,518.3 | 30.0 | 3,522.7 | 3,559.7 | 10 |
| `B1` | 4 | True | 6,041.9 | 36.2 | 6,054.0 | 6,076.8 | 10 |
| `B2_steps_1` | 1 | False | 1,179.1 | 36.3 | 1,172.5 | 1,235.7 | 10 |
| `B2_steps_2` | 2 | False | 1,950.7 | 14.7 | 1,951.7 | 1,968.6 | 10 |
| `B2_steps_3` | 3 | False | 2,743.6 | 28.4 | 2,736.4 | 2,790.5 | 10 |
| `B2_steps_4` | 4 | False | 3,517.7 | 31.8 | 3,514.2 | 3,566.3 | 10 |

## What the VAE decode costs (B1 − B0)

B0 and B1 do **identical generation work** — the only difference is whether `samples["vision"]` is decoded to RGB. The delta is therefore the decode price:

- B0 (no decode): 3,518.3 ms
- B1 (decode):    6,041.9 ms
- **decode cost:  2,523.6 ms (41.8% of B1)**

This is the *only* visual cost that `decode_video=False` removes. The latent generation itself is still paid in B0 — see the step sweep for how much of B0 is denoising.

## Denoising-step sweep (B2)

| steps | mean ms | Δ vs 1 step | ms/step (marginal) |
|---|---|---|---|
| 1 | 1,179.1 | 0.0 | — |
| 2 | 1,950.7 | 771.5 | 771.5 |
| 3 | 2,743.6 | 1,564.5 | 792.9 |
| 4 | 3,517.7 | 2,338.6 | 774.1 |

Linear fit over 1→4 steps: **779.5 ms per denoising step**, with **399.6 ms of step-independent cost** (preprocessing, encode, context, postprocess — everything paid once per request regardless of step count).

So of the 3,517.7 ms at 4 steps, roughly 89% is denoising and 11% is fixed overhead. **This bounds what any denoising-reuse method (Experiment F) can possibly save**, and it is the frontier that reduced-step baselines already reach for free.

## Where the time goes, by module

Attribution from CUDA-event timing on real submodules — no invented action/vision split. Nested modules double-count against their parents, so read these as a tree, not as a partition summing to the total.

### `B0`

| module | mean ms | % of infer() |
|---|---|---|
| `sampler` | 3,219.7 | 91.5% |
| `net` | 3,199.7 | 90.9% |
| `net.language_model` | 3,177.9 | 90.3% |

### `B1`

| module | mean ms | % of infer() |
|---|---|---|
| `sampler` | 3,247.1 | 53.7% |
| `net` | 3,222.2 | 53.3% |
| `net.language_model` | 3,197.0 | 52.9% |

### `B2_steps_1`

| module | mean ms | % of infer() |
|---|---|---|
| `sampler` | 860.8 | 73.0% |
| `net` | 854.8 | 72.5% |
| `net.language_model` | 848.9 | 72.0% |

### `B2_steps_2`

| module | mean ms | % of infer() |
|---|---|---|
| `sampler` | 1,645.5 | 84.4% |
| `net` | 1,635.2 | 83.8% |
| `net.language_model` | 1,623.7 | 83.2% |

### `B2_steps_3`

| module | mean ms | % of infer() |
|---|---|---|
| `sampler` | 2,449.0 | 89.3% |
| `net` | 2,431.5 | 88.6% |
| `net.language_model` | 2,413.1 | 88.0% |

### `B2_steps_4`

| module | mean ms | % of infer() |
|---|---|---|
| `sampler` | 3,228.0 | 91.8% |
| `net` | 3,207.0 | 91.2% |
| `net.language_model` | 3,184.9 | 90.5% |

## Token census — what the sequence is actually made of

_Not captured. Re-run the probe (see `capture_packed_sequence`)._

## Token layout and tensor shapes

Captured from forward-hook inputs. These are the evidence for whether action and vision tokens share attention (spec §11.5).

| module | observed input shapes |
|---|---|
| `net.language_model` | `position_ids=(3, 3184)` |

## Memory

| config | after load (alloc / reserved MiB) | peak (alloc / reserved MiB) |
|---|---|---|
| `B0` | 30,285 / 30,306 | 30,932 / 31,052 |
| `B1` | 30,285 / 30,306 | 33,987 / 35,152 |
| `B2_steps_1` | 30,285 / 30,306 | 30,932 / 31,032 |
| `B2_steps_2` | 30,285 / 30,306 | 30,932 / 31,052 |
| `B2_steps_3` | 30,285 / 30,306 | 30,932 / 31,052 |
| `B2_steps_4` | 30,285 / 30,306 | 30,932 / 31,052 |

Budget: 46068 MiB total per L40S (45.0 GiB).

## FLOPs and coverage

### `B0`

- `FlopCounterMode` total: **347,074,311,248,384** FLOPs
- top ops:
  - `aten.mm`: 345,135,183,298,560
  - `aten.convolution`: 1,859,284,000,768
  - `aten.addmm`: 73,299,656,704
  - `aten._scaled_dot_product_efficient_attention`: 6,262,005,760
  - `aten.bmm`: 282,286,592
- FlopCounterMode counts only ops it recognises. Fused attention (flash-attn / cudnn fmha) is opaque to it, so this is a LOWER BOUND, not the true total. Compare against the analytic attention estimate and report coverage.

### `B1`

- `FlopCounterMode` total: **553,739,187,628,544** FLOPs
- top ops:
  - `aten.mm`: 345,135,183,298,560
  - `aten.convolution`: 208,459,928,547,328
  - `aten.addmm`: 73,299,656,704
  - `aten._scaled_dot_product_efficient_attention`: 70,493,839,360
  - `aten.bmm`: 282,286,592
- FlopCounterMode counts only ops it recognises. Fused attention (flash-attn / cudnn fmha) is opaque to it, so this is a LOWER BOUND, not the true total. Compare against the analytic attention estimate and report coverage.

### `B2_steps_1`

- `FlopCounterMode` total: **89,220,021,081,728** FLOPs
- top ops:
  - `aten.mm`: 87,336,079,589,376
  - `aten.convolution`: 1,859,284,000,768
  - `aten.addmm`: 18,324,914,176
  - `aten._scaled_dot_product_efficient_attention`: 6,262,005,760
  - `aten.bmm`: 70,571,648
- FlopCounterMode counts only ops it recognises. Fused attention (flash-attn / cudnn fmha) is opaque to it, so this is a LOWER BOUND, not the true total. Compare against the analytic attention estimate and report coverage.

### `B2_steps_2`

- `FlopCounterMode` total: **175,171,451,137,280** FLOPs
- top ops:
  - `aten.mm`: 173,269,114,159,104
  - `aten.convolution`: 1,859,284,000,768
  - `aten.addmm`: 36,649,828,352
  - `aten._scaled_dot_product_efficient_attention`: 6,262,005,760
  - `aten.bmm`: 141,143,296
- FlopCounterMode counts only ops it recognises. Fused attention (flash-attn / cudnn fmha) is opaque to it, so this is a LOWER BOUND, not the true total. Compare against the analytic attention estimate and report coverage.

### `B2_steps_3`

- `FlopCounterMode` total: **261,122,881,192,832** FLOPs
- top ops:
  - `aten.mm`: 259,202,148,728,832
  - `aten.convolution`: 1,859,284,000,768
  - `aten.addmm`: 54,974,742,528
  - `aten._scaled_dot_product_efficient_attention`: 6,262,005,760
  - `aten.bmm`: 211,714,944
- FlopCounterMode counts only ops it recognises. Fused attention (flash-attn / cudnn fmha) is opaque to it, so this is a LOWER BOUND, not the true total. Compare against the analytic attention estimate and report coverage.

### `B2_steps_4`

- `FlopCounterMode` total: **347,074,311,248,384** FLOPs
- top ops:
  - `aten.mm`: 345,135,183,298,560
  - `aten.convolution`: 1,859,284,000,768
  - `aten.addmm`: 73,299,656,704
  - `aten._scaled_dot_product_efficient_attention`: 6,262,005,760
  - `aten.bmm`: 282,286,592
- FlopCounterMode counts only ops it recognises. Fused attention (flash-attn / cudnn fmha) is opaque to it, so this is a LOWER BOUND, not the true total. Compare against the analytic attention estimate and report coverage.

**Coverage caveat, stated rather than hidden:** `FlopCounterMode` only sees ops it recognises. On this stack the attention path runs through flash-attn or cuDNN fused kernels, which it cannot decompose, so the counted total is a **lower bound**. Any FLOP-based claim must quote this coverage limitation, and per spec §8 a normalized-FLOP speedup claim is not acceptable on its own — measured latency has to confirm it.

## Measurement floor (B3 — deterministic replay)

Identical input, identical seed, repeated. B3 does not measure the model; it measures **us**. Everything below is only as trustworthy as this number.

| run | n | median | MAD | mean | std | min | max | drift |
|---|---|---|---|---|---|---|---|---|
| sustained (back-to-back) | 40 | **3,396.2** | — | 3,775.3 | 748.9 | 3,377.2 | 5,973.8 | — |
| **cooled** (idle gap) | 25 | **3,430.9** | 11.5 (0.34%) | 3,432.7 | 19.8 | 3,394.5 | 3,479.1 | +0.06% |

**The medians agree to 1.0%** (3,396.2 vs 3,430.9 ms) while the sustained run's maximum is 76% above its own median. So the model's cost is stable and reproducible; the sustained run's tail is an intermittent **stall**, not a shift in the work being done.

**Not thermal, and not GPU throttling.** Across the run `sm_clock` stayed at 1,065 MHz (drop 0.0%), temperature moved 69→73 °C, and throttle reasons observed: **none**. The SM clock is in fact *locked* well below this card's 2520 MHz maximum, so it cannot fall further under load. That leaves host-side interference — this box's CPU is shared and the GPU's affinity is a subset of cores — as the explanation, which is consistent with an idle gap making the tail disappear.

> **Minimum detectable effect: ~1.0%** (3x the steady-state MAD of 0.34%), provided the comparison is made the same way: **median-of-many with an idle gap between requests, configs interleaved rather than measured in separate blocks.** Measured that way this host resolves small effects well. Measured back-to-back it does not — the same work spanned 3,377.2–5,973.8 ms. Report medians, not means: the mean of the sustained run (3,775.3 ms) describes the stalls, not the model.

Two consequences carried into every later experiment: (1) per spec §8 a FLOP reduction never substitutes for measured latency, so a method whose only evidence sits under this floor has no evidence; (2) the **sustained-load** tail is the number a deployed server actually experiences, so it is reported alongside the steady-state figure rather than replaced by it.

## Input validation — synthetic vs real captured request

The in-process probe has to supply its own observation. To show that choice does not drive the numbers, the same config was re-measured with a request captured from a live closed-loop episode (`robolab-0001`).

- this sweep: SYNTHETIC deterministic observation (seed 12345)
- comparison: REAL captured request (/scratch/agustin/robotics/action-aware-refresh/results/raw/captured_request_BananaInBowlTask.npz)

| config | tokens | counted FLOPs | peak reserved MiB | wall mean ms | wall std ms |
|---|---|---|---|---|---|
| `B0` (this) | 3184 | 347,074,311,248,384 | 31,052 | 3,518.3 | 30.0 |
| `B0` (compare) | 3188 | 347,129,877,393,920 | 31,052 | 3,722.3 | 213.8 |

**`B0`:**

- joint sequence length 3184 → 3188 tokens (+4, +0.13%) — the prompt-length difference and nothing else; the image is the same 540×640 uint8 either way, because that is the server's own default and what the client sends.
- counted FLOPs 347,074,311,248,384 → 347,129,877,393,920 (+0.016%)
- peak reserved VRAM 31,052 → 31,052 MiB
- wall 3,518.3 → 3,722.3 ms (+5.8%)

The shape-determined quantities agree to well under a percent, so **the synthetic-input tables above are sound for cost.** The wall delta is larger than the FLOP delta by two orders of magnitude, so it is *not* caused by the input — it is host/GPU interference. Note the standard deviations: 30.0 vs 213.8 ms.

> **This wall delta is a measurement artefact, not an input effect.** These two runs were taken back-to-back in separate blocks, which the B3 section above shows is the one way *not* to compare on this host: identical work spans a wide range under sustained load, while a steady-state median is reproducible to a fraction of a percent. The FLOP and token deltas — which are immune to scheduling — are the trustworthy comparison here, and they agree to well under 1%.

## Closed-loop end-to-end cost (measured, not modelled)

From the M1 smoke run (`results/reports/baseline_smoke.json`). The tables above are **in-process** server latency; these are what the deployed loop actually pays, including image composition, msgpack serialization, the websocket round-trip and the simulator. Spec §8 requires the claim to be made at this level, not at the FLOP level.

| episode | steps | server calls | policy total | **per call** | env step total | video | wall |
|---|---|---|---|---|---|---|---|
| `BananaInBowlTask_0` | 145 | 5 | 44.9 s | **8,970.4 ms** | 32.4 s | 3.1 s | 80.4 s |
| `RubiksCubeAndBananaTask_0` | 900 | 29 | 136.7 s | **4,715.3 ms** | 216.8 s | 16.7 s | 370.3 s |

Per-call figures are not comparable across rows: the **first** episode against a freshly loaded server pays one-off warm-up (kernel autotuning, cuDNN benchmarking, first tokenizer pass) amortized over very few calls, which is why a short episode can show a much larger per-call number than a long one on the same server. Read the longest episode as the warm figure.

Server-side request count over the whole smoke run: **34** — consistent with one policy call per 32 control steps (`Cosmos3Client.OPEN_LOOP_HORIZON`), which is the baseline's *existing* action-chunk reuse. Any "skip frames" proposal must be framed against this, not against a per-control-step strawman.

**The wire is not free.** In-process `infer()` costs 3,518.3 ms (B0), but the client observes 4,715.3 ms per call on the longest episode — a gap of **1,197.0 ms (25% of per-call latency)** spent outside the model: three camera resizes, a torch `interpolate` + concatenate to build the composed view, msgpack encoding of a 540×640×3 array, and the round trip. This overhead is **invariant to denoising steps and to anything done inside the model**, so it is a hard floor under every method in spec §5.

For scale, in that episode simulator stepping cost 216.8 s against 136.7 s of policy inference (37% of 370.3 s wall). **The simulator is not part of a real deployment**, so it must be excluded from any speedup denominator — but it does mean simulated closed-loop wall-clock is a misleading proxy, and evaluation throughput will be dominated by Isaac rather than by the policy.

## Profiler traces

- `B0`: `/scratch/agustin/robotics/action-aware-refresh/results/profiles/m2-anatomy/B0.chrome_trace.json`
- `B1`: `/scratch/agustin/robotics/action-aware-refresh/results/profiles/m2-anatomy/B1.chrome_trace.json`
- `B2_steps_1`: `/scratch/agustin/robotics/action-aware-refresh/results/profiles/m2-anatomy/B2_steps_1.chrome_trace.json`
- `B2_steps_2`: `/scratch/agustin/robotics/action-aware-refresh/results/profiles/m2-anatomy/B2_steps_2.chrome_trace.json`
- `B2_steps_3`: `/scratch/agustin/robotics/action-aware-refresh/results/profiles/m2-anatomy/B2_steps_3.chrome_trace.json`
- `B2_steps_4`: `/scratch/agustin/robotics/action-aware-refresh/results/profiles/m2-anatomy/B2_steps_4.chrome_trace.json`

## Go / no-go — the primary question

> Is visual imagination a substantial part of the deployed Cosmos3 policy
> cost, or is the main cost shared/action computation?

_Hand-written; inlined verbatim from `docs/compute_anatomy_conclusion.md`. Edit that file, not this one — regenerating overwrites everything here._

**Answer: visual imagination is a large part of the cost — but the cost is
matmul over a mostly-vision token sequence, not a separable "vision module", and
the cheap baseline is already strong.**

The dominant component is **the joint diffusion transformer**, not the VAE and
not any auxiliary stage. In B0, `net.language_model` accounts for **3,177.9 ms of
3,518.3 ms (90.3%)** of in-process `infer()`, and the sampler for 91.5%.

1. **Dominant cost.** Joint DiT denoising: 90.3% of a 3,518.3 ms request. FLOPs
   are **99.4% `aten.mm`** (345.1 of 347.1 TFLOP) with attention negligible
   (6.26 GFLOP), so the model is **matmul-bound, not attention-bound**, and
   per-step cost scales roughly linearly in token count. That is what makes token
   *count* — i.e. how many vision tokens are in the joint sequence — the lever
   that matters, rather than attention tricks.

2. **Denoising vs fixed.** The step sweep is linear: **779.5 ms per denoising
   step + 399.6 ms step-independent**, so at 4 steps it is **~89% denoising, ~11%
   fixed**. That bounds every denoising-reuse idea (Experiment F).

3. **VAE decode.** **2,523.6 ms (+71.7% over B0)** and +4.1 GB VRAM — and
   `decode_video=False` already avoids it, so it is *not* available as a saving.
   It is the **only** visual cost that flag removes: `generate_samples_from_batch`
   produces `samples["vision"]` unconditionally, and B0 still pays for it.

4. **Separability (spec §11.5) — outcome C, and now measured.** The token census
   put **2,720 of 3,188 tokens (85.3%)** in imagined future video — 8 of 9 latent
   frames — against 32 predicted action tokens. `attn_modes=['causal','full']`
   confirms vision and action share one full-attention split with no mask, so
   there is no *configuration* switch that removes the imagination: outcome **C**,
   as predicted. But the required patch turned out to be far smaller than
   attention surgery. `video_length` is derived from the video tensor and the
   existing plan machinery already supports a temporally subsampled video against
   a dense action stream, so the imagined horizon is a knob
   (`cosmos-framework-0004`): 9 → 2 latent frames, 3,060 → 680 vision tokens.

5. **Verdict: the imagination is load-bearing, and shortening it does NOT work.**
   Experiment E ran offline on 88 real captured requests and then closed-loop on 9
   tasks, and the closed-loop result overturned the offline one.

   Offline, shortening the imagined horizon looked nearly free: **1.14–1.40× the
   model's own seed-to-seed sampling noise**, with gripper flips in 26% of requests
   at worst — *inside* the 24–32% the baseline already flips between seeds. On that
   basis I bought episodes.

   Closed-loop, `vision_frames_9` scored **0/9**, losing all five tasks the baseline
   solved, and produced **one** contact event across nine episodes against the
   baseline's 96. The robot was not mis-manipulating; it was barely engaging. A
   reference-free metric explains why: those chunks move at a normal per-step rate
   but their **straightness** (net displacement / path length) collapses to
   0.10–0.13 versus 0.30–0.39 for the teacher, seeds and reduced steps — a clean 2.3×
   gap with no overlap. The arm meanders instead of committing to travel, so over
   ~29 policy calls it never reaches the object.

   Mechanistically that is exactly what should happen, and it answers §11.5 more
   sharply than the offline data could: **the imagination *is* the plan.** A model
   that can see only 3 latent frames ahead commits to less displacement. The
   imagined future is not decoration whose resolution can be traded away — it is
   the lookahead the action chunk is derived from. Consistent with that,
   *destroying* the imagination (E1: tokens present but frozen at noise) is worse
   still — 3.07× noise, 100% gripper flips, straightness 0.026.

   **So the visual-imagination branch is not "cache and refresh it" and not
   "shorten it" either.** Both cheap levers examined so far lose task success:
   `vision_frames_9` 0/9 and `steps_1` 2/9 against the baseline's 5/9. Whether a
   *milder* reduction sits on the frontier is being measured now (`steps_2`,
   `steps_3`), and that is where the remaining hope lies.

## Three things that constrain every later claim

- **The cheap baseline is 2.98× — but it is NOT free, and the offline screen said
  otherwise.** 1 step costs 1,179.1 ms against 3,518.3 ms at 4, and across 88 real
  requests its action deviation is **0.97× the seed-to-seed sampling noise** with
  **zero** gripper disagreement. On that evidence it looked free. Closed-loop it
  scored **2/9** against the baseline's 5/9 — losing 3 of 5 solved tasks while
  producing *more* clumsy contact (44 drops vs the baseline's 17). So a perturbation
  no larger than the model's own sampling noise still destroyed task success, because
  open-loop deviation cannot capture compounding across 29–57 policy calls. Milder
  reductions (2 and 3 steps) are the remaining candidates.

- **Fewer policy calls is not on the table — two independent measurements say so.**
  Contact-critical steps are 8.9% of an episode against the baseline's 3.2% refresh
  rate, so an oracle covering them all would make 5.25× the calls
  (`docs/oracle_temporal.md`). And the action plan is already stale at a chunk
  boundary: the jump between consecutive chunks is **9.1× a normal intra-chunk
  step**, meaning re-planning materially changes the intention and extending the
  horizon would execute a stale plan. The compute win has to come from making each
  call cheaper, not from making fewer of them.

- **~25% of per-call latency is outside the model.** Warm closed-loop cost is
  ~4,715 ms per call against 3,518 ms in-process — roughly **1,197 ms** of camera
  resizes, composed-view construction, msgpack encoding of a 540×640×3 array, and
  the websocket round trip. It is invariant to denoising steps and to anything
  done inside the model, so it is a **hard floor**: even a free model would leave
  ~1.2 s per call. Amdahl applies to every number in spec §5.

- **The baseline already reuses action chunks.** `OPEN_LOOP_HORIZON = 32` means
  one policy call per 32 control steps (confirmed: 34 server requests for
  145 + 900 control steps). "Skip redundant frames" must be framed against this,
  not against a per-control-step strawman that the baseline never was.

## What this conclusion does not rest on

- **Not on synthetic input.** Re-measured with a request captured from a live
  episode: 3,184 → 3,188 tokens (+0.13%), counted FLOPs +0.016%, peak VRAM
  identical. The shape-determined quantities agree, so the tables stand.
- **Measurement discipline, now quantified (B3).** With an idle gap between
  requests this host is *precise*: identical work has a median of 3,430.9 ms and a
  MAD of 11.5 ms — **0.34%** — with +0.06% drift, so the minimum detectable effect
  is **~1%**, not the ~10% an earlier reading of the data suggested. Measured
  back-to-back instead, the same work spans 3,377–5,974 ms. The medians of the two
  runs agree to 1%, so the tail is an intermittent **host-side stall**, not a change
  in cost: `sm_clock` never moves (it is locked at 1065 MHz, well under the card's
  2520 MHz max) and no throttle reason is ever set. Hence the protocol for every
  comparison below: **median-of-many, idle gap, configs interleaved** — and report
  medians, since the sustained run's *mean* (3,775 ms) describes the stalls rather
  than the model. Sustained-load latency is still reported separately, because that
  is what a deployed server experiences.
- **Not on task success.** Everything above is cost. The only success evidence so
  far is the M1 smoke run: 1/1 on `BananaInBowlTask` (score 1.0, 145 steps),
  0/1 on `RubiksCubeAndBananaTask` (900 steps, timeout, 11 drop events). Two
  episodes is a wiring check, not a success rate.
- **Not on the official configuration.** Guardrails were disabled throughout
  (`nvidia/Cosmos-Guardrail1` is gated and access is denied), so every latency
  here **understates** the official baseline. See `docs/upstream_patches.md`.

