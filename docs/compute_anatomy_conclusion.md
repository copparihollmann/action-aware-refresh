<!--
Hand-written on purpose, and kept in a separate file on purpose.

`scripts/write_compute_anatomy.py` regenerates docs/compute_anatomy.md wholesale
from the probe summaries, so a conclusion written *into* that file would be
destroyed by the next `make profile`. It refuses to auto-generate this paragraph
because it gates the whole M4-M8 branch; this file is where the human answer
lives, and the generator inlines it verbatim.

Written 2026-08-03 from the B0/B1/B2 sweep; updated 2026-08-04 with the
closed-loop and real-payload evidence from the M1 smoke run.
-->

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
