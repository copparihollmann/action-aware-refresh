# Session 4 report — the group's efficiency repo: audited, ported, falsified

Format per spec §17. Host: **`firesim2`** (Ubuntu 24.04.4, 4× L40S, driver 580.126.18,
CUDA 13.0). GPU spend this session: **≈0.3 GPU-h**, which is exactly what was approved.

> **Redaction note.** This repo's `origin` is public, and the group's efficiency repo is
> private and carries no licence. Their measured values, internal file/line references,
> training-side knobs, checkpoint names and wording are therefore **not** reproduced here —
> they are in `private/` (untracked; see `private/README.md`). Everything below that is *our*
> measurement is complete and unredacted.

**Bottom line — the plan's premise was wrong, and the method it pointed at does not transfer.
Both were established without spending a closed-loop episode.**

The user obtained access to `chooper1/Cosmos3-Efficient-Imagination` — the repo the group
actually uses — and asked to base our work on it. Two things followed, in order:

1. **It cannot be our substrate, and its numbers are not our comparator.** It is a standalone
   repo that patches cosmos-framework from the outside; it contains no `cosmos_framework`
   package. It runs a **different benchmark on a different checkpoint** (their own simulator benchmark,
   Cosmos3-Nano + LoRA) than ours (RoboLab, `Cosmos3-Nano-Policy-DROID`). Their headline
   number and our 31.1% measure different things. The approved plan's central phase — "redraw
   the reference arm against their baseline" — is void.
2. **Their headline method collapses on our checkpoint.** Asymmetric video/action denoising
   (fewer video steps than action steps) was ported and measured offline: cosine similarity
   to the 4-step joint teacher is **≈0** in every asymmetric cell. Not degraded — unrelated.

The cause was predicted from source *before* the measurement and recorded in advance, so the
null is a finding rather than a surprise: `nano_model_config.py:90` sets
`independent_action_schedule=False` and `action_policy_droid_nano.py` never overrides it
(its only model edit is `loss_scale=10.0` at line 247). **Our checkpoint was trained with
video and action sharing one σ schedule** — the exact condition their sampler's own
documented caveat names as the open question, distinguishing graceful degradation
from hard collapse.

**Two byproducts are worth more than the verdict**, and neither was the goal:

- **The inference path is bit-deterministic** — the B3 noise floor owed since M2, obtained
  free as a control, and it is **zero**.
- **Cost tracks action steps, not video steps** — so this axis offers no speedup even in
  principle today, which is their own unimplemented-KV-cache caveat appearing as measured
  latency.

---

## What succeeded

### Phase 0 — acquire and audit (no GPU)

The user generated an SSH key and registered it; the private repo was cloned to
`third_party/cosmos3-efficient-imagination` and audited read-only into
**`private/upstream_fork_audit.md`** (untracked — see `private/README.md`), with
file/line citations rather than inference.

What the audit settled, and what each finding cost us or saved us:

| finding | consequence |
|---|---|
| Not a fork of anything we pin — standalone, patches cosmos-framework from outside via `framework_patches/`, monkeypatches its sampler | **Phases 1–3 of the plan are void.** No second substrate to install, no venv to build, no equivalence check to run. `configs/substrates.yaml` deliberately has **no** `efficient_imagination` entry; the file's comment records why |
| Different benchmark + checkpoint (their own simulator benchmark, LoRA) | **Their baseline cannot be our reference arm.** `baseline_official` stands as the comparator |
| Two of their four framework patches don't apply to NVIDIA upstream `a904d2d` — their training-priors patch wants a correlated-noise helper module VENDORED from a private training repo we lack access to; their eval/data patch misses by one line of context | The training-side work is unreachable *and* unnecessary to reach — verified that the private training repo is **not** the blocker (that helper appears in only two hunks, both for their env-gated correlated-noise training mode) |
| **No LICENSE file** at the repo root; per-file SPDX is MIT for 8 of 9 sampler modules | We **import** their clone rather than copying it. Per-file SPDX is recorded in every result row |

The audit also mapped their results onto our experiments, which **retired four of ours** —
each because it requires training we are not doing. Most instructive: our Experiment **G**
(spatial masking of the video block) was retired *as designed*, because their whole-block
mask measures 0.000 across 6 cells for a reason that applies to us too — the serving batch
has no `pixel_values`, so the video latent is the policy's **only** channel to the world.
Masking it blinds the robot rather than removing imagination. Full table in
`docs/decision_log.md` §2026-08-10.

### Integration — their samplers, without a fork (no GPU)

**`src/action_refresh/server/warm_start_vendor.py`** makes their `sampler/` directory
importable as the `warm_start` package their own code expects, by putting the directory on
`sys.path` (satisfying their flat imports) and registering a synthetic package whose
`__path__` is that directory (satisfying their package imports). **Nothing is copied.** Three
reasons, in order of weight: their repo grants no licence at the top level, so copying is the
one use that needs a grant we don't clearly have; this project's rule is preserve-and-patch,
not fork; and a copy goes stale silently whereas an import cannot.

One recorded patch was unavoidable:
**`private/patches/cei-0001-prepare-arity.patch`** (untracked) (branch
`research/action-aware-refresh` on their clone, `8433abd`) makes `prepare_inject`
arity-agnostic — our framework's `_prepare_inference_data` returns an 8th value
(`has_noisy_actions`) their base lacked. **A wrapper cannot fix this**: on a warm call they
install `prepare_inject` as an *instance* attribute as an instance attribute, which shadows any
class-level adapter. `apply()` asserts the marker is present and fails with the remedy in the
message, because without it the failure is a bare unpacking `ValueError` three frames deep in
upstream code.

One property of their design we exploited: `WarmStartConfig.from_env()` is read **per call**, so an entire (V, A) grid sweeps in **one** model load.

### The measurement — offline V×A grid (~0.3 GPU-h, approved)

**`scripts/offline_split_schedule.py`** + **`scripts/analyze_split_schedule.py`**. Their
`KNFrozenVideoSampler` in NO-PRIOR mode (`KNFV_NO_PRIOR=1`,
`KNFV_VIDEO_ENTRY_SIGMA=1.0`, rolling off — no cross-call state, which is what makes
single-request replay faithful): the video denoises cold in K steps, then freezes for the
remaining N−K action steps. Deviation is against the 4-step joint teacher — the sampler that
produced `baseline_official` (28/90 = 31.1%). Stratified, 4 requests from each of 4 tasks,
all 16 cells verified engaged via their `[knfv]` trace.

| cell | V | A | joint L2 (rad) | linf | gripper | **cosine** | CUDA ms |
|---|---|---|---|---|---|---|---|
| teacher repeat | 4 | 4 | **0.0** | 0.0 | 0.000 | **1.000** | 7,402 |
| knfv V4/A4 | 4 | 4 | 0.052 | 0.078 | 0.000 | 0.460 | 7,721 |
| knfv V2/A4 | 2 | 4 | 1.460 | 1.46 | 0.031 | −0.028 | 7,650 |
| knfv V1/A4 | 1 | 4 | 1.695 | 1.60 | 0.031 | −0.060 | 7,335 |
| knfv V1/A8 | 1 | 8 | 1.804 | 1.74 | 0.000 | 0.055 | 15,686 |

**Verdict: COLLAPSE**, 3/3 asymmetric cells. Uniform across tasks — V1/A4 cosine per task is
−0.060 / −0.028 / −0.034 / −0.099, so it is not a task-specific artifact.

**Why this is collapse and not degradation.** Cosine ≈ 0 means the emitted chunk is
*unrelated* to the teacher's, not a noisier version of it. The σ_a floor argument predicted
the same outcome independently: action σ is sampled `logitnormal`, which puts a negligible fraction of
training mass at low σ, and their own floor bound therefore predicts that large action-step counts visit σ
values the stock checkpoint never saw. Widening that floor is training.

**Their mechanics reproduce**, which is what licenses attributing the collapse to the
checkpoint rather than to our wiring: V4/A4 — their K=N "mechanics proof" — measures 0.052
here against the value they document. Same order, different checkpoint, different benchmark.

Artifacts: `results/processed/split_schedule{,_stratified}.jsonl`,
`results/reports/split_schedule{,_stratified}.json`.

### Two findings worth more than the verdict

**1. The pipeline is bit-deterministic.** The stock-repeat control measured *exactly* 0.0
deviation and cosine 1.000 on all 16 requests: `deterministic_seed=True` holds for the whole
inference path, not merely the seed. **This is the B3 noise floor the plan has owed since
M2.** It is zero, so offline A/Bs need no minimum-detectable-effect threshold at all and any
nonzero offline deviation is signal. It came free, as a control we ran for a different reason.

**2. No speedup exists on this axis today, even in principle.** Cost tracks **A**, not V:
V1/A16 measured 30.4 s against the teacher's 7.3 s — **4× slower** — and V1/A4 measured
7,335 ms against 7,402 ms, indistinguishable. Their report flags the frozen-video KV cache
as *not yet implemented* (recorded in their own write-up); this is that caveat as measured latency, and
it is exactly the claim this project's rules forbid making from FLOPs alone. Had we trusted
the ~99% FLOP figure, we would have reported a 99% saving for a configuration that runs 4×
slower.

### Supporting infrastructure (no GPU)

- **Substrate abstraction** — `configs/substrates.yaml` + `resolve_substrate()` in
  `src/action_refresh/config.py`, precedence explicit arg > `$COSMOS_SUBSTRATE` > `default`.
  Location from the manifest, **live SHA from git**, because the manifest was found 4 commits
  stale; `manifest_stale` surfaces that drift rather than hiding it. Threaded through
  `scripts/lib/common.sh`, `start_cosmos_server.sh`, `cosmos_server_entry.py`,
  `setup_cosmos.sh` (`COSMOS_SRC`, `cosmos_install.json` now keyed by source),
  `run_anatomy_sweep.sh`, `Makefile`. Built for a second substrate that turned out not to
  exist; kept because it removes three hardcoded paths and makes provenance explicit.
- `scripts/update_manifest.py` gained `PHASE_HINTS`/`NOTE_HINTS`/`discover_unregistered`, so
  a hand-cloned private repo gets registered by regeneration rather than by hand-editing the
  manifest — which its own header forbids, and which I did once and had to unwind.
- `scripts/clone_sources.sh`: `GIT_TERMINAL_PROMPT=0` and a `MANUAL_REPOS` list, so a private
  repo fails fast instead of sitting on a credential prompt.
- 15 new substrate tests, incl. `test_live_commit_wins_over_a_stale_manifest`,
  `test_missing_venv_refuses_to_borrow_another`, and
  `test_the_efficiency_repo_is_not_registered_as_a_substrate` — which pins the audit's central
  conclusion in the test suite so a future session cannot quietly re-add it.

---

## What failed

- **The approved plan, at its premise.** Phases 1–3 (install their fork as a substrate, rebase
  our patches onto it, prove substrate equivalence) were all predicated on it being a fork of
  cosmos-framework. It isn't. `scripts/validate_substrate.py` was written before the audit
  landed and is now a harness with nothing to compare; it is kept, unrun, and honest about
  that.
- **Their headline method, on our checkpoint.** Above. This closes the direction chosen on
  2026-08-10, ~5 days after choosing it, for ~0.3 GPU-h.
- **Retraining, priced and declined.** hundreds of GPU-h per mode (from their
  their internal throughput sweep — their README and sweep file
  disagree on gradient accumulation, hence a range), no training data on this host (the 34 GB
  HF cache is models only), and very large non-pruned checkpoints. The user chose to request a
  trained checkpoint instead.
- **Three of my own errors, each caught and recorded** rather than left in the record: a false
  alarm that the sampler had not engaged (their `[knfv]` traces are gated behind
  `KNFV_DEBUG=1`, so my detector was blind and the numbers had been valid all along);
  a coverage misreport (`--limit 16` drew all 16 requests from one task because the corpus is
  task-sorted — `--per-task` was added and the stratified re-run confirmed the result); and an
  analyzer that dropped a column because `if floor_med:` treats the ideal floor of `0.0` as a
  missing value.

---

## Current repository SHAs

| source | branch | commit |
|---|---|---|
| `cosmos-framework` | `research/action-aware-refresh` | `f98f4d5` |
| `RoboLab` | `research/action-aware-refresh` | `0377dd1` (dirty) |
| `cosmos3-efficient-imagination` | `research/action-aware-refresh` | `8433abd` (main at `ecb8de4`; **no LICENSE**) |
| `cosmos` | `research/action-aware-refresh` | `404b9bf` |
| `cosmos-policy` | `research/action-aware-refresh` | `18a2acc` |
| `openpi` | `research/action-aware-refresh` | `15a9616` |

Our repo: `main` @ `6cf33dd`, with **2,294 files staged and uncommitted** (+111,023/−525).

**A provenance defect to fix before any further result cites their commit:**
`warm_start_vendor.locate()` reads `commit` from the manifest, so this session's result files
record `vendor_commit=ecb8de4` while the code that actually executed was `8433abd`
(= `ecb8de4` + the arity patch). Same class as the substrate staleness fixed above; the
vendor path never received the fix.

---

## Selected deployment topology

Unchanged: GPU 0 → Cosmos server, GPU 2 → RoboLab/Isaac (separate PCIe switches), GPUs 1 and
3 spare, pinned via `configs/topology.yaml` with the recorded UUID asserted. Attention backend
on SM 8.9 is `flash2` (`flash3` is Hopper-only), recorded in every result.

**Not available as of 2026-08-17:** all four GPUs hold ~30.7 GB from one other user's
process, leaving ~14.5 GB free against a 32.9 GB checkpoint. This is a capacity blocker, not
contention — see `docs/handover.md` §2.

---

## Next three experiments

1. **`cold-N` priced properly** — symmetric denoising-step reduction, the one lever that stays
   *in distribution* for our checkpoint. Normalized FLOPs + measured GPU time + end-to-end
   latency + energy + task success against the now-known zero noise floor. Per the group's
   decomposition, our M2 "2.98× at 1 vs 4 steps" should be relabelled `cold-N` rather than
   read as two independent axes. **Report the GPU-h estimate and get approval first.**
2. **Re-measure `docs/latency.md` in a quiet window.** Owed since M2; the box has never been
   idle when we were ready.
3. **Cross-call latent reuse** — their prior-drift gate, which has no eval
   cells behind it in their repo, and which is the closest thing to this project's original
   thesis. Carry this session's caution into it: reusing a *previous* imagination is a
   strictly larger perturbation than freezing the current one, and freezing alone already
   collapsed the action.

---

## Blockers requiring your action

1. **Send the checkpoint request** (drafted, `private/outbound_checkpoint_request.md`).
   If it arrives, note the unscoped prerequisite: evaluating
   a trained for their benchmark checkpoint needs the **their benchmark's eval stack** stack, which this host does not
   have, and their LoRA checkpoint cannot be evaluated on RoboLab/DROID instead.
2. **Authorize `git revert --no-edit 7b22280 2257031`** in `third_party/cosmos-framework` —
   blocked three times by the permission classifier. Until then `docs/upstream_patches.md`
   keeps its WITHDRAWN sections.
3. **Rule on task #18** — `make contract` would overwrite the 239-line hand-verified
   `docs/baseline_contract.md` with a 25-line stub whose regexes yield wrong values
   (`port_default: 3`). Pre-existing; I wrote to a scratch path to confirm and filed it rather
   than widening scope.
4. **The GPU-hour overage** — ≈30 cumulative against ~18 approved, flagged in session 3 and
   still yours to rule on. Session 4 added 0.3.
5. **Consider committing the staged tree.** Sessions 2–4 exist only in one machine's git
   index.

---

## A note on measurement conditions

`loadavg ≈ 33` throughout, from other users' parallel compiles; GPU CPU affinity is
`0-15,32-47`. This session's numbers are *deviation* measurements, which are unaffected — the
pipeline is bit-deterministic, which we now know rather than assume. The CUDA-ms column is
corroboration only and was taken under that load; the V1/A16-versus-teacher comparison
survives it because a 4× gap is far outside any plausible load effect, but it should not be
quoted as an absolute latency. `docs/latency.md` remains the only home for absolute numbers,
and it still needs a quiet window.

`/scratch` fell from 328 GB free to **49 GB (99% full)** during this interval, from other
users — our footprint is unchanged at ~80 GB. No install may proceed without re-checking
`df -h /scratch` and aborting rather than filling a shared volume.
