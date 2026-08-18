# Upstream patches

`CLAUDE.md`: *"Preserve upstream repos — modifications land as local branches
under `third_party/*/research/action-aware-refresh` and are exported to
`reproducibility/patches/`."* This file is the index of those patches: what each
one changes, why it was unavoidable, and what it means for the numbers.

Apply order matters (they are `git am`-able in numeric order):

```bash
for p in reproducibility/patches/cosmos-framework-*.patch; do
  git -C third_party/cosmos-framework am "$p"
done
for p in reproducibility/patches/robolab-*.patch; do
  git -C third_party/RoboLab am "$p"
done
```

Base SHAs are recorded in `reproducibility/source_manifest.json`
(`cosmos-framework` @ `a904d2d`, `RoboLab` @ `0aef241`).

## Do these patches invalidate the baseline?

No. Three categories, and — after the 2026-08-04 audit below — **none** of them changes
the baseline's numbers.

**Withdrawn: no longer needed** (`0002`, `0003`). Both skipped NCCL collectives that
segfault at single rank. They are replaced by `scripts/cosmos_server_entry.py`, which
pre-creates the one-rank process group with **gloo** so upstream's collectives *execute*,
unmodified. Measured equivalent: bitwise-identical actions. See the sections below.

**Off by default / additive** (`0004`, `robolab-0001`, `robolab-0002`). Each defaults to
upstream behaviour exactly, so the official configuration is byte-identical. They add
capability the experiments need — a shortened imagined horizon, request capture, a
configurable open-loop horizon — and each one is *recorded per run* so no result is
ambiguous about which of them was active.

**Additive, and now shown not to affect measured cost** (`0001`, guardrails). We reported
for three sessions that "every latency understates the official baseline" because
guardrails are disabled. **That was wrong**, and the correction is in the `0001` section:
the guardrail runners are never invoked on the RoboLab policy-server code path.

## Audit, 2026-08-04: which deviations are actually closable?

Prompted by "are you sure all of those are not closable?", every recorded deviation was
re-examined against the source rather than against our own notes. Result:

| deviation | verdict |
|---|---|
| `--no-guardrails` | **immaterial** — runners exist but are never called on this path (proof below). Repo is `gated: auto`, so access is one click, but it would change no number |
| `0002` / `0003` NCCL skips | **closed** — replaced by a gloo pre-init; upstream code runs unmodified, actions bitwise identical |
| `--no-decode-video` | **not a deviation** — `decode_video: bool = False` *is* the upstream default |
| `--hf-revision <sha>` | **not a behavioural deviation** — the cached `main` ref resolves to exactly the SHA we pin |
| `--num-envs 1` | **closed** — the official recipe is `--num-envs 10`; adopted, and 4.5× cheaper per episode |
| `flash3` unavailable | **not closable** — `get_backend_list` gates `flash3` on `arch_tag == 90` (Hopper); this host is sm89 |
| 9–16 tasks vs the published 120 × 10 | **closable for GPU-hours** (~41 GPU-h at `--num-envs 10`), not for free |

The two genuinely-open items are the task count (a budget question, not a technical one)
and `flash3` (hardware). Everything else was either already vanilla or has been made so.

---

## cosmos-framework-0001 — additive `--guardrails` / `--no-guardrails`

**Category:** deviation from official. *Does **not** affect measured cost — corrected
2026-08-04.*

Upstream leaves `OmniSetupArgs.guardrails` at its default `True` and gives the
RoboLab policy server no way to override it, so starting the server
unconditionally downloads **`nvidia/Cosmos-Guardrail1`**, whose weights we cannot fetch.
The official server therefore cannot start, which blocks M1 and M2 outright.

The flag is strictly additive and **defaults to `True`**, so the official
configuration is byte-identical to upstream. `scripts/start_cosmos_server.sh`
passes `--no-guardrails` by default (`COSMOS_GUARDRAILS=false`) purely so the
smoke test can run.

### Correction: the guardrail runners are never invoked on this path

For three sessions this file claimed *"every latency and energy number taken with
`--no-guardrails` excludes guardrail cost, and the guardrail runners are real deployed
cost."* Reading the call graph shows that is **false** for the RoboLab policy server:

- `_run_text_guardrail` / `_run_video_guardrail` are called only from
  `OmniInference.generate_batch` and `_generate_reasoner_batch`
  (`cosmos_framework/inference/inference.py`) — the *sample-generation script* entry
  points, which take a `sample_args.output_dir`.
- `RobolabPolicyService.infer()` calls
  `self.model.generate_samples_from_batch(...)` directly
  (`cosmos_framework/model/generator/omni_mot_model.py:2686`), bypassing `generate_batch`
  entirely. `grep -rn guardrail cosmos_framework/model/` returns **zero** hits.
- The video guardrail could only see a decoded rollout, and `decode_video` defaults to
  `False`.

So enabling guardrails would add a startup download and resident VRAM — on a card with
~14 GiB of headroom against a 32.9 GB checkpoint, that is a real constraint — and
**zero per-request cost**. No latency, energy or success number in this project is
affected by the flag.

### Access status (2026-08-04)

The HF API reports `gated: "auto"` for `nvidia/Cosmos-Guardrail1` — *auto-approved*
click-through, not a pending human review. The cache holds only its `README.md` and
`blocklist` (80 KB), so no weights were ever fetched.

**Still a user action, and still a small one:** accepting the terms is a license
acceptance, which `CLAUDE.md` forbids us from doing on your behalf. But it is one click
with automatic approval, and — per the correction above — it would change no measurement.
Its only value is being able to say the runners were resident.

## cosmos-framework-0002 — skip `_download_on_rank0` collective at `world_size == 1`

**Category:** upstream bug workaround. **WITHDRAWN 2026-08-04** — superseded by
`scripts/cosmos_server_entry.py`.

With an initialized NCCL process group of one rank,
`broadcast_object_list()` **segfaults** inside `c10d::broadcast` on this stack
(torch 2.10.0+cu130, L40S sm89), so `OmniInference.create()` never completes and
the policy server cannot start.

At `world_size == 1` there is no peer to share the resolved path with, so the
broadcast is pure overhead; returning the locally downloaded path is
semantically identical. Multi-rank behaviour is unchanged.

## cosmos-framework-0003 — `sync_model_states` no-op at `world_size == 1`

**Category:** upstream bug workaround. **WITHDRAWN 2026-08-04** — superseded by
`scripts/cosmos_server_entry.py`.

Same failure class, different call site:
`_verify_param_shape_across_processes()` / `_sync_module_states()` segfault
inside NCCL at single rank, so the VAE tokenizer cannot be constructed. With one
rank there is no peer to synchronize *from*, so the early return is semantically
identical. Multi-rank behaviour is unchanged.

## Why 0002 and 0003 were withdrawn

`maybe_init_distributed()` (`action_policy_server_utils.py:50`) deliberately builds a
one-rank **NCCL** group when the server runs outside `torchrun` — the documented
standalone path — *and returns early if a group already exists*. That early return is the
opening: create the group ourselves with **gloo** and upstream's collectives run
unmodified rather than being skipped.

Evidence, in order of cost:

1. **A 40-line probe with no model** (`scripts/validate_pg_backend.py`, docstring)
   confirmed the mechanism: at `world_size=1`, NCCL dumps core on
   `broadcast_object_list`, while gloo completes `broadcast_object_list`,
   `_verify_param_shape_across_processes` **and** `_sync_module_states`.
2. **Four timed arms** (`results/processed/pg_backend_ab.jsonl`) — patched/NCCL vs
   reverted/gloo, replicated — with the reverted tree supplied via a `PYTHONPATH` shadow
   of a git worktree, so the comparison was against genuinely unpatched source:

   | arm | wall median | MAD |
   |---|---|---|
   | patched + NCCL | 2368.7 ms / 2477.6 ms | 0.97% / 0.63% |
   | reverted + gloo | 2434.8 ms / 2603.5 ms | 0.71% / 1.68% |

   Between-instance spread *within* an arm is 4.6–6.9%, which exceeds the ~2–4% apparent
   difference between arms: **no resolvable latency effect**, taken while GPU 0 was busy
   with a closed-loop run. The final `docs/latency.md` numbers must still be re-measured
   under the adopted configuration in a quiet window.
3. **Bitwise-identical output.** With `--deterministic-seed` and one fixed request, the
   full action array digests to `5c01880496f1f66619959ad0c845f6d2dcb2baca043234b446018f1a54ec958c`
   under **both** routes. That is what makes this a free change: no prior result needs
   re-running.

Why gloo is the right backend and not a new risk: `sync_model_states` is called only from
the VAE tokenizer constructors, so its collectives are a *startup* self-copy at
`world_size 1`; nothing on the per-request path performs a collective. Backend choice
therefore cannot reach the hot path — which is what the timing above confirms rather than
assumes. `ensure_single_rank_group` refuses to act at `world_size > 1`, so this reasoning
cannot leak into a multi-GPU run.

**Trade-off, stated plainly:** upstream on working hardware uses NCCL, so the backend
still differs from theirs. What changes is *which kind* of deviation we carry — one line
of our own setup code, instead of two edits inside upstream functions. Given the ask
("use the vanilla version so we can be sure everything works as it should"), unmodified
upstream source is worth more than a matching backend name for a group that has no peer
to talk to.

`COSMOS_PG_BACKEND=none` restores the old route for anyone who wants it; the patches
remain exported in `reproducibility/patches/` and are required in that mode.

## cosmos-framework-0004 — `--vision-frames`, shortening the imagined horizon

**Category:** research capability (Experiment E2b / H). *Changes measured cost — that is
the point.* Defaults to upstream behaviour.

The token census found **85.3% of the 3,188-token sequence is imagined future video** (8
of 9 latent frames), while the deliverable is 32 action tokens. The model is matmul-bound
(99.4% of counted FLOPs are `aten.mm`) and therefore near-linear in token count, so
shortening the imagined horizon is the largest lever the compute anatomy identified.

**No attention or sequence-plan surgery was needed**, which was the pleasant surprise.
The action transform derives `video_length` from the video tensor and `action_length` from
the action tensor, and `build_sequence_plan_from_mode` already supports a temporally
subsampled video against a dense action stream (its "Case C"). That requires
`(action_length - 1) % (video_length - 1) == 0`, so with 33 action steps the valid frame
counts are **33, 17, 9, 5, 3 → 9, 5, 3, 2, 1 latent frames**, i.e. 3,060 → 340 vision
tokens (up to 6.8× fewer). Anything else is rejected at startup rather than silently
falling into a different alignment case — a misaligned plan yields plausible-looking
actions from a layout the actions do not correspond to, which is the worst failure mode
available for an experiment about action quality.

`conditioning_fps` is scaled with the frame count, because a shortened video is a
subsampled view of the *same* horizon. This was not cosmetic: the caption augmentor
computes `duration = int(num_frames / fps)`, so 9 frames at 15 fps produced **"The video
is 0.0 seconds long"** — a degenerate prompt that would have confounded the experiment
with a text change. fps is consumed as an integer, so it is rounded and floored at 1 (at
fps 0 the augmentor drops the metadata entirely, changing the prompt a different way).
Net effect: **17/9/5 frames keep the caption byte-identical** to the baseline's "2.0
seconds long", varying only the frame count. 3 frames shifts it to "3.0 seconds" and is
therefore excluded from the sweep.

## robolab-0001 — env-gated capture of one real policy request

**Category:** research instrumentation. *No effect on the eval path.*

The M2 compute-anatomy probe drives the model **in-process** (the server exposes
no per-stage timings, and `num_steps`/`decode_video` are server-*start* args),
so the probe has to supply an observation itself. Hand-authoring one risks
profiling a payload the closed loop never sends — wrong dtype, wrong composed-
view geometry, or a prompt that tokenizes to a different length, any of which
silently changes the token count and therefore the cost attribution the whole
milestone rests on.

So the patch dumps the *first genuine request* built by
`Cosmos3Client._pack_request` during a live RoboLab episode.

Off by default: nothing happens unless `ACTION_REFRESH_CAPTURE_REQUEST` names an
output path. It writes at most once per process and swallows its own exceptions
— a debug artifact must never be able to fail an eval run. `smoke_test.sh` sets
it for the primary task only.

## robolab-0002 — capture every request, and a configurable open-loop horizon

**Category:** research instrumentation + capability. *No effect on the eval path.* Both
halves default to upstream behaviour exactly.

**Capture-all.** The single-request capture from `robolab-0001` cannot support the offline
screens it feeds: a denoising-step sweep measured on one frame says nothing about how
deviation grows through a grasp, and an E0 seed-spread on one observation has no
across-observation scale to be compared against. Passing a *directory* (rather than a
`.npz` path) now captures every request, zero-padded so lexical order matches temporal
order — `req9` sorting before `req10` would silently reorder the trajectory. One episode
yields ~24–30 requests for the price of one closed-loop run; the corpus used for M3 is 88
requests across 4 tasks.

**Configurable `open_loop_horizon`.** Experiment B (spec §11.2) sweeps how many returned
actions execute before the next policy call, but upstream hard-codes 32. Without this, the
registry's `baseline_fixed_horizon_8/16/32` entries would all have run at 32 while being
recorded under different names — a fabricated comparison. The override is validated and
range-checked against the chunk the server actually returns (a larger horizon would index
past the chunk and needs an explicit extrapolation rule, not a bigger index), and it
**raises rather than falling back**: a typo must not silently yield a baseline run under
an experimental label.

---

## Not patched, deliberately

- **`openpi-client`** is imported by `policies/cosmos3/client.py` but is not a
  declared RoboLab dependency (upstream keeps it out so non-openpi backends
  don't pull it in, and vendors `image_tools` in
  `robolab/core/utils/image_utils.py` for the same reason). We install the
  published wheel into the RoboLab venv from `setup_robolab.sh` via
  `uv pip install` — **not** `uv add` — so upstream's `pyproject.toml` and
  `uv.lock` stay clean. Pinned to the version the server side resolved, since
  both ends must agree on the msgpack wire format.
- **The `isaac50` extra choice** is upstream's own documented default; no patch
  needed. `isaac50` and `isaac51` are declared `conflicts` — never install both
  into one venv.
