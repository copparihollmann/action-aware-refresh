# Handover — read this before touching anything

Written 2026-08-17, at the end of session 4. Audience: a fresh agentic session (or a human)
picking this project up cold. Everything here is either measured on this host or cited to a
file; where a number is an estimate it says so.

Companion documents, in the order worth reading:

| read | for |
|---|---|
| `docs/research_spec.md` | the research question and the non-negotiable rules (§5 methods, §14 gate, §17 operation) |
| `CLAUDE.md` (repo root) | environment invariants and the constraints that override default behaviour |
| `docs/decision_log.md` | the full chronological record, including every correction. **1,300+ lines — the last two sections are the current direction** |
| `results/reports/session_{1,2,3,4}_report.md` | per-session deliverables per spec §17 |
| this file | state, blockers, what you may and may not do next |

> **Redaction note.** This repo's `origin` is public, and the group's efficiency repo is
> private and carries no licence. Their measured values, internal file/line references,
> training-side knobs, checkpoint names and wording are therefore **not** reproduced here —
> they are in `private/` (untracked; see `private/README.md`). Everything below that is *our*
> measurement is complete and unredacted.

---

## 1. The 60-second orientation

**The question.** Can end-to-end inference compute of a pretrained world-action model
(`nvidia/Cosmos3-Nano-Policy-DROID` served to RoboLab) be cut by reusing keyframes,
imagination latents, action chunks and denoising intermediates — refreshing only what is
action-relevant — while preserving closed-loop task success? The deliverable is a
**task-success-versus-compute Pareto frontier**. The robot action is the product; images are
diagnostic only.

**Where it stands.** The compute side is answered and the success side is largely
*unmeasurable at feasible cost*, and that asymmetry is the central finding of the project so
far:

- **The compute is there and is cheap to take.** 85.3% of the model's sequence is imagined
  future video. Cutting denoising steps and imagined frames together measures **8.97×**
  (386 ms vs 3,465 ms, `steps_1 × vision_frames_5`, ±2.7%).
- **Only one success claim in the project is statistically established**, and it is a
  negative one: shortening the imagined horizon (`vision_frames_9`) gives 0/9,
  p = 0.026 — it *breaks* the policy.
- **The binding constraint is evaluation variance, not method quality.** The baseline
  disagrees with itself on 3 of 9 tasks between two seeds. Resolving the ~11-point
  `steps_2` difference needs ≈31 episodes/task/arm ≈ **87 GPU-h for one pairwise
  comparison**. The spec's §14 gate of ≤2 absolute success points is **finer than the
  official benchmark's own 10-episode protocol can resolve**.
- **The setup is verified, not assumed.** Our baseline reproduces the published leaderboard
  number: 33–39% against NVIDIA's 36.8%, where Cosmos3-Nano-Policy is the *top-ranked*
  RoboLab-120 entry. `baseline_official` = **28/90 = 31.1%**, mean score 0.502, 5.95 GPU-h
  (`results/reports/baseline_official.json`).

**The live direction, set 2026-08-10 and unchanged:** inference-only work on RoboLab/DROID,
**no retraining**, after the group's efficiency repo reshaped the plan (§3 below).

---

## 2. Current machine state — two hard blockers, neither of them ours

Checked 2026-08-17. **Both of these will silently ruin work if you don't look first.**

### Blocker A — no GPU can hold the checkpoint right now

```
GPU 0..3: 46,068 MiB total, ~30,700 MiB used, ~14,200–14,800 MiB free
one process, pid 2953978, another user, spanning all four GPUs
```

The bf16 checkpoint is **32.9 GB of weights against ~14.5 GB free**. Not a contention
problem — a *capacity* problem. **Any attempt to start the Cosmos server today will OOM.**
Note whose it is: the author of the efficiency repo this project now depends on
socially (§4). Do **not** kill, preempt, or crowd that process.

Before any GPU work: re-check `nvidia-smi`, confirm ≥34 GB free on the target GPU, and
assert the recorded UUID from `configs/topology.yaml`.

### Blocker B — `/scratch` is at 99%, 49 GB free

Was 328 GB free on 2026-08-10. **The fill is not ours**: our footprint is unchanged at ~80 GB
(`/scratch/agustin` = 80 GB, of which the repo is 42 GB — RoboLab 29 GB + cosmos-framework
12 GB; plus 34 GB of HF cache under `~/.cache/huggingface`, which is *not* on `/scratch`).
`du` cannot read other users' directories, so the remaining ~3.2 TB is invisible but
external.

Consequences, per `CLAUDE.md`'s shared-machine rule:

- **Do not start any install** (their benchmark's eval stack in particular, ~15–25 GB) without checking
  `df -h /scratch` first and **aborting and reporting** rather than filling the volume.
- 49 GB is below the headroom the setup scripts assert (`setup_cosmos.sh` has a ≥40 GB
  precheck — it would pass by 9 GB, which is not a margin).
- `uv cache prune` after any sync.

### Chronic, not new

`loadavg ≈ 33` from other users' parallel compiles; GPU CPU affinity is cores
`0-15,32-47`. This perturbs every CPU-side stage. **Correctness runs are fine on a busy box;
timing runs are not.** Record `loadavg` + a process snapshot with every timed run.

### Housekeeping

Four stale background shells of ours (pids incl. 981598, oldest ~13 days) are looping on a
self-matching `pgrep -f probe_compute_anatomy` — the pattern matches the polling shell's own
command line, so the `until` loop never exits. Harmless (sleeping), but they are noise and
can be killed. Not killed here because it wasn't asked for.

---

## 3. What session 4 established (the short version)

Full detail in `results/reports/session_4_report.md` and the last two sections of
`docs/decision_log.md`.

The user obtained access to **`chooper1/Cosmos3-Efficient-Imagination`**, the group's
efficiency repo, and asked to base our work on it. The audit
(`private/upstream_fork_audit.md`) falsified the plan's central premise:

- It is **not a substrate.** It is a standalone repo that patches cosmos-framework from the
  outside (`framework_patches/`) and monkeypatches its sampler; it contains no
  `cosmos_framework` package. So `configs/substrates.yaml` deliberately has **no**
  `efficient_imagination` entry — the comment in that file explains why.
- It is a **different benchmark and checkpoint** (their own simulator benchmark,
  Cosmos3-Nano + LoRA) than ours (RoboLab, `Cosmos3-Nano-Policy-DROID`). **Their headline number and our 31.1%
  are not comparable.** The plan's whole "redraw the reference arm against their baseline"
  phase is void.

Then their headline method was ported and **falsified on our checkpoint for ~0.3 GPU-h**:

| cell | V | A | joint L2 (rad) | **cosine** | CUDA ms |
|---|---|---|---|---|---|
| teacher repeat | 4 | 4 | **0.0** | **1.000** | 7,402 |
| knfv V4/A4 | 4 | 4 | 0.052 | 0.460 | 7,721 |
| knfv V2/A4 | 2 | 4 | 1.460 | −0.028 | 7,650 |
| knfv V1/A4 | 1 | 4 | 1.695 | −0.060 | 7,335 |
| knfv V1/A8 | 1 | 8 | 1.804 | 0.055 | 15,686 |

Verdict **COLLAPSE**, 3/3 asymmetric cells. Cosine ≈ 0 means the emitted chunk is
*unrelated* to the teacher's — not a degraded version. Uniform across 4 tasks. Cause is the
checkpoint, established from source: `nano_model_config.py:90` sets
`independent_action_schedule=False` and `action_policy_droid_nano.py` never overrides it, so
our checkpoint was trained with video and action sharing **one** σ schedule. Their own sampler documents exactly this risk for such a checkpoint. **Their mechanics
reproduce correctly** (V4/A4 = 0.052 here vs the value they document), so it is the checkpoint's
response, not a wiring artifact.

**Two byproducts are worth more than the verdict:**

1. **The inference path is bit-deterministic.** The stock-repeat control measured *exactly*
   0.0 deviation, cosine 1.000, on all 16 requests. `deterministic_seed=True` holds for the
   whole path. **This is the B3 noise floor owed since M2** — it is zero, so no
   minimum-detectable-effect threshold is needed for offline comparisons and any nonzero
   offline deviation is signal. Obtained free, as a control.
2. **There is no speedup on this axis even in principle, today.** Cost tracks **A**, not V:
   V1/A16 measured 30.4 s against a 7.3 s teacher (**4× slower**); V1/A4 = 7,335 ms vs
   7,402 ms, indistinguishable. This is their own unimplemented-KV-cache caveat
   (recorded in their own write-up) appearing as measured latency — and precisely the claim our rules
   forbid making from FLOPs alone.

Reproduce (needs ≥34 GB free VRAM, see Blocker A):

```bash
.venv/bin/python scripts/offline_split_schedule.py \
    --grid 4x4,1x4,2x4,1x8 --per-task 4 --seed 0
.venv/bin/python scripts/analyze_split_schedule.py \
    --jsonl results/processed/split_schedule_stratified.jsonl \
    --out   results/reports/split_schedule_stratified.json
```

---

## 4. Blocked on other people — do not work around these

1. **A trained checkpoint from the group.** The user chose (2026-08-10) to *ask* rather than
   train, because retraining is hundreds of GPU-h per mode, there is no training data on this host,
   and their checkpoints are very large. A request is drafted and **the user sends it, not us**:
   **`private/outbound_checkpoint_request.md`** — copied out of session scratch so it survives the
   session; it is a personal message, so untrack it if you'd rather it stayed out of git. It asks
   for their frozen-regime LoRA checkpoint (+ `_v1rollout`, + a `baseline`), the
   `--experiment` name, the V=1 serving env block, and their framework base commit; it offers
   our three transfer findings back.
   **Unscoped prerequisite if it arrives:** evaluating a trained for their benchmark checkpoint needs the
   **their benchmark's eval stack** stack, which this host does not have (we have RoboLab/Isaac), and
   their LoRA checkpoint cannot be evaluated on RoboLab/DROID instead. See Blocker B before
   installing anything.
2. **`git revert --no-edit 7b22280 2257031` in `third_party/cosmos-framework`** — dead code
   since the gloo pre-init, blocked three times by the permission classifier. Needs the
   user's authorization or their shell. Until then `docs/upstream_patches.md` keeps its
   WITHDRAWN sections and the patch set is not fully honest.
3. **Guardrails stay disabled** — `nvidia/Cosmos-Guardrail1` is gated, access denied. Recorded
   in every result as `deviations_from_official`.
4. **Never accept a licence or EULA on the user's behalf** (Cosmos, HF gated, Isaac). Stop
   and ask. Their efficiency repo has **no LICENSE file** at all (per-file SPDX is MIT for 8
   of 9 sampler modules), which is why we *import* their clone rather than copying it —
   see `src/action_refresh/server/warm_start_vendor.py`.

---

## 5. What to do next

**Sanctioned and unblocked (zero GPU):** task **#18** — guard `make contract` against
clobbering `docs/baseline_contract.md`. `scripts/derive_baseline_contract.py`'s regexes
produce wrong values (`port_default: 3`, `denoising_steps_default: 5`) and its 25-line output
would overwrite the 239-line hand-verified document. Same failure class as the
anatomy-overwrite bug already fixed in `run_anatomy_sweep.sh`. Pre-existing; flagged, not yet
ruled on by the user.

**The highest-value research step, needs a quiet box and an estimate first:** price
**`cold-N`** (symmetric step reduction) properly — normalized FLOPs, measured GPU time,
end-to-end latency, energy, and task success against the now-known zero noise floor.
Rationale: it is the one lever that stays *in distribution* for our checkpoint, and per the
group's decomposition our M2 "2.98× at 1 vs 4 steps" should be labelled `cold-N` rather than
read as two independent axes. Report the GPU-h estimate and get approval before starting.

**Also waiting on a quiet window:** re-measure `docs/latency.md`. Those absolute numbers have
wanted an idle box since M2 and have never had one.

**Untested and closest to the project's original thesis:** their cross-call latent-reuse gate
(their prior-drift probe, which has *no* eval cells behind it). Carry
session 4's caution into it: reusing a *previous* imagination is a strictly larger
perturbation than freezing the current one, and freezing alone already collapsed the action.

**Retired, with reasons — do not silently revive:** E2a/E2b attention masking, G spatial
masking of the video block (their whole-block mask = 0.000 across 6 cells; the serving batch
has no `pixel_values`, so the video latent is the policy's *only* channel to the world —
masking it blinds the robot rather than removing imagination), σ_a floor widening, and
anything from their their training-priors patch. All need training. Table in
`docs/decision_log.md` §2026-08-10.

**Do not start** LoRA, RAFT, v2e, spatial modifications, their benchmark's eval-stack install, or any
retraining prep. None is approved.

---

## 5b. Moving to another machine

Two halves, deliberately:

**1. Git carries everything of ours.** `scripts/bootstrap_new_machine.sh` rebuilds
`third_party/` from `reproducibility/source_manifest.json` + `reproducibility/patches/`.
Verified 2026-08-17 end-to-end: replaying the 4 cosmos-framework patches onto the recorded
base reproduces the **exact tree hash**, so the 42 GB of clones is genuinely redundant with
(pinned SHA + 7 patches) and is not committed.

```bash
git clone <origin> && cd action-aware-refresh
bash scripts/bootstrap_new_machine.sh --dry-run    # check disk + what it will do
bash scripts/bootstrap_new_machine.sh             # clone at pinned SHAs, apply patches, verify trees
cp configs/machine.example.yaml configs/machine.yaml   # then edit: machine.yaml is gitignored
export HF_TOKEN=...                               # env var only, never written to disk
make setup && make smoke
```

**2. Two things git does not carry, by choice.**

| not in git | why | how to get it |
|---|---|---|
| `private/` (~450 KB) | their unlicensed code + unpublished results; origin is public | `tar czf private.tar.gz private/` and scp it — see `private/README.md` |
| `results/raw/corpus/` (88 MB, 88 requests) | CLAUDE.md's no-large-outputs rule | re-capture with `scripts/capture_corpus.sh` (~0.2 GPU-h, needs a working server), or copy the directory across |

Without the corpus, every offline replay (`offline_action_study.py`,
`offline_split_schedule.py`) has nothing to read. It is the cheapest thing to copy and the
most annoying to regenerate, since regenerating it needs the GPU that Blocker A currently
denies — so if you can copy it, copy it.

Without `private/`, the efficiency-repo arm cannot be re-run: `bootstrap_new_machine.sh`
will say so explicitly and the tree check for that one clone will mismatch by exactly the one
missing commit. Nothing else depends on it.

## 6. Repo conventions that are load-bearing

- **Typed Python, dataclasses/pydantic, `structlog`, deterministic seeds, no hidden global
  state, and — most often violated by accident — no silent fallbacks.** If something can't
  be found, raise with the remedy in the message (see `warm_start_vendor.py`'s errors).
- **No hardcoded personal paths.** Use `configs/machine.yaml` (gitignored;
  `machine.example.yaml` is the template).
- **Substrates are resolved, not hardcoded.** `configs/substrates.yaml` +
  `resolve_substrate()` in `src/action_refresh/config.py`; precedence is explicit arg >
  `$COSMOS_SUBSTRATE` > `default`. Location comes from the manifest, **the live SHA from
  git** — because the manifest was found 4 commits stale, and `manifest_stale` now surfaces
  that drift instead of hiding it. Shell equivalent: `resolve_substrate` in
  `scripts/lib/common.sh`.
- **`reproducibility/source_manifest.json` is generated — never hand-edit it** (its own
  header says so; I broke this rule once and had to unwind it). Anything you want recorded
  goes in `PHASE_HINTS`/`NOTE_HINTS` in `scripts/update_manifest.py`, then regenerate.
- **Upstream repos are preserved.** Modifications live on branch
  `research/action-aware-refresh` in each clone and are exported to
  `reproducibility/patches/`. Their repo's patch is `cei-0001-prepare-arity.patch`.
- **Never commit** tokens, credentials, weights, generated videos, large parquet, or the
  `third_party/` clones. `HF_TOKEN` via env var only — **never written to disk**.
- **No run >4 GPU-h without reporting the estimate and getting explicit approval.**
- **Every speedup claim needs** normalized FLOPs + measured GPU time + end-to-end latency +
  task success. Fewer tokens or fewer theoretical FLOPs **never** imply a real speedup
  unless measured latency confirms it. Session 4's V1/A16 result is the cautionary case.
- Energy is power-integrated at 10 Hz and must always be labelled **ESTIMATED**
  (`total_energy_consumption` is not a valid NVML field on these GPUs). GPU 0 idles at ~67 W
  vs ~34 W for GPUs 1–3, so report idle-adjusted alongside gross.
- Attention backend on SM 8.9 is `flash2`/`cudnn`/`natten` — **`flash3` is Hopper-only**, so
  absolute latencies are not comparable to NVIDIA's published FA3 numbers. All comparisons
  are within-machine; record the selected backend in every result.

---

## 7. Gotchas that already cost time

Each of these was a real debugging session. They are cheaper to read than to rediscover.

- **`[knfv]` traces are gated behind `KNFV_DEBUG=1`** (their sampler module). Without it
  the sampler engages silently and an "is it engaged?" detector that greps stdout will report
  a false alarm. I raised exactly that false alarm.
- **`--limit` samples a task-sorted corpus**, so `--limit 16` drew 16 requests from *one*
  task while appearing to be a broad sample. Use `--per-task`. I misreported coverage before
  catching this.
- **Deviation metric keys are** `joint_l2_mean`, `joint_linf_max`, `endpoint_joint_l2`,
  `gripper_disagreement_rate`, `cosine_similarity_mean` — not the plausible-sounding
  `mean_l2`/`max_abs`. Guessing them yields a `NoneType.__format__` crash.
- **A floor of exactly `0.0` is the good case, not a missing value.** `if floor_med:` silently
  dropped the ratio column; it must be `is not None`, with zero → `inf`. The bit-determinism
  finding is exactly the case that trips this.
- **Their patch installs `prepare_inject` as an *instance* attribute**, which shadows
  any class-level adapter. Arity mismatches must be fixed *inside* their function (hence a
  recorded patch), not wrapped around it. Our framework returns an 8th value,
  `has_noisy_actions`, that their base lacked.
- **`WarmStartConfig.from_env()` is read per call** , so a whole (V, A) grid
  sweeps in **one** model load by mutating the environment between requests. Do not reload
  the model per cell.
- **`make test-model` needs the cosmos venv**, not the repo venv (the repo venv deliberately
  has no torch). Tests that assert on torch's absence are wrong; assert the real claim.
- **Tests inherit `$COSMOS_SUBSTRATE`** exported by Makefile recipes — an autouse
  `monkeypatch.delenv` is needed, or substrate tests pass/fail by accident.
- **A known provenance drift, unfixed:** `warm_start_vendor.locate()` reads `commit` from the
  manifest, so `results/reports/split_schedule*.json` record `vendor_commit=ecb8de4` while the
  code that actually ran was `8433abd` (= `ecb8de4` + the arity patch). Same class as the
  substrate staleness already fixed by reading live git; the vendor path never got the fix.
  Worth doing before any further result cites their commit.

---

## 8. Verification state, as of this handover

```
make test          →  117 passed, 2 skipped   (both skips expected: need the cosmos venv)
ruff check         →  71 findings, ALL in pre-M3 files (style: SIM117/UP037/N806/B905/F541);
                      every file created in sessions 3–4 is clean. There is no `lint` make
                      target, so this is not gated — treat the 71 as pre-existing debt.
/scratch           →  49 GB free (99% full) — see Blocker B
GPUs               →  all four occupied by another user — see Blocker A
GPU-h              →  ≈0.3 spent in session 4 (exactly what was approved);
                      ≈30 cumulative against ~18 approved — the overage was flagged in
                      session 3 and is the user's to rule on
git                →  NOTHING COMMITTED since 6cf33dd. 2,294 files staged
                      (+111,023/−525), 5 unstaged, 9 untracked. Sessions 2–4 are entirely
                      in the index. `.gitignore` correctly excludes npz/traces/videos;
                      largest staged file is 480 KB (a ledger jsonl), so the tree is safe
                      to commit — but review before doing so, and note `docs/decision_log.md`
                      and `reproducibility/source_manifest.json` have BOTH staged and
                      unstaged changes (`MM`).
```

**The single most useful non-research act available:** commit the staged tree. Three sessions
of work — the entire baseline, the anatomy, the Pareto machinery, the substrate abstraction,
the negative result — exists only in one machine's git index.
