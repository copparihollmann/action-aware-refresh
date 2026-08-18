#!/usr/bin/env python3
"""Sequential, resumable driver for the M3/M4 experiment chain.

Runs the phases one after another, unattended, and is safe to kill at any point:
every step is a ledger unit, so re-invoking the same command continues from where it
stopped rather than repeating finished work. That property is the whole reason this
exists — the chain is many GPU-hours on a shared box, and losing a completed phase to
an unrelated interruption is the expensive failure.

    scripts/run_phases.py                 # run everything not yet done
    scripts/run_phases.py --only p1_offline
    scripts/run_phases.py --status        # what is done / pending, run nothing
    scripts/run_phases.py --retry-failed  # re-attempt steps that errored

Design notes:

- **Steps shell out.** Each phase needs a different interpreter (repo venv for
  analysis, cosmos venv for the model, RoboLab venv via shell for Isaac), so steps are
  subprocesses rather than imports. A step's stdout/stderr is teed to a log under the
  ledger unit directory, so a failure is diagnosable after the fact.
- **Steps are idempotent themselves where possible.** `capture_corpus.sh` skips
  already-captured tasks and `offline_action_study.py` keeps its own inner ledger, so
  a step that is re-run after a partial failure resumes internally too. The outer
  ledger records step completion; the inner ones protect work *within* a step.
- **A failed step stops the chain by default.** Later phases consume earlier outputs,
  so continuing past a failure would produce results built on missing inputs.
  `--keep-going` overrides this for independent steps.
"""
from __future__ import annotations

import argparse
import os
import shlex
import subprocess
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

from action_refresh.ledger import Ledger, WorkUnit  # noqa: E402

COSMOS_PY = REPO_ROOT / "third_party" / "cosmos-framework" / ".venv" / "bin" / "python"
REPO_PY = REPO_ROOT / ".venv" / "bin" / "python"


def cache_env() -> dict[str, str]:
    """HF_HOME / UV_CACHE_DIR from configs/machine.yaml, for EVERY step.

    This is the third time this project has been bitten by the same bug: a *run* path
    that does not export what the *install* path did, so Hugging Face silently resolves
    its default `$HOME/.cache/huggingface` — a 60 GB NFS volume — and pages tens of GB
    over the network. It cost ~25 minutes on a model load in session 2, and it happened
    again here because the orchestrator passed only `os.environ` through to
    subprocesses.
    """
    import yaml  # noqa: PLC0415

    machine = REPO_ROOT / "configs" / "machine.yaml"
    if not machine.is_file():
        raise SystemExit(
            f"{machine} not found — cannot resolve the cache paths. Copy "
            "configs/machine.example.yaml and set paths.hf_cache / paths.uv_cache. "
            "Refusing to run with HF's default, which writes to the NFS home volume."
        )
    paths = (yaml.safe_load(machine.read_text()) or {}).get("paths") or {}
    for key in ("hf_cache", "uv_cache"):
        if not paths.get(key):
            raise SystemExit(f"configs/machine.yaml is missing paths.{key}")
    return {
        "HF_HOME": os.environ.get("HF_HOME") or str(paths["hf_cache"]),
        "UV_CACHE_DIR": os.environ.get("UV_CACHE_DIR") or str(paths["uv_cache"]),
        "TOKENIZERS_PARALLELISM": "false",
    }


@dataclass
class Step:
    """One resumable unit of the chain."""

    name: str
    phase: str
    argv: list[str]
    description: str
    env: dict[str, str] = field(default_factory=dict)
    # GPU-hour estimate, printed before the chain starts. The 4-GPU-hour rule needs a
    # number stated up front, not discovered afterwards.
    est_gpu_hours: float = 0.0
    # Whether this step's script keeps its own inner ledger and accepts --retry-failed.
    # Without propagating the flag, `run_phases.py --retry-failed` would re-run the
    # *step* while its inner ledger still skipped the units that failed inside it —
    # so the step would exit "successfully" having done nothing, and the failure would
    # look fixed when it was not.
    accepts_retry_failed: bool = False

    @property
    def unit(self) -> WorkUnit:
        return WorkUnit(phase=self.phase, kind="step", method=self.name)


def build_steps() -> list[Step]:
    """The chain, in dependency order."""
    steps: list[Step] = []

    # ---- Phase 0.3: the offline corpus -----------------------------------
    # Everything offline depends on this, so it runs first. Internally skips tasks
    # already captured.
    steps.append(
        Step(
            name="capture_corpus",
            phase="p0_corpus",
            argv=["bash", "scripts/capture_corpus.sh"],
            description="record real policy requests from closed-loop episodes",
            env={"COSMOS_GUARDRAILS": "false"},
            est_gpu_hours=0.4,
        )
    )

    # ---- Phases 1 + 2: offline screening ---------------------------------
    # One process sweeps every condition because the 31 GiB model load dominates;
    # the inner ledger makes it resumable per (condition, request).
    steps.append(
        Step(
            name="offline_study",
            phase="p1_offline",
            argv=[
                str(COSMOS_PY),
                "scripts/offline_action_study.py",
                "--corpus-glob",
                "results/raw/corpus/*/captured_request_*.npz",
                "--steps",
                "1,2,3,4",
                "--seeds",
                "4",
                "--action-only",
            ],
            description="Experiments A (denoising steps), E0 (seed spread), E1 (no imagination)",
            est_gpu_hours=1.5,
            accepts_retry_failed=True,
        )
    )
    # E2b as a separate step, sharing the SAME ledger. The first sweep's units are
    # already recorded, so this computes only the vision_frames conditions — no
    # recomputation of the teacher, steps or seeds. That is the whole point of a
    # content-addressed ledger: new conditions can be appended to a finished sweep
    # without paying for it twice.
    #
    # Kept separate rather than folded into the first step because the first step was
    # already running when the E2b knob was built; merging them would have discarded
    # its progress.
    steps.append(
        Step(
            name="offline_study_e2b",
            phase="p1_offline",
            argv=[
                str(COSMOS_PY),
                "scripts/offline_action_study.py",
                "--corpus-glob",
                "results/raw/corpus/*/captured_request_*.npz",
                # Only the teacher is needed as the reference; the step/seed conditions
                # already exist in the ledger and will be skipped.
                "--steps",
                "4",
                "--seeds",
                "1",
                # 17/9/5 keep the conditioning caption byte-identical to the baseline
                # (see cosmos-framework-0004), so they vary only the frame count. 3 is
                # excluded: it shifts the caption to "3.0 seconds" and would confound
                # the frame-count effect with a text change.
                "--vision-frames",
                "17,9,5",
            ],
            description="Experiment E2b: shorten the imagined horizon (9->2 latent frames)",
            est_gpu_hours=0.6,
            accepts_retry_failed=True,
        )
    )
    steps.append(
        Step(
            name="analyze_offline",
            phase="p1_offline",
            argv=[str(REPO_PY), "scripts/analyze_offline_study.py"],
            description="deviation vs teacher; writes docs/offline_action_study.md",
            est_gpu_hours=0.0,
        )
    )

    # ---- Phase 3: M3, the first Pareto frontier --------------------------
    # Breadth-first at ONE episode per task, per the agreed allocation. Episode counts
    # come later on whichever tasks turn out to discriminate; spending 2 episodes on
    # all 16 tasks up front would double the cost of learning which tasks matter.
    #
    # Budget per pass: 3.69 GPU-h worst case (all 16 tasks timing out), realistically
    # ~2-2.5 since successes terminate early. Two methods keeps M3 near 5 GPU-h and
    # leaves room for Phase 4 inside the approved ~18.
    #
    # Only two methods, chosen for what they settle: `baseline_full` is the reference
    # every normalized-compute number divides by, and `baseline_steps_1` is the
    # strongest trivial competitor (2.98x cheaper, and the offline screen puts its
    # action deviation inside the model's own sampling noise). If 1-step holds task
    # success, that pair alone is the headline Pareto result — and every later caching
    # method has to beat it rather than beat 4-step.
    for method in ("baseline_full", "baseline_steps_1"):
        steps.append(
            Step(
                name=f"closed_loop_{method}",
                phase="p3_pareto",
                argv=[
                    str(REPO_PY),
                    "scripts/run_closed_loop.py",
                    "--method",
                    method,
                    "--set",
                    "pilot_screened",
                    "--episodes",
                    "1",
                ],
                description=f"closed-loop screened pilot (9 tasks x1) for {method}",
                est_gpu_hours=1.4,
                accepts_retry_failed=True,
            )
        )

    # ---- Phase 4: Experiment B, horizon sweep ----------------------------
    # The baseline already reuses chunks at horizon 32 (confirmed: 34 server requests
    # for 1,045 control steps), so this measures whether refreshing *more often* buys
    # success — the honest framing, rather than pretending the baseline refreshes every
    # control step. Enabled by the horizon override in patch robolab-0002.
    for method in ("baseline_fixed_horizon_8", "baseline_fixed_horizon_16"):
        steps.append(
            Step(
                name=f"closed_loop_{method}",
                phase="p4_horizon",
                argv=[
                    str(REPO_PY),
                    "scripts/run_closed_loop.py",
                    "--method",
                    method,
                    "--set",
                    "pilot_screened",
                    "--episodes",
                    "1",
                ],
                description=f"Experiment B horizon sweep: {method}",
                est_gpu_hours=1.4,
                accepts_retry_failed=True,
            )
        )

    # ---- Experiment C, offline feasibility (no GPU) -----------------------
    # Placed before the closed-loop phases because it can *falsify* the
    # compute-reduction premise of Experiment C for free, and spec §14 prefers rapid
    # falsification. Uses h5py, so it runs in the RoboLab venv.
    steps.append(
        Step(
            name="analyze_oracle_temporal",
            phase="p2_oracle_offline",
            argv=[
                str(REPO_ROOT / "third_party" / "RoboLab" / ".venv" / "bin" / "python"),
                "scripts/analyze_oracle_temporal.py",
            ],
            description="Experiment C feasibility from recorded episodes (no GPU)",
            env={"PYTHONPATH": str(REPO_ROOT / "src")},
            est_gpu_hours=0.0,
        )
    )

    # ---- Trustworthy latency, after the GPU is free -----------------------
    # Deliberately last among the GPU steps. The offline study's wall times are not
    # latency measurements (it sweeps conditions in back-to-back blocks, which B3 showed
    # is exactly how not to time anything here), and this script *refuses to run* while
    # another process shares the GPU. So it has to come after the closed-loop phases have
    # torn down their servers. Every speedup quoted in the project comes from here.
    steps.append(
        Step(
            name="measure_latency",
            phase="p5_report",
            argv=[
                str(COSMOS_PY),
                "scripts/measure_latency.py",
                "--repeats",
                "12",
                "--cooldown-s",
                "6",
            ],
            description="interleaved, cooled, median-of-many latency per configuration",
            est_gpu_hours=0.4,
        )
    )

    # ---- Reporting -------------------------------------------------------
    steps.append(
        Step(
            name="build_pareto",
            phase="p5_report",
            argv=[str(REPO_PY), "scripts/build_pareto.py"],
            description="success vs normalized total compute; the M3 deliverable",
            est_gpu_hours=0.0,
        )
    )

    return steps


def run_step(step: Step, led: Ledger, *, dry_run: bool, retry_failed: bool = False) -> bool:
    """Execute one step under its ledger claim. Returns True on success."""
    argv = list(step.argv)
    if retry_failed and step.accepts_retry_failed:
        argv.append("--retry-failed")
    if dry_run:
        print(f"  DRY-RUN {step.name}: {' '.join(shlex.quote(a) for a in argv)}")
        return True

    with led.claim(step.unit) as slot:
        log_path = slot.dir / "step.log"
        # Cache env first so a step cannot override it by accident, then the step's own.
        env = {**os.environ, **cache_env(), **step.env}
        # Isaac refuses to launch without this and we must never set it ourselves;
        # fail early with a clear message rather than deep inside a subprocess.
        t0 = time.perf_counter()
        with log_path.open("w") as log:
            log.write(f"$ {' '.join(shlex.quote(a) for a in argv)}\n\n")
            log.flush()
            proc = subprocess.run(
                argv,
                cwd=REPO_ROOT,
                env=env,
                stdout=log,
                stderr=subprocess.STDOUT,
                check=False,
            )
        elapsed = time.perf_counter() - t0
        tail = "\n".join(log_path.read_text(errors="replace").splitlines()[-25:])
        if proc.returncode != 0:
            print(f"  FAILED rc={proc.returncode} after {elapsed / 60:.1f} min")
            print("  --- log tail ---")
            print("  " + tail.replace("\n", "\n  "))
            # Raising inside the claim records the error and leaves the unit
            # retryable, rather than marking it done.
            raise RuntimeError(f"step {step.name} exited {proc.returncode}; see {log_path}")
        slot.result = {
            "returncode": proc.returncode,
            "elapsed_min": round(elapsed / 60, 2),
            "log": str(log_path.relative_to(REPO_ROOT)),
            "argv": argv,
        }
        print(f"  ok in {elapsed / 60:.1f} min")
    return True


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--ledger", default=str(REPO_ROOT / "results" / "ledger" / "phases"))
    ap.add_argument("--only", action="append", default=None, help="run only these phases/steps")
    ap.add_argument("--status", action="store_true", help="report state and exit")
    ap.add_argument("--retry-failed", action="store_true")
    ap.add_argument("--keep-going", action="store_true", help="continue past a failed step")
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()

    steps = build_steps()
    if args.only:
        wanted = set(args.only)
        steps = [s for s in steps if s.name in wanted or s.phase in wanted]
        if not steps:
            raise SystemExit(f"--only {sorted(wanted)} matched no step")

    led = Ledger(args.ledger, retry_failed=args.retry_failed)

    print("=== chain ===")
    pending_hours = 0.0
    for s in steps:
        st = led.state(s.unit)
        mark = (
            "done"
            if st.done
            else "FAILED"
            if st.failed
            else "interrupted"
            if st.interrupted
            else "pending"
        )
        if not st.done and not (st.failed and not args.retry_failed):
            pending_hours += s.est_gpu_hours
        print(f"  [{mark:11s}] {s.phase:12s} {s.name:18s} ~{s.est_gpu_hours:.1f} GPU-h — {s.description}")
    print(f"=== estimated remaining: ~{pending_hours:.1f} GPU-h ===")

    if args.status:
        return 0

    # Surface the budget before spending it, per the 4-GPU-hour rule. This is a
    # notice, not a prompt: the user approved ~18 GPU-h for this phase, and stopping
    # to ask again mid-chain would defeat running unattended.
    if pending_hours > 4.0:
        print(
            f"NOTE: remaining estimate {pending_hours:.1f} GPU-h exceeds 4 h; running "
            "under the approved ~18 GPU-h budget for M3/M4, chunked per step."
        )

    if not os.environ.get("OMNI_KIT_ACCEPT_EULA") and any(
        s.phase.startswith("p0_corpus") or "robolab" in s.name for s in steps
    ):
        print(
            "WARNING: OMNI_KIT_ACCEPT_EULA is not set, so any Isaac step will fail. "
            "Export it yourself if you accept the EULA — scripts never set it."
        )

    failures: list[str] = []
    for s in steps:
        if not led.should_run(s.unit):
            st = led.state(s.unit)
            print(f"skip {s.name} ({'done' if st.done else 'failed; use --retry-failed'})")
            continue
        print(f"run  {s.name} — {s.description}", flush=True)
        try:
            run_step(s, led, dry_run=args.dry_run, retry_failed=args.retry_failed)
        except Exception as exc:  # noqa: BLE001
            failures.append(s.name)
            if not args.keep_going:
                print(
                    f"\nSTOPPED at {s.name}: {exc}\n"
                    "Later phases consume this step's output, so continuing would build "
                    "results on missing inputs. Fix the cause and re-run this command — "
                    "finished work is not repeated. Use --keep-going to override.",
                    file=sys.stderr,
                )
                return 1
            print(f"  continuing past failure in {s.name} (--keep-going)", file=sys.stderr)

    if failures:
        print(f"\ncompleted with failures: {failures}", file=sys.stderr)
        return 1
    print("\nchain complete")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
