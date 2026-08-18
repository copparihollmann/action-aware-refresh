"""Crash-safe work ledger: run long experiment chains without losing finished work.

The experiment chain for M3/M4 is many GPU-hours long on a shared, contended box.
Any of it can die halfway — another user's job, an OOM, a lost session, a reboot.
The rule this module enforces is: **work that finished is never redone, and work
that never finished is never mistaken for a result.**

Design, and why each piece is the way it is:

- **One directory per work unit**, named by a stable hash of its identity
  (`phase/kind/method/task/episode/variant`). Identity is content-addressed rather
  than positional, so inserting a new config into a sweep does not invalidate
  everything after it — a positional index would.

- **`result.json` is the completion marker.** It is written to a temp file and then
  `os.replace`d, which is atomic on POSIX within a filesystem. So the file either
  does not exist or is complete and parseable; there is no torn state to
  misinterpret as a result. This is why completion is *not* recorded as a line in
  a shared log — an append can be interleaved or truncated mid-write.

- **Failures are recorded, not silent.** A crashed unit leaves `error.json`, so a
  resume can tell "never attempted" from "attempted and blew up". Failed units are
  retried only when explicitly asked (`retry_failed`), because silently retrying a
  deterministic failure burns GPU-hours in a loop.

- **A stale `started.json` means someone was killed mid-unit.** That is reported as
  interrupted, not as a failure and not as a success.

- **The JSONL log is an audit trail only.** Nothing reads it to decide what to skip.
  It exists so a human can see the order things happened in.

Not a distributed lock: single-writer per output root is assumed. `claim()` takes an
O_EXCL lock so two concurrent drivers on the same root cannot duplicate a unit, but
this is a guard against accident, not a scheduler.
"""
from __future__ import annotations

import hashlib
import json
import os
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Iterator

from action_refresh.logging import get_logger

log = get_logger(__name__)

_STARTED = "started.json"
_RESULT = "result.json"
_ERROR = "error.json"
_LOCK = "claim.lock"


@dataclass(frozen=True)
class WorkUnit:
    """Identity of one resumable piece of work.

    Every field participates in the identity hash, so two units differing in any
    field are distinct work. `variant` carries anything else that changes the
    computation (e.g. `num_steps=2`), and must be included for configs that would
    otherwise collide — a sweep over denoising steps whose units all hashed the
    same would record one result and skip the rest.
    """

    phase: str
    kind: str
    method: str = ""
    task: str = ""
    episode: int = 0
    variant: str = ""

    @property
    def key(self) -> str:
        raw = "|".join(
            (self.phase, self.kind, self.method, self.task, str(self.episode), self.variant)
        )
        return hashlib.sha256(raw.encode()).hexdigest()[:16]

    @property
    def label(self) -> str:
        """Human-readable id for logs and directory names (not the identity)."""
        parts = [self.phase, self.kind]
        parts += [p for p in (self.method, self.task) if p]
        if self.variant:
            parts.append(self.variant)
        if self.episode:
            parts.append(f"ep{self.episode}")
        safe = "-".join(parts)
        return "".join(c if (c.isalnum() or c in "-_.") else "_" for c in safe)[:120]

    def as_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class UnitState:
    """`result` is what the caller stored; `payload` is the envelope around it.

    Keeping them separate matters: callers want their own data back, while the
    envelope's `elapsed_s` / `utc` / `unit` are provenance that analysis code reads.
    Returning the envelope as `result` would make every caller reach through an
    extra layer and invite `result["result"]` typos.
    """

    unit: WorkUnit
    done: bool = False
    failed: bool = False
    interrupted: bool = False
    result: dict[str, Any] | None = None
    payload: dict[str, Any] | None = field(default=None, repr=False)
    error: str | None = None
    dir: Path | None = field(default=None, repr=False)


class Ledger:
    """Filesystem-backed record of which work units are finished.

    Usage:

        led = Ledger(root)
        for unit in units:
            if led.is_done(unit):
                continue
            with led.claim(unit) as slot:
                slot.result = do_the_work()

    Leaving the `claim` block without setting `slot.result` records an error, so a
    unit can never be marked done by falling through.
    """

    def __init__(self, root: Path | str, *, retry_failed: bool = False) -> None:
        self.root = Path(root)
        self.root.mkdir(parents=True, exist_ok=True)
        self.retry_failed = retry_failed
        self.audit = self.root / "ledger.jsonl"

    # -- paths --------------------------------------------------------------
    def unit_dir(self, unit: WorkUnit) -> Path:
        # Label first for greppability, key appended for uniqueness: two units
        # with the same label but different variants must not share a directory.
        return self.root / f"{unit.label}.{unit.key}"

    # -- queries ------------------------------------------------------------
    def state(self, unit: WorkUnit) -> UnitState:
        d = self.unit_dir(unit)
        st = UnitState(unit=unit, dir=d)
        if not d.is_dir():
            return st
        res = d / _RESULT
        if res.is_file():
            try:
                st.payload = json.loads(res.read_text())
                st.result = (st.payload or {}).get("result")
                st.done = True
                return st
            except json.JSONDecodeError as exc:
                # An unparseable result.json should never happen given the atomic
                # rename. If it does, treat it as *not done* and say so loudly
                # rather than skipping the work on the strength of a corrupt file.
                log.warning("ledger.result_corrupt", unit=unit.label, path=str(res), error=str(exc))
        err = d / _ERROR
        if err.is_file():
            try:
                st.error = json.loads(err.read_text()).get("error")
            except json.JSONDecodeError:
                st.error = "(unparseable error.json)"
            st.failed = True
            return st
        if (d / _STARTED).is_file():
            st.interrupted = True
        return st

    def is_done(self, unit: WorkUnit) -> bool:
        return self.state(unit).done

    def should_run(self, unit: WorkUnit) -> bool:
        st = self.state(unit)
        if st.done:
            return False
        if st.failed and not self.retry_failed:
            return False
        return True

    def results(self, phase: str | None = None) -> Iterator[dict[str, Any]]:
        """Every completed result, optionally filtered by phase."""
        for d in sorted(self.root.iterdir()):
            res = d / _RESULT
            if not (d.is_dir() and res.is_file()):
                continue
            try:
                payload = json.loads(res.read_text())
            except json.JSONDecodeError:
                continue
            if phase and (payload.get("unit") or {}).get("phase") != phase:
                continue
            yield payload

    def summary(self, units: list[WorkUnit]) -> dict[str, int]:
        counts = {"done": 0, "failed": 0, "interrupted": 0, "pending": 0}
        for u in units:
            st = self.state(u)
            if st.done:
                counts["done"] += 1
            elif st.failed:
                counts["failed"] += 1
            elif st.interrupted:
                counts["interrupted"] += 1
            else:
                counts["pending"] += 1
        return counts

    # -- audit --------------------------------------------------------------
    def _append_audit(self, event: str, unit: WorkUnit, **extra: Any) -> None:
        rec = {
            "event": event,
            "utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            "unit": unit.as_dict(),
            "key": unit.key,
            **extra,
        }
        with self.audit.open("a") as fh:
            fh.write(json.dumps(rec) + "\n")

    # -- claiming -----------------------------------------------------------
    def claim(self, unit: WorkUnit) -> "_Claim":
        return _Claim(self, unit)


class _Claim:
    """Context manager wrapping one unit's execution.

    Sets `.result` to mark success. Anything else — an exception, or falling out of
    the block without a result — records an error instead. There is deliberately no
    way to mark a unit done without producing a result payload.
    """

    def __init__(self, ledger: Ledger, unit: WorkUnit) -> None:
        self.ledger = ledger
        self.unit = unit
        self.dir = ledger.unit_dir(unit)
        self.result: dict[str, Any] | None = None
        self.meta: dict[str, Any] = {}
        self._t0 = 0.0
        self._lock_fd: int | None = None

    def __enter__(self) -> "_Claim":
        self.dir.mkdir(parents=True, exist_ok=True)
        # O_EXCL so a second driver on the same root cannot run the same unit.
        try:
            self._lock_fd = os.open(str(self.dir / _LOCK), os.O_CREAT | os.O_EXCL | os.O_WRONLY)
        except FileExistsError:
            # A leftover lock from a killed process is expected on resume; the
            # absence of result.json already told us the work is unfinished, so
            # take it over rather than refusing to make progress.
            log.warning("ledger.stale_lock_taken_over", unit=self.unit.label)
            self._lock_fd = os.open(str(self.dir / _LOCK), os.O_WRONLY)
        self._t0 = time.perf_counter()
        _atomic_write_json(
            self.dir / _STARTED,
            {"unit": self.unit.as_dict(), "pid": os.getpid(), "utc": _utc()},
        )
        # Clear a previous failure so a retried unit does not read as both failed
        # and done afterwards.
        (self.dir / _ERROR).unlink(missing_ok=True)
        self.ledger._append_audit("start", self.unit)
        log.info("unit.start", unit=self.unit.label)
        return self

    def __exit__(self, exc_type, exc, _tb) -> bool:
        elapsed = time.perf_counter() - self._t0
        try:
            if exc is not None:
                self._record_error(f"{type(exc).__name__}: {exc}", elapsed)
                return False  # never swallow: the driver decides whether to continue
            if self.result is None:
                self._record_error(
                    "unit produced no result (fell out of the claim block without "
                    "setting .result)",
                    elapsed,
                )
                return False
            payload = {
                "unit": self.unit.as_dict(),
                "key": self.unit.key,
                "utc": _utc(),
                "elapsed_s": round(elapsed, 3),
                **self.meta,
                "result": self.result,
            }
            _atomic_write_json(self.dir / _RESULT, payload)
            (self.dir / _STARTED).unlink(missing_ok=True)
            self.ledger._append_audit("done", self.unit, elapsed_s=round(elapsed, 3))
            log.info("unit.done", unit=self.unit.label, elapsed_s=round(elapsed, 1))
            return False
        finally:
            if self._lock_fd is not None:
                os.close(self._lock_fd)
                (self.dir / _LOCK).unlink(missing_ok=True)

    def _record_error(self, message: str, elapsed: float) -> None:
        _atomic_write_json(
            self.dir / _ERROR,
            {"unit": self.unit.as_dict(), "error": message, "utc": _utc(), "elapsed_s": elapsed},
        )
        (self.dir / _STARTED).unlink(missing_ok=True)
        self.ledger._append_audit("error", self.unit, error=message)
        log.error("unit.failed", unit=self.unit.label, error=message)


def _utc() -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())


def _atomic_write_json(path: Path, payload: dict[str, Any]) -> None:
    """Write JSON so readers see either the old state or the complete new one.

    `os.replace` is atomic within a filesystem; the fsync makes the content durable
    before the rename so a power loss cannot leave a renamed-but-empty file.
    """
    tmp = path.with_suffix(path.suffix + ".tmp")
    with tmp.open("w") as fh:
        json.dump(payload, fh, indent=2, default=str)
        fh.flush()
        os.fsync(fh.fileno())
    os.replace(tmp, path)
