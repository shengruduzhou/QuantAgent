"""Persistent multiple-testing ledger: the authoritative family trial count N.

The deflated Sharpe ratio corrects a winner for the number of configurations
that were tried to find it. Until Round 32 that number was a hand estimate in
``docs/RESEARCH.md`` §3 ("非线性/融合族 >= 134") and every command fed DSR the
size of *its own* run, so a researcher who ran ``search-factor-fusion`` forty
times was deflated as if it were the first search ever made (ROADMAP P1-4).

This module is the single source for that count:

* ``begin_run`` reserves a run's planned trials *before* any compute, and
  reads the prior family N under the same lock, so a run that later crashes
  still counts (GOVERNANCE R6) and two concurrent runs cannot both miss each
  other's reservation. Outcome entries (one per arm / blend / candidate,
  controls, failures and rejections included) then carry ``trials=0`` unless
  they exceed the reservation. Nothing is ever rewritten or deleted.
* ``family_count(family)`` is the cumulative N: a *declared historical
  baseline* (the pre-ledger count, with its source) plus the sum of every
  entry's ``trials``. Callers add their own run's trials on top; they can
  never lower N.
* Entries are hash-chained (``previous_hash`` -> ``record_hash``) so an edited,
  reordered or removed line breaks verification at exactly that point, and
  ``family_count`` refuses to count a chain that does not verify.
* The read-tail -> hash -> append decision runs under an exclusive
  cross-process file lock, re-reading the tail *inside* the lock. Without it two
  writers chain onto the same predecessor and fork the log (Round 22 A-04
  measured four processes x forty entries: zero reachable from genesis). A
  batch is written with a single ``write`` + ``fsync``.

There was no live generic hash-chain helper to reuse: ``domain.ledger`` and
``paper.ledger`` embed theirs in economic-event classes, and the generic
``governance.audit`` chain was deleted as production-dead in Round 31
(8115e4d). This module follows that design (lock around tail read, latch on a
failed durable write) and stays deliberately small.

The chain is tamper-*evident*, not tamper-proof: whoever can write the file can
rewrite the whole chain. It protects against accidental edits and silent
undercounting, which is the failure that actually happened.
"""

from __future__ import annotations

from contextlib import contextmanager
from dataclasses import dataclass, field
from datetime import datetime, timezone
import hashlib
import json
import math
import os
from pathlib import Path
import subprocess
from typing import Any, Iterator, Mapping, Sequence

try:  # POSIX research/CI hosts
    import fcntl as _fcntl
except ImportError:  # pragma: no cover - Windows
    _fcntl = None

try:  # Windows hosts
    import msvcrt as _msvcrt
except ImportError:  # pragma: no cover - Unix
    _msvcrt = None


SCHEMA_VERSION = "quantagent.trial_ledger.v1"
GENESIS_HASH = "0" * 64
LEDGER_FILENAME = "trial_ledger.jsonl"

#: Model-class comparison (``audit-nonlinear-factors``) and factor-blend search
#: (``search-factor-fusion``) answer the same question -- "which combination of
#: the reviewed factors survives?" -- and RESEARCH.md has always counted them as
#: one family.
FAMILY_NONLINEAR_FUSION = "nonlinear_fusion"
#: Single-factor discovery by the RD-Agent loop (``synthesize-factors-v7``).
FAMILY_RDAGENT_FACTOR = "rdagent_factor_synthesis"

_REPO_ROOT = Path(__file__).resolve().parents[3]


@dataclass(frozen=True)
class FamilyBaseline:
    """Trials consumed before this ledger existed, with where the number comes from."""

    family: str
    count: int
    source: str


#: Pre-ledger trial counts. These are *lower bounds*: they are what could be
#: reconstructed from documents and artifacts, not a census. Raising one is
#: always safe; lowering one requires evidence that the old count was wrong.
HISTORICAL_BASELINES: Mapping[str, FamilyBaseline] = {
    # Every term is counted from an artifact; where a term is a bound, the upper
    # bound is used (round up, never down):
    #   117  H-001..H-027 whole-library cumulative (docs/research/ledgers/
    #        HYPOTHESIS_REGISTRY.md:313 "全库累计 102 -> 117"; H-027 counted at its
    #        max 15 incl. conditional arms; other families included => upper bound)
    #    18  Round-29 E1: 6 arms (a,s,b,c,d,e) x run 1 + determinism run 2, plus the
    #        6-arm sealed-limit-up sensitivity re-scoring (git show 2c4e296:
    #        docs/audits/round29/roles/R3_factor_findings.md:22-34,128-135,194-195)
    #     6  Round-29 governed audit-nonlinear-factors run1 (runtime/reports/
    #        nonlinearity/round29_r29_panel/run1_20261003/model_comparison.json, nTrials 6)
    #     5  Round-31 descriptive replication X1 (runtime/archives/audits/round31/
    #        audit/quant/notes.md:15,21 "5 trials consumed")
    #   ---
    #   146  (the old RESEARCH.md figure 134 counted E1 once, not per executed run)
    FAMILY_NONLINEAR_FUSION: FamilyBaseline(
        family=FAMILY_NONLINEAR_FUSION,
        count=146,
        source=(
            "117 H-001..H-027 (HYPOTHESIS_REGISTRY.md:313) + 18 R29 E1 (6 arms x 2 runs + "
            "6-arm sensitivity, R3_factor_findings.md @2c4e296) + 6 R29 governed run1 "
            "(round29_r29_panel/run1_20261003) + 5 R31 X1 (round31 audit/quant/notes.md)"
        ),
    ),
    FAMILY_RDAGENT_FACTOR: FamilyBaseline(
        family=FAMILY_RDAGENT_FACTOR,
        count=153,
        source=(
            "sum of candidate rows in the 7 rd_agent_task_feedback.json artifacts under "
            "runtime/ on 2026-11-25 (6+31+7+27+25+28+29; v89_closed_loop + void "
            "pre-round20 discovery runs); runs that saved no artifact are not counted"
        ),
    ),
}


class TrialLedgerCorruption(RuntimeError):
    """The chain does not verify, so no count derived from it can be trusted."""


class TrialLedgerWriteUnavailable(RuntimeError):
    """A durable write failed; this ledger object accepts no further appends.

    After a failed ``fsync`` nobody knows whether the bytes landed. Chaining onto
    a tail the OS refused to guarantee would break the chain permanently, so the
    writer latches closed (same policy as ``domain.ledger``, DEF-017). A new
    process re-reads the file honestly.
    """


class TrialLedgerLocationUnknown(RuntimeError):
    """No durable ledger location could be determined; refuse rather than guess."""


def _main_checkout_root(repo_root: Path) -> Path | None:
    """Main checkout of the git repository containing ``repo_root`` (worktree-safe).

    ``git rev-parse --git-common-dir`` names the shared ``.git`` of every linked
    worktree; its parent is the main checkout. ``None`` when git cannot answer.
    """
    try:
        result = subprocess.run(
            ["git", "rev-parse", "--git-common-dir"],
            capture_output=True,
            text=True,
            cwd=repo_root,
            timeout=5,
            check=False,
        )
    except Exception:  # noqa: BLE001 - git missing => location unknown
        return None
    common = result.stdout.strip()
    if result.returncode != 0 or not common:
        return None
    common_dir = Path(common)
    if not common_dir.is_absolute():
        common_dir = (repo_root / common_dir).resolve()
    if common_dir.name != ".git":
        return None  # bare/unusual layout: no checkout to put runtime/ in
    return common_dir.parent


def default_trial_ledger_path() -> Path:
    """``<QUANTAGENT_HOME>/state/trial_ledger.jsonl``, else the MAIN checkout's runtime.

    With ``QUANTAGENT_HOME`` set (tests set it to a tmp dir) that home is used.
    Otherwise ``quant_paths()`` would resolve to *this* checkout's ``runtime/``,
    which for a linked git worktree is deleted with the worktree -- a silent
    family-N undercount. The default therefore follows the shared git dir to the
    main checkout, and refuses when that cannot be determined.
    """
    from quantagent.config.paths import DEFAULT_QUANT_HOME_ENV, quant_paths

    if os.environ.get(DEFAULT_QUANT_HOME_ENV):
        return quant_paths().home / "state" / LEDGER_FILENAME
    main = _main_checkout_root(_REPO_ROOT)
    if main is None:
        raise TrialLedgerLocationUnknown(
            "cannot locate a durable trial ledger: QUANTAGENT_HOME is unset and the main git "
            f"checkout of {_REPO_ROOT} could not be determined; set QUANTAGENT_HOME or pass an "
            "explicit TrialLedger(path)"
        )
    return main / "runtime" / "state" / LEDGER_FILENAME


def current_git_hash() -> str:
    try:
        result = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            capture_output=True,
            text=True,
            cwd=_REPO_ROOT,
            timeout=5,
            check=False,
        )
    except Exception:  # noqa: BLE001 - provenance only; never block a run on git
        return "unknown"
    return result.stdout.strip() or "unknown"


def _clean(value: Any) -> Any:
    """JSON-safe, deterministic representation. Non-finite floats become None."""
    if isinstance(value, bool) or value is None or isinstance(value, (str, int)):
        return value
    if isinstance(value, float):
        return value if math.isfinite(value) else None
    if isinstance(value, Mapping):
        return {str(key): _clean(value[key]) for key in sorted(value, key=str)}
    if isinstance(value, (list, tuple)):
        return [_clean(item) for item in value]
    if hasattr(value, "item") and callable(value.item):  # numpy scalars
        return _clean(value.item())
    if hasattr(value, "isoformat"):
        return value.isoformat()
    return str(value)


def _canonical_json(payload: Mapping[str, Any]) -> str:
    return json.dumps(
        _clean(payload), ensure_ascii=False, sort_keys=True, separators=(",", ":"), allow_nan=False
    )


def config_hash(config: Mapping[str, Any]) -> str:
    """Stable sha256 of a run configuration (key order and numpy types do not matter)."""
    return hashlib.sha256(_canonical_json(config).encode("utf-8")).hexdigest()


def _record_hash(previous_hash: str, body: Mapping[str, Any]) -> str:
    return hashlib.sha256(f"{previous_hash}{_canonical_json(body)}".encode("utf-8")).hexdigest()


_BODY_KEYS = (
    "schema",
    "sequence",
    "family",
    "trial_id",
    "trials",
    "command",
    "git_hash",
    "config_hash",
    "window",
    "metrics",
    "verdict",
    "recorded_at",
)


@dataclass(frozen=True)
class TrialRecord:
    """One evaluated trial. ``window`` is the data window the trial was scored on."""

    family: str
    trial_id: str
    command: str
    git_hash: str
    config_hash: str
    window: Mapping[str, Any]
    metrics: Mapping[str, Any] = field(default_factory=dict)
    verdict: str = ""
    #: Multiple-testing trials this entry adds to the family N. ``1`` for a
    #: plain trial; ``0`` for an outcome whose trial was reserved by ``begin_run``.
    trials: int = 1

    def __post_init__(self) -> None:
        if isinstance(self.trials, bool) or int(self.trials) != self.trials or int(self.trials) < 0:
            raise ValueError("trials must be a non-negative integer")
        missing = [
            name
            for name in ("family", "trial_id", "command", "git_hash", "config_hash")
            if not str(getattr(self, name) or "").strip()
        ]
        if missing:
            raise ValueError(f"trial record fields cannot be empty: {missing}")


class TrialLedger:
    """Append-only, hash-chained, lock-protected trial ledger (one JSONL file)."""

    def __init__(self, path: str | os.PathLike[str] | None = None) -> None:
        self.path = Path(path) if path is not None else default_trial_ledger_path()
        self._write_failure: str | None = None

    # -- writing --------------------------------------------------------------
    @contextmanager
    def _exclusive_lock(self) -> Iterator[None]:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        lock_path = self.path.with_suffix(self.path.suffix + ".lock")
        with lock_path.open("a+b") as handle:
            if _fcntl is not None:
                _fcntl.flock(handle.fileno(), _fcntl.LOCK_EX)
                try:
                    yield
                finally:
                    _fcntl.flock(handle.fileno(), _fcntl.LOCK_UN)
                return
            if _msvcrt is not None:  # pragma: no cover - Windows host
                handle.seek(0, os.SEEK_END)
                if handle.tell() == 0:
                    handle.write(b"\0")
                    handle.flush()
                    os.fsync(handle.fileno())
                handle.seek(0)
                _msvcrt.locking(handle.fileno(), _msvcrt.LK_LOCK, 1)
                try:
                    yield
                finally:
                    handle.seek(0)
                    _msvcrt.locking(handle.fileno(), _msvcrt.LK_UNLCK, 1)
                return
            raise RuntimeError("no supported cross-process file locking primitive")

    def record_trial(
        self,
        family: str,
        trial_id: str,
        command: str,
        git_hash: str,
        config_hash: str,
        window: Mapping[str, Any],
        metrics: Mapping[str, Any] | None = None,
        verdict: str = "",
        *,
        trials: int = 1,
    ) -> dict[str, Any]:
        return self.record_trials(
            [
                TrialRecord(
                    family=family,
                    trial_id=trial_id,
                    command=command,
                    git_hash=git_hash,
                    config_hash=config_hash,
                    window=dict(window),
                    metrics=dict(metrics or {}),
                    verdict=verdict,
                    trials=trials,
                )
            ]
        )[0]

    def record_trials(self, trials: Sequence[TrialRecord]) -> list[dict[str, Any]]:
        """Append a batch atomically: one lock, one ``write``, one ``fsync``.

        Refuses to chain onto a ledger that does not verify (including a torn
        tail from a killed writer): an operator must look at it first.
        """
        if self._write_failure is not None:
            raise TrialLedgerWriteUnavailable(self._write_failure)
        trials = list(trials)
        if not trials:
            return []
        with self._exclusive_lock():
            return self._append_locked(trials)

    def begin_run(
        self,
        family: str,
        *,
        run_id: str,
        command: str,
        git_hash: str,
        config_hash: str,
        window: Mapping[str, Any],
        planned_trials: int,
    ) -> dict[str, Any]:
        """Atomically read the prior family N and reserve this run's trials.

        Under one lock: verify the chain, compute ``prior`` (baseline + every
        entry's trials), append a ``started`` entry carrying ``planned_trials``.
        The returned ``priorFamilyTrials`` is what the run adds its own trials
        to for DSR; the reservation is already durable if the run then dies.
        """
        if self._write_failure is not None:
            raise TrialLedgerWriteUnavailable(self._write_failure)
        baseline = HISTORICAL_BASELINES.get(family)
        base = baseline.count if baseline is not None else 0
        with self._exclusive_lock():
            report = self.verify()
            if not report["valid"]:
                raise TrialLedgerCorruption(
                    f"refusing to start a run on {self.path}: {report.get('reason')} at line "
                    f"{report.get('brokenAt')}"
                )
            recorded = self._sum_trials(family)
            head = report["headHash"]
            self._append_locked(
                [
                    TrialRecord(
                        family=family,
                        trial_id=f"{run_id}:start",
                        command=command,
                        git_hash=git_hash,
                        config_hash=config_hash,
                        window=dict(window),
                        metrics={"plannedTrials": int(planned_trials)},
                        verdict="started",
                        trials=int(planned_trials),
                    )
                ]
            )
        return {
            "ledgerPath": str(self.path),
            "family": family,
            "runId": run_id,
            "historicalBaseline": base,
            "historicalBaselineSource": baseline.source if baseline is not None else "none declared",
            "recordedBeforeRun": recorded,
            "priorFamilyTrials": base + recorded,
            "plannedTrials": int(planned_trials),
            "headHashBeforeRun": head,
        }

    def _append_locked(self, trials: list[TrialRecord]) -> list[dict[str, Any]]:
        """Chain ``trials`` onto the verified tail. Caller holds the lock."""
        report = self.verify()
        if not report["valid"]:
            raise TrialLedgerCorruption(
                f"refusing to append to {self.path}: {report.get('reason')} at line "
                f"{report.get('brokenAt')}; inspect the file before recording more trials"
            )
        previous = str(report["headHash"])
        sequence = int(report["records"])
        recorded_at = datetime.now(timezone.utc).isoformat(timespec="seconds")
        entries: list[dict[str, Any]] = []
        for trial in trials:
            body = {
                "schema": SCHEMA_VERSION,
                "sequence": sequence,
                "family": trial.family,
                "trial_id": trial.trial_id,
                "trials": int(trial.trials),
                "command": trial.command,
                "git_hash": trial.git_hash,
                "config_hash": trial.config_hash,
                "window": _clean(dict(trial.window)),
                "metrics": _clean(dict(trial.metrics)),
                "verdict": str(trial.verdict),
                "recorded_at": recorded_at,
            }
            record_hash = _record_hash(previous, body)
            entries.append({**body, "previous_hash": previous, "record_hash": record_hash})
            previous = record_hash
            sequence += 1
        blob = "".join(_canonical_json(entry) + "\n" for entry in entries).encode("utf-8")
        fd = os.open(self.path, os.O_APPEND | os.O_CREAT | os.O_WRONLY, 0o644)
        try:
            written = os.write(fd, blob)
            if written != len(blob):
                raise OSError(f"short write: {written} of {len(blob)} bytes")
            os.fsync(fd)
        except OSError as exc:
            self._write_failure = f"trial ledger write failed ({exc}); restart to re-read {self.path}"
            raise TrialLedgerWriteUnavailable(self._write_failure) from exc
        finally:
            os.close(fd)
        return entries

    # -- reading --------------------------------------------------------------
    def _lines(self) -> list[str]:
        if not self.path.exists():
            return []
        return self.path.read_text(encoding="utf-8").split("\n")

    def verify(self) -> dict[str, Any]:
        """Recompute the chain; report the first break instead of a bare bool."""
        lines = self._lines()
        # A well-formed file ends with "\n", so the final split element is "".
        if lines and lines[-1] != "":
            return {
                "valid": False,
                "brokenAt": len(lines),
                "reason": "torn tail (last line not newline-terminated)",
                "records": len(lines) - 1,
            }
        previous = GENESIS_HASH
        count = 0
        for index, line in enumerate(lines[:-1], start=1):
            try:
                row = json.loads(line)
            except json.JSONDecodeError:
                return {"valid": False, "brokenAt": index, "reason": "unparsable line", "records": count}
            body = {key: row.get(key) for key in _BODY_KEYS}
            if (
                row.get("previous_hash") != previous
                or row.get("sequence") != count
                or row.get("record_hash") != _record_hash(previous, body)
                or set(row) != set(_BODY_KEYS) | {"previous_hash", "record_hash"}
            ):
                return {"valid": False, "brokenAt": index, "reason": "hash chain mismatch", "records": count}
            previous = str(row["record_hash"])
            count += 1
        return {"valid": True, "records": count, "headHash": previous}

    def require_intact(self) -> dict[str, Any]:
        report = self.verify()
        if not report["valid"]:
            raise TrialLedgerCorruption(
                f"trial ledger {self.path} does not verify: {report.get('reason')} at line "
                f"{report.get('brokenAt')}; family trial counts cannot be trusted"
            )
        return report

    def read(self, family: str | None = None) -> list[dict[str, Any]]:
        self.require_intact()
        rows = [json.loads(line) for line in self._lines() if line]
        return [row for row in rows if family is None or row.get("family") == family]

    def _sum_trials(self, family: str) -> int:
        return sum(int(row.get("trials", 0)) for row in self.read(family))

    def recorded_count(self, family: str) -> int:
        """Trials recorded for ``family`` (sum of entry weights, not line count)."""
        return self._sum_trials(family)

    def family_count(self, family: str) -> int:
        """Cumulative N for ``family``: declared historical baseline + recorded trials.

        Raises ``TrialLedgerCorruption`` when the chain does not verify: an
        undercount is exactly the failure this ledger exists to prevent, so a
        damaged ledger fails closed rather than reporting a smaller number.
        """
        baseline = HISTORICAL_BASELINES.get(family)
        return (baseline.count if baseline is not None else 0) + self.recorded_count(family)

    def family_summary(self, family: str, *, run_trials: int = 0) -> dict[str, Any]:
        """Provenance block for an artifact: where N came from and what it was."""
        report = self.require_intact()
        baseline = HISTORICAL_BASELINES.get(family)
        recorded = self.recorded_count(family)
        base = baseline.count if baseline is not None else 0
        return {
            "ledgerPath": str(self.path),
            "family": family,
            "historicalBaseline": base,
            "historicalBaselineSource": baseline.source if baseline is not None else "none declared",
            "recordedBeforeRun": recorded,
            "priorFamilyTrials": base + recorded,
            "runTrials": int(run_trials),
            "cumulativeTrials": base + recorded + int(run_trials),
            "headHashBeforeRun": report["headHash"],
        }


def outcome_records(
    outcomes: Sequence[TrialRecord], *, planned_trials: int, run_id: str
) -> list[TrialRecord]:
    """Outcome entries for a run whose trials ``begin_run`` already reserved.

    Each outcome is re-weighted to ``trials=0`` (it was counted by the
    reservation); if the run evaluated more than it reserved, one ``overflow``
    entry adds the difference, so N can only go up.
    """
    records = [
        TrialRecord(
            family=item.family,
            trial_id=item.trial_id,
            command=item.command,
            git_hash=item.git_hash,
            config_hash=item.config_hash,
            window=item.window,
            metrics=item.metrics,
            verdict=item.verdict,
            trials=0,
        )
        for item in outcomes
    ]
    extra = len(outcomes) - int(planned_trials)
    if extra > 0 and outcomes:
        first = outcomes[0]
        records.append(
            TrialRecord(
                family=first.family,
                trial_id=f"{run_id}:overflow",
                command=first.command,
                git_hash=first.git_hash,
                config_hash=first.config_hash,
                window=first.window,
                metrics={"plannedTrials": int(planned_trials), "evaluatedTrials": len(outcomes)},
                verdict="overflow",
                trials=extra,
            )
        )
    return records


__all__ = [
    "FAMILY_NONLINEAR_FUSION",
    "FAMILY_RDAGENT_FACTOR",
    "FamilyBaseline",
    "GENESIS_HASH",
    "HISTORICAL_BASELINES",
    "SCHEMA_VERSION",
    "TrialLedger",
    "TrialLedgerCorruption",
    "TrialLedgerLocationUnknown",
    "TrialLedgerWriteUnavailable",
    "TrialRecord",
    "config_hash",
    "current_git_hash",
    "default_trial_ledger_path",
    "outcome_records",
]
