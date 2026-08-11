"""Canonical runtime layout for paper/shadow execution evidence.

The daily signal loop, continuous paper consumer, API and UI must resolve the
same files.  Keeping these paths behind ``QUANTAGENT_HOME`` prevents a common
operator failure mode where execution writes one journal while the control
plane quietly reads a repository-local look-alike.
"""

from __future__ import annotations

from dataclasses import dataclass
from os import PathLike
from pathlib import Path

from quantagent.config.paths import quant_paths


@dataclass(frozen=True, slots=True)
class PaperRuntimePaths:
    root: Path
    pending_signals: Path
    execution_journal: Path
    canonical_ledger: Path
    operational_ledger: Path
    idempotency: Path
    paper_book: Path
    account_identity: Path

    def ensure(self) -> "PaperRuntimePaths":
        self.root.mkdir(parents=True, exist_ok=True)
        self.pending_signals.mkdir(parents=True, exist_ok=True)
        return self

    def as_dict(self) -> dict[str, str]:
        return {
            "root": str(self.root),
            "pending_signals": str(self.pending_signals),
            "execution_journal": str(self.execution_journal),
            "canonical_ledger": str(self.canonical_ledger),
            "operational_ledger": str(self.operational_ledger),
            "idempotency": str(self.idempotency),
            "paper_book": str(self.paper_book),
            "account_identity": str(self.account_identity),
        }


def paper_runtime_paths(
    home: str | PathLike[str] | None = None,
) -> PaperRuntimePaths:
    """Resolve the paper account under the canonical QuantAgent runtime root.

    ``home`` has the same semantics as :func:`quantagent.config.paths.quant_paths`:
    an explicit value wins, otherwise ``QUANTAGENT_HOME`` and then the standard
    repository runtime are used.  This function is side-effect free; writers may
    call ``ensure()`` when they are ready to create artifacts.
    """

    root = quant_paths(home=home).home / "paper"
    return PaperRuntimePaths(
        root=root,
        pending_signals=root / "pending_signals",
        execution_journal=root / "execution_journal.jsonl",
        canonical_ledger=root / "canonical_ledger.jsonl",
        operational_ledger=root / "operational_ledger.jsonl",
        idempotency=root / "idempotency.jsonl",
        paper_book=root / "paper_book.parquet",
        account_identity=root / "account_identity.json",
    )


__all__ = ["PaperRuntimePaths", "paper_runtime_paths"]
