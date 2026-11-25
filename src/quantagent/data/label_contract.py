"""Label files must say which label they are, and certified ones must stay put.

R3-F11: ``runtime/data/gold/full_universe/labels.parquet`` was overwritten after
certification by ``build-labels-v7`` (the workstation's "重建 Labels" action
writes to the selected labels path). The replacement used the v7 convention
``close(t+h) / close(t) - 1`` (same close, no delay) while the certified dataset
uses ``close(t+1+h) / close(t+1) - 1`` (delay-1 executable); on the E1 sample
1,918,850 of 1,927,661 rows differed. Readiness only checked that the file
existed, and the manifest's ``label_hash`` hashed column *names*, so nothing
noticed.

Three rules close that:

1. every labels parquet carries its convention id in the file's schema metadata;
2. a certified gold directory (one holding ``quality_certificate.json``) is
   never a write target for another builder;
3. readiness verifies the file's content hash and convention against the
   manifest instead of its existence.

Round 32 (ROADMAP P1-2) adds the consumer side. ``build-labels-v7``'s delay-0
labels inflate short-horizon IC by up to 53% relative to the delay-1 labels the
certified data and ``_exec_`` datasets use, and nothing stopped the two from
meeting in one training set:

4. every label producer stamps its artifact (``write_labels_file``; CSV gets a
   ``.label_convention.json`` sidecar because CSV has no schema metadata);
5. every consumer that merges labels calls ``require_label_convention``: an
   unstamped file is refused unless the caller passes an explicit legacy flag,
   and ``assert_consistent_conventions`` refuses two different conventions in
   one consumer (fail closed, with the fix in the error message).
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any, Mapping

import pandas as pd

#: Schema-metadata key carrying the convention id.
LABEL_CONVENTION_METADATA_KEY = "quantagent.label_convention"

#: ``forward_return_{h}d = close(t+1+h) / close(t+1) - 1`` -- the gold bridge.
GOLD_DELAY1 = "delay1_close_t1_to_close_t1_plus_h_v1"
#: ``forward_return_{h}d = close(t+h) / close(t) - 1`` -- the v7 label builder.
V7_SAME_CLOSE = "same_close_t_to_close_t_plus_h_v1"
#: ``forward_return_{h}d`` on the exact global market-session clock with entry at
#: session T+1 (missing entry/exit bar => NaN, never the symbol's next row) --
#: ``scripts/build_executable_labels_dataset.py`` (the ``*_exec_*`` datasets).
EXEC_DELAY1_SESSION = "delay1_exact_global_session_t1_to_t1_plus_h_v1"

CONVENTION_TEXT: dict[str, str] = {
    GOLD_DELAY1: (
        "forward_return_{h}d = close(r+1+h) / close(r+1) - 1 where r+k is the symbol's k-th next "
        "ROW (gold bridge; entry-infeasible rows dropped; exit span not session-bounded)"
    ),
    V7_SAME_CLOSE: "forward_return_{h}d = close(t+h) / close(t) - 1 (same close, no entry delay)",
    EXEC_DELAY1_SESSION: (
        "forward_return_{h}d = close(S[T+1+h]) / close(S[T+1]) - 1 on the exact global "
        "market-session clock; missing entry/exit bar => NaN (delay-1 executable)"
    ),
}

#: Sessions between the signal close and the entry price, per convention.
ENTRY_DELAY_SESSIONS: dict[str, int] = {GOLD_DELAY1: 1, V7_SAME_CLOSE: 0, EXEC_DELAY1_SESSION: 1}

#: ``build-labels-v7 --label-convention`` choices. ``delay1`` is the exact
#: global-session clock (same as ``_exec_``): entry must be the next session
#: and exit exactly h sessions later, else NaN. It is deliberately NOT the gold
#: bridge's per-symbol next-row id, so the two cannot be silently mixed.
BUILDER_CONVENTIONS: dict[str, str] = {"delay1": EXEC_DELAY1_SESSION, "delay0": V7_SAME_CLOSE}

#: ``DataFrame.attrs`` key carrying the convention of an in-memory label frame.
LABEL_CONVENTION_ATTR = "label_convention"
#: Sidecar for formats without schema metadata (CSV).
SIDECAR_SUFFIX = ".label_convention.json"
#: Bases that prove a convention was declared by the producer (not guessed).
STAMPED_BASES = frozenset({"metadata", "sidecar"})

LEGACY_FLAG_HINT = (
    "pass --allow-legacy-unstamped-labels (allow_legacy_unstamped_labels=True) to "
    "accept it explicitly; the inferred convention is then recorded as legacy"
)

#: File whose presence marks a directory as a certified, immutable build.
CERTIFIED_MARKER = "quality_certificate.json"


class CertifiedArtifactError(PermissionError):
    """Raised when a writer targets a certified build directory."""


class LabelConventionError(ValueError):
    """A labels artifact is unstamped, has an unknown stamp, or mixes conventions."""


def file_sha256(path: str | Path, *, length: int = 16) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        while block := handle.read(1 << 20):
            digest.update(block)
    return digest.hexdigest()[:length]


def refuse_certified_overwrite(path: str | Path) -> None:
    """Refuse to write into a directory that holds a certified build.

    A build is certified when its directory holds ``quality_certificate.json``,
    or when its ``manifest.json`` pins the labels file by content hash
    (``labels_file_sha256``) -- the r29 gold build has the latter only, and the
    workstation's "补齐 Labels" action targets the selected labels path.
    """
    target = Path(path)
    marker = target.parent / CERTIFIED_MARKER
    if marker.exists():
        raise CertifiedArtifactError(
            f"{target} is inside a certified build directory ({marker}); certified "
            "artifacts are immutable -- write to a new versioned directory instead"
        )
    manifest = target.parent / "manifest.json"
    if target.name == "labels.parquet" and manifest.is_file():
        try:
            pinned = json.loads(manifest.read_text(encoding="utf-8")).get("labels_file_sha256")
        except (OSError, ValueError, AttributeError):
            pinned = None
        if pinned:
            raise CertifiedArtifactError(
                f"{target} is pinned by content hash in {manifest} (labels_file_sha256); "
                "certified labels are immutable -- write to a new versioned directory instead"
            )


def write_labels_parquet(
    frame: pd.DataFrame, path: str | Path, *, convention: str
) -> str:
    """Write a labels parquet stamped with its convention; return its sha256[:16]."""
    import pyarrow as pa
    import pyarrow.parquet as pq

    if convention not in CONVENTION_TEXT:
        raise ValueError(f"unknown label convention {convention!r}")
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    table = pa.Table.from_pandas(frame, preserve_index=False)
    metadata = dict(table.schema.metadata or {})
    metadata[LABEL_CONVENTION_METADATA_KEY.encode()] = json.dumps(
        {"id": convention, "text": CONVENTION_TEXT[convention]}
    ).encode()
    pq.write_table(table.replace_schema_metadata(metadata), target)
    return file_sha256(target)


def _sidecar_path(path: str | Path) -> Path:
    target = Path(path)
    return target.with_name(target.name + SIDECAR_SUFFIX)


def write_labels_file(
    frame: pd.DataFrame, path: str | Path, *, convention: str
) -> tuple[Path, str]:
    """Write labels with their convention stamp; return ``(path, sha256[:16])``.

    Parquet carries the stamp in its schema metadata. Any other suffix is
    written as CSV plus a ``<file>.label_convention.json`` sidecar.
    """
    if convention not in CONVENTION_TEXT:
        raise ValueError(f"unknown label convention {convention!r}")
    target = Path(path)
    if target.suffix == ".parquet":
        return target, write_labels_parquet(frame, target, convention=convention)
    target.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(target, index=False)
    sha = file_sha256(target)
    _sidecar_path(target).write_text(
        json.dumps(
            {"id": convention, "text": CONVENTION_TEXT[convention], "labels_file_sha256": sha},
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )
    return target, sha


def read_label_convention(path: str | Path) -> tuple[str | None, str]:
    """``(convention id, basis)``; basis is ``metadata`` / ``sidecar`` / ``inferred`` / ``unknown``.

    Inference covers files written before the stamp existed: the gold bridge
    emits ``entry_close_t1`` and no ``label_end_*``; the v7 builder emits
    ``label_end_*`` and no entry column. An inferred convention is a guess from
    column names, not a declaration, and consumers treat it as unstamped.
    """
    target = Path(path)
    if target.suffix != ".parquet":
        sidecar = _sidecar_path(target)
        if sidecar.exists():
            try:
                payload = json.loads(sidecar.read_text(encoding="utf-8"))
                declared_sha = payload.get("labels_file_sha256")
                # A sidecar is only a stamp when it is bound to this exact file.
                if not declared_sha or declared_sha != file_sha256(target):
                    return None, "unknown"
                return str(payload["id"]), "sidecar"
            except (ValueError, KeyError):
                return None, "unknown"
        try:
            names = set(pd.read_csv(target, nrows=0).columns)
        except Exception:  # noqa: BLE001 - unreadable header => unknown
            return None, "unknown"
        return _infer_from_names(names)

    import pyarrow.parquet as pq

    schema = pq.read_schema(path)
    raw = (schema.metadata or {}).get(LABEL_CONVENTION_METADATA_KEY.encode())
    if raw:
        try:
            return str(json.loads(raw.decode())["id"]), "metadata"
        except (ValueError, KeyError):
            return None, "unknown"
    return _infer_from_names(set(schema.names))


def _infer_from_names(names: set[str]) -> tuple[str | None, str]:
    has_end = any(name.startswith("label_end_") for name in names)
    if "entry_close_t1" in names and not has_end:
        return GOLD_DELAY1, "inferred"
    if has_end and "entry_close_t1" not in names:
        return V7_SAME_CLOSE, "inferred"
    return None, "unknown"


def require_label_convention(
    path: str | Path,
    *,
    allow_legacy_unstamped: bool = False,
    consumer: str = "label consumer",
) -> dict[str, Any]:
    """Return the declared convention of a labels file, or refuse it.

    Fail-closed: an unstamped (or unreadably stamped) file is an error unless
    ``allow_legacy_unstamped`` is passed, in which case the column-inferred
    convention (possibly ``None``) is returned with a ``legacy_unstamped_*``
    basis so the downstream artifact records that it was a guess.
    """
    target = Path(path)
    if not target.is_file():
        raise LabelConventionError(f"{consumer}: labels file does not exist: {target}")
    convention, basis = read_label_convention(target)
    if basis in STAMPED_BASES:
        if convention not in CONVENTION_TEXT:
            raise LabelConventionError(
                f"{consumer}: {target} declares unknown label convention {convention!r}; "
                f"known conventions: {sorted(CONVENTION_TEXT)}"
            )
        return {
            "label_convention": convention,
            "label_convention_basis": basis,
            "label_convention_text": CONVENTION_TEXT[convention],
            "entry_delay_sessions": ENTRY_DELAY_SESSIONS[convention],
            "labels_path": str(target),
        }
    if not allow_legacy_unstamped:
        guess = f" (columns suggest {convention!r})" if convention else ""
        raise LabelConventionError(
            f"{consumer}: {target} carries no label-convention stamp{guess}. Labels without a "
            "declared convention cannot be merged: delay-0 (close(t+h)/close(t)) and delay-1 "
            "(close(t+1+h)/close(t+1)) labels differ by up to 53% in short-horizon IC. Rebuild "
            "with `quantagent build-labels-v7 --label-convention delay1` (or delay0), use the "
            f"certified gold labels, or {LEGACY_FLAG_HINT}."
        )
    return {
        "label_convention": convention,
        "label_convention_basis": f"legacy_unstamped_{basis}",
        "label_convention_text": CONVENTION_TEXT.get(convention or "", "unknown"),
        "entry_delay_sessions": ENTRY_DELAY_SESSIONS.get(convention or ""),
        "labels_path": str(target),
    }


def assert_consistent_conventions(
    sources: Mapping[str, str | None], *, consumer: str = "label consumer"
) -> str | None:
    """Refuse more than one known convention among ``{source name: convention}``.

    ``None`` (an accepted legacy file whose convention could not be inferred)
    cannot be shown to differ and is ignored; it is already recorded as legacy.
    """
    known = {name: conv for name, conv in sources.items() if conv}
    distinct = sorted(set(known.values()))
    if len(distinct) > 1:
        raise LabelConventionError(
            f"{consumer} refuses to mix label conventions: {known}. Rebuild every label "
            "source with one convention (delay-1 is canonical: certified gold labels or "
            "`build-labels-v7 --label-convention delay1`)."
        )
    return distinct[0] if distinct else None


def verify_labels_against_manifest(
    labels_path: str | Path, manifest: Mapping[str, Any] | None
) -> tuple[bool | None, dict[str, Any]]:
    """True / False / None(unknown) that the labels file is the one certified.

    Unknown when the manifest predates the contract (no declared convention or
    content hash): the file cannot be shown to be the certified one, which a
    readiness gate must not read as a pass.
    """
    path = Path(labels_path)
    evidence: dict[str, Any] = {"path": str(path)}
    if not path.exists():
        evidence["reason"] = "labels file missing"
        return False, evidence
    convention, basis = read_label_convention(path)
    evidence.update({"file_convention": convention, "convention_basis": basis})
    declared = (manifest or {}).get("label_convention_id")
    declared_hash = (manifest or {}).get("labels_file_sha256")
    evidence.update({"declared_convention": declared, "declared_sha256": declared_hash})
    if declared and convention and convention != declared:
        evidence["reason"] = "label convention differs from the manifest"
        return False, evidence
    if declared_hash:
        actual = file_sha256(path)
        evidence["file_sha256"] = actual
        if actual != declared_hash:
            evidence["reason"] = "labels file content differs from the certified build"
            return False, evidence
    if not declared or not declared_hash:
        evidence["reason"] = (
            "manifest does not declare label_convention_id/labels_file_sha256; the "
            "file cannot be shown to be the certified labels")
        return None, evidence
    return True, evidence


__all__ = [
    "BUILDER_CONVENTIONS", "CERTIFIED_MARKER", "CONVENTION_TEXT", "CertifiedArtifactError",
    "ENTRY_DELAY_SESSIONS", "EXEC_DELAY1_SESSION", "GOLD_DELAY1", "LABEL_CONVENTION_ATTR",
    "LABEL_CONVENTION_METADATA_KEY", "LabelConventionError", "STAMPED_BASES",
    "V7_SAME_CLOSE", "assert_consistent_conventions", "file_sha256",
    "read_label_convention", "refuse_certified_overwrite", "require_label_convention",
    "verify_labels_against_manifest", "write_labels_file", "write_labels_parquet",
]
