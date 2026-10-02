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

CONVENTION_TEXT: dict[str, str] = {
    GOLD_DELAY1: "forward_return_{h}d = close(t+1+h) / close(t+1) - 1 (delay-1 executable)",
    V7_SAME_CLOSE: "forward_return_{h}d = close(t+h) / close(t) - 1 (same close, no entry delay)",
}

#: File whose presence marks a directory as a certified, immutable build.
CERTIFIED_MARKER = "quality_certificate.json"


class CertifiedArtifactError(PermissionError):
    """Raised when a writer targets a certified build directory."""


def file_sha256(path: str | Path, *, length: int = 16) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        while block := handle.read(1 << 20):
            digest.update(block)
    return digest.hexdigest()[:length]


def refuse_certified_overwrite(path: str | Path) -> None:
    """Refuse to write into a directory that holds a certified build."""
    target = Path(path)
    marker = target.parent / CERTIFIED_MARKER
    if marker.exists():
        raise CertifiedArtifactError(
            f"{target} is inside a certified build directory ({marker}); certified "
            "artifacts are immutable -- write to a new versioned directory instead"
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


def read_label_convention(path: str | Path) -> tuple[str | None, str]:
    """``(convention id, basis)`` where basis is ``metadata`` / ``inferred`` / ``unknown``.

    Inference covers files written before the stamp existed: the gold bridge
    emits ``entry_close_t1`` and no ``label_end_*``; the v7 builder emits
    ``label_end_*`` and no entry column.
    """
    import pyarrow.parquet as pq

    schema = pq.read_schema(path)
    raw = (schema.metadata or {}).get(LABEL_CONVENTION_METADATA_KEY.encode())
    if raw:
        try:
            return str(json.loads(raw.decode())["id"]), "metadata"
        except (ValueError, KeyError):
            return None, "unknown"
    names = set(schema.names)
    has_end = any(name.startswith("label_end_") for name in names)
    if "entry_close_t1" in names and not has_end:
        return GOLD_DELAY1, "inferred"
    if has_end and "entry_close_t1" not in names:
        return V7_SAME_CLOSE, "inferred"
    return None, "unknown"


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
    "CERTIFIED_MARKER", "CONVENTION_TEXT", "CertifiedArtifactError", "GOLD_DELAY1",
    "LABEL_CONVENTION_METADATA_KEY", "V7_SAME_CLOSE", "file_sha256",
    "read_label_convention", "refuse_certified_overwrite",
    "verify_labels_against_manifest", "write_labels_parquet",
]
