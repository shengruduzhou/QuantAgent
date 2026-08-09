from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

from quantagent.backtest.execution_timing import (
    EXECUTION_TIMING_SEMANTICS,
    execution_trace_sha256,
)
from quantagent.execution import live_model_trust_v2_execution_policy as policy
from quantagent.execution.live_model_trust_v2 import V2VerificationResult, sha256_file


def _trace(path: Path) -> pd.DataFrame:
    frame = pd.DataFrame(
        [
            {
                "record_type": "session_mapping",
                "signal_date": "2026-01-05",
                "execution_date": "2026-01-06",
                "status": "mapped",
                "reason": "",
                "symbol": "",
                "client_order_id": "",
                "price_source": "close",
                "reference_price": None,
                "target_weight": None,
                "execution_timing_semantics": EXECUTION_TIMING_SEMANTICS,
                "trace_schema": "execution_trace_v1",
            },
            {
                "record_type": "order",
                "signal_date": "2026-01-05",
                "execution_date": "2026-01-06",
                "status": "filled",
                "reason": "",
                "symbol": "600000.SH",
                "client_order_id": "order-1",
                "price_source": "close",
                "reference_price": 10.5,
                "target_weight": 0.1,
                "execution_timing_semantics": EXECUTION_TIMING_SEMANTICS,
                "trace_schema": "execution_trace_v1",
            },
        ]
    )
    frame.to_csv(path, index=False)
    return frame


def _payload(root: Path, trace_path: Path) -> dict:
    artifacts = {
        role: {"root": "bundle", "path": f"unused/{role}", "sha256": "0" * 64}
        for role in policy.GOVERNED_REQUIRED_ARTIFACT_ROLES
    }
    artifacts[policy.GOVERNED_EXECUTION_TRACE_ROLE] = {
        "root": "bundle",
        "path": trace_path.relative_to(root).as_posix(),
        "sha256": sha256_file(trace_path),
    }
    return {
        "schema_version": 2,
        "status": "governed_evidence_accepted",
        "trust_class": "fresh_oos_evidence",
        "artifacts": artifacts,
    }


def _stub_base(monkeypatch, strict_summary: Path) -> None:
    monkeypatch.setattr(
        policy,
        "_verify_base_governed_v2",
        lambda *args, **kwargs: V2VerificationResult(
            ok=True,
            reasons=(),
            evidence={"economic_live_eligible": False},
            resolved_paths={"strict_backtest": str(strict_summary)},
        ),
    )


def test_trace_layer_requires_byte_hash_semantics_and_strict_cross_binding(
    tmp_path: Path,
    monkeypatch,
) -> None:
    root = tmp_path / "bundle"
    root.mkdir()
    trace_path = root / "execution_trace.csv"
    frame = _trace(trace_path)
    canonical = execution_trace_sha256(frame)
    summary = root / "strict_summary.json"
    summary.write_text(
        json.dumps(
            {
                "execution_trace_sha256": canonical,
                "execution_timing_semantics": EXECUTION_TIMING_SEMANTICS,
            }
        ),
        encoding="utf-8",
    )
    _stub_base(monkeypatch, summary)

    result = policy.verify_trace_proven_live_model_trust_v2(
        _payload(root, trace_path),
        artifact_roots={"bundle": root},
    )
    assert result.ok is True, result.reasons
    assert result.evidence["execution_trace_sha256"] == canonical
    assert result.evidence["execution_timing_assurance"] == policy.TRACE_PROVEN_EXECUTION_ASSURANCE
    assert result.evidence["economic_live_eligible"] is False


def test_trace_byte_tamper_fails_even_before_semantic_validation(tmp_path: Path, monkeypatch) -> None:
    root = tmp_path / "bundle"
    root.mkdir()
    trace_path = root / "execution_trace.csv"
    frame = _trace(trace_path)
    summary = root / "strict_summary.json"
    summary.write_text(
        json.dumps(
            {
                "execution_trace_sha256": execution_trace_sha256(frame),
                "execution_timing_semantics": EXECUTION_TIMING_SEMANTICS,
            }
        ),
        encoding="utf-8",
    )
    _stub_base(monkeypatch, summary)
    payload = _payload(root, trace_path)
    trace_path.write_text(trace_path.read_text(encoding="utf-8") + "\n", encoding="utf-8")

    result = policy.verify_trace_proven_live_model_trust_v2(
        payload,
        artifact_roots={"bundle": root},
    )
    assert result.ok is False
    assert any("strict_execution_trace:sha256_mismatch" in reason for reason in result.reasons)


def test_rehashed_same_session_trace_still_fails_semantic_validation(tmp_path: Path, monkeypatch) -> None:
    root = tmp_path / "bundle"
    root.mkdir()
    trace_path = root / "execution_trace.csv"
    frame = _trace(trace_path)
    frame["execution_date"] = frame["signal_date"]
    frame.to_csv(trace_path, index=False)
    canonical = execution_trace_sha256(frame)
    summary = root / "strict_summary.json"
    summary.write_text(
        json.dumps(
            {
                "execution_trace_sha256": canonical,
                "execution_timing_semantics": EXECUTION_TIMING_SEMANTICS,
            }
        ),
        encoding="utf-8",
    )
    _stub_base(monkeypatch, summary)

    result = policy.verify_trace_proven_live_model_trust_v2(
        _payload(root, trace_path),
        artifact_roots={"bundle": root},
    )
    assert result.ok is False
    assert any("execution_trace_same_or_prior_session_execution" in reason for reason in result.reasons)
    assert result.evidence["execution_timing_assurance"] == "not_trace_proven"


def test_strict_summary_must_match_canonical_trace_digest(tmp_path: Path, monkeypatch) -> None:
    root = tmp_path / "bundle"
    root.mkdir()
    trace_path = root / "execution_trace.csv"
    _trace(trace_path)
    summary = root / "strict_summary.json"
    summary.write_text(
        json.dumps(
            {
                "execution_trace_sha256": "f" * 64,
                "execution_timing_semantics": EXECUTION_TIMING_SEMANTICS,
            }
        ),
        encoding="utf-8",
    )
    _stub_base(monkeypatch, summary)

    result = policy.verify_trace_proven_live_model_trust_v2(
        _payload(root, trace_path),
        artifact_roots={"bundle": root},
    )
    assert result.ok is False
    assert "strict_backtest:execution_trace_sha256_mismatch" in result.reasons
