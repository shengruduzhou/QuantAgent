from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

from quantagent.backtest.execution_timing import (
    EXECUTION_TIMING_SEMANTICS,
    execution_trace_sha256,
    signal_schedule_sha256,
)
from quantagent.execution import live_model_trust_v2_execution_policy as policy
from quantagent.execution.live_model_trust_v2 import V2VerificationResult, sha256_file


def _targets(path: Path, dates: tuple[str, ...] = ("2026-01-05",)) -> pd.DataFrame:
    frame = pd.DataFrame(
        {
            "signal_date": list(dates),
            "600000.SH": [0.1 for _ in dates],
        }
    )
    frame.to_csv(path, index=False)
    return frame


def _trace(
    path: Path,
    dates: tuple[tuple[str, str], ...] = (("2026-01-05", "2026-01-06"),),
) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for idx, (signal, execution) in enumerate(dates):
        rows.append(
            {
                "record_type": "session_mapping",
                "signal_date": signal,
                "execution_date": execution,
                "status": "mapped",
                "reason": "",
                "symbol": "",
                "client_order_id": "",
                "price_source": "close",
                "reference_price": None,
                "target_weight": None,
                "execution_timing_semantics": EXECUTION_TIMING_SEMANTICS,
                "trace_schema": "execution_trace_v1",
            }
        )
        rows.append(
            {
                "record_type": "order",
                "signal_date": signal,
                "execution_date": execution,
                "status": "filled",
                "reason": "",
                "symbol": "600000.SH",
                "client_order_id": f"order-{idx + 1}",
                "price_source": "close",
                "reference_price": 10.5 + idx,
                "target_weight": 0.1,
                "execution_timing_semantics": EXECUTION_TIMING_SEMANTICS,
                "trace_schema": "execution_trace_v1",
            }
        )
    frame = pd.DataFrame(rows)
    frame.to_csv(path, index=False)
    return frame


def _summary(path: Path, targets: pd.DataFrame, trace: pd.DataFrame) -> None:
    path.write_text(
        json.dumps(
            {
                "execution_trace_sha256": execution_trace_sha256(trace),
                "strict_target_signal_schedule_sha256": signal_schedule_sha256(
                    targets["signal_date"]
                ),
                "execution_timing_semantics": EXECUTION_TIMING_SEMANTICS,
            }
        ),
        encoding="utf-8",
    )


def _payload(
    root: Path,
    target_path: Path,
    trace_path: Path,
    summary_path: Path,
) -> dict:
    artifacts = {
        role: {"root": "bundle", "path": f"unused/{role}", "sha256": "0" * 64}
        for role in policy.GOVERNED_REQUIRED_ARTIFACT_ROLES
    }
    artifacts["strict_backtest"] = {
        "root": "bundle",
        "path": summary_path.relative_to(root).as_posix(),
        "sha256": sha256_file(summary_path),
    }
    artifacts[policy.GOVERNED_TARGET_WEIGHTS_ROLE] = {
        "root": "bundle",
        "path": target_path.relative_to(root).as_posix(),
        "sha256": sha256_file(target_path),
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


def _valid_fixture(tmp_path: Path, monkeypatch):
    root = tmp_path / "bundle"
    root.mkdir()
    target_path = root / "target_weights.csv"
    trace_path = root / "execution_trace.csv"
    targets = _targets(target_path)
    trace = _trace(trace_path)
    summary = root / "strict_summary.json"
    _summary(summary, targets, trace)
    _stub_base(monkeypatch, summary)
    return root, target_path, trace_path, targets, trace, summary


def test_trace_layer_requires_target_schedule_byte_hash_semantics_and_cross_binding(
    tmp_path: Path,
    monkeypatch,
) -> None:
    root, target_path, trace_path, targets, trace, summary = _valid_fixture(tmp_path, monkeypatch)
    canonical = execution_trace_sha256(trace)
    schedule_sha = signal_schedule_sha256(targets["signal_date"])

    result = policy.verify_trace_proven_live_model_trust_v2(
        _payload(root, target_path, trace_path, summary),
        artifact_roots={"bundle": root},
    )
    assert result.ok is True, result.reasons
    assert result.evidence["execution_trace_sha256"] == canonical
    assert result.evidence["strict_target_signal_schedule_sha256"] == schedule_sha
    assert result.evidence["execution_timing_assurance"] == policy.TRACE_PROVEN_EXECUTION_ASSURANCE
    assert result.evidence["economic_live_eligible"] is False


def test_trace_byte_tamper_fails_even_before_semantic_validation(tmp_path: Path, monkeypatch) -> None:
    root, target_path, trace_path, _, _, summary = _valid_fixture(tmp_path, monkeypatch)
    payload = _payload(root, target_path, trace_path, summary)
    trace_path.write_text(trace_path.read_text(encoding="utf-8") + "\n", encoding="utf-8")

    result = policy.verify_trace_proven_live_model_trust_v2(
        payload,
        artifact_roots={"bundle": root},
    )
    assert result.ok is False
    assert any("strict_execution_trace:sha256_mismatch" in reason for reason in result.reasons)


def test_target_schedule_byte_tamper_fails(tmp_path: Path, monkeypatch) -> None:
    root, target_path, trace_path, _, _, summary = _valid_fixture(tmp_path, monkeypatch)
    payload = _payload(root, target_path, trace_path, summary)
    target_path.write_text(
        target_path.read_text(encoding="utf-8").replace("0.1", "0.2"),
        encoding="utf-8",
    )
    result = policy.verify_trace_proven_live_model_trust_v2(
        payload,
        artifact_roots={"bundle": root},
    )
    assert result.ok is False
    assert any("strict_target_weights:sha256_mismatch" in reason for reason in result.reasons)


def test_rehashed_same_session_trace_still_fails_semantic_validation(tmp_path: Path, monkeypatch) -> None:
    root, target_path, trace_path, targets, trace, summary = _valid_fixture(tmp_path, monkeypatch)
    trace["execution_date"] = trace["signal_date"]
    trace.to_csv(trace_path, index=False)
    _summary(summary, targets, trace)

    result = policy.verify_trace_proven_live_model_trust_v2(
        _payload(root, target_path, trace_path, summary),
        artifact_roots={"bundle": root},
    )
    assert result.ok is False
    assert any("execution_trace_same_or_prior_session_execution" in reason for reason in result.reasons)
    assert result.evidence["execution_timing_assurance"] == "not_trace_proven"


def test_rehashed_trace_cannot_delete_a_strict_signal_mapping(tmp_path: Path, monkeypatch) -> None:
    root = tmp_path / "bundle"
    root.mkdir()
    target_path = root / "target_weights.csv"
    trace_path = root / "execution_trace.csv"
    targets = _targets(target_path, ("2026-01-05", "2026-01-06"))
    # Adversary omits the second signal entirely and rehashes every visible artifact.
    trace = _trace(trace_path, (("2026-01-05", "2026-01-06"),))
    summary = root / "strict_summary.json"
    _summary(summary, targets, trace)
    _stub_base(monkeypatch, summary)

    result = policy.verify_trace_proven_live_model_trust_v2(
        _payload(root, target_path, trace_path, summary),
        artifact_roots={"bundle": root},
    )
    assert result.ok is False
    assert "strict_execution_trace:signal_schedule_not_equal_strict_targets" in result.reasons
    assert result.evidence["execution_timing_assurance"] == "not_trace_proven"


def test_strict_summary_must_match_trace_and_target_schedule_digests(tmp_path: Path, monkeypatch) -> None:
    root, target_path, trace_path, _, _, summary = _valid_fixture(tmp_path, monkeypatch)
    summary.write_text(
        json.dumps(
            {
                "execution_trace_sha256": "f" * 64,
                "strict_target_signal_schedule_sha256": "e" * 64,
                "execution_timing_semantics": EXECUTION_TIMING_SEMANTICS,
            }
        ),
        encoding="utf-8",
    )
    result = policy.verify_trace_proven_live_model_trust_v2(
        _payload(root, target_path, trace_path, summary),
        artifact_roots={"bundle": root},
    )
    assert result.ok is False
    assert "strict_backtest:execution_trace_sha256_mismatch" in result.reasons
    assert "strict_backtest:target_signal_schedule_sha256_mismatch" in result.reasons


def test_hash_and_semantic_parse_use_the_same_trace_bytes(
    tmp_path: Path,
    monkeypatch,
) -> None:
    root, target_path, trace_path, _, original_trace, summary = _valid_fixture(tmp_path, monkeypatch)
    payload = _payload(root, target_path, trace_path, summary)
    original_read_bytes = Path.read_bytes
    swapped = False

    def read_then_swap(path: Path) -> bytes:
        nonlocal swapped
        data = original_read_bytes(path)
        if path == trace_path and not swapped:
            swapped = True
            malicious = original_trace.copy()
            malicious["execution_date"] = malicious["signal_date"]
            malicious.to_csv(trace_path, index=False)
        return data

    monkeypatch.setattr(Path, "read_bytes", read_then_swap)
    first = policy.verify_trace_proven_live_model_trust_v2(
        payload,
        artifact_roots={"bundle": root},
    )
    # The current verification parses exactly the immutable bytes it already
    # hashed, so the post-read filesystem swap cannot change its semantics.
    assert first.ok is True, first.reasons
    assert swapped is True

    monkeypatch.setattr(Path, "read_bytes", original_read_bytes)
    second = policy.verify_trace_proven_live_model_trust_v2(
        payload,
        artifact_roots={"bundle": root},
    )
    # A subsequent verification observes the replacement and rejects the byte
    # digest before accepting any semantic claim from it.
    assert second.ok is False
    assert any("strict_execution_trace:sha256_mismatch" in reason for reason in second.reasons)
