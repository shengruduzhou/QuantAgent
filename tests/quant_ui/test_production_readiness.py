from __future__ import annotations

from datetime import datetime, timezone
import json
from pathlib import Path

from fastapi.testclient import TestClient
import pytest

from services.quant_api.app import create_app
from services.quant_api.config import ApiSettings
from services.quant_api.services.production_readiness import ProductionReadinessService


NOW = datetime(2026, 8, 8, 12, 0, tzinfo=timezone.utc)


def _service(tmp_path: Path) -> ProductionReadinessService:
    return ProductionReadinessService(
        tmp_path,
        tmp_path / "runtime",
        now_fn=lambda: NOW,
    )


def _cards(service: ProductionReadinessService) -> dict[str, dict]:
    return {card["key"]: card for card in service.status()["cards"]}


def _write(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload), encoding="utf-8")


def _valid_until() -> str:
    return "2026-08-09T12:00:00+00:00"


def test_missing_runtime_evidence_never_becomes_green(tmp_path: Path) -> None:
    service = _service(tmp_path)
    status = service.status()
    cards = _cards(service)

    assert status["aggregateTradingReady"] is None
    assert len(status["cards"]) == 8
    assert cards["modelTrust"]["state"] == "BLOCKED"
    assert cards["brokerQuery"]["state"] == "NOT_CERTIFIED"
    assert cards["killSwitch"]["state"] == "NOT_CONFIGURED"
    assert cards["reconciliation"]["state"] == "NOT_CERTIFIED"
    assert cards["hostCertification"]["state"] == "NOT_CERTIFIED"
    assert cards["productArming"]["state"] == "NOT_ARMED"
    assert cards["targetRisk"]["state"] == "WIRED"
    assert cards["orderRisk"]["state"] == "WIRED"


def test_runtime_certificates_upgrade_only_their_own_dimensions(tmp_path: Path) -> None:
    root = tmp_path / "runtime" / "certificates"
    _write(
        root / "broker_query_readiness.json",
        {
            "as_of": "2026-08-08T11:00:00+00:00",
            "valid_until": _valid_until(),
            "query_only_ready": True,
            "preflight_ok": True,
            "health_ok": True,
        },
    )
    _write(
        root / "reconciliation.json",
        {
            "as_of": "2026-08-08T11:00:00+00:00",
            "valid_until": _valid_until(),
            "complete": True,
            "unexplained_order_differences": 0,
            "unexplained_trade_differences": 0,
            "unexplained_position_differences": 0,
            "unexplained_cash_differences": 0,
        },
    )
    _write(
        root / "qmt_host_certification.json",
        {
            "as_of": "2026-08-08T11:00:00+00:00",
            "valid_until": _valid_until(),
            "platform": "Windows 11",
            "broker": "MiniQMT",
            "controlled_host": True,
            "certified": True,
        },
    )

    cards = _cards(_service(tmp_path))
    assert cards["brokerQuery"]["state"] == "READY"
    assert cards["reconciliation"]["state"] == "CLEAN"
    assert cards["hostCertification"]["state"] == "CERTIFIED"
    # Independent cards stay visible and blocked; healthy infrastructure cannot
    # promote an untrusted model or arm the product.
    assert cards["modelTrust"]["state"] == "BLOCKED"
    assert cards["productArming"]["state"] == "NOT_ARMED"


def test_canonical_persistent_kill_switch_schema_is_read_without_invented_fields(tmp_path: Path) -> None:
    state = tmp_path / "runtime" / "live" / "kill_switch_state.json"
    _write(state, {"version": 1, "manual_triggered": False, "reasons": []})
    assert _cards(_service(tmp_path))["killSwitch"]["state"] == "CLEAR"

    _write(state, {"version": 1, "manual_triggered": False, "reasons": ["data_provider_failure"]})
    killed = _cards(_service(tmp_path))["killSwitch"]
    assert killed["state"] == "KILLED"
    assert killed["reasons"] == ["data_provider_failure"]


def test_expired_certificate_fails_closed(tmp_path: Path) -> None:
    path = tmp_path / "runtime" / "certificates" / "broker_query_readiness.json"
    _write(
        path,
        {
            "valid_until": "2026-08-08T11:59:59+00:00",
            "query_only_ready": True,
            "preflight_ok": True,
            "health_ok": True,
        },
    )
    card = _cards(_service(tmp_path))["brokerQuery"]
    assert card["state"] == "EXPIRED"
    assert card["severity"] == "blocked"


def test_runtime_certificate_symlink_is_rejected(tmp_path: Path) -> None:
    external = tmp_path / "outside.json"
    _write(
        external,
        {
            "valid_until": _valid_until(),
            "query_only_ready": True,
            "preflight_ok": True,
            "health_ok": True,
        },
    )
    target = tmp_path / "runtime" / "certificates" / "broker_query_readiness.json"
    target.parent.mkdir(parents=True, exist_ok=True)
    try:
        target.symlink_to(external)
    except (OSError, NotImplementedError):
        pytest.skip("symlink creation is unavailable on this platform")

    card = _cards(_service(tmp_path))["brokerQuery"]
    assert card["state"] == "INVALID"
    assert any("symlink_not_allowed" in reason for reason in card["reasons"])


def test_http_endpoint_returns_exactly_eight_independent_cards(tmp_path: Path) -> None:
    settings = ApiSettings(
        project_root=tmp_path,
        runtime_root=tmp_path / "runtime",
        cache_root=tmp_path / "runtime" / "cache",
        jobs_root=tmp_path / "runtime" / "jobs",
    ).ensure()
    app = create_app(settings)
    with TestClient(app) as client:
        response = client.get("/api/governance/production-readiness")
    app.state.services.paper_orders.close()

    assert response.status_code == 200
    payload = response.json()
    assert payload["status"] == "ready"
    assert payload["data"]["aggregateTradingReady"] is None
    assert {card["key"] for card in payload["data"]["cards"]} == {
        "modelTrust",
        "brokerQuery",
        "targetRisk",
        "orderRisk",
        "killSwitch",
        "reconciliation",
        "productArming",
        "hostCertification",
    }
