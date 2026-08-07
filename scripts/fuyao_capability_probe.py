#!/usr/bin/env python3
"""Authenticated Fuyao capability probe for QuantAgent.

Only small representative calls are made. Market Dump probes only request a
signed URL and record that one was returned; the URL itself is never printed or
written. This keeps entitlement evidence useful without leaking a temporary
credential-bearing URL.
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))

from quantagent.data.ashare.env import load_repo_env  # noqa: E402
from quantagent.data.ashare.fuyao import (  # noqa: E402
    CORPORATE_ACTIONS,
    FINANCIAL_INDICATORS,
    FUYAO_API_KEY_ENV,
    FuyaoClient,
    LIMIT_UP_POOL,
    MARKET_DUMP_ENDPOINTS,
    PRICE_HISTORICAL,
    PRICE_SNAPSHOT,
    THS_INDEX_LIST,
    VALUATION_SNAPSHOT,
)

OUT = REPO / "runtime/data/u0/capability/fuyao_capability.json"
SYMBOL = "600519.SH"


def _now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def _rows(data: Any) -> int:
    if not isinstance(data, dict):
        return 0
    item = data.get("item")
    return len(item) if isinstance(item, list) else 0


def _probe(api: FuyaoClient, family: str, path: str, params: dict[str, Any] | None = None) -> dict[str, Any]:
    started = time.monotonic()
    outcome = api.get(path, params=params)
    payload = outcome.payload if isinstance(outcome.payload, dict) else {}
    data = payload.get("data") if isinstance(payload, dict) else None
    status = "SUPPORTED" if outcome.ok else (
        "UNAUTHORIZED" if outcome.retry_class == "ENTITLEMENT" else outcome.retry_class
    )
    return {
        "provider": "fuyao",
        "dataset_family": family,
        "endpoint": path,
        "status": status,
        "business_code": payload.get("code") if isinstance(payload, dict) else None,
        "rows": _rows(data),
        "latency_s": round(time.monotonic() - started, 3),
        "detail": "" if outcome.ok else (outcome.error or "")[:200],
        "probed_at": _now(),
    }


def _probe_dump(api: FuyaoClient, kind: str, path: str) -> dict[str, Any]:
    started = time.monotonic()
    outcome = api.get(path)
    payload = outcome.payload if isinstance(outcome.payload, dict) else {}
    data = payload.get("data") if isinstance(payload, dict) else {}
    signed = bool(isinstance(data, dict) and data.get("presigned_url"))
    status = "SUPPORTED" if outcome.ok and signed else (
        "UNAUTHORIZED" if outcome.retry_class == "ENTITLEMENT" else outcome.retry_class
    )
    return {
        "provider": "fuyao",
        "dataset_family": f"market_dump_{kind}",
        "endpoint": path,
        "status": status,
        "business_code": payload.get("code") if isinstance(payload, dict) else None,
        "rows": 0,
        "signed_url_received": signed,
        "signed_url_persisted": False,
        "latency_s": round(time.monotonic() - started, 3),
        "detail": "" if status == "SUPPORTED" else (outcome.error or "")[:200],
        "probed_at": _now(),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--allow-network", action="store_true")
    parser.add_argument("--output", type=Path, default=OUT)
    args = parser.parse_args()
    if not args.allow_network:
        print("refusing to probe: --allow-network was not confirmed")
        return 2

    load_repo_env()
    api = FuyaoClient()
    if not api.configured:
        report = {
            "provider": "fuyao",
            "generated_at": _now(),
            "configured": False,
            "credential_env": FUYAO_API_KEY_ENV,
            "status": "NO_CREDENTIAL",
            "probes": [],
        }
        print(json.dumps(report, indent=2, ensure_ascii=False))
        return 3

    # Keep all calls intentionally small. Historical timestamps are a recent
    # finite window; no probe should accidentally launch a full backfill.
    recent_start = int((time.time() - 30 * 86400) * 1000)
    recent_end = int(time.time() * 1000)
    probes = [
        _probe(api, "quotes", PRICE_SNAPSHOT, {"thscodes": SYMBOL}),
        _probe(api, "daily_bars", PRICE_HISTORICAL, {
            "thscode": SYMBOL, "interval": "1d", "start": recent_start,
            "end": recent_end, "adjust": "none", "offset": 0,
        }),
        _probe(api, "valuation_metrics", VALUATION_SNAPSHOT, {"thscodes": SYMBOL, "limit": 1, "offset": 0}),
        _probe(api, "financial_indicators", FINANCIAL_INDICATORS, {"thscode": SYMBOL, "limit": 1, "offset": 0}),
        _probe(api, "corporate_actions", CORPORATE_ACTIONS, {"thscode": SYMBOL}),
        _probe(api, "limit_up_pool", LIMIT_UP_POOL, {"limit": 1, "offset": 0}),
        _probe(api, "ths_index_catalog", THS_INDEX_LIST, {"limit": 1, "offset": 0}),
    ]
    probes.extend(_probe_dump(api, kind, path) for kind, path in MARKET_DUMP_ENDPOINTS.items())
    supported = sorted({p["dataset_family"] for p in probes if p["status"] == "SUPPORTED"})
    report = {
        "provider": "fuyao",
        "generated_at": _now(),
        "configured": True,
        "credential_env": FUYAO_API_KEY_ENV,
        "supported_families": supported,
        "probes": probes,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps(report, indent=2, ensure_ascii=False))
    return 0 if supported else 4


if __name__ == "__main__":
    raise SystemExit(main())
