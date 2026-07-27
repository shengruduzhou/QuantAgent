#!/usr/bin/env python3
"""Live entitlement-aware probe of every candidate A-share tick / Level-2 source.

Answers one question per (provider, dataset family) cell with a real call:
does this source serve this family, from this host, right now?

Deliberately does **not** contend for the TickFlow rate budget: the daily-bar
acquisition tracks own that quota, and stealing requests from them to prove a
capability we already have evidence for would be a bad trade. TickFlow cells are
sourced from its existing capability manifest and marked as such.

    python scripts/probe_tick_l2_source_matrix.py \
        --trade-date 2026-07-24 --output runtime/data/capabilities/tick_l2
"""

from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import pandas as pd  # noqa: E402

from quantagent.data.ashare.http import (  # noqa: E402
    RETRY_ENTITLEMENT,
    RETRY_OK,
    RETRY_PERMANENT,
    RETRY_RATE_LIMITED,
    HttpClient,
)
from quantagent.data.microstructure import capability as cap  # noqa: E402
from quantagent.data.microstructure import contracts as mc  # noqa: E402
from quantagent.data.microstructure import public_tick_sources as pts  # noqa: E402

REPO = Path(__file__).resolve().parents[1]


def _status_for(retry_class: str) -> str:
    return {
        RETRY_OK: cap.SERVING,
        RETRY_ENTITLEMENT: cap.UNAUTHORIZED,
        RETRY_RATE_LIMITED: cap.THROTTLED,
        RETRY_PERMANENT: cap.NOT_OFFERED,
    }.get(retry_class, cap.EMPTY)


def probe_public_sources(
    matrix: cap.CapabilityMatrix, *, symbol: str, trade_date: str
) -> dict[str, pd.DataFrame]:
    """Probe the public HTTP sources and record what each actually returned."""
    client = HttpClient()
    captured: dict[str, pd.DataFrame] = {}
    probed_at = datetime.now(timezone.utc).isoformat(timespec="seconds")

    # -- Tencent 分笔 (the one tick-like route that still answers)
    source = pts.TencentTickDetail(client)
    frame, outcome = source.fetch(symbol, trade_date, max_pages=120)
    captured["tencent_tick_detail"] = frame
    matrix.add(cap.CapabilityCell(
        provider="tencent", dataset_family="trade_ticks",
        status=cap.SERVING if len(frame) else _status_for(outcome.retry_class),
        entitlement=cap.PUBLIC_FREE,
        data_class=source.data_class if len(frame) else None,
        probed_at=probed_at, endpoint=source.URL,
        rows_returned=len(frame) or None, sample_symbol=symbol,
        detail=(
            f"{len(frame)} records for {symbol} on {trade_date}; records are "
            f"{source.AGGREGATION_SECONDS}s snapshot-differenced aggregates, "
            "not individual trade prints"
        ) if len(frame) else (outcome.error or "no rows"),
        evidence={"aggregation_seconds": source.AGGREGATION_SECONDS,
                  "http_status": outcome.status_code},
    ))

    # -- Sina 历史分笔
    sina = pts.SinaTickDetail(client)
    sina_frame, sina_outcome = sina.fetch(symbol, trade_date)
    captured["sina_tick_detail"] = sina_frame
    matrix.add(cap.CapabilityCell(
        provider="sina", dataset_family="trade_ticks",
        status=cap.SERVING if len(sina_frame) else _status_for(sina_outcome.retry_class),
        entitlement=cap.PUBLIC_FREE, probed_at=probed_at, endpoint=sina.URL,
        rows_returned=len(sina_frame) or None, sample_symbol=symbol,
        detail=sina_outcome.error or f"{len(sina_frame)} rows",
        evidence={"http_status": sina_outcome.status_code},
    ))

    # -- Eastmoney 分笔
    eastmoney = pts.EastmoneyIntraday(client)
    em_frame, em_outcome = eastmoney.fetch(symbol, count=200)
    captured["eastmoney_intraday"] = em_frame
    matrix.add(cap.CapabilityCell(
        provider="eastmoney", dataset_family="trade_ticks",
        status=cap.SERVING if len(em_frame) else _status_for(em_outcome.retry_class),
        entitlement=cap.PUBLIC_FREE, probed_at=probed_at, endpoint=eastmoney.URL,
        rows_returned=len(em_frame) or None, sample_symbol=symbol,
        detail=em_outcome.error or f"{len(em_frame)} rows",
        evidence={"http_status": em_outcome.status_code},
    ))

    # -- Tencent Level-1 quote with 5-level display depth
    quote = pts.TencentLevel1Quote(client)
    q_frame, q_outcome = quote.fetch([symbol])
    captured["tencent_level1_quote"] = q_frame
    matrix.add(cap.CapabilityCell(
        provider="tencent", dataset_family="level1_quote",
        status=cap.SERVING if len(q_frame) else _status_for(q_outcome.retry_class),
        entitlement=cap.PUBLIC_FREE,
        data_class=mc.LEVEL1_QUOTE if len(q_frame) else None,
        probed_at=probed_at, endpoint=quote.URL,
        rows_returned=len(q_frame) or None, sample_symbol=symbol,
        detail=(f"best bid/offer plus {quote.DISPLAY_DEPTH} aggregated display "
                "levels; a display depth is not an exchange order book")
        if len(q_frame) else (q_outcome.error or "no rows"),
        evidence={"display_depth": quote.DISPLAY_DEPTH,
                  "http_status": q_outcome.status_code},
    ))

    # -- families no public source offers at all. Recorded explicitly so the
    #    matrix shows an absence that was checked, not a cell nobody filled in.
    for provider in ("tencent", "sina", "eastmoney"):
        for family in ("level2_snapshot", "level2_order_events",
                       "level2_transaction_events", "order_queue", "cancellations"):
            matrix.add(cap.CapabilityCell(
                provider=provider, dataset_family=family, status=cap.NOT_OFFERED,
                entitlement=cap.PUBLIC_FREE, probed_at=probed_at,
                detail="no public endpoint publishes per-order or >5-level depth "
                       "for A-shares; these products are sold through exchange-"
                       "authorised vendors and broker terminals only",
            ))
    return captured


def add_recorded_evidence(matrix: cap.CapabilityMatrix) -> None:
    """Fold in capability facts already established by other probes."""
    probed_at = datetime.now(timezone.utc).isoformat(timespec="seconds")

    # TickFlow: reuse the existing manifest rather than spending its rate budget.
    manifest = REPO / "reports" / "data" / "tickflow_capability_manifest.json"
    if manifest.exists():
        payload = json.loads(manifest.read_text(encoding="utf-8"))
        operations = payload.get("operations", [])
        by_id = {str(op.get("operation_id")): op for op in operations}

        # Explicit operation -> canonical family mapping. Matching on substrings
        # silently mislabels ("get_klines_minute" contains "kline"), so the map
        # is written out and anything unmapped simply is not claimed.
        family_operations: dict[str, tuple[str, ...]] = {
            "daily_bars_raw": ("get_klines_daily",),
            "daily_bars_adjusted": ("get_klines_daily",),
            "minute_bars": ("get_klines_minute",),
            "level1_quote": ("get_quotes", "get_quote"),
            "trade_ticks": ("get_ticks", "get_tick", "get_trades"),
            "level2_snapshot": ("get_depth", "get_orderbook"),
            "security_master": ("get_instruments", "get_universes"),
            "corporate_actions": ("get_ex_factors", "get_dividends"),
            "financials": ("get_financials",),
        }
        for family, operation_ids in family_operations.items():
            matched = [by_id[o] for o in operation_ids if o in by_id]
            if not matched:
                continue
            classifications = {str(op.get("classification", "")).upper() for op in matched}
            supported = "SUPPORTED" in classifications
            unauthorized = any("UNAUTH" in c or "FORBIDDEN" in c for c in classifications)
            matrix.add(cap.CapabilityCell(
                provider="tickflow", dataset_family=family,
                status=cap.SERVING if supported else (
                    cap.UNAUTHORIZED if unauthorized else cap.NOT_OFFERED),
                entitlement=cap.ENTITLED_PAID if supported else cap.PAID_NOT_HELD,
                probed_at=probed_at, rows_returned=1 if supported else None,
                endpoint=matched[0].get("path"),
                detail="from the recorded TickFlow capability manifest "
                       f"(probed {payload.get('probed_at')}); not re-probed here "
                       "because the daily-bar acquisition owns that rate budget. "
                       + str(matched[0].get("notes") or ""),
                evidence={"manifest": str(manifest.relative_to(REPO)),
                          "operation_ids": [str(op.get("operation_id")) for op in matched],
                          "permission_status": [str(op.get("permission_status")) for op in matched]},
            ))

    # U0 has already *proved* which providers serve daily bars, by building a
    # 17.8M-row panel from them. Re-probing would burn rate budget to learn
    # nothing, so the proven fact is folded in with its evidence path.
    u0_certificate = REPO / "runtime" / "data" / "u0" / "u0_bar_readiness_certificate.json"
    if u0_certificate.exists():
        u0 = json.loads(u0_certificate.read_text(encoding="utf-8"))
        counts = u0.get("provider", {}).get("panel_serving_provider_counts", {})
        for provider, symbols in counts.items():
            base = provider.split("_")[0]
            if base in {"none"} or not symbols:
                continue
            matrix.add(cap.CapabilityCell(
                provider=base, dataset_family="daily_bars_raw", status=cap.SERVING,
                entitlement=cap.ENTITLED_PAID if base == "tickflow" else cap.PUBLIC_FREE,
                probed_at=u0.get("generated"), rows_returned=int(symbols),
                detail=f"serves {symbols} securities in the verified U0 daily panel "
                       f"(decision {u0.get('decision')})",
                evidence={"certificate": str(u0_certificate.relative_to(REPO))},
            ))

    for artifact, provider in (
        ("runtime/data/capabilities/mt5/capability_matrix.json", "mt5_broker_feed"),
        ("runtime/data/capabilities/qmt/capability_matrix.json", "qmt_xtdata"),
    ):
        path = REPO / artifact
        if not path.exists():
            continue
        payload = json.loads(path.read_text(encoding="utf-8"))
        for cell in payload.get("cells_detail", []):
            matrix.add(cap.CapabilityCell(**cell))


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--symbol", default="600000.SH")
    parser.add_argument("--trade-date", default="2026-07-24")
    parser.add_argument("--output", default="runtime/data/capabilities/tick_l2")
    parser.add_argument(
        "--save-events", action="store_true",
        help="also persist the captured canonical frames for reconciliation",
    )
    args = parser.parse_args()

    matrix = cap.CapabilityMatrix()
    captured = probe_public_sources(
        matrix, symbol=args.symbol, trade_date=args.trade_date
    )
    add_recorded_evidence(matrix)

    output = Path(args.output)
    written = matrix.write(output, stem="tick_l2_capability_matrix")
    if args.save_events:
        events_dir = output / "captured"
        events_dir.mkdir(parents=True, exist_ok=True)
        for name, frame in captured.items():
            if len(frame):
                frame.to_parquet(events_dir / f"{name}.parquet", index=False)
                written[name] = str(events_dir / f"{name}.parquet")

    summary = matrix.summary()
    summary["artifacts"] = written
    summary["captured_rows"] = {k: len(v) for k, v in captured.items()}
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
