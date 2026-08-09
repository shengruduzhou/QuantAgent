#!/usr/bin/env python3
"""Research-only real-data factor validity cycle.

This command can evaluate an existing canonical market panel or explicitly pull
public BaoStock history.  Governed next-session labels never infer their clock
from the selected stocks: they use either a separately supplied market-calendar
artifact or BaoStock's independent ``query_trade_dates`` endpoint.

It intentionally does NOT:
- certify a public-data pull as production PIT evidence;
- mutate the active factor lifecycle ledger;
- arm a model, broker, or Product LIVE state;
- select factors using the final holdout.
"""

from __future__ import annotations

import argparse
from dataclasses import asdict
from hashlib import sha256
import json
from pathlib import Path
import sys

import numpy as np
import pandas as pd

from quantagent.backtest.execution_timing import EXECUTION_TIMING_SEMANTICS
from quantagent.data.providers.base import ProviderRequest
from quantagent.data.providers.baostock_provider import BaoStockConfig, BaoStockProvider
from quantagent.factors import default_registry
from quantagent.factors.executable_labels import (
    build_executable_forward_returns,
    canonical_market_sessions,
    market_session_schedule_sha256,
)
from quantagent.factors.governance_metrics import (
    FactorGateConfig,
    correlation_clusters,
    evaluate_factor_candidate,
)
from quantagent.factors.lifecycle import build_factor_lifecycle_report


RESEARCH_CYCLE_SCHEMA = "factor_research_cycle_v2_explicit_market_calendar"


def _load_table(path: Path) -> pd.DataFrame:
    suffix = path.suffix.lower()
    if suffix in {".parquet", ".pq"}:
        return pd.read_parquet(path)
    if suffix in {".csv", ".txt"}:
        return pd.read_csv(path)
    raise ValueError(f"unsupported market-panel format: {path.suffix}")


def _parse_csv(value: str) -> list[str]:
    return [item.strip() for item in str(value or "").split(",") if item.strip()]


def _research_baostock(
    *,
    symbols: list[str],
    start_date: str,
    end_date: str,
) -> tuple[pd.DataFrame, dict[str, object]]:
    if not symbols:
        raise ValueError("BaoStock research requires explicit --symbols")
    provider = BaoStockProvider(config=BaoStockConfig(adjust_flag="3"))
    request = ProviderRequest(
        start_date=start_date,
        end_date=end_date,
        symbols=tuple(symbols),
        use_cache=False,
    )
    result = provider.daily_ohlcv(request)
    if result.frame is None or result.frame.empty:
        raise RuntimeError("BaoStock returned no research rows")
    metadata = {
        "mode": "public_network_research",
        "provider": result.source,
        "quality_score": float(result.quality_score),
        "provider_point_in_time_claim": bool(result.point_in_time),
        "provider_metadata": dict(result.metadata),
        "warnings": list(result.warnings),
        "adjustment": "raw",
        "production_integrity_certified": False,
        "production_note": (
            "network/public-data research evidence only; production use requires "
            "the repository daily integrity/PIT/calendar contract"
        ),
    }
    return result.frame.copy(), metadata


def _research_baostock_calendar(
    *,
    start_date: str,
    end_date: str,
) -> tuple[pd.DatetimeIndex, dict[str, object]]:
    """Fetch BaoStock's market calendar independently of researched symbols."""

    provider = BaoStockProvider(config=BaoStockConfig(adjust_flag="3"))
    bs = provider._get_module()  # research adapter; provider owns login semantics
    try:
        provider._login(bs)
        rs = bs.query_trade_dates(start_date=start_date, end_date=end_date)
        if getattr(rs, "error_code", "0") != "0":
            raise RuntimeError(
                f"BaoStock query_trade_dates failed: {getattr(rs, 'error_msg', '')}"
            )
        rows: list[list[str]] = []
        while rs.next():
            rows.append(rs.get_row_data())
    finally:
        provider._logout(bs)
    if not rows:
        raise RuntimeError("BaoStock returned no market-calendar rows")
    columns = list(getattr(rs, "fields", []) or ["calendar_date", "is_trading_day"])
    calendar = pd.DataFrame(rows, columns=columns)
    if not {"calendar_date", "is_trading_day"}.issubset(calendar.columns):
        raise RuntimeError(
            f"BaoStock calendar missing expected fields: {calendar.columns.tolist()}"
        )
    mask = calendar["is_trading_day"].astype(str).str.strip().isin({"1", "true", "True"})
    sessions = canonical_market_sessions(calendar.loc[mask, "calendar_date"].tolist())
    return sessions, {
        "mode": "baostock_query_trade_dates",
        "provider": "baostock",
        "start_date": start_date,
        "end_date": end_date,
        "returned_calendar_rows": int(len(calendar)),
        "trading_sessions": int(len(sessions)),
        "independent_of_symbol_bars": True,
        "production_integrity_certified": False,
    }


def _load_market_calendar(path: Path) -> tuple[pd.DatetimeIndex, dict[str, object]]:
    frame = _load_table(path)
    if frame.empty:
        raise ValueError(f"market calendar is empty: {path}")
    date_column = next(
        (name for name in ("trade_date", "calendar_date", "date") if name in frame.columns),
        None,
    )
    if date_column is None:
        if len(frame.columns) != 1:
            raise ValueError(
                "market calendar needs trade_date/calendar_date/date or exactly one date column"
            )
        date_column = str(frame.columns[0])
    work = frame.copy()
    if "is_trading_day" in work.columns:
        flag = work["is_trading_day"].astype(str).str.strip().str.lower()
        work = work[flag.isin({"1", "true", "yes"})]
    sessions = canonical_market_sessions(work[date_column].tolist())
    return sessions, {
        "mode": "provided_market_calendar",
        "path": str(path),
        "input_sha256": sha256(path.read_bytes()).hexdigest(),
        "date_column": date_column,
        "trading_sessions": int(len(sessions)),
        "production_integrity_certified": False,
        "production_note": (
            "calendar file is bound and validated but not independently production-certified "
            "by this research command"
        ),
    }


def _normalise_market(frame: pd.DataFrame) -> pd.DataFrame:
    required = {"trade_date", "symbol", "close", "amount"}
    missing = sorted(required - set(frame.columns))
    if missing:
        raise ValueError(f"factor research market panel missing columns: {missing}")
    data = frame.copy()
    data["trade_date"] = pd.to_datetime(data["trade_date"], errors="coerce").dt.normalize()
    for column in ("open", "high", "low", "close", "volume", "amount"):
        if column in data.columns:
            data[column] = pd.to_numeric(data[column], errors="coerce")
    data = data.dropna(subset=["trade_date", "symbol", "close"]).sort_values(
        ["symbol", "trade_date"]
    )
    if data.duplicated(["trade_date", "symbol"]).any():
        count = int(data.duplicated(["trade_date", "symbol"]).sum())
        raise ValueError(f"factor research panel has duplicate trade_date/symbol rows: {count}")
    if (data["close"] <= 0).any() or (data["amount"].fillna(-1) < 0).any():
        raise ValueError("factor research panel has non-positive close or negative amount")
    data["adv20_cny"] = (
        data.groupby("symbol", sort=False)["amount"]
        .transform(lambda series: series.rolling(20, min_periods=10).mean())
    )
    return data.reset_index(drop=True)


def _select_registered_factors(
    market: pd.DataFrame,
    requested: list[str],
    max_factors: int,
) -> tuple[list[str], list[dict[str, object]]]:
    names = requested or default_registry.names()
    selected: list[str] = []
    skipped: list[dict[str, object]] = []
    for name in names:
        try:
            meta = default_registry.get(name)
        except KeyError:
            skipped.append({"factor": name, "reason": "not_registered"})
            continue
        missing = sorted(set(meta.required_columns) - set(market.columns))
        if missing:
            skipped.append({"factor": name, "reason": f"missing_columns:{','.join(missing)}"})
            continue
        if not bool(meta.pit_safe):
            skipped.append({"factor": name, "reason": "meta_not_pit_safe"})
            continue
        if str(meta.frequency).lower() != "daily":
            skipped.append({"factor": name, "reason": f"frequency:{meta.frequency}"})
            continue
        selected.append(name)
        if max_factors > 0 and len(selected) >= max_factors:
            break
    return selected, skipped


def _materialise_factors(
    market: pd.DataFrame,
    names: list[str],
) -> tuple[pd.DataFrame, list[dict[str, object]]]:
    base = market.copy()
    failures: list[dict[str, object]] = []
    for name in names:
        meta = default_registry.get(name)
        try:
            output = default_registry.compute(name, market).frame
            values = output[["trade_date", "symbol", "factor_value"]].copy()
            values["trade_date"] = pd.to_datetime(values["trade_date"], errors="coerce").dt.normalize()
            direction = int(meta.expected_direction if meta.expected_direction is not None else meta.direction)
            if direction == 0:
                failures.append({"factor": name, "reason": "expected_direction_unspecified"})
                continue
            values[name] = pd.to_numeric(values["factor_value"], errors="coerce") * float(direction)
            values = values.drop(columns=["factor_value"])
            base = base.merge(values, on=["trade_date", "symbol"], how="left", validate="one_to_one")
        except Exception as exc:
            failures.append({"factor": name, "reason": f"compute_error:{type(exc).__name__}:{exc}"})
    return base, failures


def _frame_digest(frame: pd.DataFrame) -> str:
    ordered = frame.sort_values(["trade_date", "symbol"]).copy()
    ordered["trade_date"] = pd.to_datetime(ordered["trade_date"]).dt.strftime("%Y-%m-%d")
    payload = ordered.to_csv(index=False, float_format="%.12g").encode("utf-8")
    return sha256(payload).hexdigest()


def run_cycle(args: argparse.Namespace) -> dict[str, object]:
    output = Path(args.output_dir)
    output.mkdir(parents=True, exist_ok=True)

    if args.market_panel:
        input_path = Path(args.market_panel)
        market = _load_table(input_path)
        source_meta: dict[str, object] = {
            "mode": "provided_market_panel",
            "path": str(input_path),
            "input_sha256": sha256(input_path.read_bytes()).hexdigest(),
            "production_integrity_certified": False,
            "production_note": "provided file was not independently production-certified by this research command",
        }
    elif args.provider == "baostock":
        market, source_meta = _research_baostock(
            symbols=_parse_csv(args.symbols),
            start_date=args.start_date,
            end_date=args.end_date,
        )
    else:
        raise ValueError("pass --market-panel or --provider baostock")

    market_calendar = str(getattr(args, "market_calendar", "") or "").strip()
    if market_calendar:
        market_sessions, calendar_meta = _load_market_calendar(Path(market_calendar))
    elif args.provider == "baostock" and not args.market_panel:
        market_sessions, calendar_meta = _research_baostock_calendar(
            start_date=args.start_date,
            end_date=args.end_date,
        )
    else:
        raise ValueError(
            "provided market panels require --market-calendar; strict labels cannot "
            "infer the global execution clock from selected-symbol rows"
        )

    market = _normalise_market(market)
    horizons = tuple(sorted({int(value) for value in _parse_csv(args.horizons)}))
    if args.target_horizon not in horizons:
        horizons = tuple(sorted({*horizons, int(args.target_horizon)}))
    labeled = build_executable_forward_returns(
        market,
        horizons=horizons,
        market_sessions=market_sessions,
    )
    working = labeled.frame

    requested = _parse_csv(args.factors)
    names, skipped = _select_registered_factors(working, requested, int(args.max_factors))
    working, failures = _materialise_factors(working, names)
    names = [name for name in names if name in working.columns]
    if not names:
        raise RuntimeError("no registered factors could be materialised from the supplied panel")

    config = FactorGateConfig(
        target_book_cny=float(args.target_book_cny),
        max_adv_participation=float(args.max_adv_participation),
    )
    target_col = f"forward_executable_return_{int(args.target_horizon)}d"
    decay_cols = {h: f"forward_executable_return_{h}d" for h in horizons}
    reports: list[dict[str, object]] = []
    lifecycle: list[dict[str, object]] = []
    for name in names:
        report = evaluate_factor_candidate(
            working,
            factor_name=name,
            target_return_col=target_col,
            target_horizon_days=int(args.target_horizon),
            decay_return_columns=decay_cols,
            library_columns=(),
            adv_col="adv20_cny",
            label_semantics=EXECUTION_TIMING_SEMANTICS,
            promotion_context=None,
            config=config,
        )
        reports.append(report.to_dict())
        lifecycle_report = build_factor_lifecycle_report(
            working,
            name,
            target_col,
            existing_factor_columns=[other for other in names if other != name],
            amount_column="amount",
            market_sessions=market_sessions,
        )
        lifecycle.append(asdict(lifecycle_report))

    valid_candidates = [str(row["factor_name"]) for row in reports if bool(row["passed"])]
    clusters = correlation_clusters(
        working,
        factor_columns=valid_candidates,
        threshold=float(args.cluster_correlation),
    )

    market_path = output / "market_panel.csv"
    calendar_path = output / "market_calendar.csv"
    factor_path = output / "factor_values.csv"
    reports_path = output / "factor_validity.json"
    lifecycle_path = output / "factor_lifecycle_diagnostics.json"
    market.to_csv(market_path, index=False)
    pd.DataFrame({"trade_date": market_sessions}).to_csv(calendar_path, index=False)
    working[["trade_date", "symbol", *names]].to_csv(factor_path, index=False)
    reports_path.write_text(json.dumps(reports, ensure_ascii=False, indent=2, default=str), encoding="utf-8")
    lifecycle_path.write_text(json.dumps(lifecycle, ensure_ascii=False, indent=2, default=str), encoding="utf-8")

    schedule_digest = market_session_schedule_sha256(market_sessions)
    if schedule_digest != labeled.schema["market_session_schedule_sha256"]:
        raise RuntimeError("label schema market-calendar digest does not match research-cycle schedule")
    manifest = {
        "schema_version": RESEARCH_CYCLE_SCHEMA,
        "research_only": True,
        "economic_live_eligible": False,
        "automatic_factor_activation": False,
        "execution_timing_semantics": EXECUTION_TIMING_SEMANTICS,
        "label_schema": labeled.schema,
        "source": source_meta,
        "market_calendar": {
            **calendar_meta,
            "market_session_schedule_sha256": schedule_digest,
            "first_session": market_sessions[0].date().isoformat(),
            "last_session": market_sessions[-1].date().isoformat(),
        },
        "market_rows": int(len(market)),
        "market_symbols": int(market["symbol"].nunique()),
        "market_dates": int(market["trade_date"].nunique()),
        "market_snapshot_sha256": _frame_digest(market),
        "requested_factors": requested,
        "materialised_factors": names,
        "skipped_factors": skipped,
        "factor_failures": failures,
        "core_valid_candidates": valid_candidates,
        "correlation_clusters": clusters,
        "target_horizon_days": int(args.target_horizon),
        "horizons": list(horizons),
        "factor_gate_config": asdict(config),
        "files": {
            "market_panel": str(market_path),
            "market_calendar": str(calendar_path),
            "factor_values": str(factor_path),
            "factor_validity": str(reports_path),
            "factor_lifecycle_diagnostics": str(lifecycle_path),
        },
    }
    manifest_path = output / "manifest.json"
    manifest_path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2, default=str), encoding="utf-8")
    return manifest


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--market-panel", default="", help="Existing CSV/Parquet market panel")
    parser.add_argument(
        "--market-calendar",
        default="",
        help=(
            "Independent CSV/Parquet market calendar. Required with --market-panel; "
            "BaoStock network research queries its calendar endpoint automatically."
        ),
    )
    parser.add_argument("--provider", choices=("none", "baostock"), default="none")
    parser.add_argument("--symbols", default="", help="Comma-separated canonical symbols for network research")
    parser.add_argument("--start-date", default="2020-01-01")
    parser.add_argument("--end-date", default=pd.Timestamp.utcnow().strftime("%Y-%m-%d"))
    parser.add_argument("--factors", default="", help="Comma-separated registered factor names; empty = available registry")
    parser.add_argument("--max-factors", type=int, default=80)
    parser.add_argument("--horizons", default="1,3,5,10,20")
    parser.add_argument("--target-horizon", type=int, default=5)
    parser.add_argument("--target-book-cny", type=float, default=10_000_000.0)
    parser.add_argument("--max-adv-participation", type=float, default=0.10)
    parser.add_argument("--cluster-correlation", type=float, default=0.85)
    parser.add_argument("--output-dir", default="runtime/research/factor_cycle")
    return parser


def main() -> int:
    parser = build_parser()
    args = parser.parse_args()
    try:
        manifest = run_cycle(args)
    except Exception as exc:
        print(f"factor research cycle failed: {type(exc).__name__}: {exc}", file=sys.stderr)
        return 2
    print(json.dumps(manifest, ensure_ascii=False, indent=2, default=str))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
