#!/usr/bin/env python3
"""Build public-data state-team inference without a curated-symbol bias.

The command requires an explicit full-universe source (CSV/Parquet) or an
explicit symbol list.  It does not silently scan a hand-picked list of known
large caps.  Holder evidence is timestamped by the filing announcement date;
a quarter-end +45 business-day estimate is disabled unless explicitly enabled.

Examples
--------
python scripts/fetch_state_team.py \
  --universe-file runtime/data/v7/silver/universe/universe.parquet \
  --dates 20251231,20260331 --output-root runtime/data/v7
"""

from __future__ import annotations

import argparse
import datetime as dt
from pathlib import Path
import time

import pandas as pd

from quantagent.data.state_team import (
    StateTeamInferenceBuilder,
    StateTeamInferenceConfig,
    holder_filings_to_events,
)


def _ak_symbol(symbol: str) -> str:
    code, _, exchange = str(symbol).partition(".")
    if exchange:
        return f"{exchange.lower()}{code}"
    if code.startswith(("5", "6", "9")):
        return f"sh{code}"
    return f"sz{code}"


def _canonical_symbol(value: object) -> str:
    text = str(value).strip().upper()
    if "." in text:
        return text
    digits = "".join(ch for ch in text if ch.isdigit())
    if len(digits) != 6:
        raise ValueError(f"invalid A-share symbol: {value!r}")
    exchange = "SH" if digits.startswith(("5", "6", "9")) else "SZ"
    return f"{digits}.{exchange}"


def _load_universe(path: Path) -> list[str]:
    if not path.exists():
        raise FileNotFoundError(path)
    if path.suffix.lower() in {".parquet", ".pq"}:
        frame = pd.read_parquet(path)
    else:
        frame = pd.read_csv(path)
    symbol_col = next((c for c in ("symbol", "instrument", "code", "证券代码") if c in frame.columns), None)
    if symbol_col is None:
        raise ValueError(f"universe file has no symbol column: {list(frame.columns)}")
    symbols: list[str] = []
    for value in frame[symbol_col].dropna().unique():
        try:
            symbols.append(_canonical_symbol(value))
        except ValueError:
            continue
    return sorted(set(symbols))


def _first_present(row: pd.Series, names: tuple[str, ...]) -> object | None:
    for name in names:
        if name in row.index and pd.notna(row.get(name)):
            return row.get(name)
    return None


def _scan(ak, symbol: str, report_period: str) -> list[dict[str, object]]:
    try:
        frame = ak.stock_gdfx_top_10_em(symbol=_ak_symbol(symbol), date=report_period)
    except Exception as exc:  # network/vendor failures are recorded by caller count
        return [{"_error": type(exc).__name__, "symbol": symbol, "report_period": report_period}]
    if frame is None or frame.empty or "股东名称" not in frame.columns:
        return []
    share_col = next(
        (c for c in ("占总股本持股比例", "持股占总股本比例", "持股比例") if c in frame.columns),
        None,
    )
    rows: list[dict[str, object]] = []
    for _, row in frame.iterrows():
        share_raw = str(row.get(share_col, "")).replace("%", "") if share_col else ""
        share = pd.to_numeric(share_raw, errors="coerce")
        announcement = _first_present(
            row,
            ("公告日期", "公告日", "披露日期", "最新公告日期", "变动公告日期"),
        )
        rows.append(
            {
                "report_period": pd.Timestamp(report_period),
                "announcement_date": pd.to_datetime(announcement, errors="coerce"),
                "symbol": symbol,
                "holder_name": str(row.get("股东名称") or ""),
                "share_pct": float(share) if pd.notna(share) else 0.0,
                "source": "akshare:stock_gdfx_top_10_em",
            }
        )
    return rows


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--symbols", help="comma-separated canonical A-share symbols")
    source.add_argument("--universe-file", type=Path, help="CSV/Parquet full-universe file")
    parser.add_argument("--dates", required=True, help="comma-separated report periods YYYYMMDD")
    parser.add_argument("--output-root", type=Path, default=Path("runtime/data/v7"))
    parser.add_argument("--min-events", type=int, default=3)
    parser.add_argument("--sleep-seconds", type=float, default=0.05)
    parser.add_argument("--max-symbols", type=int, default=0, help="0 means no cap")
    parser.add_argument("--allow-estimated-availability", action="store_true")
    parser.add_argument("--raw-output", type=Path)
    args = parser.parse_args()

    if args.universe_file:
        symbols = _load_universe(args.universe_file)
    else:
        symbols = sorted({_canonical_symbol(s) for s in args.symbols.split(",") if s.strip()})
    if args.max_symbols > 0:
        symbols = symbols[: args.max_symbols]
    if not symbols:
        raise ValueError("resolved universe is empty")
    dates = [value.strip() for value in args.dates.split(",") if value.strip()]
    if not dates:
        raise ValueError("dates is empty")

    import akshare as ak

    rows: list[dict[str, object]] = []
    failures = 0
    for report_period in dates:
        for symbol in symbols:
            result = _scan(ak, symbol, report_period)
            failures += sum(1 for row in result if "_error" in row)
            rows.extend(row for row in result if "_error" not in row)
            if args.sleep_seconds > 0:
                time.sleep(args.sleep_seconds)

    filings = pd.DataFrame(rows)
    raw_path = args.raw_output or (
        args.output_root / "raw" / "state_team" / f"holder_filings_{dt.date.today():%Y%m%d}.parquet"
    )
    raw_path.parent.mkdir(parents=True, exist_ok=True)
    filings.to_parquet(raw_path, index=False)
    if filings.empty:
        print(f"no holder rows; vendor_failures={failures}; raw={raw_path}")
        return 0

    events = holder_filings_to_events(
        filings,
        allow_estimated_availability=args.allow_estimated_availability,
    )
    cfg = StateTeamInferenceConfig(
        source="akshare:stock_gdfx_top_10_em",
        source_version=dt.date.today().strftime("%Y%m%d"),
        output_root=args.output_root,
        min_events=args.min_events,
    )
    builder = StateTeamInferenceBuilder(cfg)
    result = builder.write(builder.build(extra_events=events))
    print(
        f"symbols={len(symbols)} periods={len(dates)} filings={len(filings)} "
        f"events={len(result.frame)} vendor_failures={failures} "
        f"gate={result.coverage.get('gate')} raw={raw_path}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
