"""Tickflow-backed fundamentals fetcher.

Populates ``runtime/data/v7/silver/fundamentals/`` by walking the
universe and pulling 4 endpoints per symbol from the tickflow SDK:

* ``tf.financials.metrics``         — PE/PB-style + margins + growth
* ``tf.financials.income``          — revenue, profit, cost lines
* ``tf.financials.balance_sheet``   — assets, liabilities, equity
* ``tf.financials.cash_flow``       — operating / investing / financing cash flow

Outputs:

* ``silver/fundamentals/by_symbol/<symbol>.parquet`` — long-form, all
  four endpoints union-merged on (symbol, period_end, announce_date),
  one parquet per stock.
* ``silver/fundamentals/metrics_panel.parquet``      — wide form for the
  FundamentalRanker (spec §4), keyed on (symbol, period_end). Columns
  match the tickflow ``metrics`` schema plus the most useful income /
  balance / cash-flow lines.

PIT contract: ``available_at = announce_date + 1d`` so the FundamentalRanker
can join into a daily panel without ever seeing a future earnings report.

Env vars:
  TICKFLOW_API_KEY   — required
  QA_AS_OF_DATE      — fetched_at timestamp (default: today)
  QA_OUTPUT_ROOT     — silver layer root (default: runtime/data/v7)
  QA_UNIVERSE_PATH   — parquet whose ``symbol`` column gates output
  QA_UNIVERSE_TXT    — alternatively a comma-separated symbols file
  QA_RATE_SLEEP      — seconds between *endpoint calls* (default 0.55 —
                       just under tickflow's 120/min financials cap)
  QA_MAX_SYMBOLS     — limit for smoke tests (default 0 = no limit)
  QA_METRICS_ONLY    — set 1 to pull only ``financials.metrics`` (fast path
                       for FundamentalRanker; skips income / balance / cash)

Runtime:
  metrics-only:  ~3 600 × 1 × 0.55 s ≈ 35 min
  full 4 endpoints: ~3 600 × 4 × 0.55 s ≈ 2.3 h
tickflow caps financials at 120 requests / minute so the sleep cannot be
lower than ~0.5 s without retries.
"""

from __future__ import annotations

import json
import os
import sys
import time
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from quantagent.data.providers.tickflow_provider import TickflowProvider


METRICS_PANEL_KEEP = (
    # tickflow metrics schema (16 cols probed; we keep the ones spec §4 names)
    "eps_basic", "bps", "ocfps",
    "roe", "roe_diluted",
    "gross_margin", "net_margin",
    "revenue_yoy", "net_income_yoy",
    "debt_to_asset_ratio",
    "inventory_turnover",
    "operating_cash_to_revenue",
    "eps_diluted",
)


def _load_universe() -> tuple[str, ...]:
    txt_path = os.environ.get("QA_UNIVERSE_TXT")
    if txt_path and Path(txt_path).exists():
        text = Path(txt_path).read_text(encoding="utf-8")
        return tuple(sorted({s.strip() for s in text.split(",") if s.strip()}))
    parquet_path = os.environ.get("QA_UNIVERSE_PATH")
    if parquet_path and Path(parquet_path).exists():
        df = pd.read_parquet(parquet_path, columns=["symbol"])
        return tuple(sorted({str(s).strip() for s in df["symbol"]}))
    return ()


def _tag_pit(df: pd.DataFrame, *, announce_col: str = "announce_date") -> pd.DataFrame:
    """Add ``available_at`` based on the announce date."""
    if df is None or df.empty:
        return df if df is not None else pd.DataFrame()
    out = df.copy()
    out[announce_col] = pd.to_datetime(out.get(announce_col), errors="coerce")
    out["available_at"] = out[announce_col] + pd.Timedelta(days=1)
    return out


def fetch_one(
    provider: TickflowProvider,
    symbol: str,
    *,
    sleep_between_endpoints: float,
    metrics_only: bool,
) -> dict[str, pd.DataFrame]:
    """Fetch financials endpoints for one symbol with per-endpoint rate sleep.

    ``sleep_between_endpoints`` is applied **before** each call (including
    the first call for a given symbol, since the previous symbol's last
    call also counts toward the 120/min cap). Rate-limit errors are
    auto-retried with the wait advised by the server (up to 3 attempts).
    """
    endpoints = [("metrics", provider.financials_metrics)]
    if not metrics_only:
        endpoints += [
            ("income",        provider.financials_income),
            ("balance_sheet", provider.financials_balance_sheet),
            ("cash_flow",     provider.financials_cash_flow),
        ]
    out: dict[str, pd.DataFrame] = {}
    for kind, call in endpoints:
        time.sleep(sleep_between_endpoints)
        df = pd.DataFrame()
        for attempt in range(3):
            try:
                df = call(symbol)
                break
            except Exception as exc:  # noqa: BLE001
                exc_name = type(exc).__name__
                if exc_name == "RateLimitError" and attempt < 2:
                    msg = str(exc)
                    wait_ms = 5_000
                    import re
                    m = re.search(r"(\d+)\s*ms", msg)
                    if m:
                        wait_ms = int(m.group(1))
                    time.sleep(max(wait_ms / 1000.0, 1.0) + 0.5)
                    continue
                print(f"  WARN {symbol} {kind}: {exc_name}: {exc}", flush=True)
                df = pd.DataFrame()
                break
        out[kind] = _tag_pit(df) if df is not None else pd.DataFrame()
    return out


def main() -> int:
    universe = _load_universe()
    if not universe:
        raise SystemExit("set QA_UNIVERSE_PATH or QA_UNIVERSE_TXT to a non-empty source")
    max_syms = int(os.environ.get("QA_MAX_SYMBOLS", "0") or "0")
    if max_syms > 0:
        universe = universe[:max_syms]
    sleep_s = float(os.environ.get("QA_RATE_SLEEP", "0.55") or "0.55")
    metrics_only = os.environ.get("QA_METRICS_ONLY", "0") == "1"
    output_root = Path(os.environ.get("QA_OUTPUT_ROOT", "runtime/data/v7"))
    silver = output_root / "silver" / "fundamentals"
    by_symbol = silver / "by_symbol"
    by_symbol.mkdir(parents=True, exist_ok=True)

    endpoints_per_sym = 1 if metrics_only else 4
    est_min = len(universe) * endpoints_per_sym * sleep_s / 60.0
    print(f"universe       : {len(universe)} symbols")
    print(f"output         : {silver}")
    print(f"endpoints/sym  : {endpoints_per_sym}  (metrics_only={metrics_only})")
    print(f"sleep/endpoint : {sleep_s}s   est runtime ≈ {est_min:.0f} min")

    provider = TickflowProvider(allow_network=True)

    metrics_panel_rows: list[pd.DataFrame] = []
    n_ok = 0
    n_empty = 0
    t0 = time.time()
    for idx, sym in enumerate(universe, start=1):
        data = fetch_one(provider, sym,
                         sleep_between_endpoints=sleep_s,
                         metrics_only=metrics_only)
        # Per-symbol parquet: stash all 4 frames under a JSON-named struct
        # by writing them as a single long-form table with a `kind` column.
        long_rows = []
        for kind, df in data.items():
            if df is None or df.empty:
                continue
            d = df.copy()
            d["__kind__"] = kind
            long_rows.append(d)
        if long_rows:
            merged = pd.concat(long_rows, ignore_index=True)
            merged.to_parquet(by_symbol / f"{sym}.parquet", index=False)
            n_ok += 1
        else:
            n_empty += 1
        # Append metrics rows to the panel
        m = data.get("metrics")
        if m is not None and not m.empty:
            keep_cols = [c for c in ("symbol", "period_end", "announce_date", "available_at") + METRICS_PANEL_KEEP if c in m.columns]
            metrics_panel_rows.append(m[keep_cols])
        # Progress
        if idx % 50 == 0 or idx == len(universe):
            elapsed = time.time() - t0
            rate = idx / max(elapsed, 1e-9)
            eta = (len(universe) - idx) / max(rate, 1e-9)
            print(f"  [{idx}/{len(universe)}] ok={n_ok} empty={n_empty}  rate={rate:.2f} sym/s  ETA={eta/60:.1f}min", flush=True)

    provider.close()

    # Write the wide-form metrics panel
    panel = pd.concat(metrics_panel_rows, ignore_index=True) if metrics_panel_rows else pd.DataFrame()
    panel_path = silver / "metrics_panel.parquet"
    if not panel.empty:
        panel = panel.sort_values(["symbol", "period_end"]).reset_index(drop=True)
        panel.to_parquet(panel_path, index=False)

    # Manifest sidecar
    manifest = {
        "dataset": "fundamentals",
        "source": "tickflow.financials",
        "endpoints": ["metrics", "income", "balance_sheet", "cash_flow"],
        "universe_size": len(universe),
        "n_symbols_written": n_ok,
        "n_symbols_empty": n_empty,
        "metrics_panel_rows": int(len(panel)),
        "metrics_panel_path": str(panel_path) if not panel.empty else None,
        "by_symbol_dir": str(by_symbol),
        "elapsed_seconds": round(time.time() - t0, 1),
        "available_at_rule": "announce_date + 1d",
    }
    (silver / "manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2, default=str),
        encoding="utf-8",
    )
    print(f"\ndone. wrote {n_ok} per-symbol parquets + metrics_panel ({len(panel)} rows)")
    print(f"manifest: {silver / 'manifest.json'}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
