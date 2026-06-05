"""Tickflow-backed sector_map fetcher.

Pulls the full SH/SZ/BJ listed-stock table from ``tf.exchanges`` and joins
the SW1/SW2 Shenwan industry classification from ``tf.universes``. Writes
``runtime/data/v7/silver/sector_map/sector_map.parquet`` + coverage
sidecars via :class:`SectorMapBuilder`.

Env vars:
  TICKFLOW_API_KEY      — required by TickflowProvider
  QA_AS_OF_DATE         — available_at timestamp (default: today)
  QA_OUTPUT_ROOT        — silver layer root (default: runtime/data/v7)
  QA_UNIVERSE_PATH      — parquet whose ``symbol`` column gates output

Typical runtime: ~60s (~120 universe walks against the SW1+SW2 IDs).
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from quantagent.data.providers.tickflow_provider import TickflowProvider
from quantagent.data.sector.sector_mapping import (
    SectorMapBuilder, SectorMapConfig,
)


def _load_universe(path: Path | None) -> tuple[str, ...]:
    if path is None or not path.exists():
        return ()
    df = pd.read_parquet(path, columns=["symbol"])
    return tuple(sorted(set(df["symbol"].astype(str).str.strip())))


def main() -> int:
    output_root = Path(os.environ.get("QA_OUTPUT_ROOT", "runtime/data/v7"))
    universe_path = os.environ.get("QA_UNIVERSE_PATH")
    universe = _load_universe(Path(universe_path)) if universe_path else ()
    as_of_date = os.environ.get("QA_AS_OF_DATE") or pd.Timestamp.utcnow().date().isoformat()

    print(f"output_root  : {output_root}")
    print(f"universe path: {universe_path or '(none)'}  -> {len(universe)} symbols")
    print(f"as_of_date   : {as_of_date}")

    provider = TickflowProvider(allow_network=True)
    basic = provider.stock_basic()
    if "symbol" not in basic.columns or "industry" not in basic.columns:
        raise SystemExit("tickflow stock_basic missing required columns 'symbol' / 'industry'")

    fetched_at = pd.Timestamp.utcnow().tz_localize(None)
    eff = pd.to_datetime(as_of_date).normalize()
    sector_frame = pd.DataFrame({
        "symbol": basic["symbol"].astype(str).str.strip(),
        "sector_level_1": basic["industry"].where(basic["industry"].notna(), None),
        "sector_level_2": basic.get("industry_sub", pd.Series([None]*len(basic))),
        "source": "tickflow:stock_basic+universes",
        "source_version": "tickflow_v1",
        "effective_date": eff,
        "fetched_at": fetched_at,
        "available_at": eff,
        "coverage_status": "current_snapshot",
    })
    if universe:
        sector_frame = sector_frame[sector_frame["symbol"].isin(universe)].reset_index(drop=True)

    coverage_pct = sector_frame["sector_level_1"].notna().mean()
    print(f"industry coverage: {coverage_pct*100:.1f}% "
          f"({sector_frame['sector_level_1'].notna().sum()}/{len(sector_frame)})")
    print("top 10 SW1 sectors:")
    print(sector_frame["sector_level_1"].value_counts(dropna=True).head(10).to_string())
    if coverage_pct < 0.80:
        print(f"WARN: industry coverage below 80% — sector_pool gate will be weak", flush=True)

    builder = SectorMapBuilder(SectorMapConfig(
        symbols=universe, as_of_date=as_of_date,
        source="tickflow:stock_basic+universes", source_version="tickflow_v1",
        coverage_status="current_snapshot",
        output_root=str(output_root),
    ))
    result = builder.build(source_frame=sector_frame)
    written = builder.write(result)
    print(f"sector_map: {written.output_paths.get('sector_map')}")
    print(f"coverage  : {written.coverage}")
    provider.close()
    return 0


if __name__ == "__main__":
    sys.exit(main())
