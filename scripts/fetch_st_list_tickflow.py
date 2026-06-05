"""Tickflow-backed ST/*ST flag fetcher.

Derives current ST/*ST status from the ``name`` field returned by
``tf.exchanges.get_instruments``. Tickflow does not expose a name-history
endpoint, so the output is a current snapshot tagged
``coverage_status='current_snapshot'`` — joining to historical OOS
predictions is unsafe and the silver schema documents this.

Env vars:
  TICKFLOW_API_KEY   — required
  QA_AS_OF_DATE      — available_at timestamp (default: today)
  QA_OUTPUT_ROOT     — silver layer root (default: runtime/data/v7)
  QA_UNIVERSE_PATH   — parquet whose ``symbol`` column gates output
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from quantagent.data.providers.tickflow_provider import TickflowProvider
from quantagent.data.sector.st_history import StFlagBuilder, StFlagConfig


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

    provider = TickflowProvider(allow_network=True)
    basic = provider.stock_basic()

    fetched_at = pd.Timestamp.utcnow().tz_localize(None)
    eff = pd.to_datetime(as_of_date).normalize()
    name_series = basic["name"].astype(str)
    is_st_series = name_series.str.upper().str.contains("ST")
    frame = pd.DataFrame({
        "symbol": basic["symbol"].astype(str).str.strip(),
        "is_st": is_st_series.astype(bool),
        "st_known": True,
        "block_weight": is_st_series.map(lambda v: 1.0 if v else 0.0),
        "st_source": "tickflow:exchanges.get_instruments",
        "source_version": "tickflow_v1",
        "effective_date": eff,
        "fetched_at": fetched_at,
        "available_at": eff,
        "coverage_status": "current_snapshot",
    })
    print(f"current ST/*ST count (universe-wide): {int(is_st_series.sum())} of {len(frame)}")
    if universe:
        present = set(frame["symbol"])
        missing = [
            {"symbol": s, "is_st": False, "st_known": False,
             "block_weight": 0.0, "st_source": "missing",
             "source_version": "tickflow_v1",
             "effective_date": eff, "fetched_at": fetched_at,
             "available_at": eff, "coverage_status": "missing"}
            for s in universe if s not in present
        ]
        if missing:
            frame = pd.concat([frame, pd.DataFrame(missing)], ignore_index=True)
        frame = frame[frame["symbol"].isin(universe)].reset_index(drop=True)

    builder = StFlagBuilder(StFlagConfig(
        symbols=universe, as_of_date=as_of_date,
        source="tickflow:exchanges.get_instruments", source_version="tickflow_v1",
        output_root=str(output_root),
    ))
    result = builder.build(source_frame=frame)
    written = builder.write(result)
    print(f"st_flags: {written.output_paths.get('st_flags')}")
    provider.close()
    return 0


if __name__ == "__main__":
    sys.exit(main())
