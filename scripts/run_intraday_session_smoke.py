#!/usr/bin/env python3
"""Validate the canonical session resampler against a real BaoStock 5-minute pull.

The parent ``run_market_data_smoke.py`` performs the network pull and writes a raw
5-minute sample. This stage consumes that exact artifact so the 10m/60m
aggregation contract is exercised on real public-provider data without turning a
fixed smoke symbol into factor/performance evidence.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd

from quantagent.data.intraday_sessions import (
    aggregate_ashare_bars,
    assert_raw_execution_prices,
)


SCHEMA = "intraday_session_smoke_v1"


def _validate(frame: pd.DataFrame, *, minutes: int) -> dict[str, object]:
    if frame.empty:
        raise RuntimeError(f"session-resampled {minutes}m frame is empty")
    if set(frame["bar_minutes"].astype(int).unique()) != {minutes}:
        raise RuntimeError(f"session-resampled {minutes}m has wrong bar_minutes")
    if set(frame["price_adjustment"].astype(str).unique()) != {"raw"}:
        raise RuntimeError(f"session-resampled {minutes}m is not raw")
    if not frame["execution_eligible"].fillna(False).astype(bool).all():
        raise RuntimeError(f"session-resampled {minutes}m is not execution eligible")
    assert_raw_execution_prices(frame)

    local = pd.to_datetime(frame["timestamp"], errors="coerce", utc=True).dt.tz_convert(
        "Asia/Shanghai"
    )
    starts = pd.to_datetime(frame["bar_start"], errors="coerce", utc=True).dt.tz_convert(
        "Asia/Shanghai"
    )
    if local.isna().any() or starts.isna().any():
        raise RuntimeError(f"session-resampled {minutes}m has invalid timestamps")

    clock = local.dt.strftime("%H:%M")
    bad_lunch = (clock > "11:30") & (clock <= "13:00")
    if bad_lunch.any():
        raise RuntimeError(f"session-resampled {minutes}m emitted a lunch-break bar")

    morning = frame["session"].astype(str).eq("morning")
    afternoon = frame["session"].astype(str).eq("afternoon")
    if not (morning | afternoon).all():
        raise RuntimeError(f"session-resampled {minutes}m has an unknown session")
    if (morning & ((clock > "11:30") | (clock <= "09:30"))).any():
        raise RuntimeError(f"session-resampled {minutes}m crossed morning boundaries")
    if (afternoon & ((clock > "15:00") | (clock <= "13:00"))).any():
        raise RuntimeError(f"session-resampled {minutes}m crossed afternoon boundaries")

    if minutes == 60:
        allowed = {"10:30", "11:30", "14:00", "15:00"}
        observed = set(clock.unique())
        if not observed.issubset(allowed):
            raise RuntimeError(
                f"60m session labels must be subset of {sorted(allowed)}, observed={sorted(observed)}"
            )

    return {
        "minutes": minutes,
        "rows": int(len(frame)),
        "symbols": sorted(frame["symbol"].astype(str).unique()),
        "sessions": sorted(frame["session"].astype(str).unique()),
        "timestamp_min": local.min().isoformat(),
        "timestamp_max": local.max().isoformat(),
        "raw_execution_prices": True,
        "right_labelled": True,
        "right_closed": True,
        "emit_partial": False,
        "lunch_break_isolated": True,
    }


def run(*, input_csv: Path, output_dir: Path) -> dict[str, object]:
    if not input_csv.exists():
        raise RuntimeError(f"missing raw 5m smoke input: {input_csv}")
    raw = pd.read_csv(input_csv)
    if raw.empty:
        raise RuntimeError("raw 5m smoke input is empty")
    assert_raw_execution_prices(raw)

    output_dir.mkdir(parents=True, exist_ok=True)
    evidence: dict[str, object] = {}
    for minutes in (10, 60):
        bars = aggregate_ashare_bars(raw, minutes=minutes, emit_partial=False)
        evidence[str(minutes)] = _validate(bars, minutes=minutes)
        bars.to_csv(output_dir / f"session_{minutes}m_raw_sample.csv", index=False)

    manifest: dict[str, object] = {
        "schema": SCHEMA,
        "research_smoke_only": True,
        "performance_evidence": False,
        "factor_promotion": False,
        "model_promotion": False,
        "economic_live_eligible": False,
        "source_input": str(input_csv),
        "source_frequency_minutes": 5,
        "source_adjustment": "raw",
        "timezone": "Asia/Shanghai",
        "session_windows": [["09:30", "11:30"], ["13:00", "15:00"]],
        "aggregation": evidence,
    }
    (output_dir / "session_manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2, sort_keys=True),
        encoding="utf-8",
    )
    return manifest


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--input-csv",
        default="runtime/research/market_data_smoke/minute_5m_raw_sample.csv",
    )
    parser.add_argument(
        "--output-dir",
        default="runtime/research/market_data_smoke",
    )
    args = parser.parse_args()
    manifest = run(input_csv=Path(args.input_csv), output_dir=Path(args.output_dir))
    print(json.dumps(manifest, ensure_ascii=False, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
