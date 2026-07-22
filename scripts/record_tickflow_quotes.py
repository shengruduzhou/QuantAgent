#!/usr/bin/env python3
"""Record real TickFlow quote snapshots as bounded Runtime parquet partitions."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import time

import pandas as pd

from quantagent.config.paths import quant_paths
from quantagent.data.manifest import build_manifest_for_frame


OUT_DIR = quant_paths().home / "data/v7/silver/tick_snapshots"


def _symbols(value: str, symbols_file: str) -> list[str]:
    items = [item.strip() for item in value.split(",") if item.strip()]
    if symbols_file:
        text = Path(symbols_file).read_text(encoding="utf-8")
        items.extend(item.strip() for item in text.replace(",", "\n").splitlines() if item.strip())
    return list(dict.fromkeys(items))


def _client():
    from dotenv import load_dotenv

    load_dotenv(".env", override=False)
    import tickflow

    return tickflow.TickFlow(
        api_key=os.environ["TICKFLOW_API_KEY"],
        base_url=os.environ.get("TICKFLOW_API_ENDPOINT") or None,
    )


def _persist(frame: pd.DataFrame, now: pd.Timestamp) -> Path | None:
    if frame is None or frame.empty:
        return None
    frame = frame.copy()
    if "symbol" not in frame.columns:
        raise ValueError("TickFlow quote response is missing symbol")
    frame["snapshot_time"] = now.isoformat()
    day = now.strftime("%Y-%m-%d")
    root = OUT_DIR / day
    root.mkdir(parents=True, exist_ok=True)
    path = root / f"{now.strftime('%H%M%S_%f')}.parquet"
    frame.to_parquet(path, index=False)
    build_manifest_for_frame(
        dataset_name="tickflow_quote_snapshots",
        vendor="tickflow.quotes",
        frame=frame,
        output_paths=(path,),
        start_date=day,
        end_date=day,
        symbols=frame["symbol"].astype(str).unique(),
        required_columns=("symbol", "snapshot_time"),
        warnings=("forward-only realtime quote snapshot; not historical trade ticks",),
    ).write(path.with_suffix(".manifest.json"))
    return path


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--symbols", default="")
    parser.add_argument("--symbols-file", default="")
    parser.add_argument("--loop-seconds", type=int, default=0, help="0 records exactly one snapshot")
    parser.add_argument("--max-iterations", type=int, default=0, help="0 continues until market close")
    args = parser.parse_args()
    symbols = _symbols(args.symbols, args.symbols_file)
    if not symbols:
        raise SystemExit("provide --symbols or --symbols-file")
    tf = _client()

    iteration = 0
    while True:
        now = pd.Timestamp.now(tz="Asia/Shanghai")
        try:
            frame = tf.quotes.get(symbols=symbols, as_dataframe=True)
        except Exception as exc:  # noqa: BLE001
            raise SystemExit(f"TickFlow quote request failed: {type(exc).__name__}: {exc}") from exc
        path = _persist(frame, now)
        iteration += 1
        progress = iteration / args.max_iterations if args.max_iterations else None
        print(json.dumps({
            "iteration": iteration,
            "total_iterations": args.max_iterations or None,
            "progress": min(progress, 0.99) if progress is not None else None,
            "rows": 0 if frame is None else len(frame),
            "output": str(path) if path else None,
        }, ensure_ascii=False), flush=True)
        if args.loop_seconds <= 0 or (args.max_iterations and iteration >= args.max_iterations):
            break
        if now.strftime("%H:%M") >= "15:00":
            break
        time.sleep(max(1, args.loop_seconds))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
