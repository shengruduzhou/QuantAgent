#!/usr/bin/env python3
"""Stream validated Runtime imports and filtered exports without browser uploads."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import sqlite3
from tempfile import NamedTemporaryFile
from typing import Iterator

import pandas as pd

from quantagent.config.paths import quant_paths


CHUNK_SIZE = 100_000


def _inside(path: Path, root: Path) -> bool:
    path = path.resolve()
    root = root.resolve()
    return path == root or root in path.parents


def _frames(path: Path) -> Iterator[pd.DataFrame]:
    if path.suffix.lower() == ".csv":
        yield from pd.read_csv(path, chunksize=CHUNK_SIZE)
        return
    if path.suffix.lower() != ".parquet":
        raise ValueError("only CSV and Parquet transfers are supported")
    import pyarrow.parquet as pq

    parquet = pq.ParquetFile(path)
    for batch in parquet.iter_batches(batch_size=CHUNK_SIZE):
        yield batch.to_pandas()


def _write_frame(frame: pd.DataFrame, output: Path, writer):
    if output.suffix.lower() == ".csv":
        frame.to_csv(output, mode="a", header=not output.exists(), index=False)
        return writer
    if output.suffix.lower() != ".parquet":
        raise ValueError("output must be CSV or Parquet")
    import pyarrow as pa
    import pyarrow.parquet as pq

    table = pa.Table.from_pandas(frame, preserve_index=False)
    if writer is None:
        writer = pq.ParquetWriter(output, table.schema, compression="zstd")
    writer.write_table(table)
    return writer


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--operation", choices=("import", "export"), required=True)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--date-column", default="trade_date")
    parser.add_argument("--symbol-column", default="symbol")
    parser.add_argument("--start-date", default="")
    parser.add_argument("--end-date", default="")
    parser.add_argument("--symbols", default="")
    args = parser.parse_args()

    runtime = quant_paths().home.resolve()
    source = args.source.resolve()
    output = args.output.resolve()
    if not source.is_file() or not _inside(source, runtime):
        raise SystemExit("source must be an existing file inside Runtime")
    if not _inside(output, runtime):
        raise SystemExit("output must remain inside Runtime")
    if args.operation == "import" and not _inside(source, runtime / "import_quarantine"):
        raise SystemExit("imports must originate in Runtime/import_quarantine")
    output.parent.mkdir(parents=True, exist_ok=True)
    partial = output.with_name(f".{output.stem}.partial{output.suffix}")
    partial.unlink(missing_ok=True)

    wanted_symbols = {item.strip() for item in args.symbols.split(",") if item.strip()}
    seen_path: str | None = None
    seen: sqlite3.Connection | None = None
    if args.operation == "import":
        cache = runtime / "cache"
        cache.mkdir(parents=True, exist_ok=True)
        temp = NamedTemporaryFile(prefix="qa-import-", suffix=".sqlite", dir=cache, delete=False)
        seen_path = temp.name
        temp.close()
        seen = sqlite3.connect(seen_path)
        seen.execute("CREATE TABLE keys(symbol TEXT NOT NULL, stamp TEXT NOT NULL, PRIMARY KEY(symbol, stamp)) WITHOUT ROWID")

    rows_read = rows_written = duplicates = 0
    writer = None
    try:
        for batch_index, frame in enumerate(_frames(source), start=1):
            rows_read += len(frame)
            missing = [name for name in (args.date_column, args.symbol_column) if name not in frame.columns]
            if missing:
                raise ValueError(f"missing required columns: {missing}")
            dates = pd.to_datetime(frame[args.date_column], errors="coerce", utc=True).dt.tz_convert(None)
            symbols = frame[args.symbol_column].astype("string").fillna("").str.strip()
            valid = dates.notna() & symbols.ne("")
            if args.start_date:
                valid &= dates >= pd.Timestamp(args.start_date)
            if args.end_date:
                valid &= dates <= pd.Timestamp(args.end_date) + pd.Timedelta(days=1) - pd.Timedelta(microseconds=1)
            if wanted_symbols:
                valid &= symbols.isin(wanted_symbols)
            frame = frame.loc[valid].copy()
            dates = dates.loc[valid]
            symbols = symbols.loc[valid]
            if seen is not None and not frame.empty:
                keep: list[bool] = []
                for symbol, stamp in zip(symbols.astype(str), dates.dt.strftime("%Y-%m-%dT%H:%M:%S.%f")):
                    before = seen.total_changes
                    seen.execute("INSERT OR IGNORE INTO keys(symbol, stamp) VALUES (?, ?)", (symbol, stamp))
                    unique = seen.total_changes > before
                    keep.append(unique)
                    duplicates += int(not unique)
                frame = frame.loc[keep]
            if not frame.empty:
                writer = _write_frame(frame, partial, writer)
                rows_written += len(frame)
            print(json.dumps({"batch": batch_index, "rows_read": rows_read, "rows_written": rows_written, "duplicates": duplicates}, ensure_ascii=False), flush=True)
        if writer is not None:
            writer.close()
            writer = None
        if rows_written == 0 or not partial.exists():
            raise ValueError("no valid rows matched the transfer contract")
        partial.replace(output)
        manifest = {
            "schema_version": "quantagent.data-transfer.v1",
            "operation": args.operation,
            "created_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
            "source": source.relative_to(runtime).as_posix(),
            "output": output.relative_to(runtime).as_posix(),
            "rows_read": rows_read,
            "rows_written": rows_written,
            "duplicates_removed": duplicates,
            "sha256": _sha256(output),
            "filters": {"symbols": sorted(wanted_symbols), "start_date": args.start_date, "end_date": args.end_date},
        }
        manifest_path = output.with_suffix(f"{output.suffix}.manifest.json")
        manifest_path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
        print(json.dumps({"progress": 1.0, "output": str(output), "manifest": str(manifest_path), **manifest}, ensure_ascii=False), flush=True)
    finally:
        if writer is not None:
            writer.close()
        if seen is not None:
            seen.close()
        if seen_path:
            Path(seen_path).unlink(missing_ok=True)
        if partial.exists():
            partial.unlink()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
