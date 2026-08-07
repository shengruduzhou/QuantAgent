"""Fuyao full-market Parquet download and local U0 adapter.

Use this path for bulk A-share history. The official Financial-API repository
explicitly warns against fetching ~5000 securities one-by-one when the three
Market Dump endpoints can provide full daily bars, the recent 10-session delta,
and all adjustment events.

Presigned URLs are short-lived secrets. They are used in memory and never
written to manifests, logs, or source provenance.
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

import pandas as pd
import pyarrow.parquet as pq
import requests

from quantagent.data.ashare import contracts
from quantagent.data.ashare.fuyao import FuyaoClient
from quantagent.data.ashare.http import RETRY_EMPTY, RETRY_OK, RETRY_PERMANENT, utc_now
from quantagent.data.ashare.sources import SourceResult
from quantagent.data.ashare.symbols import identify

DEFAULT_DUMP_ROOT = Path("runtime/data/fuyao")
DUMP_PATH_ENV = "FUYAO_DAILY_K_DUMP"


@dataclass(frozen=True)
class DumpArtifact:
    kind: str
    path: Path
    bytes: int
    rows: int
    columns: tuple[str, ...]

    def as_dict(self) -> dict[str, Any]:
        return {
            "kind": self.kind,
            "path": str(self.path),
            "bytes": self.bytes,
            "rows": self.rows,
            "columns": list(self.columns),
        }


def _expected_columns(kind: str) -> set[str]:
    if kind in {"daily-k", "daily-k-10d"}:
        return {
            "thscode", "currency", "interval", "adjusted", "date_ms",
            "open_price", "high_price", "low_price", "close_price", "volume", "turnover",
        }
    if kind == "adjustment-factors":
        return {
            "thscode", "ticker", "ex_date_ms", "dividend_per_share",
            "per_share_bonus", "allotment_ratio", "allotment_price", "currency",
        }
    raise ValueError(f"unknown Fuyao dump kind {kind!r}")


def validate_dump(path: Path, kind: str) -> DumpArtifact:
    """Validate metadata/schema without loading a multi-million-row dump in RAM."""
    if not path.is_file() or path.stat().st_size <= 0:
        raise RuntimeError(f"Fuyao dump is missing or empty: {path}")
    parquet = pq.ParquetFile(path)
    columns = tuple(parquet.schema_arrow.names)
    missing = sorted(_expected_columns(kind) - set(columns))
    if missing:
        raise RuntimeError(f"Fuyao {kind} dump missing columns: {missing}")
    return DumpArtifact(
        kind=kind,
        path=path,
        bytes=path.stat().st_size,
        rows=int(parquet.metadata.num_rows),
        columns=columns,
    )


def download_dump(
    kind: str,
    *,
    api: FuyaoClient | None = None,
    root: Path = DEFAULT_DUMP_ROOT,
    chunk_bytes: int = 1 << 20,
) -> DumpArtifact:
    """Sign, download with Range resume, atomically publish, then validate.

    A fresh signing call is made on every invocation. If a partial download
    exists, the presigned object is resumed when the server supports HTTP 206;
    otherwise it is restarted. The URL itself is never persisted.
    """
    api = api or FuyaoClient()
    root = Path(root)
    root.mkdir(parents=True, exist_ok=True)
    signed = api.market_dump_download_url(kind)
    url = str(signed.get("presigned_url") or "")
    if not url:
        raise RuntimeError(f"Fuyao {kind} signing response did not contain presigned_url")

    final = root / f"{kind}.parquet"
    partial = final.with_suffix(".parquet.part")
    existing = partial.stat().st_size if partial.exists() else 0
    headers = {"Range": f"bytes={existing}-"} if existing else {}
    mode = "ab" if existing else "wb"

    try:
        response = requests.get(url, headers=headers, stream=True, timeout=60)
    except requests.RequestException as exc:
        raise RuntimeError(f"Fuyao {kind} download transport failure: {type(exc).__name__}") from exc

    if existing and response.status_code == 200:
        # Range was ignored. Restart so the output cannot contain duplicated bytes.
        existing = 0
        mode = "wb"
    elif response.status_code not in {200, 206}:
        raise RuntimeError(f"Fuyao {kind} download HTTP {response.status_code}")

    with response:
        with partial.open(mode) as handle:
            for chunk in response.iter_content(chunk_size=chunk_bytes):
                if chunk:
                    handle.write(chunk)
    partial.replace(final)
    return validate_dump(final, kind)


def configured_daily_dump(root: Path = DEFAULT_DUMP_ROOT) -> Path:
    override = os.environ.get(DUMP_PATH_ENV)
    return Path(override) if override else Path(root) / "daily-k.parquet"


class FuyaoDumpSource:
    """Predicate-pushdown reader over the local Fuyao full-market daily dump."""

    name = "fuyao_dump"

    def __init__(self, path: Path | None = None) -> None:
        self.path = Path(path) if path is not None else configured_daily_dump()

    @property
    def configured(self) -> bool:
        return self.path.is_file()

    @staticmethod
    def _trade_date(ms: pd.Series) -> pd.Series:
        # Vendor schema documents date_ms at Asia/Shanghai midnight. Convert via
        # UTC then Shanghai before stripping timezone; this survives DST-agnostic
        # host environments and never depends on the workstation timezone.
        return (
            pd.to_datetime(ms, unit="ms", utc=True)
            .dt.tz_convert(contracts.TIMEZONE_CST)
            .dt.tz_localize(None)
            .dt.normalize()
        )

    def daily_bars(
        self,
        symbol: str,
        start: str | pd.Timestamp,
        end: str | pd.Timestamp,
    ) -> SourceResult:
        cols = list(contracts.DAILY_BARS.columns)
        now = utc_now()
        endpoint = str(self.path)
        if not self.configured:
            return SourceResult(
                pd.DataFrame(columns=cols), self.name, endpoint, RETRY_PERMANENT,
                now, 0, error=f"local Fuyao daily dump not found: {self.path}",
            )
        ident = identify(symbol)
        start_ts = pd.Timestamp(start).normalize()
        end_ts = pd.Timestamp(end).normalize()
        try:
            # PyArrow applies Parquet row-group statistics when available; only
            # the target security columns are materialised into Python memory.
            table = pq.read_table(
                self.path,
                columns=[
                    "thscode", "date_ms", "open_price", "high_price", "low_price",
                    "close_price", "volume", "turnover", "adjusted", "currency", "interval",
                ],
                filters=[("thscode", "=", ident.symbol)],
            )
        except Exception as exc:  # noqa: BLE001 - classify local artifact/schema failure
            return SourceResult(
                pd.DataFrame(columns=cols), self.name, endpoint, RETRY_PERMANENT,
                now, 0, error=f"dump read failed: {type(exc).__name__}: {str(exc)[:160]}",
            )
        raw = table.to_pandas()
        if raw.empty:
            return SourceResult(
                pd.DataFrame(columns=cols), self.name, endpoint, RETRY_EMPTY,
                now, 0, error="symbol absent from Fuyao dump",
            )
        # Fail closed on a schema-semantic change; never silently accept adjusted
        # prices into the canonical raw-price U0 panel.
        if set(raw["adjusted"].dropna().astype(str).str.lower()) - {"none"}:
            return SourceResult(
                pd.DataFrame(columns=cols), self.name, endpoint, RETRY_PERMANENT,
                now, 0, error="Fuyao dump contains non-raw adjusted rows",
            )
        if set(raw["currency"].dropna().astype(str).str.upper()) - {"CNY"}:
            return SourceResult(
                pd.DataFrame(columns=cols), self.name, endpoint, RETRY_PERMANENT,
                now, 0, error="Fuyao dump contains non-CNY A-share rows",
            )
        if set(raw["interval"].dropna().astype(str).str.lower()) - {"1d"}:
            return SourceResult(
                pd.DataFrame(columns=cols), self.name, endpoint, RETRY_PERMANENT,
                now, 0, error="Fuyao dump contains non-daily rows",
            )

        trade_date = self._trade_date(raw["date_ms"])
        frame = pd.DataFrame(
            {
                "symbol": ident.symbol,
                "trade_date": trade_date,
                "open": pd.to_numeric(raw["open_price"], errors="coerce"),
                "high": pd.to_numeric(raw["high_price"], errors="coerce"),
                "low": pd.to_numeric(raw["low_price"], errors="coerce"),
                "close": pd.to_numeric(raw["close_price"], errors="coerce"),
                "volume": pd.to_numeric(raw["volume"], errors="coerce"),
                "amount": pd.to_numeric(raw["turnover"], errors="coerce"),
            }
        )
        frame = frame[(frame["trade_date"] >= start_ts) & (frame["trade_date"] <= end_ts)]
        frame = frame.drop_duplicates(["symbol", "trade_date"], keep="last").sort_values("trade_date")
        if frame.empty:
            return SourceResult(
                pd.DataFrame(columns=cols), self.name, endpoint, RETRY_EMPTY,
                now, 0, error="no Fuyao dump rows inside requested range",
            )
        frame["source"] = self.name
        frame["source_endpoint"] = endpoint
        frame["retrieved_at"] = now
        frame["available_at"] = frame["trade_date"].dt.strftime("%Y-%m-%d") + " 15:00:00"
        frame["quality_status"] = contracts.QUALITY_OK
        return SourceResult(
            frame[cols].reset_index(drop=True), self.name, endpoint, RETRY_OK, now, len(frame),
            metadata={
                "adjustment": contracts.ADJUST_NONE,
                "volume_unit": contracts.VOLUME_SHARES,
                "amount_unit": contracts.AMOUNT_CNY,
                "artifact": str(self.path),
            },
        )


def available_dump_kinds() -> Iterable[str]:
    return ("daily-k", "daily-k-10d", "adjustment-factors")


__all__ = [
    "DEFAULT_DUMP_ROOT",
    "DUMP_PATH_ENV",
    "DumpArtifact",
    "FuyaoDumpSource",
    "available_dump_kinds",
    "configured_daily_dump",
    "download_dump",
    "validate_dump",
]
