"""Fetch current A-share ST / *ST list from AkShare and write the
PIT-safe st_flags silver dataset.

Like ``fetch_sector_map.py`` this is a one-shot live-network operation.
The output is a current snapshot tagged with
``st_source='akshare:stock_zh_a_st_em'`` and
``coverage_status='current_snapshot'`` so historical OOS joins cannot
backfill 2020 predictions with 2026 ST status.

Stocks NOT in the AkShare ST list get ``is_st=False`` AND
``st_known=True`` — meaning we know they are not ST right now, not that
ST status is unknown. Symbols missing entirely from AkShare (delisted
already, BSE, etc.) get ``coverage_status='missing'`` and
``st_known=False`` so downstream filters can distinguish "confirmed
not ST" from "no data".

Env vars:
  QA_AS_OF_DATE     — available_at timestamp (default: today)
  QA_OUTPUT_ROOT    — output root (default: runtime/data/v7)
  QA_UNIVERSE_PATH  — parquet whose ``symbol`` column defines the
                      universe. When set, every universe symbol gets
                      a row in st_flags (ST flag if AkShare lists it,
                      not-ST otherwise). Default: include only the
                      AkShare ST list.
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from quantagent.data.sector.st_history import StFlagBuilder, StFlagConfig


def _suffix_from_code(code: str) -> str:
    code = str(code).strip()
    if not code:
        return ""
    if code.startswith("6") or code.startswith("9") or code.startswith("5"):
        return f"{code}.SH"
    if code.startswith(("0", "3", "1", "2")):
        return f"{code}.SZ"
    if code.startswith(("4", "8")):
        return f"{code}.BJ"
    return code


def _fetch_st_list() -> pd.DataFrame:
    import time as _time
    try:
        import akshare as ak  # type: ignore
    except ImportError as exc:  # pragma: no cover - environment issue
        raise SystemExit(f"akshare not installed in the venv: {exc}") from exc
    fn = getattr(ak, "stock_zh_a_st_em", None)
    if fn is None:
        raise SystemExit("akshare.stock_zh_a_st_em endpoint not available in this akshare version")
    # Retry on transient RemoteDisconnected from Eastmoney upstream.
    raw = None
    last_exc: Exception | None = None
    for attempt in range(1, 6):
        try:
            raw = fn()
            if raw is not None and not raw.empty:
                break
        except Exception as exc:
            last_exc = exc
            wait_s = min(30, 2 ** attempt)
            print(f"  attempt {attempt}/5 failed: {type(exc).__name__}: {str(exc)[:120]}  retrying in {wait_s}s", flush=True)
            _time.sleep(wait_s)
    if raw is None or raw.empty:
        raise SystemExit(f"stock_zh_a_st_em exhausted retries; last error: {last_exc!r}")
    code_col = next((c for c in ("代码", "code", "股票代码") if c in raw.columns), None)
    name_col = next((c for c in ("名称", "name", "股票名称") if c in raw.columns), None)
    if code_col is None:
        raise SystemExit(f"could not find code column in akshare ST output; columns={list(raw.columns)}")
    out = pd.DataFrame({
        "symbol": raw[code_col].astype(str).map(_suffix_from_code),
        "name": raw[name_col].astype(str) if name_col else "",
        "is_st": True,
        "st_known": True,
    })
    out = out[out["symbol"] != ""].drop_duplicates("symbol", keep="first").reset_index(drop=True)
    return out


def _load_universe(path: Path | None) -> list[str]:
    if path is None or not path.exists():
        return []
    df = pd.read_parquet(path, columns=["symbol"])
    return sorted(set(df["symbol"].astype(str).str.strip()))


def main() -> None:
    as_of = os.environ.get("QA_AS_OF_DATE") or pd.Timestamp.today().strftime("%Y-%m-%d")
    output_root = Path(os.environ.get("QA_OUTPUT_ROOT", "runtime/data/v7"))
    universe_path = Path(os.environ["QA_UNIVERSE_PATH"]) if os.environ.get("QA_UNIVERSE_PATH") else None

    print(f"as_of_date  : {as_of}")
    print(f"output_root : {output_root}")
    print(f"universe    : {universe_path or '(AkShare ST list only)'}")
    print()

    print("fetching current ST list from AkShare ...", flush=True)
    st_df = _fetch_st_list()
    print(f"  ST entries: {len(st_df):,}")
    print(f"  sample: {st_df.head(3).to_dict('records')}")
    print()

    universe_symbols = _load_universe(universe_path)
    if universe_symbols:
        # Build a frame where every universe symbol has a row. Symbols on
        # AkShare's ST list → is_st=True, st_known=True. Symbols in the
        # universe but absent from the ST list → is_st=False, st_known=True
        # ("confirmed not ST as of as_of_date"). Symbols not in the
        # universe at all are simply dropped.
        st_set = set(st_df["symbol"])
        universe_frame = pd.DataFrame({
            "symbol": universe_symbols,
            "is_st": [s in st_set for s in universe_symbols],
            "st_known": True,
        })
        source_frame = universe_frame.copy()
    else:
        source_frame = st_df[["symbol", "is_st", "st_known"]].copy()
    source_frame["available_at"] = pd.to_datetime(as_of)
    source_frame["fetched_at"] = pd.Timestamp.utcnow()
    source_frame["st_source"] = "akshare:stock_zh_a_st_em"
    source_frame["source_version"] = str(pd.Timestamp.today().date())
    source_frame["coverage_status"] = "current_snapshot"

    print("building st_flags via StFlagBuilder ...", flush=True)
    config = StFlagConfig(
        symbols=tuple(universe_symbols) if universe_symbols else (),
        as_of_date=as_of,
        fetched_at=pd.Timestamp.utcnow().isoformat(),
        source="akshare:stock_zh_a_st_em",
        source_version=str(pd.Timestamp.today().date()),
        output_root=output_root,
    )
    builder = StFlagBuilder(config)
    built = builder.build(source_frame)
    written = builder.write(built)

    print()
    print("=== coverage report ===")
    for key, value in built.coverage.items():
        if isinstance(value, dict):
            print(f"  {key}:")
            for sub_k, sub_v in value.items():
                print(f"    {sub_k}: {sub_v}")
        else:
            print(f"  {key}: {value}")
    print()
    print("=== validation report ===")
    for key, value in built.validation.items():
        print(f"  {key}: {value}")
    print()
    print("=== output paths ===")
    for key, path in written.output_paths.items():
        print(f"  {key}: {path}")


if __name__ == "__main__":
    main()
