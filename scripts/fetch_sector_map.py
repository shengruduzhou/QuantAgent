"""Fetch current A-share sector mapping from AkShare and write the
PIT-safe sector_map silver dataset.

This is a one-shot live-network operation. The output is a current
snapshot — every row is labelled ``coverage_status='current_snapshot'``
and ``source='akshare:stock_board_industry_cons_em'`` so downstream
historical joins cannot silently backfill 2020 OOS predictions with
2026 classifications.

Env vars:
  QA_AS_OF_DATE — override the ``available_at`` timestamp (default: today).
                  Useful when you want the snapshot timestamped against
                  the trading day boundary, not the wall clock.
  QA_OUTPUT_ROOT — output root (default: runtime/data/v7)
  QA_UNIVERSE_SYMBOLS — comma-separated symbol allowlist. When set, only
                       these symbols appear in the output (missing ones
                       get ``coverage_status='missing'``). Default: no
                       filter, use whatever AkShare returns.

The script never modifies target_weights or alpha; it only produces a
silver-layer data product. Stage 2.2 gate decisions about whether the
sector map is "usable_for_optimization" live in the coverage_report
JSON sidecar and the manifest.
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from quantagent.data.providers.akshare_valuation_provider import AkShareSectorProvider
from quantagent.data.sector.sector_mapping import (
    SectorMapBuilder,
    SectorMapConfig,
    normalize_sector_source,
)


def _resolve_symbols(env_value: str | None) -> tuple[str, ...]:
    if not env_value:
        return ()
    return tuple(s.strip() for s in env_value.split(",") if s.strip())


def _load_universe(path: Path | None) -> list[str]:
    if path is None or not path.exists():
        return []
    df = pd.read_parquet(path, columns=["symbol"])
    return sorted(set(df["symbol"].astype(str).str.strip()))


def _per_symbol_fetch(symbols: list[str], rate_limit_seconds: float = 0.15) -> pd.DataFrame:
    """Fallback path: query AkShare's per-symbol endpoint
    ``stock_individual_info_em`` for each symbol's ``行业`` field.

    Slower than bulk board fetch (~13 min for full universe) but
    survives upstream RemoteDisconnected on the bulk endpoints when
    only the per-symbol path is reachable. Failures are recorded as
    blanks; the SectorMapBuilder labels those rows ``missing``.
    """
    import time
    try:
        import akshare as ak  # type: ignore
    except ImportError as exc:  # pragma: no cover
        raise SystemExit(f"akshare not available: {exc}") from exc
    fn = getattr(ak, "stock_individual_info_em", None)
    if fn is None:
        raise SystemExit("akshare.stock_individual_info_em endpoint missing in this version")

    rows: list[dict] = []
    success = 0
    failures = 0
    last_print = time.time()
    for idx, sym in enumerate(symbols, start=1):
        code = sym.split(".")[0]
        industry = ""
        try:
            info = fn(symbol=code)
            if info is not None and not info.empty and {"item", "value"}.issubset(info.columns):
                ind_row = info[info["item"] == "行业"]
                if not ind_row.empty:
                    industry = str(ind_row.iloc[0]["value"]).strip()
        except Exception:
            failures += 1
        if industry:
            success += 1
        rows.append({"symbol": sym, "industry": industry or pd.NA})
        if rate_limit_seconds > 0:
            time.sleep(rate_limit_seconds)
        # Progress log every 30s
        if time.time() - last_print > 30:
            pct = idx * 100 // max(len(symbols), 1)
            print(f"  per-symbol fetch: {idx}/{len(symbols)} ({pct}%) — {success} success, {failures} fail", flush=True)
            last_print = time.time()
    print(f"  per-symbol fetch DONE: {success} success, {failures} fail of {len(symbols)} symbols", flush=True)
    return pd.DataFrame(rows)


def main() -> None:
    as_of = os.environ.get("QA_AS_OF_DATE") or pd.Timestamp.today().strftime("%Y-%m-%d")
    output_root = Path(os.environ.get("QA_OUTPUT_ROOT", "runtime/data/v7"))
    symbols_filter = _resolve_symbols(os.environ.get("QA_UNIVERSE_SYMBOLS"))
    universe_path = Path(os.environ["QA_UNIVERSE_PATH"]) if os.environ.get("QA_UNIVERSE_PATH") else None
    fetch_mode = os.environ.get("QA_FETCH_MODE", "bulk")  # bulk | per_symbol

    print(f"as_of_date    : {as_of}")
    print(f"output_root   : {output_root}")
    print(f"universe path : {universe_path or 'none'}")
    print(f"fetch mode    : {fetch_mode}")
    print(f"symbol filter : {len(symbols_filter)} symbols" if symbols_filter else "symbol filter : none")
    print()

    raw_frame: pd.DataFrame | None = None
    used_source = ""
    if fetch_mode == "per_symbol":
        # Per-symbol path: needs a universe list.
        universe_symbols = list(symbols_filter) if symbols_filter else _load_universe(universe_path)
        if not universe_symbols:
            raise SystemExit("per_symbol mode requires either QA_UNIVERSE_SYMBOLS or QA_UNIVERSE_PATH")
        print(f"fetching per-symbol industry via stock_individual_info_em ({len(universe_symbols)} symbols) ...", flush=True)
        raw_frame = _per_symbol_fetch(universe_symbols)
        used_source = "akshare:stock_individual_info_em"
    else:
        print("fetching bulk industry mapping from AkShare board endpoints ...", flush=True)
        provider = AkShareSectorProvider(allow_network=True)
        request = None
        if symbols_filter:
            from quantagent.data.providers.base import ProviderRequest
            request = ProviderRequest(symbols=symbols_filter)
        last_exc: Exception | None = None
        result = None
        for attempt in range(1, 4):
            try:
                result = provider.industry_classification(request=request, as_of_date=as_of)
                break
            except Exception as exc:
                last_exc = exc
                wait_s = min(15, 2 ** attempt)
                print(f"  attempt {attempt}/3 failed: {type(exc).__name__}: {str(exc)[:120]}  retrying in {wait_s}s", flush=True)
                import time as _time
                _time.sleep(wait_s)
        if result is None or result.frame is None or result.frame.empty:
            raise SystemExit(
                "bulk endpoint exhausted retries — re-run with QA_FETCH_MODE=per_symbol "
                "QA_UNIVERSE_PATH=runtime/data/v7/silver/market_panel/market_panel.parquet "
                f"(last error: {last_exc!r})"
            )
        raw_frame = result.frame.rename(columns={"industry": "industry"})
        used_source = "akshare:stock_board_industry_cons_em"
        print(f"  bulk rows: {len(raw_frame):,}  unique symbols: {raw_frame['symbol'].nunique():,}")
    print()

    print("normalizing into canonical 9-column schema ...", flush=True)
    raw = normalize_sector_source(
        raw_frame,
        source=used_source,
        source_version=str(pd.Timestamp.today().date()),
        as_of_date=as_of,
        fetched_at=pd.Timestamp.utcnow().isoformat(),
    )
    print(f"  normalized rows: {len(raw):,}")
    print()

    print("building sector_map via SectorMapBuilder ...", flush=True)
    # If we have a universe path, expand the symbols list so missing
    # rows are explicitly emitted (rather than just omitted).
    builder_symbols = symbols_filter
    if not builder_symbols and universe_path is not None:
        builder_symbols = tuple(_load_universe(universe_path))
    config = SectorMapConfig(
        symbols=builder_symbols,
        as_of_date=as_of,
        fetched_at=pd.Timestamp.utcnow().isoformat(),
        source=used_source,
        source_version=str(pd.Timestamp.today().date()),
        coverage_status="current_snapshot",
        output_root=output_root,
    )
    builder = SectorMapBuilder(config)
    built = builder.build(raw)
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
