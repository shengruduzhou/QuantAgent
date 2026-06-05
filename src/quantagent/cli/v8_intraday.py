"""v8 intraday feature CLI.

Builds day-level 分时 volume-price factors from a 1-minute panel and,
optionally, PIT-merges them into a gold training dataset. The command is
feed-agnostic: operators can provide a tickflow-exported minute parquet/csv
or a qlib 1min provider root.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Optional

import pandas as pd
import typer

from quantagent.cli._utils import app


def _parse_symbols(symbols: Optional[str], symbols_file: Optional[Path]) -> list[str]:
    if symbols:
        return [s.strip() for s in symbols.split(",") if s.strip()]
    if symbols_file:
        text = symbols_file.read_text(encoding="utf-8")
        return [s.strip() for s in text.replace("\n", ",").split(",") if s.strip()]
    return []


def _read_minute_panel(path: Path) -> pd.DataFrame:
    suffix = path.suffix.lower()
    if suffix in {".parquet", ".pq"}:
        return pd.read_parquet(path)
    if suffix in {".csv", ".txt"}:
        return pd.read_csv(path)
    raise typer.BadParameter(f"unsupported minute panel format: {path}")


def _normalise_minute_panel(df: pd.DataFrame) -> pd.DataFrame:
    """Normalise common tickflow/qlib minute-panel column spellings."""
    if df is None or df.empty:
        return pd.DataFrame()
    out = df.copy()
    rename = {}
    if "timestamp" in out.columns and "datetime" not in out.columns:
        rename["timestamp"] = "datetime"
    if "date_time" in out.columns and "datetime" not in out.columns:
        rename["date_time"] = "datetime"
    if "vol" in out.columns and "volume" not in out.columns:
        rename["vol"] = "volume"
    if rename:
        out = out.rename(columns=rename)
    required = {"symbol", "datetime", "open", "high", "low", "close", "volume"}
    missing = sorted(required - set(out.columns))
    if missing:
        raise typer.BadParameter(f"minute panel missing required columns: {missing}")
    out["datetime"] = pd.to_datetime(out["datetime"], errors="coerce")
    out = out.dropna(subset=["symbol", "datetime"])
    if "trade_date" not in out.columns:
        out["trade_date"] = out["datetime"].dt.normalize()
    else:
        out["trade_date"] = pd.to_datetime(out["trade_date"], errors="coerce").dt.normalize()
    out["symbol"] = out["symbol"].astype(str).str.strip()
    return out


def _business_available_at(trade_date: pd.Series) -> pd.Series:
    """Same-day intraday factors become usable after the close / next session."""
    return pd.to_datetime(trade_date, errors="coerce") + pd.tseries.offsets.BDay(1)


@app.command("build-intraday-factors-v8")
def build_intraday_factors_v8(
    minute_panel_path: Optional[Path] = typer.Option(
        None, help="tickflow/other extracted 1-minute panel parquet/csv",
    ),
    provider_uri: Optional[Path] = typer.Option(
        None, help="qlib 1min provider root; requires --symbols or --symbols-file",
    ),
    symbols: Optional[str] = typer.Option(None, help="comma-separated canonical symbols for qlib root"),
    symbols_file: Optional[Path] = typer.Option(None, exists=True, dir_okay=False),
    start_date: Optional[str] = typer.Option(None),
    end_date: Optional[str] = typer.Option(None),
    output_path: Path = typer.Option(
        Path("runtime/data/v7/silver/intraday_factors/intraday_factors.parquet"),
        help="output day-level intraday factor parquet",
    ),
    dataset_path: Optional[Path] = typer.Option(
        None, exists=True, dir_okay=False,
        help="optional gold training dataset to merge these factors into",
    ),
    output_dataset_path: Optional[Path] = typer.Option(
        None, help="where to write the merged training dataset",
    ),
    manifest_path: Path = typer.Option(
        Path("runtime/data/v7/manifests/intraday_factors.json"),
        help="DataManifest path for the factor artifact",
    ),
    symbol_batch_size: int = typer.Option(
        200,
        help="provider-uri mode: process symbols in batches to avoid materialising all 1min bars",
    ),
):
    """Collapse 1-minute bars into PIT day-level 分时量价 factors."""
    from quantagent.data.manifest import build_manifest_for_frame
    from quantagent.data.providers.qlib_intraday_reader import build_intraday_panel
    from quantagent.factors.intraday_volume_price import FACTOR_COLUMNS, compute_intraday_factors

    if minute_panel_path is None and provider_uri is None:
        raise typer.BadParameter("provide --minute-panel-path or --provider-uri")
    if minute_panel_path is not None and provider_uri is not None:
        raise typer.BadParameter("choose only one input: --minute-panel-path or --provider-uri")

    raw_paths: list[str] = []
    vendor = "minute_panel"
    if minute_panel_path is not None:
        if not minute_panel_path.exists():
            raise typer.BadParameter(f"minute panel missing: {minute_panel_path}")
        minute = _normalise_minute_panel(_read_minute_panel(minute_panel_path))
        raw_paths.append(str(minute_panel_path))
        vendor = "tickflow_or_external_1min"
        if start_date is not None:
            minute = minute[minute["datetime"] >= pd.Timestamp(start_date)]
        if end_date is not None:
            minute = minute[minute["datetime"] <= pd.Timestamp(end_date)]
        if minute.empty:
            raise typer.BadParameter("minute input produced zero rows after filtering")
        factors = compute_intraday_factors(minute)
        minute_rows = int(len(minute))
        batches = 1
    else:
        syms = _parse_symbols(symbols, symbols_file)
        if not syms:
            raise typer.BadParameter("--provider-uri requires --symbols or --symbols-file")
        raw_paths.append(str(provider_uri))
        vendor = "qlib_1min"
        if symbol_batch_size <= 0:
            raise typer.BadParameter("--symbol-batch-size must be positive")
        factor_parts = []
        minute_rows = 0
        batches = 0
        for i in range(0, len(syms), symbol_batch_size):
            batch_symbols = syms[i:i + symbol_batch_size]
            minute = build_intraday_panel(
                provider_uri, batch_symbols, start=start_date, end=end_date, adjust=True,
            )
            minute = _normalise_minute_panel(minute)
            if minute.empty:
                continue
            factor_parts.append(compute_intraday_factors(minute))
            minute_rows += int(len(minute))
            batches += 1
            typer.echo(json.dumps({
                "batch": batches,
                "symbols": len(batch_symbols),
                "minute_rows": int(len(minute)),
                "factor_rows": int(len(factor_parts[-1])),
            }, ensure_ascii=False))
        if not factor_parts:
            raise typer.BadParameter("minute input produced zero rows after filtering")
        factors = pd.concat(factor_parts, ignore_index=True)

    factors["trade_date"] = pd.to_datetime(factors["trade_date"], errors="coerce").dt.normalize()
    factors["intraday_available_at"] = _business_available_at(factors["trade_date"])
    factors["source"] = vendor
    factors["point_in_time_valid"] = True

    output_path.parent.mkdir(parents=True, exist_ok=True)
    factors.to_parquet(output_path, index=False)

    outputs = [output_path]
    merged_rows = None
    if dataset_path is not None:
        if output_dataset_path is None:
            raise typer.BadParameter("--dataset-path requires --output-dataset-path")
        ds = pd.read_parquet(dataset_path)
        ds["trade_date"] = pd.to_datetime(ds["trade_date"], errors="coerce").dt.normalize()
        merge_cols = ["symbol", "trade_date", *FACTOR_COLUMNS, "intraday_available_at"]
        merged = ds.merge(factors[merge_cols], on=["symbol", "trade_date"], how="left")
        # Guard against accidentally using same-day intraday data for pre-close
        # signals: when a dataset has available_at, the intraday feature must
        # not be considered available earlier than next business day.
        if "available_at" in merged.columns:
            merged["available_at"] = pd.to_datetime(merged["available_at"], errors="coerce")
            merged["available_at"] = merged[["available_at", "intraday_available_at"]].max(axis=1)
        output_dataset_path.parent.mkdir(parents=True, exist_ok=True)
        merged.to_parquet(output_dataset_path, index=False)
        outputs.append(output_dataset_path)
        merged_rows = int(len(merged))

    manifest = build_manifest_for_frame(
        dataset_name="intraday_factors",
        vendor=vendor,
        frame=factors,
        output_paths=outputs,
        raw_paths=raw_paths,
        start_date=str(factors["trade_date"].min()),
        end_date=str(factors["trade_date"].max()),
        symbols=sorted(factors["symbol"].astype(str).unique()),
        required_columns=["symbol", "trade_date", *FACTOR_COLUMNS, "intraday_available_at"],
        warnings=(),
        extra={
            "minute_rows": int(minute_rows),
            "symbol_batches": int(batches),
            "factor_columns": list(FACTOR_COLUMNS),
            "merged_training_rows": merged_rows,
            "pit_rule": "intraday factors are available at next business day after trade_date",
        },
    )
    manifest.write(manifest_path)

    summary = {
        "output_path": str(output_path),
        "rows": int(len(factors)),
        "symbols": int(factors["symbol"].nunique()),
        "start": str(factors["trade_date"].min()),
        "end": str(factors["trade_date"].max()),
        "manifest": str(manifest_path),
        "output_dataset_path": str(output_dataset_path) if output_dataset_path else None,
    }
    typer.echo(json.dumps(summary, ensure_ascii=False, indent=2, default=str))
    return output_path


@app.command("run-do-t-overlay-v8")
def run_do_t_overlay_v8(
    target_weights_path: Path = typer.Option(..., exists=True, dir_okay=False),
    market_panel_path: Path = typer.Option(
        Path("runtime/data/v7/silver/market_panel/market_panel.parquet"),
        exists=True,
        dir_okay=False,
    ),
    base_nav_path: Optional[Path] = typer.Option(None, exists=True, dir_okay=False),
    minute_panel_path: Optional[Path] = typer.Option(None, help="tickflow/1min parquet/csv"),
    provider_uri: Optional[Path] = typer.Option(None, help="qlib 1min provider root"),
    output_dir: Path = typer.Option(Path("runtime/reports/v8/do_t_overlay")),
    initial_cash: float = typer.Option(1_000_000.0),
    trade_fraction: float = typer.Option(0.30),
    min_edge_pct: float = typer.Option(0.025),
    min_minutes_between_legs: int = typer.Option(5),
    max_trades_per_day: int = typer.Option(50),
    symbol_batch_size: int = typer.Option(50),
):
    """Run a legal T+1 Do-T overlay on existing target weights.

    The overlay never emits live orders.  It simulates same-day round trips
    only when yesterday-settled inventory exists.
    """
    from quantagent.data.providers.qlib_intraday_reader import build_intraday_panel
    from quantagent.portfolio.do_t_overlay import DoTOverlayConfig, simulate_do_t_overlay
    from quantagent.training.do_t_labels import build_do_t_training_labels

    if minute_panel_path is None and provider_uri is None:
        raise typer.BadParameter("provide --minute-panel-path or --provider-uri")
    if minute_panel_path is not None and provider_uri is not None:
        raise typer.BadParameter("choose only one input: --minute-panel-path or --provider-uri")

    output_dir.mkdir(parents=True, exist_ok=True)
    tw = pd.read_parquet(target_weights_path)
    tw.index = pd.to_datetime(tw.index, errors="coerce")
    tw = tw[tw.index.notna()].sort_index()
    tw = tw.loc[:, tw.abs().sum(axis=0) > 0]
    if tw.empty:
        raise typer.BadParameter("target weights are empty after dropping zero columns")

    market = pd.read_parquet(market_panel_path, columns=["trade_date", "symbol", "close"])
    market["trade_date"] = pd.to_datetime(market["trade_date"], errors="coerce")
    market["symbol"] = market["symbol"].astype(str)
    market = market[
        (market["trade_date"] >= tw.index.min() - pd.Timedelta(days=10))
        & (market["trade_date"] <= tw.index.max())
        & market["symbol"].isin(tw.columns)
    ]
    close = market.pivot_table(index="trade_date", columns="symbol", values="close", aggfunc="last")
    close = close.reindex(tw.index).ffill()

    nav = _read_or_make_base_nav(base_nav_path, tw.index, initial_cash)
    inventory = _target_weights_to_available_inventory(tw, close, nav, lot=100)
    inventory = inventory[inventory["available_shares"] >= 100].reset_index(drop=True)
    if inventory.empty:
        raise typer.BadParameter("no T+1 available inventory from target weights")
    inventory.to_parquet(output_dir / "available_inventory.parquet", index=False)
    symbols = sorted(inventory["symbol"].unique())
    start = str(inventory["trade_date"].min().date())
    end = str(inventory["trade_date"].max().date())
    cfg = DoTOverlayConfig(
        trade_fraction=trade_fraction,
        min_edge_pct=min_edge_pct,
        min_minutes_between_legs=min_minutes_between_legs,
        max_trades_per_day=max_trades_per_day,
    )

    trade_parts: list[pd.DataFrame] = []
    label_parts: list[pd.DataFrame] = []
    minute_rows = 0
    if minute_panel_path is not None:
        minute = _normalise_minute_panel(_read_minute_panel(minute_panel_path))
        minute = minute[
            (minute["trade_date"] >= pd.Timestamp(start))
            & (minute["trade_date"] <= pd.Timestamp(end))
            & minute["symbol"].isin(symbols)
        ]
        minute_rows = int(len(minute))
        trades = simulate_do_t_overlay(minute, inventory, config=cfg)
        labels = build_do_t_training_labels(minute, inventory, config=cfg)
        trade_parts.append(trades)
        label_parts.append(labels)
    else:
        for i in range(0, len(symbols), symbol_batch_size):
            batch_symbols = symbols[i:i + symbol_batch_size]
            minute = build_intraday_panel(provider_uri, batch_symbols, start=start, end=end, adjust=True)
            minute = _normalise_minute_panel(minute)
            if minute.empty:
                continue
            batch_inventory = inventory[inventory["symbol"].isin(batch_symbols)]
            minute_rows += int(len(minute))
            trades = simulate_do_t_overlay(minute, batch_inventory, config=cfg)
            labels = build_do_t_training_labels(minute, batch_inventory, config=cfg)
            if not trades.empty:
                trade_parts.append(trades)
            if not labels.empty:
                label_parts.append(labels)
            typer.echo(json.dumps({
                "batch": i // symbol_batch_size + 1,
                "symbols": len(batch_symbols),
                "minute_rows": int(len(minute)),
                "trades": int(len(trades)),
            }, ensure_ascii=False))

    trades_all = pd.concat(trade_parts, ignore_index=True) if trade_parts else pd.DataFrame()
    labels_all = pd.concat(label_parts, ignore_index=True) if label_parts else pd.DataFrame()
    trades_path = output_dir / "do_t_trades.parquet"
    labels_path = output_dir / "do_t_labels.parquet"
    trades_all.to_parquet(trades_path, index=False)
    labels_all.to_parquet(labels_path, index=False)

    overlay_nav, metrics = _combine_do_t_nav(nav, trades_all, initial_cash=initial_cash)
    overlay_nav.to_csv(output_dir / "combined_nav.csv", index=False)
    (output_dir / "metrics.json").write_text(
        json.dumps(metrics, ensure_ascii=False, indent=2, default=str),
        encoding="utf-8",
    )
    summary = {
        "output_dir": str(output_dir),
        "target_dates": int(len(tw)),
        "target_symbols": int(len(symbols)),
        "minute_rows": int(minute_rows),
        "do_t_trades": int(len(trades_all)),
        "do_t_labels": int(len(labels_all)),
        **metrics,
    }
    typer.echo(json.dumps(summary, ensure_ascii=False, indent=2, default=str))
    return output_dir


def _read_or_make_base_nav(base_nav_path: Optional[Path], dates: pd.DatetimeIndex, initial_cash: float) -> pd.Series:
    if base_nav_path is None:
        return pd.Series(float(initial_cash), index=dates, name="base_nav")
    nav = pd.read_csv(base_nav_path)
    if "trade_date" not in nav.columns or "nav" not in nav.columns:
        raise typer.BadParameter("base nav CSV must include trade_date,nav columns")
    nav["trade_date"] = pd.to_datetime(nav["trade_date"], errors="coerce")
    s = nav.dropna(subset=["trade_date"]).set_index("trade_date")["nav"].astype(float)
    return s.reindex(dates).ffill().fillna(float(initial_cash)).rename("base_nav")


def _target_weights_to_available_inventory(
    target_weights: pd.DataFrame,
    close: pd.DataFrame,
    nav: pd.Series,
    *,
    lot: int,
) -> pd.DataFrame:
    previous_weights = target_weights.shift(1).fillna(0.0)
    previous_close = close.shift(1).reindex(previous_weights.index).ffill()
    previous_nav = nav.shift(1).reindex(previous_weights.index).ffill().fillna(float(nav.iloc[0]))
    value = previous_weights.mul(previous_nav, axis=0)
    shares = (value / previous_close.replace(0.0, pd.NA)).replace([float("inf"), float("-inf")], pd.NA)
    shares = (shares.fillna(0.0) // lot * lot).astype("int64")
    long = shares.stack().rename("available_shares").reset_index()
    long.columns = ["trade_date", "symbol", "available_shares"]
    return long


def _combine_do_t_nav(
    base_nav: pd.Series,
    trades: pd.DataFrame,
    *,
    initial_cash: float,
) -> tuple[pd.DataFrame, dict[str, float]]:
    out = pd.DataFrame({"trade_date": base_nav.index, "base_nav": base_nav.to_numpy(dtype=float)})
    if trades is None or trades.empty:
        out["do_t_net_pnl"] = 0.0
    else:
        t = trades.copy()
        t["trade_date"] = pd.to_datetime(t["trade_date"], errors="coerce")
        pnl = t.groupby("trade_date")["net_pnl"].sum()
        out["do_t_net_pnl"] = out["trade_date"].map(pnl).fillna(0.0)
    out["do_t_cum_pnl"] = out["do_t_net_pnl"].cumsum()
    out["combined_nav"] = out["base_nav"] + out["do_t_cum_pnl"]
    metrics = _nav_metrics(out["base_nav"], out["combined_nav"], initial_cash=initial_cash)
    metrics["do_t_total_net_pnl"] = float(out["do_t_net_pnl"].sum())
    metrics["do_t_return_on_initial_cash"] = float(out["do_t_net_pnl"].sum() / max(1.0, initial_cash))
    return out, metrics


def _nav_metrics(base_nav: pd.Series, combined_nav: pd.Series, *, initial_cash: float) -> dict[str, float]:
    base = pd.Series(base_nav, dtype=float).reset_index(drop=True)
    combined = pd.Series(combined_nav, dtype=float).reset_index(drop=True)
    n = max(1, len(combined))
    base_total = float(base.iloc[-1] / initial_cash - 1.0) if len(base) else 0.0
    combined_total = float(combined.iloc[-1] / initial_cash - 1.0) if len(combined) else 0.0
    combined_ann = float((1.0 + combined_total) ** (252.0 / n) - 1.0) if combined_total > -1.0 else -1.0
    peak = combined.cummax()
    dd = (combined / peak.replace(0.0, pd.NA) - 1.0).fillna(0.0)
    return {
        "base_total_return": base_total,
        "combined_total_return": combined_total,
        "combined_annualized_return": combined_ann,
        "combined_max_drawdown": float(-dd.min()),
        "overlay_total_return_delta": float(combined_total - base_total),
    }


__all__ = ["build_intraday_factors_v8", "run_do_t_overlay_v8"]
