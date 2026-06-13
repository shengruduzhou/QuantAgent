"""train-v8-deep — GPU FT-Transformer per horizon → predictions → strict v8 backtest.

This is the GPU entry point for the v8 pipeline. The existing
``train-v8-pipeline`` runs a pure-numpy GA over a couple of simple
factors; that's quick but does not exercise the hardware. This CLI:

1. Loads the gold training dataset (``training_dataset_alpha181_*.parquet``)
   which already carries OHLCV + 181 alpha factors + macro indicators
   + 6 forward-return horizons with PIT-correct ``available_at``.
2. Selects a horizon class (short_5d / mid_5d_30d / long_30d_120d)
   → maps to the appropriate ``forward_return_{H}d`` label.
3. Splits by **date** (no leakage) into train / OOS using purged
   walk-forward with embargo.
4. Trains the existing :class:`FTTransformerTrainer` with
   ``require_gpu=True``. Mixed-precision + AMP on by default.
5. Predicts on the OOS window → emits ``predictions.parquet``.
6. Builds top-K equal-weight target weights from predictions.
7. Runs :func:`run_strict_backtest_v8` on the OOS predictions using
   the silver market panel → 9 metrics + 10 output files.
8. Writes ``daily_decision_report.md``.

The CLI is a thin orchestrator; every heavy step is a module already
under test.
"""

from __future__ import annotations

import json
import os
from datetime import datetime
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
import typer

from quantagent.cli._utils import app, default_reports_root
from quantagent.factors.core_policy import CORE_FEATURE_COLUMNS, core_feature_columns


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _ts() -> str:
    return datetime.now().isoformat(timespec="seconds")


def _resolve_label_column(horizon_class: str) -> str:
    return {
        "short_5d": "forward_return_5d",
        "mid_5d_30d": "forward_return_20d",
        "long_30d_120d": "forward_return_120d",
    }[horizon_class]


_FEATURE_COMMON_DROP = {
    "symbol", "trade_date", "available_at", "source", "source_type",
    "source_reliability", "point_in_time_valid",
    "is_suspended", "is_st", "is_limit_up", "is_limit_down",
}
_FEATURE_LABEL_DROP = {f"forward_return_{h}d" for h in (1, 5, 20, 60, 120, 126)} | {
    f"label_end_{h}d" for h in (1, 5, 20, 60, 120, 126)
}
_INTRADAY_FEATURES = {
    "first30_return", "last30_return", "vwap_deviation", "intraday_range_pos",
    "net_buy_pressure", "volume_concentration", "spike_minutes",
    "am_pm_volume_ratio", "minute_ret_skew", "liq_amihud_1min",
    "liq_amihud_1min_m20", "corr_prv", "corr_prv_m20",
    "open30_volume_share", "close30_volume_share", "close3_volume_share",
}
_SELECTION_FEATURES = {
    "cicc_stock_selection_score",
    "cicc_sector_selection_score",
    "cicc_aggressive_momentum_score",
    "cicc_defensive_quality_score",
    "cicc_liquidity_defense_score",
    "agent_stock_score",
    "agent_sector_score",
    "agent_conviction_score",
}
_NO_CROSS_SECTIONAL_NORM = {
    "core_policy_score",
    "core_sentiment_score",
    "flow_north_total",
    "flow_margin_sh",
    "idx_csi300_ret5",
}


def _is_selection_feature(name: str) -> bool:
    return (
        name in _SELECTION_FEATURES
        or name.startswith("cicc_")
        or name.startswith("agent_")
        or name.endswith("_agent_score")
        or name.endswith("_selection_score")
    )


_JUDGMENT_ASSIGNMENT_PATH = os.getenv(
    "QUANTAGENT_HORIZON_ASSIGNMENT",
    "runtime/reports/v8/factor_full_judgment/horizon_factor_assignment.json",
)
_JUDGMENT_HORIZON_KEY = {
    "short_5d": "short_5d",
    "mid_5d_30d": "mid_20d",
    "long_30d_120d": "long_60d",
}


def _judgment_factors(horizon_class: str) -> list[str] | None:
    """Measured-best-horizon factor routing from the unified factor judgment.

    Unlike the ``auto`` policy (which buckets by alpha NUMBER and silently
    ignores gtja*/synth_*/llm_* columns), this reads
    ``horizon_factor_assignment.json`` produced by
    ``scripts/factor_full_judgment.py`` and routes every accepted factor to
    the horizon where its |ICIR| is actually strongest.
    """
    path = Path(_JUDGMENT_ASSIGNMENT_PATH)
    if not path.exists():
        return None
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None
    key = _JUDGMENT_HORIZON_KEY.get(horizon_class)
    items = payload.get(key, []) if key else []
    names = [str(item.get("factor")) for item in items if item.get("factor")]
    # The assignment lists are sorted by |ICIR| desc; an optional cap keeps
    # the strongest head when the full list blows GPU memory (long_60d has
    # 150+ factors, many redundant size proxies).
    max_factors = int(os.getenv("QUANTAGENT_JUDGMENT_MAX_FACTORS", "0") or 0)
    if max_factors > 0:
        names = names[:max_factors]
    return names or None


def _candidate_feature_names(
    all_columns: list[str],
    horizon_class: str,
    feature_policy: str = "auto",
) -> list[str]:
    """Name-based candidate feature columns for a horizon (no data needed).

    Lets the caller read ONLY the columns a horizon could use from the
    246-column gold parquet, instead of materialising all of it in RAM.
    The numeric / non-null filter is applied later by
    :func:`_select_feature_columns` once the (much smaller) subset is loaded.
    """
    if feature_policy == "core30":
        return core_feature_columns(all_columns)
    candidate = [c for c in all_columns
                 if c not in _FEATURE_COMMON_DROP and c not in _FEATURE_LABEL_DROP]
    if feature_policy == "judgment":
        judged = _judgment_factors(horizon_class)
        if judged is None:
            raise RuntimeError(
                "--feature-policy judgment needs the factor judgment assignment at "
                f"{_JUDGMENT_ASSIGNMENT_PATH} (run scripts/factor_full_judgment.py first)"
            )
        judged_set = set(judged)
        base = {
            "short_5d": ("return_1d", "momentum_5d", "intraday_return",
                         "volatility_20d", "volume_mean_20d", "amount_mean_20d"),
            "mid_5d_30d": ("momentum_20d", "volatility_20d", "amount_mean_20d", "volume_mean_20d"),
            "long_30d_120d": ("momentum_20d", "amount_mean_20d"),
        }[horizon_class]
        return [c for c in candidate if (
            c in judged_set
            or c in base
            or (horizon_class == "long_30d_120d" and c.startswith("idx_"))
            or c in _INTRADAY_FEATURES
            or _is_selection_feature(c)
        )]
    if horizon_class == "short_5d":
        return [c for c in candidate if (
            c.startswith("alpha") and c <= "alpha060"
        ) or c in (
            "return_1d", "momentum_5d", "intraday_return",
            "volatility_20d", "volume_mean_20d", "amount_mean_20d",
        ) or c in _INTRADAY_FEATURES or _is_selection_feature(c)]
    if horizon_class == "mid_5d_30d":
        return [c for c in candidate if (
            c.startswith("alpha") and "060" <= c <= "alpha120"
        ) or c in (
            "momentum_20d", "volatility_20d", "amount_mean_20d", "volume_mean_20d",
        ) or c in _INTRADAY_FEATURES or _is_selection_feature(c)]
    # long_30d_120d
    return [c for c in candidate if (
        c.startswith("alpha") and c >= "alpha100"
    ) or c.startswith("idx_") or c in (
        "momentum_20d", "amount_mean_20d",
    ) or _is_selection_feature(c)]


def _select_feature_columns(
    panel: pd.DataFrame,
    horizon_class: str,
    feature_policy: str = "auto",
) -> list[str]:
    """Pick FT-Transformer features for a horizon class.

    Short ⊇ alpha 1-60 + return_1d / momentum_5d / volatility_20d /
    volume_mean_20d.
    Mid ⊇ alpha 60-120 + momentum_20d + ma-style features.
    Long ⊇ alpha 120-181 + macro indicators (idx_*).
    """
    if feature_policy == "core30":
        feats = [c for c in CORE_FEATURE_COLUMNS if c in panel.columns][:30]
    else:
        feats = _candidate_feature_names(list(panel.columns), horizon_class, feature_policy=feature_policy)
    # Drop columns whose dtype is non-numeric / mostly null
    out = []
    for c in feats:
        if c in panel.columns and pd.api.types.is_numeric_dtype(panel[c]):
            non_null = panel[c].notna().mean()
            min_coverage = 0.05 if c in _INTRADAY_FEATURES else 0.50
            if non_null >= min_coverage:
                out.append(c)
    return out


def _split_by_date(
    panel: pd.DataFrame,
    *,
    train_end: pd.Timestamp,
    embargo_days: int,
    test_end: pd.Timestamp,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    panel = panel.copy()
    panel["trade_date"] = pd.to_datetime(panel["trade_date"], errors="coerce")
    train = panel[panel["trade_date"] <= train_end]
    test_start = train_end + pd.tseries.offsets.BDay(embargo_days)
    test = panel[(panel["trade_date"] >= test_start) & (panel["trade_date"] <= test_end)]
    return train.reset_index(drop=True), test.reset_index(drop=True)


def _filter_by_regime_dates(
    panel: pd.DataFrame,
    regime_by_date: pd.Series | pd.DataFrame,
    *,
    regimes: list[str],
    min_rows: int = 1000,
) -> pd.DataFrame:
    """Keep rows whose trade_date belongs to one of the requested regimes."""
    if panel.empty or not regimes:
        return panel
    if isinstance(regime_by_date, pd.DataFrame):
        if "regime" not in regime_by_date.columns:
            raise ValueError("regime_by_date DataFrame must include a regime column")
        regime_series = regime_by_date["regime"].copy()
        if "trade_date" in regime_by_date.columns:
            regime_series.index = pd.to_datetime(regime_by_date["trade_date"], errors="coerce")
    else:
        regime_series = regime_by_date.copy()
    regime_series.index = pd.to_datetime(regime_series.index, errors="coerce")
    regime_series = regime_series[regime_series.index.notna()].dropna().astype(str)
    requested = {str(r).strip() for r in regimes if str(r).strip()}
    allowed_dates = set(regime_series[regime_series.isin(requested)].index)
    out = panel.copy()
    out["trade_date"] = pd.to_datetime(out["trade_date"], errors="coerce")
    out = out[out["trade_date"].isin(allowed_dates)].reset_index(drop=True)
    if len(out) < int(min_rows):
        raise ValueError(
            f"regime-filtered dataset has only {len(out)} rows for regimes={sorted(requested)}; "
            f"minimum required is {min_rows}"
        )
    return out


def _predictions_to_target_weights(
    predictions: pd.DataFrame,
    *,
    top_k: int,
    score_column: str,
) -> pd.DataFrame:
    pf = predictions[["trade_date", "symbol", score_column]].copy()
    pf = pf.dropna(subset=[score_column])
    pf = pf.sort_values(["trade_date", score_column], ascending=[True, False])
    pf["rank"] = pf.groupby("trade_date").cumcount()
    pf["weight"] = (pf["rank"] < top_k).astype(float) / float(top_k)
    wide = pf.pivot_table(index="trade_date", columns="symbol", values="weight", fill_value=0.0)
    return wide


def _cross_sectional_normalize(
    df: pd.DataFrame,
    feature_cols: list[str],
    *,
    method: str = "rank",
) -> pd.DataFrame:
    """Per-date cross-sectional normalisation of features (leak-free).

    For each ``trade_date`` independently, transform every feature into its
    cross-sectional position within that day. This is the single biggest
    generalisation lever for a full-universe model: it forces the network
    to learn *relative* strength (is this stock cheap / strong vs today's
    peers) instead of absolute levels, so a 1300-yuan blue-chip and a
    3-yuan micro-cap become directly comparable.

    Leak-free by construction: date ``t``'s transform uses only date ``t``'s
    cross-section, all of which is known at ``t``.

    * ``rank``   → percentile rank centred to [-0.5, 0.5] (robust to the
                   extreme feature values micro-caps produce).
    * ``zscore`` → per-date mean/std standardisation.
    """
    if not feature_cols:
        return df
    normalize_cols = [c for c in feature_cols if c not in _NO_CROSS_SECTIONAL_NORM]
    passthrough_cols = [c for c in feature_cols if c in _NO_CROSS_SECTIONAL_NORM]
    if passthrough_cols:
        for c in passthrough_cols:
            df[c] = pd.to_numeric(df[c], errors="coerce").replace([np.inf, -np.inf], np.nan).fillna(0.0).astype("float32")
    if not normalize_cols:
        return df
    # Mutate feature columns in place (no whole-frame copy — the frame can be
    # 7M+ rows and a copy doubled RAM into the OOM killer).
    grouped = df.groupby("trade_date", sort=False)
    if method == "rank":
        ranked = grouped[normalize_cols].rank(pct=True) - 0.5
        for c in normalize_cols:
            df[c] = ranked[c].fillna(0.0).astype("float32")
        del ranked
    elif method == "zscore":
        mean = grouped[normalize_cols].transform("mean")
        std = grouped[normalize_cols].transform("std").replace(0.0, np.nan)
        for c in normalize_cols:
            df[c] = ((df[c] - mean[c]) / std[c]).fillna(0.0).astype("float32")
        del mean, std
    else:
        raise ValueError(f"unknown cross-sectional method: {method}")
    return df


def _normalize_label_per_date(
    df: pd.DataFrame,
    label_col: str,
    *,
    winsor: float = 0.01,
) -> pd.DataFrame:
    """Per-date winsorise + z-score the training label.

    Micro-cap limit-up streaks produce forward returns of +50 % that
    dominate the Huber regression term and make the model chase garbage.
    Winsorising at the 1/99 % per-date quantiles then z-scoring within the
    day turns the target into a stable cross-sectional score, while the
    per-date rank loss continues to drive the portfolio objective.
    """
    out = df.copy()
    grp = out.groupby("trade_date", sort=False)[label_col]
    lo = grp.transform(lambda s: s.quantile(winsor))
    hi = grp.transform(lambda s: s.quantile(1.0 - winsor))
    clipped = out[label_col].clip(lower=lo, upper=hi)
    by_date = clipped.groupby(out["trade_date"], sort=False)
    mean = by_date.transform("mean")
    std = by_date.transform("std").replace(0.0, np.nan)
    out[label_col] = ((clipped - mean) / std).fillna(0.0)
    return out


def _gpu_probe() -> dict[str, object]:
    try:
        import torch
    except Exception as exc:  # noqa: BLE001
        return {"cuda_available": False, "reason": f"torch import failed: {exc}"}
    if not torch.cuda.is_available():
        return {"cuda_available": False, "reason": "torch.cuda.is_available() is False"}
    info: dict[str, object] = {
        "cuda_available": True,
        "torch_version": torch.__version__,
        "device_count": torch.cuda.device_count(),
        "devices": [],
    }
    for i in range(torch.cuda.device_count()):
        p = torch.cuda.get_device_properties(i)
        info["devices"].append({
            "index": i,
            "name": p.name,
            "total_memory_gb": round(p.total_memory / 1e9, 2),
            "capability": f"{p.major}.{p.minor}",
        })
    return info


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

@app.command("train-v8-deep")
def train_v8_deep(
    horizon_class: str = typer.Option(
        "short_5d", help="short_5d | mid_5d_30d | long_30d_120d",
    ),
    dataset_path: Path = typer.Option(
        Path("runtime/data/v7/gold/training_dataset/training_dataset_alpha181_full_nosynth.parquet"),
        exists=True, dir_okay=False,
        help="gold training dataset with OHLCV + alphas + forward_return labels",
    ),
    silver_panel_path: Path = typer.Option(
        Path("runtime/data/v7/silver/market_panel/market_panel.parquet"),
        exists=True, dir_okay=False,
        help="silver market panel for the strict backtest",
    ),
    symbols: Optional[str] = typer.Option(
        None,
        help="comma-separated symbols. omit for ALL symbols in the dataset.",
    ),
    symbols_file: Optional[Path] = typer.Option(
        None,
        help="optional file with comma-separated symbol list",
    ),
    train_start: str = typer.Option("2018-01-02"),
    train_end: str = typer.Option("2023-06-30"),
    test_end: str = typer.Option("2024-12-31"),
    embargo_days: int = typer.Option(20),
    top_k: int = typer.Option(30),
    max_epochs: int = typer.Option(20),
    batch_size: int = typer.Option(8192),
    d_token: int = typer.Option(128),
    n_blocks: int = typer.Option(4),
    n_heads: int = typer.Option(8),
    dates_per_step: int = typer.Option(
        8, help="# trading days per optimisation step; lower → smaller activation footprint",
    ),
    train_micro_batch: Optional[int] = typer.Option(
        None,
        help="If set, split each date-chunk into sub-batches of ≤ N rows for forward/backward; used to fit large universes on tight VRAM.",
    ),
    cross_sectional_norm: str = typer.Option(
        "rank",
        help="per-date cross-sectional feature normalisation: rank | zscore | none",
    ),
    label_norm: bool = typer.Option(
        True, "--label-norm/--no-label-norm",
        help="per-date winsorise+zscore the training label (anti micro-cap overfit)",
    ),
    feature_policy: str = typer.Option(
        "auto",
        "--feature-policy",
        help="auto | core30. core30 uses the auditable <=30 A-share factor policy.",
    ),
    attention_dropout: float = typer.Option(0.10, help="FT-Transformer attention dropout"),
    ffn_dropout: float = typer.Option(0.10, help="FT-Transformer FFN dropout"),
    weight_decay: float = typer.Option(1e-4, help="AdamW weight decay"),
    early_stopping_patience: int = typer.Option(10, help="epochs without val improvement before stop"),
    learning_rate: float = typer.Option(1e-3),
    regime_filter: Optional[str] = typer.Option(
        None,
        help="comma-separated PIT regimes to train this expert on, e.g. bull,neutral_up",
    ),
    regime_min_rows: int = typer.Option(1000, help="minimum rows after regime filter"),
    require_gpu: bool = typer.Option(True, "--require-gpu/--no-require-gpu"),
    output_dir: Path = typer.Option(
        Path("runtime/reports/v8/deep") / datetime.now().strftime("%Y%m%d_%H%M%S"),
    ),
):
    """Train FT-Transformer on GPU for one horizon class + emit OOS backtest."""
    typer.echo(f"[{_ts()}] train-v8-deep starting horizon={horizon_class}")
    gpu_info = _gpu_probe()
    typer.echo(f"[{_ts()}] gpu probe: {json.dumps(gpu_info)[:200]}")
    if require_gpu and not gpu_info.get("cuda_available"):
        typer.echo("[fatal] --require-gpu set but CUDA not available", err=True)
        raise typer.Exit(code=1)

    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "run_config.json").write_text(json.dumps({
        "horizon_class": horizon_class,
        "dataset_path": str(dataset_path),
        "silver_panel_path": str(silver_panel_path),
        "train_start": train_start, "train_end": train_end, "test_end": test_end,
        "embargo_days": embargo_days, "top_k": top_k,
        "max_epochs": max_epochs, "batch_size": batch_size,
        "d_token": d_token, "n_blocks": n_blocks, "n_heads": n_heads,
        "dates_per_step": dates_per_step, "train_micro_batch": train_micro_batch,
        "cross_sectional_norm": cross_sectional_norm, "label_norm": label_norm,
        "feature_policy": feature_policy,
        "attention_dropout": attention_dropout, "ffn_dropout": ffn_dropout,
        "weight_decay": weight_decay, "early_stopping_patience": early_stopping_patience,
        "learning_rate": learning_rate,
        "regime_filter": regime_filter, "regime_min_rows": regime_min_rows,
        "require_gpu": require_gpu, "gpu_probe": gpu_info,
    }, indent=2, default=str), encoding="utf-8")

    # ── 1. Load + filter dataset ───────────────────────────────────────
    label_col = _resolve_label_column(horizon_class)
    typer.echo(f"[{_ts()}] loading dataset → label={label_col}")
    sym_filter: list[str] | None = None
    if symbols:
        sym_filter = [s.strip() for s in symbols.split(",") if s.strip()]
    elif symbols_file:
        sym_filter = [s.strip() for s in symbols_file.read_text(encoding="utf-8").split(",") if s.strip()]
    # Read ONLY the columns this horizon needs. Loading all 246 gold columns
    # × 7.3M rows as float64 (~14 GB) then copying for normalisation blew
    # past system RAM (SIGKILL). We read keys + label + the horizon's
    # name-based feature candidates, and downcast features to float32.
    import pyarrow.parquet as pq

    all_names = list(pq.ParquetFile(dataset_path).schema.names)
    candidate_feats = _candidate_feature_names(all_names, horizon_class, feature_policy=feature_policy)
    if feature_policy == "core30" and len(candidate_feats) < 20:
        typer.echo(
            f"[fatal] --feature-policy core30 found only {len(candidate_feats)} core features. "
            "Run build-core-factor-dataset-v8 first.",
            err=True,
        )
        raise typer.Exit(code=1)
    read_cols = ["symbol", "trade_date"]
    if "available_at" in all_names:
        read_cols.append("available_at")
    if label_col not in all_names:
        typer.echo(f"[fatal] dataset lacks label column {label_col}", err=True)
        raise typer.Exit(code=1)
    read_cols.append(label_col)
    read_cols += [c for c in candidate_feats if c not in read_cols]
    df = pd.read_parquet(dataset_path, columns=read_cols)
    # Downcast feature columns float64 → float32 to halve the footprint.
    f32_cols = [c for c in candidate_feats
                if c in df.columns and pd.api.types.is_float_dtype(df[c])]
    if f32_cols:
        df[f32_cols] = df[f32_cols].astype("float32")
    typer.echo(f"[{_ts()}] dataset rows={len(df)} symbols={df['symbol'].nunique()} "
               f"cols={len(df.columns)} (of {len(all_names)})")
    if sym_filter:
        df = df[df["symbol"].isin(sym_filter)].reset_index(drop=True)
        typer.echo(f"[{_ts()}] after symbol filter rows={len(df)} symbols={df['symbol'].nunique()}")
    # Restrict date range to [train_start, test_end]
    df["trade_date"] = pd.to_datetime(df["trade_date"], errors="coerce")
    df = df[(df["trade_date"] >= pd.Timestamp(train_start))
            & (df["trade_date"] <= pd.Timestamp(test_end))].reset_index(drop=True)
    typer.echo(f"[{_ts()}] after date filter rows={len(df)}")
    if df.empty:
        typer.echo("[fatal] no rows after filtering", err=True)
        raise typer.Exit(code=1)
    if regime_filter:
        from quantagent.risk.regime_family import compute_regime_family

        regimes = [item.strip() for item in regime_filter.split(",") if item.strip()]
        typer.echo(f"[{_ts()}] applying PIT regime filter: {regimes}")
        regime_panel = pd.read_parquet(silver_panel_path)
        regime_panel["trade_date"] = pd.to_datetime(regime_panel["trade_date"], errors="coerce")
        regime_panel = regime_panel[
            (regime_panel["trade_date"] >= pd.Timestamp(train_start) - pd.Timedelta(days=260))
            & (regime_panel["trade_date"] <= pd.Timestamp(test_end))
        ]
        regime_by_date = compute_regime_family(regime_panel)
        before = len(df)
        try:
            df = _filter_by_regime_dates(
                df,
                regime_by_date,
                regimes=regimes,
                min_rows=regime_min_rows,
            )
        except ValueError as exc:
            typer.echo(f"[fatal] {exc}", err=True)
            raise typer.Exit(code=1) from exc
        typer.echo(
            f"[{_ts()}] after regime filter rows={len(df)} "
            f"({len(df) / max(1, before):.1%} of date-filtered rows)"
        )

    # Drop rows where the primary label is missing (no forward return)
    df = df[df[label_col].notna()].reset_index(drop=True)
    typer.echo(f"[{_ts()}] after label dropna rows={len(df)}")
    if df.empty:
        typer.echo(f"[fatal] no rows with non-null {label_col}", err=True)
        raise typer.Exit(code=1)

    # ── 2. Feature subset for the horizon class ────────────────────────
    feature_cols = _select_feature_columns(df, horizon_class, feature_policy=feature_policy)
    typer.echo(f"[{_ts()}] horizon={horizon_class} feature_cols={len(feature_cols)}")
    if feature_policy == "core30" and len(feature_cols) > 30:
        feature_cols = feature_cols[:30]
        typer.echo(f"[{_ts()}] capped core30 feature_cols={len(feature_cols)}")
    if len(feature_cols) < 8:
        typer.echo(f"[warn] only {len(feature_cols)} features selected — check dataset coverage")

    # ── 2b. Per-date cross-sectional feature normalisation (A) ─────────
    # Leak-free: each trade_date is normalised within its own cross-section.
    # Applied to the whole frame before the split so train / OOS / inference
    # all receive the identical transform.
    if cross_sectional_norm and cross_sectional_norm != "none":
        typer.echo(f"[{_ts()}] cross-sectional feature normalisation: {cross_sectional_norm}")
        df = _cross_sectional_normalize(df, feature_cols, method=cross_sectional_norm)

    # ── 3. Train / OOS split by date ───────────────────────────────────
    train_df, test_df = _split_by_date(
        df, train_end=pd.Timestamp(train_end), embargo_days=embargo_days,
        test_end=pd.Timestamp(test_end),
    )
    typer.echo(f"[{_ts()}] train rows={len(train_df)}, test rows={len(test_df)}")
    if train_df.empty or test_df.empty:
        typer.echo("[fatal] empty train or test split", err=True)
        raise typer.Exit(code=1)

    # ── 3b. Per-date label normalisation (B) — train rows only ─────────
    # The OOS label stays raw (backtest PnL uses raw panel returns); only
    # the training target is winsorised+zscored per date.
    if label_norm:
        typer.echo(f"[{_ts()}] per-date label winsorise+zscore on {label_col}")
        train_df = _normalize_label_per_date(train_df, label_col, winsor=0.01)

    # ── 4. FT-Transformer training ─────────────────────────────────────
    from quantagent.training.ft_transformer_trainer import (
        FTTransformerTrainer, FTTransformerTrainerConfig,
        predict_ft_transformer_artifact,
    )

    primary_horizon = int(label_col.removeprefix("forward_return_").removesuffix("d"))
    artifact_dir = output_dir / "ft"
    artifact_dir.mkdir(parents=True, exist_ok=True)
    trainer = FTTransformerTrainer(FTTransformerTrainerConfig(
        horizons=(primary_horizon,),
        feature_columns=tuple(feature_cols),
        d_token=d_token, n_blocks=n_blocks, n_heads=n_heads,
        attention_dropout=attention_dropout, ffn_dropout=ffn_dropout,
        weight_decay=weight_decay,
        early_stopping_patience=early_stopping_patience,
        batch_size=batch_size, max_epochs=max_epochs,
        dates_per_step=dates_per_step,
        train_micro_batch=train_micro_batch,
        learning_rate=learning_rate,
        require_gpu=require_gpu,
        device="cuda" if require_gpu else "auto",
        output_dir=str(artifact_dir),
        use_amp=True,
        log_gpu_memory=True,
    ))
    # Keep train_df schema lean — only keys + features + label needed
    keep_cols = ["symbol", "trade_date"] + feature_cols + [label_col]
    train_slice = train_df[keep_cols].copy()
    val_slice = test_df[keep_cols].copy()
    typer.echo(f"[{_ts()}] fitting FT-Transformer (epochs={max_epochs}, batch={batch_size}, d_token={d_token}, blocks={n_blocks})")
    artifacts = trainer.fit_and_save(train_slice, val_slice)
    typer.echo(f"[{_ts()}] training complete — device={artifacts.device} gpu={artifacts.gpu_name}")

    # ── 5. Predict on OOS ─────────────────────────────────────────────
    typer.echo(f"[{_ts()}] running OOS inference on {len(test_df)} rows")
    pred_input = test_df[["symbol", "trade_date"] + feature_cols].copy()
    pred = predict_ft_transformer_artifact(
        artifact_dir=str(artifact_dir),
        feature_frame=pred_input,
        primary_horizon=primary_horizon,
        device="cuda" if require_gpu else "cpu",
    )
    predictions = pred.predictions.copy()
    score_col = next(
        (c for c in predictions.columns if c.startswith("pred_") or c == "alpha_score"),
        None,
    )
    if score_col is None:
        # Fall back: pick the only numeric non-key column
        candidates = [c for c in predictions.columns
                      if c not in ("symbol", "trade_date")
                      and pd.api.types.is_numeric_dtype(predictions[c])]
        score_col = candidates[0] if candidates else None
    if score_col is None:
        typer.echo("[fatal] no prediction score column found", err=True)
        raise typer.Exit(code=1)
    predictions = predictions.rename(columns={score_col: "alpha_score"})
    predictions[["trade_date", "symbol", "alpha_score"]].to_parquet(
        output_dir / "predictions.parquet"
    )
    typer.echo(f"[{_ts()}] wrote predictions.parquet ({len(predictions)} rows)")

    # ── 6. Target weights ──────────────────────────────────────────────
    target_weights = _predictions_to_target_weights(
        predictions, top_k=top_k, score_column="alpha_score",
    )
    target_weights.to_parquet(output_dir / "target_weights.parquet")
    typer.echo(f"[{_ts()}] wrote target_weights.parquet ({len(target_weights)} rows × {len(target_weights.columns)} cols)")

    # ── 7. Strict v8 backtest on OOS predictions ───────────────────────
    from quantagent.backtest.ashare_execution_simulator import AShareExecutionSimulationConfig
    from quantagent.backtest.strict_v8 import run_strict_backtest_v8

    bt_oos_start = test_df["trade_date"].min()
    bt_oos_end = test_df["trade_date"].max()
    typer.echo(f"[{_ts()}] strict backtest {bt_oos_start} → {bt_oos_end}")
    panel = pd.read_parquet(silver_panel_path)
    panel["trade_date"] = pd.to_datetime(panel["trade_date"], errors="coerce")
    panel = panel[(panel["trade_date"] >= bt_oos_start) & (panel["trade_date"] <= bt_oos_end)]
    panel = panel[panel["symbol"].isin(target_weights.columns)].reset_index(drop=True)
    for col in ("is_suspended", "is_st", "is_limit_up", "is_limit_down"):
        if col not in panel.columns:
            panel[col] = False
    bt_result = run_strict_backtest_v8(
        target_weights, panel,
        config=AShareExecutionSimulationConfig(slippage_bps=8.0, initial_cash=1_000_000.0),
    )
    bt_paths = bt_result.write(output_dir / "backtest")
    typer.echo(f"[{_ts()}] backtest metrics: {bt_result.metrics.to_dict()}")
    typer.echo(f"[{_ts()}] wrote {len(bt_paths)} backtest files")

    # ── 8. Daily decision report ───────────────────────────────────────
    from quantagent.diagnostics.daily_decision_report import (
        DailyDecisionInputs, build_daily_decision_report,
    )
    last_day_weights = target_weights.iloc[-1] if not target_weights.empty else None
    report_inputs = DailyDecisionInputs(
        as_of_date=bt_oos_end,
        target_weights=last_day_weights if last_day_weights is not None else None,
        risk_events=bt_result.risk_events,
        global_conviction=0.70,
        gross_exposure=float(last_day_weights.sum()) if last_day_weights is not None else 0.0,
    )
    report = build_daily_decision_report(report_inputs)
    report.write(output_dir / "daily_decision_report.md")
    typer.echo(f"[{_ts()}] wrote daily_decision_report.md")

    typer.echo(f"\n[{_ts()}] DONE.  artifacts: {output_dir}")
    typer.echo(f"  total_return = {bt_result.metrics.total_return:.4f}")
    typer.echo(f"  sharpe       = {bt_result.metrics.sharpe:.3f}")
    typer.echo(f"  max_drawdown = {bt_result.metrics.max_drawdown:.4f}")
    return output_dir


__all__ = ["train_v8_deep"]
