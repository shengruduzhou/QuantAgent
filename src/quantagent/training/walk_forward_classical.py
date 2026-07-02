"""Schema-locked walk-forward training with CROSS-SECTIONAL features + a tree/linear model.

The deep MLP failed on the full universe partly because it standardised features
*globally* (one mean/scale over the whole train set), so it could not see the
per-day cross-sectional rank structure that alpha factors actually live in. This
trainer fixes that:

* features are transformed **per trading day** (cross-sectional rank or zscore)
  before fitting — causal/PIT-safe (each day uses only that day's cross-section);
* the model is LightGBM (handles NaN natively → no complete-case row attrition)
  or Ridge; one model per horizon;
* the label can be the raw forward return or its per-day cross-sectional rank
  (a listwise-style ranking target).

It is schema-locked (one pinned ``feature_schema.json`` across all folds) and
emits OOS predictions + a reproducibility manifest in the SAME shape as
``run_walk_forward_deep_training`` so the predictions flow straight into the
policy-search / strict-backtest / rank-IC eval loop.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass, field
import json
from pathlib import Path

import numpy as np
import pandas as pd

from quantagent.training.v7_deep_trainer import _load_feature_schema


@dataclass(frozen=True)
class ClassicalWFConfig:
    horizons: tuple[int, ...] = (1, 5, 20)
    model: str = "lightgbm"            # lightgbm | ridge
    cross_sectional: str = "rank"      # rank | zscore | none  (per-day feature transform)
    label_transform: str = "raw"       # raw | csrank          (per-day rank of the label)
    seed: int = 1729
    # LightGBM
    n_estimators: int = 400
    learning_rate: float = 0.03
    num_leaves: int = 63
    min_child_samples: int = 200
    subsample: float = 0.8
    colsample_bytree: float = 0.7
    reg_lambda: float = 1.0
    # Ridge
    ridge_alpha: float = 10.0
    feature_version: str = "classical"


@dataclass(frozen=True)
class ClassicalWFResult:
    oos_predictions: pd.DataFrame
    fold_metadata: pd.DataFrame
    schema_hash: str
    feature_version: str
    feature_columns: list[str]
    run_manifest: dict = field(default_factory=dict)
    manifest_path: str | None = None


def _cs_transform(df: pd.DataFrame, feature_cols: list[str], method: str) -> pd.DataFrame:
    """Per-day cross-sectional transform of the feature block (vectorised)."""
    if method == "none":
        return df[feature_cols].apply(pd.to_numeric, errors="coerce")
    block = df[["trade_date", *feature_cols]].copy()
    g = block.groupby("trade_date", sort=False)[feature_cols]
    if method == "rank":
        return g.rank(method="average", pct=True)
    if method == "zscore":
        mean = g.transform("mean")
        std = g.transform("std").replace(0.0, np.nan)
        return (block[feature_cols] - mean) / std
    raise ValueError(f"unknown cross_sectional method: {method}")


def _csrank_label(df: pd.DataFrame, label_col: str) -> pd.Series:
    return df.groupby("trade_date", sort=False)[label_col].rank(method="average", pct=True)


def _fit_predict(train_x, train_y, valid_x, cfg: ClassicalWFConfig):
    if cfg.model == "lightgbm":
        import lightgbm as lgb

        model = lgb.LGBMRegressor(
            n_estimators=cfg.n_estimators, learning_rate=cfg.learning_rate,
            num_leaves=cfg.num_leaves, min_child_samples=cfg.min_child_samples,
            subsample=cfg.subsample, colsample_bytree=cfg.colsample_bytree,
            reg_lambda=cfg.reg_lambda, random_state=cfg.seed, n_jobs=-1, verbose=-1,
        )
        model.fit(train_x, train_y)   # LightGBM handles NaN natively
        return model.predict(valid_x)
    if cfg.model == "ridge":
        from sklearn.linear_model import Ridge

        # Ridge needs finite inputs: fill missing (post-transform) with the
        # cross-section-neutral value (0.5 for rank, 0 for zscore/raw).
        fill = 0.5 if cfg.cross_sectional == "rank" else 0.0
        model = Ridge(alpha=cfg.ridge_alpha, random_state=cfg.seed)
        model.fit(np.nan_to_num(train_x, nan=fill), train_y)
        return model.predict(np.nan_to_num(valid_x, nan=fill))
    raise ValueError(f"unknown model: {cfg.model}")


def run_walk_forward_classical(
    dataset: pd.DataFrame,
    *,
    feature_schema_path: str,
    config: ClassicalWFConfig | None = None,
    split_config=None,
    output_dir: str | Path | None = None,
) -> ClassicalWFResult:
    from datetime import datetime, timezone

    from quantagent.training.splitters import WalkForwardSplitConfig, split_walk_forward

    if dataset is None or dataset.empty:
        raise ValueError("walk-forward classical training requires a non-empty dataset")
    cfg = config or ClassicalWFConfig()
    feats, schema_version, schema_hash = _load_feature_schema(feature_schema_path)
    feature_version = cfg.feature_version or schema_version
    model_version = f"{feature_version}@{schema_hash[:12]}"

    frame = dataset.copy()
    frame["trade_date"] = pd.to_datetime(frame["trade_date"], errors="coerce")
    frame = frame.sort_values(["trade_date", "symbol"]).reset_index(drop=True)
    missing = [c for c in feats if c not in frame.columns]
    if missing:
        raise ValueError(f"dataset missing {len(missing)} pinned feature columns: {missing[:15]}")
    horizons = [h for h in cfg.horizons if f"forward_return_{h}d" in frame.columns]
    if not horizons:
        raise ValueError("dataset has no forward_return_*d labels for the configured horizons")

    split_cfg = split_config or WalkForwardSplitConfig()
    folds = split_walk_forward(frame, config=split_cfg)
    if not folds:
        raise ValueError("walk-forward split produced no folds; check split_config vs date span")

    out_root = Path(output_dir) if output_dir else None
    pred_parts: list[pd.DataFrame] = []
    meta_rows: list[dict] = []
    for fold in folds:
        tr = frame.iloc[fold.train_idx]
        va = frame.iloc[fold.valid_idx]
        tr_x = _cs_transform(tr, feats, cfg.cross_sectional)
        va_x = _cs_transform(va, feats, cfg.cross_sectional)
        preds_h = {"symbol": va["symbol"].to_numpy(), "trade_date": va["trade_date"].to_numpy()}
        for h in horizons:
            label_col = f"forward_return_{h}d"
            y = _csrank_label(tr, label_col) if cfg.label_transform == "csrank" else tr[label_col]
            ok = pd.to_numeric(y, errors="coerce").notna().to_numpy()
            yhat = _fit_predict(tr_x.to_numpy(dtype=float)[ok],
                                pd.to_numeric(y, errors="coerce").to_numpy()[ok],
                                va_x.to_numpy(dtype=float), cfg)
            preds_h[f"alpha_{h}d"] = yhat
        pred = pd.DataFrame(preds_h).assign(
            fold_id=fold.fold_id,
            train_start=fold.train_dates[0], train_end=fold.train_dates[1],
            valid_start=fold.valid_dates[0], valid_end=fold.valid_dates[1],
            model_version=model_version, schema_hash=schema_hash, feature_version=feature_version,
        )
        pred_parts.append(pred)
        meta_rows.append({
            "fold_id": fold.fold_id,
            "train_start": fold.train_dates[0], "train_end": fold.train_dates[1],
            "valid_start": fold.valid_dates[0], "valid_end": fold.valid_dates[1],
            "n_train": int(fold.train_idx.size), "n_valid": int(fold.valid_idx.size),
            "embargo_days": int(fold.embargo_days), "model": cfg.model,
            "schema_hash": schema_hash, "feature_version": feature_version,
            "feature_count": len(feats),
        })

    oos = pd.concat(pred_parts, ignore_index=True) if pred_parts else pd.DataFrame()
    fold_plan = pd.DataFrame(meta_rows)
    manifest = {
        "created_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "model_version": model_version, "schema_hash": schema_hash,
        "feature_version": feature_version, "feature_count": len(feats),
        "feature_columns": list(feats), "feature_schema_path": str(feature_schema_path),
        "config": asdict(cfg), "split_config": asdict(split_cfg),
        "n_folds": len(folds), "n_oos_predictions": int(len(oos)),
        "dataset_rows": int(len(frame)), "dataset_symbols": int(frame["symbol"].nunique()),
        "dataset_dates": int(frame["trade_date"].nunique()),
        "fold_plan": [{k: (str(v) if isinstance(v, pd.Timestamp) else v) for k, v in r.items()} for r in meta_rows],
    }
    manifest_path = None
    if out_root is not None:
        out_root.mkdir(parents=True, exist_ok=True)
        fold_plan.to_csv(out_root / "fold_plan.csv", index=False)
        if not oos.empty:
            try:
                oos.to_parquet(out_root / "walkforward_predictions.parquet", index=False)
                manifest["predictions_path"] = str(out_root / "walkforward_predictions.parquet")
            except Exception:  # noqa: BLE001
                oos.to_csv(out_root / "walkforward_predictions.csv", index=False)
                manifest["predictions_path"] = str(out_root / "walkforward_predictions.csv")
        manifest_path = str(out_root / "run_manifest.json")
        Path(manifest_path).write_text(json.dumps(manifest, ensure_ascii=False, indent=2, sort_keys=True), encoding="utf-8")

    return ClassicalWFResult(
        oos_predictions=oos, fold_metadata=fold_plan, schema_hash=schema_hash,
        feature_version=feature_version, feature_columns=list(feats),
        run_manifest=manifest, manifest_path=manifest_path,
    )


__all__ = ["ClassicalWFConfig", "ClassicalWFResult", "run_walk_forward_classical"]
