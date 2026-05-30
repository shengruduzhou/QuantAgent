"""End-to-end training pipeline tests (P9.3).

Uses an in-memory stub provider registered into the router so the
pipeline can run without network access — but the contract under
test is the *production* path (router → forward labels → horizon
bundles → GA → strict backtest → daily report).
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from quantagent.data.providers.base import ProviderRequest, ProviderResult
from quantagent.data.router import (
    MultiSourceDataRouter,
    RouterConfig,
    RoutedProvider,
    RouterAllSourcesUnavailable,
)
from quantagent.training.horizon_models import HorizonClass
from quantagent.training.v8_pipeline import (
    V8TrainingConfig,
    build_default_factor_panel,
    build_forward_returns,
    build_top_k_target_weights,
    factor_blend_to_predictions,
    run_v8_training_pipeline,
)


# ---------------------------------------------------------------------------
# Synthetic provider with deterministic OHLCV
# ---------------------------------------------------------------------------

class _SyntheticProvider:
    """In-memory deterministic OHLCV with trending closes + noise.

    This is a *test fixture* — the router knows it as a registered
    real-source impostor so the production code path is exercised.
    """

    def __init__(self, name: str = "stub", n_days: int = 120, seed: int = 7):
        self.name = name
        self.n_days = n_days
        self.seed = seed
        self.rng = np.random.default_rng(seed)
        # Build the cached panel once so the random draws are stable
        dates = pd.bdate_range("2024-01-01", periods=n_days)
        syms = [f"60000{i:02d}.SH" for i in range(15)]
        rows = []
        prices: dict[str, float] = {s: 100.0 + i * 0.5 for i, s in enumerate(syms)}
        for d in dates:
            for s in syms:
                drift = self.rng.normal(0.0005, 0.012)
                prices[s] *= (1.0 + drift)
                close = prices[s]
                rows.append({
                    "symbol": s, "trade_date": d,
                    "open": close * 0.998, "high": close * 1.005,
                    "low": close * 0.995, "close": close,
                    "volume": 1_000_000.0, "amount": close * 1_000_000.0,
                    "is_suspended": False, "is_st": False,
                    "is_limit_up": False, "is_limit_down": False,
                })
        self._cache = pd.DataFrame(rows)

    def daily_ohlcv(self, request: ProviderRequest) -> ProviderResult:
        df = self._cache.copy()
        if request.symbols:
            df = df[df["symbol"].isin(request.symbols)]
        df = df[
            (df["trade_date"] >= pd.Timestamp(request.start_date))
            & (df["trade_date"] <= pd.Timestamp(request.end_date))
        ]
        df["available_at"] = df.groupby("symbol")["trade_date"].shift(-1)
        df["available_at"] = df["available_at"].fillna(df["trade_date"] + pd.Timedelta(days=1))
        return ProviderResult(df.reset_index(drop=True), source=self.name, quality_score=0.85)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _stub_router() -> MultiSourceDataRouter:
    router = MultiSourceDataRouter(RouterConfig(daily_priority=("stub",)))
    router.register(RoutedProvider(name="stub", provider=_SyntheticProvider()))
    return router


# ---------------------------------------------------------------------------
# Forward returns
# ---------------------------------------------------------------------------

def test_build_forward_returns_emits_one_column_per_horizon():
    provider = _SyntheticProvider(n_days=30)
    panel = provider._cache
    fwd = build_forward_returns(panel, horizons=(1, 5, 20))
    for h in (1, 5, 20):
        assert f"forward_return_{h}d" in fwd.columns
    assert "forward_return" in fwd.columns
    # PIT: forward returns at the tail should be NaN
    assert fwd[f"forward_return_20d"].iloc[-1] != fwd[f"forward_return_20d"].iloc[-1]  # NaN


def test_build_default_factor_panel_has_two_factors():
    provider = _SyntheticProvider(n_days=60)
    panel = build_default_factor_panel(provider._cache)
    assert "mr_5d" in panel.columns
    assert "mom_20d" in panel.columns
    assert not panel.empty


def test_factor_blend_to_predictions_renames_to_alpha_score():
    provider = _SyntheticProvider(n_days=60)
    panel = build_default_factor_panel(provider._cache)
    preds = factor_blend_to_predictions(panel, {"mr_5d": 0.5, "mom_20d": 0.5})
    assert "alpha_score" in preds.columns
    assert set(preds.columns) == {"trade_date", "symbol", "alpha_score"}


def test_build_top_k_target_weights_pivots_to_wide():
    dates = pd.bdate_range("2024-03-01", periods=3)
    preds = pd.DataFrame([
        {"trade_date": d, "symbol": sym, "alpha_score": rank}
        for d in dates for rank, sym in enumerate(("A", "B", "C", "D"))
    ])
    wide = build_top_k_target_weights(preds, top_k=2)
    # top-2 of 4 → 0.5 + 0.5 = 1.0 per day
    assert (wide.sum(axis=1) <= 1.0 + 1e-9).all()


# ---------------------------------------------------------------------------
# Full pipeline
# ---------------------------------------------------------------------------

def test_full_pipeline_produces_backtest_and_report(tmp_path):
    router = _stub_router()
    artifacts = run_v8_training_pipeline(
        router=router,
        symbols=tuple(f"60000{i:02d}.SH" for i in range(15)),
        start_date="2024-01-01", end_date="2024-06-30",
        config=V8TrainingConfig(
            horizon_class=HorizonClass.SHORT,
            top_k=5, ga_population=6, ga_generations=2,
            walk_forward_folds=2, min_train_days=30, min_test_days=10,
            embargo_days=2,
        ),
        output_dir=tmp_path,
    )
    assert not artifacts.market_panel.empty
    assert not artifacts.factor_panel.empty
    assert artifacts.backtest.metrics.n_trades >= 0
    # Every expected output file exists
    for name in (
        "market_panel.parquet", "forward_returns.parquet",
        "factor_panel.parquet", "router_diagnostics.json",
        "daily_report.md",
    ):
        assert (tmp_path / name).exists(), name
    bt_dir = tmp_path / "backtest"
    assert (bt_dir / "metrics.json").exists()
    assert (bt_dir / "trades.csv").exists()
    assert (bt_dir / "factor_weights.json").exists()


def test_pipeline_fails_loud_when_no_provider_serves_request():
    router = _stub_router()
    with pytest.raises(RouterAllSourcesUnavailable):
        run_v8_training_pipeline(
            router=router,
            symbols=("UNKNOWN.SH",),    # not in synthetic cache → empty
            start_date="1990-01-01", end_date="1990-01-31",
            config=V8TrainingConfig(top_k=5, ga_population=4, ga_generations=1,
                                     walk_forward_folds=2,
                                     min_train_days=30, min_test_days=10),
        )


def test_pipeline_records_router_diagnostics(tmp_path):
    router = _stub_router()
    artifacts = run_v8_training_pipeline(
        router=router,
        symbols=tuple(f"60000{i:02d}.SH" for i in range(10)),
        start_date="2024-01-01", end_date="2024-06-30",
        config=V8TrainingConfig(top_k=3, ga_population=4, ga_generations=1,
                                 walk_forward_folds=2,
                                 min_train_days=30, min_test_days=10,
                                 embargo_days=2),
        output_dir=tmp_path,
    )
    diag = artifacts.router_diagnostics
    assert diag["primary_source"] == "stub"
    assert "stub" in diag["per_source"]
    on_disk = json.loads((tmp_path / "router_diagnostics.json").read_text(encoding="utf-8"))
    assert on_disk["primary_source"] == "stub"


def test_train_v8_pipeline_cli_registered_on_app():
    from quantagent.cli import app
    names = {c.name for c in app.registered_commands}
    assert "train-v8-pipeline" in names
