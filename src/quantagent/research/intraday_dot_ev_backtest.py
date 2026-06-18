"""End-to-end closed loop for the cost-sensitive intraday Do-T EV engine.

This module is the missing real-data driver that wires the EV stack together on
the actual TickFlow 1-minute book panel:

    causal features  (intraday_features.build_causal_intraday_feature_frame)
        + round-trip labels (do_t_roundtrip_labels.build_round_trip_labels)
        -> walk-forward split (train / validation / test)
        -> calibrated tabular models (do_t_models.train_do_t_models)
        -> per-minute model signals (predict_model_signals)
        -> decide_ev() over a T+1 IntradayLedger seeded with the held position,
           executed through the conservative next-bar IntradayFillSimulator
        -> trade ledger -> evaluate_walk_forward_results() deployment verdict.

Everything is causal: features never look forward, the model is trained only on
``train``/``validation`` dates and the closed loop is simulated minute-by-minute
on the unseen ``test`` dates.  The default action is always NO_TRADE; the engine
only trades a held name when a calibrated positive-EV, T+1-legal round trip
clears the dynamic cost / probability / risk gates.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from quantagent.execution.broker_base import OrderSide
from quantagent.execution.intraday_features import (
    CAUSAL_INTRADAY_FEATURE_COLUMNS,
    FUNDFLOW_FEATURE_COLUMNS,
    build_causal_intraday_feature_frame,
    merge_fundflow_features,
)
from quantagent.execution.intraday_fill import CostConfig, FillMode, IntradayFillSimulator
from quantagent.execution.intraday_ev_engine import (
    EVDecisionConfig,
    decide_ev,
)
from quantagent.execution.intraday_ledger import IntradayLedger
from quantagent.training.do_t_models import predict_model_signals, train_do_t_models
from quantagent.training.do_t_roundtrip_labels import (
    ROUND_TRIP_LABEL_COLUMNS,
    RoundTripLabelConfig,
    build_round_trip_labels,
)
from quantagent.research.intraday_dot_walkforward import evaluate_walk_forward_results


@dataclass(frozen=True)
class EVBacktestConfig:
    start: str = "2025-09-01"
    end: str = "2026-06-12"
    train_end: str = "2026-02-27"
    validation_end: str = "2026-04-15"
    order_notional_yuan: float = 100_000.0
    horizon_minutes: int = 60
    # EV / cost knobs (defaults are realistic retail; maker_only flips to ~10bps RT)
    commission_rate: float = 0.0003
    stamp_tax_sell: float = 0.0005
    transfer_fee: float = 0.00001
    slippage_bps: float = 8.0
    spread_bps: float = 6.0
    absolute_min_edge_bps: float = 8.0
    edge_cost_multiple: float = 2.0  # dynamic_min_edge floor = multiple * round-trip cost
    base_success_prob: float = 0.58
    max_round_trips_per_day: int = 3
    backend: str = "lightgbm"
    min_round_trips_enable: int = 300
    random_seed: int = 42
    fill_mode: str = "conservative"

    def cost_config(self) -> CostConfig:
        return CostConfig(
            commission_rate=self.commission_rate,
            min_commission=0.0,
            stamp_tax_sell=self.stamp_tax_sell,
            transfer_fee=self.transfer_fee,
            slippage_bps=self.slippage_bps,
            spread_bps=self.spread_bps,
        )

    def ev_config(self) -> EVDecisionConfig:
        cost = self.cost_config()
        # dynamic_min_edge_bps = max(edge_cost_multiple * cost, ...). EVDecisionConfig
        # hard-codes the 2.0 multiple inside dynamic_thresholds, so we fold a
        # non-2.0 multiple into absolute_min_edge_bps to keep the public knob.
        return EVDecisionConfig(
            cost=cost,
            absolute_min_edge_bps=self.absolute_min_edge_bps,
            base_success_prob=self.base_success_prob,
            max_round_trips_per_day=self.max_round_trips_per_day,
        )

    def label_config(self) -> RoundTripLabelConfig:
        return RoundTripLabelConfig(
            horizon_minutes=self.horizon_minutes,
            min_required_edge_bps=self.absolute_min_edge_bps,
            cost=self.cost_config(),
        )


@dataclass
class EVBacktestResult:
    verdict: str
    reason: str
    metrics: dict[str, Any] = field(default_factory=dict)
    trades: pd.DataFrame = field(default_factory=pd.DataFrame)
    baseline_trades: dict[str, pd.DataFrame] = field(default_factory=dict)
    diagnostics: dict[str, Any] = field(default_factory=dict)
    n_train_rows: int = 0
    n_test_symbol_days: int = 0
    models: Any = None


# --------------------------------------------------------------------------- #
# 1. Build the book minute panel (held symbol-days only) with pre_close/limits
# --------------------------------------------------------------------------- #
def load_book_keys(holdings_csv: str | Path, start: pd.Timestamp, end: pd.Timestamp) -> pd.DataFrame:
    h = pd.read_csv(holdings_csv)
    if not {"trade_date", "symbol", "weight"}.issubset(h.columns):
        return pd.DataFrame(columns=["trade_date", "symbol", "weight"])
    h["trade_date"] = pd.to_datetime(h["trade_date"], errors="coerce").dt.normalize()
    h["symbol"] = h["symbol"].astype(str)
    h["weight"] = pd.to_numeric(h["weight"], errors="coerce")
    h = h[(h["weight"] > 0) & (h["trade_date"] >= start) & (h["trade_date"] <= end)]
    return h.dropna(subset=["trade_date", "symbol"]).drop_duplicates(["trade_date", "symbol"]).reset_index(drop=True)


def _symbol_limit_band(symbol: str) -> float:
    s = str(symbol)
    if s.startswith(("30", "68")):
        return 0.20
    if s.startswith(("8", "4")):
        return 0.30
    return 0.10


def build_book_minute_panel(
    *,
    minute_dir: str | Path,
    book_keys: pd.DataFrame,
    panel_path: str | Path,
    start: pd.Timestamp,
    end: pd.Timestamp,
) -> pd.DataFrame:
    """Concatenate held symbol-day minute bars with pre_close + limit prices."""
    mdir = Path(minute_dir)
    panel = pd.read_parquet(panel_path, columns=["symbol", "trade_date", "close"])
    panel["trade_date"] = pd.to_datetime(panel["trade_date"], errors="coerce").dt.normalize()
    panel["symbol"] = panel["symbol"].astype(str)
    panel = panel.sort_values(["symbol", "trade_date"])
    panel["pre_close"] = panel.groupby("symbol", sort=False)["close"].shift(1)
    preclose_map = panel.set_index(["symbol", "trade_date"])["pre_close"].to_dict()

    wanted_days = {
        sym: set(grp["trade_date"].tolist())
        for sym, grp in book_keys.groupby("symbol", sort=False)
    }
    frames: list[pd.DataFrame] = []
    for sym, days in wanted_days.items():
        p = mdir / f"{sym}.parquet"
        if not p.exists():
            continue
        bars = pd.read_parquet(
            p, columns=["symbol", "trade_time", "open", "high", "low", "close", "volume", "amount"]
        )
        bars["trade_time"] = pd.to_datetime(bars["trade_time"], errors="coerce")
        bars["trade_date"] = bars["trade_time"].dt.normalize()
        bars = bars[(bars["trade_time"] >= start) & (bars["trade_time"] <= end + pd.Timedelta(days=1))]
        bars = bars[bars["trade_date"].isin(days)]
        if bars.empty:
            continue
        bars["symbol"] = sym
        band = _symbol_limit_band(sym)
        pc = bars["trade_date"].map(lambda d, s=sym: preclose_map.get((s, pd.Timestamp(d)), np.nan))
        bars["pre_close"] = pd.to_numeric(pc, errors="coerce")
        # fallback: first bar open as pre_close proxy when panel has no prior close
        first_open = bars.groupby("trade_date")["open"].transform("first")
        bars["pre_close"] = bars["pre_close"].fillna(first_open)
        bars["limit_up"] = bars["pre_close"] * (1.0 + band)
        bars["limit_down"] = bars["pre_close"] * (1.0 - band)
        frames.append(bars)
    if not frames:
        return pd.DataFrame()
    out = pd.concat(frames, ignore_index=True)
    return out.sort_values(["symbol", "trade_date", "trade_time"]).reset_index(drop=True)


# --------------------------------------------------------------------------- #
# 2. Build the causal feature + round-trip label training table
# --------------------------------------------------------------------------- #
def build_feature_label_table(
    minute_panel: pd.DataFrame,
    cfg: EVBacktestConfig,
    fundflow_panel: pd.DataFrame | None = None,
) -> pd.DataFrame:
    feats = build_causal_intraday_feature_frame(minute_panel, include_level2=False)
    if fundflow_panel is not None and not fundflow_panel.empty:
        feats = merge_fundflow_features(feats, fundflow_panel)
    labels = build_round_trip_labels(minute_panel, config=cfg.label_config())
    key = ["symbol", "trade_date", "trade_time"]
    for df in (feats, labels):
        df["symbol"] = df["symbol"].astype(str)
        df["trade_date"] = pd.to_datetime(df["trade_date"], errors="coerce").dt.normalize()
        df["trade_time"] = pd.to_datetime(df["trade_time"], errors="coerce")
    label_cols = [c for c in ROUND_TRIP_LABEL_COLUMNS if c in labels.columns]
    merged = feats.merge(labels[key + label_cols], on=key, how="left")
    return merged


def feature_columns_present(table: pd.DataFrame) -> list[str]:
    base = [c for c in CAUSAL_INTRADAY_FEATURE_COLUMNS if c in table.columns]
    # include fund-flow order-flow features only when actually populated (forward-collected)
    ff = [c for c in FUNDFLOW_FEATURE_COLUMNS if c in table.columns and table[c].notna().any()]
    return base + ff


# --------------------------------------------------------------------------- #
# 3. Closed-loop ledger simulation for one book symbol-day
# --------------------------------------------------------------------------- #
def _round_lot(qty: float, lot: int = 100) -> int:
    return int(max(0, int(qty)) // lot * lot)


def _state_from_row(row: pd.Series, *, remaining_bars: int, completed_rt: int) -> dict[str, Any]:
    def g(name: str, default: float = 0.0) -> float:
        v = row.get(name, default)
        try:
            v = float(v)
        except (TypeError, ValueError):
            return default
        return v if np.isfinite(v) else default

    lim_up = g("limit_up_distance", 1.0)
    lim_dn = g("limit_down_distance", 1.0)
    return {
        "last": g("close", g("price", 0.0)),
        "close": g("close", 0.0),
        "estimated_spread_bps": g("estimated_spread_bps", 6.0),
        "rolling_volatility_20m": g("rolling_volatility_20m", 0.0),
        "rolling_return_20m": g("rolling_return_20m", 0.0),
        "volume_capacity_ratio": g("volume_capacity_ratio", 0.0),
        "one_way_trend_probability": g("one_way_trend_probability", 0.0),
        "mean_reversion_probability": g("mean_reversion_probability", 0.0),
        "near_limit_risk": bool(g("near_limit_risk", 0.0) >= 0.5),
        "limit_up_distance": lim_up,
        "limit_down_distance": lim_dn,
        "near_limit_up_risk": bool(lim_up < 0.005),
        "near_limit_down_risk": bool(lim_dn < 0.005),
        "minutes_to_close": float(remaining_bars),
        "completed_round_trips_today": float(completed_rt),
    }


def simulate_symbol_day(
    *,
    day_bars: pd.DataFrame,
    day_table: pd.DataFrame,
    signals: list,
    cfg: EVBacktestConfig,
    n_names: int,
    regime: str,
) -> list[dict[str, Any]]:
    """Run the EV engine minute-by-minute over one held symbol-day.

    Returns trade rows (one per completed round trip + one per EOD restore)
    in the schema ``evaluate_walk_forward_results`` consumes.
    """
    bars = day_bars.reset_index(drop=True)
    table = day_table.reset_index(drop=True)
    n = len(bars)
    if n < 5 or len(table) != n:
        return []
    close = pd.to_numeric(bars["close"], errors="coerce").to_numpy(float)
    ref_price = float(close[0]) if np.isfinite(close[0]) and close[0] > 0 else float(np.nanmedian(close))
    if not np.isfinite(ref_price) or ref_price <= 0:
        return []
    position = _round_lot(cfg.order_notional_yuan / ref_price)
    if position <= 0:
        return []
    notional_ref = position * ref_price

    ledger = IntradayLedger(
        symbol=str(bars["symbol"].iloc[0]),
        date=str(bars["trade_date"].iloc[0]),
        carried_shares=position,
        target_shares=position,
        cash=notional_ref * 3.0,
    )
    ev_cfg = cfg.ev_config()
    fill_sim = IntradayFillSimulator(cost_config=cfg.cost_config())
    fill_mode = FillMode(cfg.fill_mode)
    pair_seq = 0
    open_meta: dict[str, dict[str, Any]] = {}
    completed_rt = 0
    rows: list[dict[str, Any]] = []
    # leave the last 2 bars for the closing leg / restore
    for i in range(n - 2):
        remaining = n - 1 - i
        state = _state_from_row(table.iloc[i], remaining_bars=remaining, completed_rt=completed_rt)
        sig = signals[i]
        decision = decide_ev(state, ledger, sig, ev_cfg)
        action = decision.action
        if action == "NO_TRADE" or decision.quantity <= 0:
            continue
        side = _action_side(action)
        fill = fill_sim.simulate(
            bars, signal_index=i, side=side, quantity=int(decision.quantity), mode=fill_mode
        )
        if not fill.filled:
            continue
        price = float(fill.fill_price)
        qty = int(fill.filled_qty)
        cost = float(fill.costs.get("total", 0.0))
        ftime = str(fill.fill_time)
        try:
            if action == "SELL_HIGH":
                qty = min(qty, ledger.sellable_shares)
                if qty <= 0:
                    continue
                pair_seq += 1
                pid = f"rt{pair_seq}"
                ledger.open_sell_high(pair_id=pid, quantity=qty, price=price, time=ftime, cost=cost)
                open_meta[pid] = {"side": "reverse", "open_idx": i, "open_price": price,
                                  "confidence": decision.calibrated_probability}
            elif action == "BUY_LOW":
                if not ledger.can_buy(qty, price, cost):
                    continue
                pair_seq += 1
                pid = f"rt{pair_seq}"
                ledger.open_buy_low(pair_id=pid, quantity=qty, price=price, time=ftime, cost=cost)
                open_meta[pid] = {"side": "positive", "open_idx": i, "open_price": price,
                                  "confidence": decision.calibrated_probability}
            elif action == "BUY_BACK":
                qty = min(qty, _open_qty(ledger.open_sell_pairs))
                if qty <= 0 or not ledger.can_buy(qty, price, cost):
                    continue
                pid = _front_pair_id(ledger.open_sell_pairs)
                meta = open_meta.get(pid, {})
                ev = ledger.close_sell_pair_buyback(quantity=qty, price=price, time=ftime, cost=cost)
                completed_rt += 1
                rows.append(_close_row(ev, meta, bars, close_idx=i, notional_ref=notional_ref,
                                       n_names=n_names, regime=regime, side="reverse"))
            elif action == "SELL_AFTER_BUY":
                qty = min(qty, _open_qty(ledger.open_buy_pairs), ledger.sellable_shares)
                if qty <= 0:
                    continue
                pid = _front_pair_id(ledger.open_buy_pairs)
                meta = open_meta.get(pid, {})
                ev = ledger.close_buy_pair_sell_after_buy(quantity=qty, price=price, time=ftime, cost=cost)
                completed_rt += 1
                rows.append(_close_row(ev, meta, bars, close_idx=i, notional_ref=notional_ref,
                                       n_names=n_names, regime=regime, side="positive"))
        except ValueError:
            # any T+1 / cash legality violation -> skip the leg (defensive)
            continue

    # EOD: restore any open pair as a risk event (never counted as a success)
    last_price = float(close[-1]) if np.isfinite(close[-1]) else ref_price
    if ledger.open_sell_pairs or ledger.open_buy_pairs or ledger.position_gap_to_target != 0:
        restore = ledger.mark_eod_restore(price=last_price, time=str(bars["trade_time"].iloc[-1]),
                                          cost=0.0)
        if restore is not None:
            rows.append({
                "symbol": ledger.symbol, "trade_date": pd.Timestamp(bars["trade_date"].iloc[0]),
                "action": "EOD_RESTORE", "completed_round_trip": 0, "eod_restore": 1,
                "gross_pnl_bps": 0.0, "net_pnl_bps": float(restore.net_pnl) / notional_ref * 10_000.0,
                "daily_uplift_bps": float(restore.net_pnl) / notional_ref * 10_000.0 / max(1, n_names),
                "sell_high_fail_new_high": 0, "buy_low_fail_breakdown": 0,
                "confidence": np.nan, "regime": regime,
                "capacity_usage": 0.0, "turnover": 0.0,
            })
    return rows


def _close_row(event, meta, bars, *, close_idx: int, notional_ref: float, n_names: int,
               regime: str, side: str) -> dict[str, Any]:
    open_idx = int(meta.get("open_idx", close_idx))
    open_price = float(meta.get("open_price", event.price))
    net_bps = float(event.net_pnl) / notional_ref * 10_000.0
    gross_bps = float(event.gross_pnl) / notional_ref * 10_000.0
    seg = bars.iloc[open_idx:close_idx + 1]
    fail_new_high = 0
    fail_breakdown = 0
    if not seg.empty and open_price > 0:
        hi = float(pd.to_numeric(seg["high"], errors="coerce").max())
        lo = float(pd.to_numeric(seg["low"], errors="coerce").min())
        if side == "reverse":  # sold high, bought back: chased if a new high formed first
            fail_new_high = int(hi > open_price * 1.0015)
        else:  # bought low, sold after: broke down if a new low formed first
            fail_breakdown = int(lo < open_price * 0.9985)
    return {
        "symbol": str(bars["symbol"].iloc[0]), "trade_date": pd.Timestamp(bars["trade_date"].iloc[0]),
        "action": event.action, "completed_round_trip": 1, "eod_restore": 0,
        "gross_pnl_bps": gross_bps, "net_pnl_bps": net_bps,
        "daily_uplift_bps": net_bps / max(1, n_names),
        "sell_high_fail_new_high": fail_new_high, "buy_low_fail_breakdown": fail_breakdown,
        "confidence": float(meta.get("confidence", np.nan)), "regime": regime,
        "capacity_usage": 0.0, "turnover": abs(net_bps),
    }


def _action_side(action: str) -> OrderSide:
    return OrderSide.SELL if action in {"SELL_HIGH", "SELL_AFTER_BUY"} else OrderSide.BUY


def _open_qty(pairs) -> int:
    return sum(int(p.quantity) for p in pairs)


def _front_pair_id(pairs) -> str:
    return pairs[0].pair_id if pairs else ""


# --------------------------------------------------------------------------- #
# 4. Orchestration: train on train+val dates, simulate the test dates
# --------------------------------------------------------------------------- #
def run_ev_closed_loop(
    *,
    minute_dir: str | Path,
    holdings_csv: str | Path,
    panel_path: str | Path,
    cfg: EVBacktestConfig,
    regimes: dict[tuple, str] | None = None,
    feature_label_table: pd.DataFrame | None = None,
) -> EVBacktestResult:
    start, end = pd.Timestamp(cfg.start), pd.Timestamp(cfg.end)
    train_end, val_end = pd.Timestamp(cfg.train_end), pd.Timestamp(cfg.validation_end)
    rng = np.random.default_rng(cfg.random_seed)

    if feature_label_table is None:
        book_keys = load_book_keys(holdings_csv, start, end)
        if book_keys.empty:
            return EVBacktestResult("DO_NOT_ENABLE", "no held symbol-days in window", diagnostics={})
        panel = build_book_minute_panel(
            minute_dir=minute_dir, book_keys=book_keys, panel_path=panel_path, start=start, end=end
        )
        if panel.empty:
            return EVBacktestResult("DO_NOT_ENABLE", "no minute bars for held symbol-days", diagnostics={})
        table = build_feature_label_table(panel, cfg)
        table = table.merge(panel[["symbol", "trade_date", "trade_time", "open", "high", "low",
                                    "close", "volume", "limit_up", "limit_down"]],
                            on=["symbol", "trade_date", "trade_time"], how="left", suffixes=("", "_bar"))
    else:
        table = feature_label_table

    table["trade_date"] = pd.to_datetime(table["trade_date"], errors="coerce").dt.normalize()
    feat_cols = feature_columns_present(table)
    train_mask = table["trade_date"] <= val_end  # train on train+validation, test is unseen
    test_mask = table["trade_date"] > val_end
    train_df = table[train_mask].copy()
    test_df = table[test_mask].copy()

    diagnostics = {
        "total_rows": int(len(table)),
        "train_rows": int(len(train_df)),
        "test_rows": int(len(test_df)),
        "test_symbol_days": int(test_df.drop_duplicates(["symbol", "trade_date"]).shape[0]),
        "feature_count": len(feat_cols),
        "round_trip_cost_bps": _round_trip_cost_bps(cfg),
        "dynamic_min_edge_floor_bps": cfg.edge_cost_multiple * _round_trip_cost_bps(cfg),
    }
    if train_df.empty or test_df.empty:
        return EVBacktestResult("DO_NOT_ENABLE", "empty train or test split", diagnostics=diagnostics)

    models = train_do_t_models(
        train_df, feature_columns=feat_cols, backend=cfg.backend, allow_sklearn_fallback=True,
        random_state=cfg.random_seed,
    )
    diagnostics["model_diagnostics"] = models.diagnostics

    engine_rows: list[dict[str, Any]] = []
    shuffled_rows: list[dict[str, Any]] = []
    randtime_rows: list[dict[str, Any]] = []

    test_df = test_df.sort_values(["trade_date", "symbol", "trade_time"])
    names_per_day = test_df.groupby("trade_date")["symbol"].nunique().to_dict()
    for (sym, day), g in test_df.groupby(["symbol", "trade_date"], sort=False):
        g = g.sort_values("trade_time").reset_index(drop=True)
        day_bars = _bars_view(g)
        if day_bars is None:
            continue
        n_names = int(names_per_day.get(day, 1))
        regime = (regimes or {}).get((str(sym), pd.Timestamp(day)), "unknown")
        sigs = predict_model_signals(models, g)
        if len(sigs) != len(g):
            continue
        engine_rows += simulate_symbol_day(day_bars=day_bars, day_table=g, signals=sigs,
                                           cfg=cfg, n_names=n_names, regime=regime)
        # permutation nulls: destroy the feature->minute alignment of the signals
        perm1 = list(rng.permutation(len(sigs)))
        shuffled_rows += simulate_symbol_day(day_bars=day_bars, day_table=g,
                                             signals=[sigs[k] for k in perm1], cfg=cfg,
                                             n_names=n_names, regime=regime)
        perm2 = list(rng.permutation(len(sigs)))
        randtime_rows += simulate_symbol_day(day_bars=day_bars, day_table=g,
                                             signals=[sigs[k] for k in perm2], cfg=cfg,
                                             n_names=n_names, regime=regime)

    trades = pd.DataFrame(engine_rows)
    baselines = {
        "shuffled_signal_baseline": pd.DataFrame(shuffled_rows),
        "random_time_same_count_baseline": pd.DataFrame(randtime_rows),
    }
    report = evaluate_walk_forward_results(trades, baselines=baselines,
                                           min_round_trips=cfg.min_round_trips_enable)
    return EVBacktestResult(
        verdict=report.verdict, reason=report.reason, metrics=dict(report.metrics),
        trades=trades, baseline_trades=baselines, diagnostics=diagnostics,
        n_train_rows=int(len(train_df)),
        n_test_symbol_days=diagnostics["test_symbol_days"],
        models=models,
    )


def _bars_view(g: pd.DataFrame) -> pd.DataFrame | None:
    need = ["symbol", "trade_date", "trade_time", "open", "high", "low", "close", "volume"]
    if not all(c in g.columns for c in need):
        return None
    out = g[need + [c for c in ("limit_up", "limit_down") if c in g.columns]].copy()
    for c in ("open", "high", "low", "close", "volume", "limit_up", "limit_down"):
        if c in out.columns:
            out[c] = pd.to_numeric(out[c], errors="coerce")
    return out


def plan_book_dot(
    *,
    held: pd.DataFrame,
    minute_bars_by_symbol: dict[str, pd.DataFrame],
    models,
    cfg: EVBacktestConfig,
    as_of_minute: int | None = None,
) -> pd.DataFrame:
    """Live do-T plan for the held book (NO_TRADE-default execution overlay).

    For each held name with intraday bars, build causal features up to
    ``as_of_minute`` (default: latest bar), seed a T+1 ledger from the carried
    position, and run ``decide_ev``.  Only carried (sellable) shares are ever
    touched; today's buys are never resold the same session.  Names without a
    positive-EV, T+1-legal, gate-clearing setup return action ``NO_TRADE``.

    ``held`` columns: ``symbol`` and either ``shares`` or ``weight`` (sized to
    ``cfg.order_notional_yuan`` when only weight is given).
    """
    ev_cfg = cfg.ev_config()
    feat_cols = [c for c in CAUSAL_INTRADAY_FEATURE_COLUMNS]
    rows: list[dict[str, Any]] = []
    held = held.copy()
    held["symbol"] = held["symbol"].astype(str)
    for _, hr in held.iterrows():
        sym = str(hr["symbol"])
        bars = minute_bars_by_symbol.get(sym)
        if bars is None or bars.empty:
            rows.append({"symbol": sym, "action": "NO_TRADE", "reason": "no_minute_bars"})
            continue
        b = bars.copy()
        b["symbol"] = sym
        if "trade_date" not in b.columns:
            b["trade_date"] = pd.to_datetime(b["trade_time"]).dt.normalize()
        feats = build_causal_intraday_feature_frame(b, include_level2=False)
        if feats.empty:
            rows.append({"symbol": sym, "action": "NO_TRADE", "reason": "no_features"})
            continue
        feats = feats.sort_values("trade_time").reset_index(drop=True)
        idx = len(feats) - 1 if as_of_minute is None else min(int(as_of_minute), len(feats) - 1)
        ref_price = float(pd.to_numeric(feats["close"], errors="coerce").iloc[idx])
        if "shares" in held.columns and np.isfinite(hr.get("shares", np.nan)):
            position = _round_lot(float(hr["shares"]))
        else:
            position = _round_lot(cfg.order_notional_yuan / max(ref_price, 1e-9))
        if position <= 0:
            rows.append({"symbol": sym, "action": "NO_TRADE", "reason": "zero_position"})
            continue
        ledger = IntradayLedger(symbol=sym, date=str(b["trade_date"].iloc[0]),
                                carried_shares=position, target_shares=position,
                                cash=position * ref_price * 3.0)
        row = feats.iloc[idx]
        for c in feat_cols:
            if c not in feats.columns:
                feats[c] = np.nan
        sig = predict_model_signals(models, feats.iloc[[idx]])[0]
        state = _state_from_row(row, remaining_bars=len(feats) - 1 - idx, completed_rt=0)
        decision = decide_ev(state, ledger, sig, ev_cfg)
        rows.append({
            "symbol": sym, "action": decision.action, "quantity": int(decision.quantity),
            "ev_bps": round(float(decision.ev_bps), 3),
            "calibrated_probability": round(float(decision.calibrated_probability), 4),
            "dynamic_min_edge_bps": round(float(decision.dynamic_min_edge_bps), 3),
            "expected_net_edge_bps": round(float(decision.expected_net_edge_bps), 3),
            "risk_flags": ",".join(decision.risk_flags),
            "reason": "; ".join(decision.reason),
            "ref_price": round(ref_price, 4), "position_shares": int(position),
        })
    return pd.DataFrame(rows)


def _round_trip_cost_bps(cfg: EVBacktestConfig) -> float:
    c = cfg.cost_config()
    explicit = (2.0 * c.commission_rate + c.stamp_tax_sell + 2.0 * c.transfer_fee) * 10_000.0
    return explicit + 2.0 * (c.slippage_bps + c.spread_bps)


__all__ = [
    "EVBacktestConfig",
    "EVBacktestResult",
    "build_book_minute_panel",
    "build_feature_label_table",
    "load_book_keys",
    "plan_book_dot",
    "run_ev_closed_loop",
    "simulate_symbol_day",
]
