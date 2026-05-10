from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import numpy as np
import pandas as pd

FlowSource = Literal[
    "northbound_holding",
    "northbound_net_buy",
    "dragon_tiger",
    "main_money_flow",
    "margin_financing",
    "etf_fund_flow",
    "block_trade",
    "institutional_research",
    "public_fund_holding",
]


@dataclass(frozen=True)
class FlowRecord:
    trade_date: pd.Timestamp
    symbol: str
    source: FlowSource
    net_amount: float
    buy_amount: float = 0.0
    sell_amount: float = 0.0
    holding_value: float | None = None
    sector: str | None = None
    evidence_quality: float = 0.5


@dataclass(frozen=True)
class FlowFeatureFrame:
    frame: pd.DataFrame
    source_columns: tuple[str, ...]


@dataclass(frozen=True)
class FlowSignal:
    symbol: str
    source: FlowSource
    horizon_days: int
    score: float
    confidence: float
    evidence_quality: float


SOURCE_QUALITY: dict[str, float] = {
    "northbound_holding": 0.75,
    "northbound_net_buy": 0.75,
    "dragon_tiger": 0.55,
    "main_money_flow": 0.50,
    "margin_financing": 0.60,
    "etf_fund_flow": 0.65,
    "block_trade": 0.55,
    "institutional_research": 0.45,
    "public_fund_holding": 0.70,
}

SOURCE_HORIZON: dict[str, int] = {
    "dragon_tiger": 3,
    "main_money_flow": 5,
    "northbound_holding": 20,
    "northbound_net_buy": 10,
    "margin_financing": 10,
    "etf_fund_flow": 30,
    "block_trade": 10,
    "institutional_research": 20,
    "public_fund_holding": 60,
}


def normalize_flow_records(records: list[FlowRecord]) -> pd.DataFrame:
    rows = [record.__dict__ for record in records]
    frame = pd.DataFrame(rows)
    if frame.empty:
        return _empty_flow_frame()
    frame["trade_date"] = pd.to_datetime(frame["trade_date"])
    return frame


def normalize_northbound_holdings(frame: pd.DataFrame) -> pd.DataFrame:
    data = _normalize_source(frame, "northbound_holding", value_column="holding_value")
    data["net_amount"] = data.groupby("symbol", sort=False)["holding_value"].diff()
    return data


def normalize_northbound_net_buy(frame: pd.DataFrame) -> pd.DataFrame:
    return _normalize_source(frame, "northbound_net_buy", value_column="net_amount")


def normalize_dragon_tiger(frame: pd.DataFrame) -> pd.DataFrame:
    data = frame.copy()
    data["net_amount"] = data.get("inst_buy", 0.0) - data.get("inst_sell", 0.0) - data.get("retail_buy", 0.0) + data.get("retail_sell", 0.0)
    return _normalize_source(data, "dragon_tiger", value_column="net_amount")


def normalize_main_money_flow(frame: pd.DataFrame) -> pd.DataFrame:
    return _normalize_source(frame, "main_money_flow", value_column="net_amount")


def normalize_margin_financing(frame: pd.DataFrame) -> pd.DataFrame:
    data = frame.copy()
    if "margin_balance" in data.columns and "net_amount" not in data.columns:
        data["net_amount"] = data.groupby("symbol", sort=False)["margin_balance"].diff()
    return _normalize_source(data, "margin_financing", value_column="net_amount")


def normalize_etf_fund_flow(frame: pd.DataFrame) -> pd.DataFrame:
    return _normalize_source(frame, "etf_fund_flow", value_column="net_amount")


def normalize_block_trades(frame: pd.DataFrame) -> pd.DataFrame:
    return _normalize_source(frame, "block_trade", value_column="net_amount")


def normalize_institutional_research_visits(frame: pd.DataFrame) -> pd.DataFrame:
    data = frame.copy()
    if "visit_count" in data.columns and "net_amount" not in data.columns:
        data["net_amount"] = data["visit_count"].astype(float)
    return _normalize_source(data, "institutional_research", value_column="net_amount")


def normalize_public_fund_holdings(frame: pd.DataFrame) -> pd.DataFrame:
    data = frame.copy()
    if "holding_value" in data.columns and "net_amount" not in data.columns:
        data["net_amount"] = data.groupby("symbol", sort=False)["holding_value"].diff()
    return _normalize_source(data, "public_fund_holding", value_column="net_amount")


def build_flow_feature_frame(
    source_frames: dict[str, pd.DataFrame],
    window: int = 20,
) -> FlowFeatureFrame:
    normalizers = {
        "northbound_holding": normalize_northbound_holdings,
        "northbound_net_buy": normalize_northbound_net_buy,
        "dragon_tiger": normalize_dragon_tiger,
        "main_money_flow": normalize_main_money_flow,
        "margin_financing": normalize_margin_financing,
        "etf_fund_flow": normalize_etf_fund_flow,
        "block_trade": normalize_block_trades,
        "institutional_research": normalize_institutional_research_visits,
        "public_fund_holding": normalize_public_fund_holdings,
    }
    frames = [normalizers[source](frame) for source, frame in source_frames.items() if source in normalizers and not frame.empty]
    if not frames:
        return FlowFeatureFrame(_empty_feature_frame(), ())
    data = pd.concat(frames, ignore_index=True).sort_values(["source", "symbol", "trade_date"])
    features = compute_flow_features(data, window=window)
    return FlowFeatureFrame(features, tuple(sorted(data["source"].dropna().unique())))


def compute_flow_features(flow_frame: pd.DataFrame, window: int = 20) -> pd.DataFrame:
    data = flow_frame.copy()
    data["trade_date"] = pd.to_datetime(data["trade_date"])
    data = data.sort_values(["source", "symbol", "trade_date"]).reset_index(drop=True)
    grouped = data.groupby(["source", "symbol"], sort=False)
    mean = grouped["net_amount"].transform(lambda s: s.rolling(window, min_periods=max(3, window // 4)).mean())
    std = grouped["net_amount"].transform(lambda s: s.rolling(window, min_periods=max(3, window // 4)).std())
    data["flow_zscore"] = (data["net_amount"] - mean) / std.replace(0.0, np.nan)
    data["flow_acceleration"] = grouped["net_amount"].diff()
    data["flow_persistence"] = grouped["net_amount"].transform(lambda s: s.rolling(window, min_periods=max(3, window // 4)).apply(lambda x: float(np.sign(x).mean()), raw=True))
    data["flow_reversal"] = -grouped["net_amount"].diff(window)
    total_abs = data.groupby(["trade_date", "source"], sort=False)["net_amount"].transform(lambda s: s.abs().sum())
    data["flow_concentration"] = data["net_amount"].abs() / total_abs.replace(0.0, np.nan)
    data["institution_retail_imbalance"] = _imbalance(data)
    data["large_order_pressure"] = data["flow_zscore"] * data["flow_concentration"]
    data["margin_balance_change"] = np.where(data["source"] == "margin_financing", grouped["net_amount"].pct_change(), np.nan)
    data["etf_sector_flow"] = _sector_flow(data, "etf_fund_flow")
    return data.replace([np.inf, -np.inf], np.nan)


def flow_signals_from_features(features: pd.DataFrame, z_threshold: float = 1.0) -> list[FlowSignal]:
    latest = features.sort_values("trade_date").groupby(["source", "symbol"], sort=False).tail(1)
    signals: list[FlowSignal] = []
    for _, row in latest.iterrows():
        z = _finite(row.get("flow_zscore", 0.0))
        pressure = _finite(row.get("large_order_pressure", 0.0))
        if abs(z) < z_threshold and abs(pressure) < z_threshold * 0.05:
            continue
        source = str(row["source"])
        score = float(np.tanh(0.6 * z + 4.0 * pressure))
        signals.append(
            FlowSignal(
                symbol=str(row["symbol"]),
                source=source,
                horizon_days=SOURCE_HORIZON.get(source, 10),
                score=score,
                confidence=float(np.clip(abs(score), 0.0, 1.0)),
                evidence_quality=SOURCE_QUALITY.get(source, 0.5),
            )
        )
    return signals


def _normalize_source(frame: pd.DataFrame, source: str, value_column: str) -> pd.DataFrame:
    data = frame.copy()
    required = {"trade_date", "symbol"}
    missing = required.difference(data.columns)
    if missing:
        raise ValueError(f"Missing required columns: {sorted(missing)}")
    data["trade_date"] = pd.to_datetime(data["trade_date"])
    if value_column not in data.columns:
        data[value_column] = 0.0
    if "net_amount" not in data.columns:
        data["net_amount"] = data[value_column]
    if "buy_amount" not in data.columns:
        data["buy_amount"] = data["net_amount"].clip(lower=0.0)
    if "sell_amount" not in data.columns:
        data["sell_amount"] = (-data["net_amount"]).clip(lower=0.0)
    if "holding_value" not in data.columns:
        data["holding_value"] = np.nan
    if "sector" not in data.columns:
        data["sector"] = None
    data["source"] = source
    data["evidence_quality"] = SOURCE_QUALITY.get(source, 0.5)
    columns = ["trade_date", "symbol", "source", "net_amount", "buy_amount", "sell_amount", "holding_value", "sector", "evidence_quality"]
    return data[columns]


def _imbalance(data: pd.DataFrame) -> pd.Series:
    denom = (data["buy_amount"].abs() + data["sell_amount"].abs()).replace(0.0, np.nan)
    return (data["buy_amount"] - data["sell_amount"]) / denom


def _sector_flow(data: pd.DataFrame, source: str) -> pd.Series:
    mask = data["source"] == source
    out = pd.Series(np.nan, index=data.index, dtype=float)
    if "sector" not in data.columns:
        return out
    sector_total = data.loc[mask].groupby(["trade_date", "sector"], sort=False)["net_amount"].transform("sum")
    out.loc[mask] = sector_total.to_numpy(dtype=float)
    return out


def _finite(value: object) -> float:
    try:
        numeric = float(value)
    except (TypeError, ValueError):
        return 0.0
    return numeric if np.isfinite(numeric) else 0.0


def _empty_flow_frame() -> pd.DataFrame:
    return pd.DataFrame(columns=["trade_date", "symbol", "source", "net_amount", "buy_amount", "sell_amount", "holding_value", "sector", "evidence_quality"])


def _empty_feature_frame() -> pd.DataFrame:
    columns = list(_empty_flow_frame().columns) + [
        "flow_zscore",
        "flow_acceleration",
        "flow_persistence",
        "flow_reversal",
        "flow_concentration",
        "institution_retail_imbalance",
        "large_order_pressure",
        "margin_balance_change",
        "etf_sector_flow",
    ]
    return pd.DataFrame(columns=columns)
