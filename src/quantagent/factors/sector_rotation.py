from __future__ import annotations

import numpy as np
import pandas as pd

from quantagent.quant_math.ashare import limit_up_mask


def sector_return_strength(frame: pd.DataFrame, sector_column: str = "sector", window: int = 20) -> pd.DataFrame:
    data = _prepare(frame, sector_column)
    data["return_strength"] = data.groupby(sector_column, sort=False)["stock_return"].transform(
        lambda s: s.rolling(window, min_periods=window).mean()
    )
    return _sector_mean(data, sector_column, "return_strength", "sector_return_strength")


def sector_relative_strength(frame: pd.DataFrame, sector_column: str = "sector", window: int = 20) -> pd.DataFrame:
    data = _prepare(frame, sector_column)
    market = data.groupby("trade_date", sort=False)["stock_return"].transform("mean")
    data["relative_return"] = data["stock_return"] - market
    data["relative_strength"] = data.groupby(sector_column, sort=False)["relative_return"].transform(
        lambda s: s.rolling(window, min_periods=window).mean()
    )
    return _sector_mean(data, sector_column, "relative_strength", "sector_relative_strength")


def sector_volume_expansion(frame: pd.DataFrame, sector_column: str = "sector", window: int = 20) -> pd.DataFrame:
    data = _prepare(frame, sector_column)
    sector_amount = data.groupby(["trade_date", sector_column], sort=False)["amount"].sum().rename("sector_amount").reset_index()
    sector_amount["sector_volume_expansion"] = sector_amount.groupby(sector_column, sort=False)["sector_amount"].transform(
        lambda s: s / s.rolling(window, min_periods=window).mean()
    )
    return sector_amount[["trade_date", sector_column, "sector_volume_expansion"]]


def sector_breadth(frame: pd.DataFrame, sector_column: str = "sector") -> pd.DataFrame:
    data = _prepare(frame, sector_column)
    data["positive"] = data["stock_return"] > 0.0
    return _sector_mean(data, sector_column, "positive", "sector_breadth")


def sector_limit_up_count(frame: pd.DataFrame, sector_column: str = "sector") -> pd.DataFrame:
    data = _prepare(frame, sector_column)
    data["limit_up"] = _limit_up_series(data)
    return data.groupby(["trade_date", sector_column], sort=False)["limit_up"].sum().rename("sector_limit_up_count").reset_index()


def sector_limit_up_ratio(frame: pd.DataFrame, sector_column: str = "sector") -> pd.DataFrame:
    data = _prepare(frame, sector_column)
    data["limit_up"] = _limit_up_series(data)
    return _sector_mean(data, sector_column, "limit_up", "sector_limit_up_ratio")


def sector_turnover_share(frame: pd.DataFrame, sector_column: str = "sector") -> pd.DataFrame:
    data = _prepare(frame, sector_column)
    sector_amount = data.groupby(["trade_date", sector_column], sort=False)["amount"].sum().rename("sector_amount").reset_index()
    total = sector_amount.groupby("trade_date", sort=False)["sector_amount"].transform("sum")
    sector_amount["sector_turnover_share"] = sector_amount["sector_amount"] / total.replace(0.0, np.nan)
    return sector_amount[["trade_date", sector_column, "sector_turnover_share"]]


def sector_money_flow_share(frame: pd.DataFrame, sector_column: str = "sector") -> pd.DataFrame:
    data = _prepare(frame, sector_column)
    intraday_return = (data["close"] - data["open"]) / data["open"].replace(0.0, np.nan)
    data["money_flow"] = intraday_return * data["amount"]
    sector_flow = data.groupby(["trade_date", sector_column], sort=False)["money_flow"].sum().rename("sector_flow").reset_index()
    total_abs = sector_flow.groupby("trade_date", sort=False)["sector_flow"].transform(lambda s: s.abs().sum())
    sector_flow["sector_money_flow_share"] = sector_flow["sector_flow"] / total_abs.replace(0.0, np.nan)
    return sector_flow[["trade_date", sector_column, "sector_money_flow_share"]]


def sector_rotation_score(frame: pd.DataFrame, sector_column: str = "sector", window: int = 20) -> pd.DataFrame:
    factors = _merge_sector_factors(
        [
            sector_return_strength(frame, sector_column, window),
            sector_relative_strength(frame, sector_column, window),
            sector_volume_expansion(frame, sector_column, window),
            sector_breadth(frame, sector_column),
            sector_limit_up_ratio(frame, sector_column),
            sector_turnover_share(frame, sector_column),
            sector_money_flow_share(frame, sector_column),
        ],
        sector_column,
    )
    score_columns = [column for column in factors.columns if column not in {"trade_date", sector_column}]
    for column in score_columns:
        factors[f"{column}_z"] = factors.groupby("trade_date", sort=False)[column].transform(_zscore)
    z_columns = [f"{column}_z" for column in score_columns]
    factors["sector_rotation_score"] = factors[z_columns].mean(axis=1)
    return factors.drop(columns=z_columns)


def concept_rotation_score(frame: pd.DataFrame, concept_column: str = "concept", window: int = 20) -> pd.DataFrame:
    data = frame.copy()
    if concept_column != "sector" and "sector" in data.columns:
        data = data.drop(columns=["sector"])
    result = sector_rotation_score(data.rename(columns={concept_column: "sector"}), "sector", window)
    return result.rename(columns={"sector": concept_column, "sector_rotation_score": "concept_rotation_score"})


def compute_sector_rotation_factors(
    frame: pd.DataFrame,
    sector_column: str = "sector",
    concept_column: str | None = None,
    window: int = 20,
) -> pd.DataFrame:
    result = sector_rotation_score(frame, sector_column=sector_column, window=window)
    if concept_column and concept_column in frame.columns:
        concept = concept_rotation_score(frame, concept_column=concept_column, window=window)
        result = result.merge(
            concept[["trade_date", concept_column, "concept_rotation_score"]],
            left_on=["trade_date", sector_column],
            right_on=["trade_date", concept_column],
            how="left",
        ).drop(columns=[concept_column])
    return result


def _prepare(frame: pd.DataFrame, sector_column: str) -> pd.DataFrame:
    required = {"trade_date", "symbol", "open", "high", "low", "close", "volume", "amount", sector_column}
    missing = required.difference(frame.columns)
    if missing:
        raise ValueError(f"Missing required columns: {sorted(missing)}")
    data = frame.copy()
    data["trade_date"] = pd.to_datetime(data["trade_date"])
    data = data.sort_values(["symbol", "trade_date"]).reset_index(drop=True)
    data["stock_return"] = data.groupby("symbol", sort=False)["close"].pct_change()
    return data


def _sector_mean(data: pd.DataFrame, sector_column: str, source: str, target: str) -> pd.DataFrame:
    return data.groupby(["trade_date", sector_column], sort=False)[source].mean().rename(target).reset_index()


def _limit_up_series(data: pd.DataFrame) -> pd.Series:
    if "is_limit_up" in data.columns:
        return data["is_limit_up"].fillna(False).astype(bool)
    return limit_up_mask(data).fillna(False)


def _merge_sector_factors(frames: list[pd.DataFrame], sector_column: str) -> pd.DataFrame:
    merged = frames[0]
    for frame in frames[1:]:
        merged = merged.merge(frame, on=["trade_date", sector_column], how="outer")
    return merged.sort_values(["trade_date", sector_column]).reset_index(drop=True)


def _zscore(series: pd.Series) -> pd.Series:
    std = series.std(ddof=0)
    if not np.isfinite(std) or std <= 1e-12:
        return pd.Series(np.nan, index=series.index, dtype=float)
    return (series - series.mean()) / std
