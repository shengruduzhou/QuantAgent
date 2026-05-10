from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class IncomeStatement:
    symbol: str
    report_date: pd.Timestamp
    revenue: float
    cogs: float
    operating_profit: float
    net_income: float


@dataclass(frozen=True)
class BalanceSheet:
    symbol: str
    report_date: pd.Timestamp
    total_assets: float
    total_liabilities: float
    total_equity: float
    cash: float = 0.0
    debt: float = 0.0


@dataclass(frozen=True)
class CashFlowStatement:
    symbol: str
    report_date: pd.Timestamp
    operating_cash_flow: float
    capex: float
    free_cash_flow: float


@dataclass(frozen=True)
class KeyMetrics:
    symbol: str
    report_date: pd.Timestamp
    gross_margin: float
    net_margin: float
    roe: float
    roic: float
    debt_to_assets: float


def standardize_statement_frame(frame: pd.DataFrame) -> pd.DataFrame:
    data = frame.copy()
    if "report_date" not in data.columns:
        raise ValueError("Missing required column: report_date")
    data["report_date"] = pd.to_datetime(data["report_date"])
    return data.sort_values(["symbol", "report_date"]).reset_index(drop=True)


def trailing_twelve_months(frame: pd.DataFrame, value_columns: list[str], periods: int = 4) -> pd.DataFrame:
    data = standardize_statement_frame(frame)
    for column in value_columns:
        data[f"{column}_ttm"] = data.groupby("symbol", sort=False)[column].transform(
            lambda s: s.rolling(periods, min_periods=periods).sum()
        )
    return data


def quarterly_delta(frame: pd.DataFrame, value_columns: list[str]) -> pd.DataFrame:
    data = standardize_statement_frame(frame)
    for column in value_columns:
        data[f"{column}_qoq_delta"] = data.groupby("symbol", sort=False)[column].diff()
    return data


def growth_rates(frame: pd.DataFrame, value_columns: list[str], yoy_periods: int = 4) -> pd.DataFrame:
    data = standardize_statement_frame(frame)
    for column in value_columns:
        data[f"{column}_qoq_growth"] = data.groupby("symbol", sort=False)[column].pct_change()
        data[f"{column}_yoy_growth"] = data.groupby("symbol", sort=False)[column].pct_change(yoy_periods)
    return data.replace([np.inf, -np.inf], np.nan)


def build_key_metrics(income: pd.DataFrame, balance: pd.DataFrame, cash_flow: pd.DataFrame) -> pd.DataFrame:
    income_data = standardize_statement_frame(income)
    balance_data = standardize_statement_frame(balance)
    cash_data = standardize_statement_frame(cash_flow)
    data = income_data.merge(balance_data, on=["symbol", "report_date"], how="left").merge(cash_data, on=["symbol", "report_date"], how="left")
    data["gross_margin"] = (data["revenue"] - data["cogs"]) / data["revenue"].replace(0.0, np.nan)
    data["net_margin"] = data["net_income"] / data["revenue"].replace(0.0, np.nan)
    data["roe"] = data["net_income"] / data["total_equity"].replace(0.0, np.nan)
    invested_capital = (data["total_assets"] - data.get("cash", 0.0)).replace(0.0, np.nan)
    data["roic"] = data["operating_profit"] * 0.75 / invested_capital
    data["debt_to_assets"] = data["total_liabilities"] / data["total_assets"].replace(0.0, np.nan)
    return data.replace([np.inf, -np.inf], np.nan)

