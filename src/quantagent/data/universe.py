from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

import pandas as pd

from quantagent.quant_math.ashare import AshareRuleEngine


@dataclass(frozen=True)
class UniverseConfig:
    name: str = "custom"
    min_amount_20d: float = 0.0
    min_listed_days: int = 0
    exclude_st: bool = True
    exclude_suspended: bool = True


class UniverseBuilder:
    """Build deterministic A-share research universes from local snapshots."""

    def __init__(self, config: UniverseConfig | None = None, rule_engine: AshareRuleEngine | None = None) -> None:
        self.config = config or UniverseConfig()
        self.rule_engine = rule_engine or AshareRuleEngine()

    def from_symbols(self, symbols: Iterable[str], trade_dates: Iterable[pd.Timestamp | str]) -> pd.DataFrame:
        rows = [
            {"trade_date": pd.Timestamp(date), "symbol": str(symbol), "is_member": True}
            for date in trade_dates
            for symbol in symbols
        ]
        return pd.DataFrame(rows).sort_values(["trade_date", "symbol"]).reset_index(drop=True)

    def csi_placeholder(self, panel: pd.DataFrame, universe: str | None = None) -> pd.DataFrame:
        name = (universe or self.config.name).upper()
        frame = panel[["trade_date", "symbol"]].drop_duplicates().copy()
        frame["trade_date"] = pd.to_datetime(frame["trade_date"])
        frame["universe"] = name
        frame["is_member"] = True
        return frame.sort_values(["trade_date", "symbol"]).reset_index(drop=True)

    def build_mask(self, panel: pd.DataFrame, snapshots: pd.DataFrame | None = None) -> pd.Series:
        data = panel.copy()
        data["trade_date"] = pd.to_datetime(data["trade_date"])
        mask = pd.Series(True, index=data.index)
        if snapshots is not None and not snapshots.empty:
            snap = snapshots.copy()
            snap["trade_date"] = pd.to_datetime(snap["trade_date"])
            members = data[["trade_date", "symbol"]].merge(
                snap[["trade_date", "symbol", "is_member"]],
                on=["trade_date", "symbol"],
                how="left",
            )["is_member"].fillna(False)
            mask &= members.to_numpy(dtype=bool)
        if self.config.exclude_st and "is_st" in data.columns:
            mask &= ~data["is_st"].fillna(False)
        if self.config.exclude_suspended:
            mask &= data.apply(self.rule_engine.is_tradable, axis=1)
        if "amount_mean_20d" in data.columns:
            mask &= data["amount_mean_20d"].fillna(0.0) >= self.config.min_amount_20d
        elif "amount" in data.columns and self.config.min_amount_20d > 0:
            amount_mean = data.groupby("symbol")["amount"].transform(lambda s: s.rolling(20, min_periods=1).mean())
            mask &= amount_mean >= self.config.min_amount_20d
        if "listed_days" in data.columns and self.config.min_listed_days > 0:
            mask &= data["listed_days"].fillna(0) >= self.config.min_listed_days
        return mask.astype(bool)

    def filter(self, panel: pd.DataFrame, snapshots: pd.DataFrame | None = None) -> pd.DataFrame:
        mask = self.build_mask(panel, snapshots=snapshots)
        return panel.loc[mask].sort_values(["trade_date", "symbol"]).reset_index(drop=True)
