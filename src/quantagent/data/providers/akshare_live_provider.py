from __future__ import annotations

from dataclasses import dataclass

import pandas as pd

from quantagent.data.providers.base import ProviderRequest, ProviderResult, ProviderUnavailable


@dataclass
class AkShareLiveProvider:
    """Optional AkShare downloader; network is disabled unless explicitly enabled."""

    allow_network: bool = False
    adjust: str = "qfq"

    def daily_ohlcv(self, request: ProviderRequest) -> ProviderResult:
        if not self.allow_network:
            raise ProviderUnavailable("AkShare live download is disabled; set data.allow_network=true explicitly")
        try:
            import akshare as ak  # type: ignore
        except Exception as exc:  # pragma: no cover - optional dependency
            raise ProviderUnavailable("akshare is not available") from exc
        if not request.symbols:
            raise ProviderUnavailable("AkShare live daily_ohlcv requires explicit symbols")
        frames = []
        for symbol in request.symbols:
            raw = ak.stock_zh_a_hist(
                symbol=_plain_a_code(symbol),
                period="daily",
                start_date=request.start_date.replace("-", ""),
                end_date=request.end_date.replace("-", ""),
                adjust=self.adjust,
            )
            if raw.empty:
                continue
            frames.append(_normalize_akshare_daily(raw, symbol))
        frame = pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()
        return ProviderResult(
            frame,
            source="akshare_live_provider",
            point_in_time=True,
            quality_score=0.78 if not frame.empty else 0.0,
            warnings=() if not frame.empty else ("akshare_empty_daily_ohlcv",),
        )


def _plain_a_code(symbol: str) -> str:
    return str(symbol).split(".")[0]


def _normalize_akshare_daily(frame: pd.DataFrame, symbol: str) -> pd.DataFrame:
    columns = {
        "日期": "trade_date",
        "开盘": "open",
        "最高": "high",
        "最低": "low",
        "收盘": "close",
        "成交量": "volume",
        "成交额": "amount",
    }
    data = frame.rename(columns=columns)
    keep = [column for column in columns.values() if column in data.columns]
    data = data[keep].copy()
    data["symbol"] = symbol
    data["available_at"] = data["trade_date"]
    data["source"] = "akshare"
    data["source_type"] = "market_data"
    data["source_reliability"] = 0.72
    data["point_in_time_valid"] = True
    return data
