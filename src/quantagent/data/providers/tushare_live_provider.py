from __future__ import annotations

from dataclasses import dataclass
import os

import pandas as pd

from quantagent.data.providers.base import ProviderRequest, ProviderResult, ProviderUnavailable


@dataclass
class TuShareLiveProvider:
    """Optional TuShare downloader with explicit token and network boundary."""

    allow_network: bool = False
    token_env: str = "TUSHARE_TOKEN"

    def daily_ohlcv(self, request: ProviderRequest) -> ProviderResult:
        if not self.allow_network:
            raise ProviderUnavailable("TuShare live download is disabled; set data.allow_network=true explicitly")
        token = os.getenv(self.token_env)
        if not token:
            raise ProviderUnavailable(f"{self.token_env} is required for TuShare live download")
        try:
            import tushare as ts  # type: ignore
        except Exception as exc:  # pragma: no cover - optional dependency
            raise ProviderUnavailable("tushare is not available") from exc
        if not request.symbols:
            raise ProviderUnavailable("TuShare live daily_ohlcv requires explicit symbols")
        pro = ts.pro_api(token)
        frames = []
        for symbol in request.symbols:
            raw = pro.daily(
                ts_code=_tushare_code(symbol),
                start_date=request.start_date.replace("-", ""),
                end_date=request.end_date.replace("-", ""),
            )
            if not raw.empty:
                frames.append(_normalize_tushare_daily(raw))
        frame = pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()
        return ProviderResult(
            frame,
            source="tushare_live_provider",
            point_in_time=True,
            quality_score=0.82 if not frame.empty else 0.0,
            warnings=() if not frame.empty else ("tushare_empty_daily_ohlcv",),
        )


def _tushare_code(symbol: str) -> str:
    text = str(symbol).upper()
    if "." in text:
        code, suffix = text.split(".", 1)
        return f"{code}.{suffix}"
    if text.startswith("6"):
        return f"{text}.SH"
    return f"{text}.SZ"


def _normalize_tushare_daily(frame: pd.DataFrame) -> pd.DataFrame:
    data = frame.rename(
        columns={
            "ts_code": "symbol",
            "trade_date": "trade_date",
            "vol": "volume",
        }
    ).copy()
    if "trade_date" in data.columns:
        data["trade_date"] = pd.to_datetime(data["trade_date"], format="%Y%m%d").dt.strftime("%Y-%m-%d")
    data["available_at"] = data["trade_date"]
    data["source"] = "tushare"
    data["source_type"] = "market_data"
    data["source_reliability"] = 0.82
    data["point_in_time_valid"] = True
    return data
