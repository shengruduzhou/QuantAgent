"""SilverPanelProvider — read the v7 silver market_panel.parquet as a provider.

The v7 silver layer already aggregates Qlib + AkShare + refresh
streams into a single PIT-correct ``market_panel.parquet`` with
``available_at`` on every row. Treating it as a router source lets
the v8 training pipeline pull data without re-running the upstream
v7 build commands — useful for offline / reproducibility runs.

This is NOT synthetic data and NOT a mock fallback — it consumes the
production silver lake. It still obeys the
``allow_mock_fallback=False`` contract because the lake itself was
materialised from real sources earlier; the router treats this
provider as a regular real source.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import pandas as pd

from quantagent.data.providers.base import (
    ProviderRequest,
    ProviderResult,
    ProviderUnavailable,
)


@dataclass
class SilverPanelProvider:
    """Read the v7 silver market panel as a router-compatible source."""

    panel_path: str | Path
    _cache: pd.DataFrame | None = None

    def daily_ohlcv(self, request: ProviderRequest) -> ProviderResult:
        if self._cache is None:
            p = Path(self.panel_path)
            if not p.exists():
                raise ProviderUnavailable(f"silver panel missing: {p}")
            try:
                self._cache = pd.read_parquet(p)
            except Exception as exc:  # noqa: BLE001
                raise ProviderUnavailable(f"failed to read {p}: {exc}") from exc
            self._cache["trade_date"] = pd.to_datetime(
                self._cache["trade_date"], errors="coerce"
            )
            if "available_at" in self._cache.columns:
                self._cache["available_at"] = pd.to_datetime(
                    self._cache["available_at"], errors="coerce"
                )
        df = self._cache
        mask = (
            (df["trade_date"] >= pd.Timestamp(request.start_date))
            & (df["trade_date"] <= pd.Timestamp(request.end_date))
        )
        if request.symbols:
            mask = mask & df["symbol"].isin(request.symbols)
        slice_df = df[mask].reset_index(drop=True)
        if slice_df.empty:
            return ProviderResult(
                pd.DataFrame(),
                source="silver_panel_provider",
                quality_score=0.0,
                warnings=("silver_panel_empty_slice",),
            )
        # ensure flags are present (the silver panel may not have them
        # populated; downstream backtest expects bool columns).
        for col in ("is_suspended", "is_st", "is_limit_up", "is_limit_down"):
            if col not in slice_df.columns:
                slice_df[col] = False
        return ProviderResult(
            slice_df,
            source="silver_panel_provider",
            point_in_time=True,
            quality_score=0.92,   # already PIT-corrected by upstream build
            metadata={"rows": int(len(slice_df))},
        )


__all__ = ["SilverPanelProvider"]
