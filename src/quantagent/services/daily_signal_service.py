from __future__ import annotations

import pandas as pd

from quantagent.models.v6_model_system import V6ModelSystem


def infer_v4_alpha(features: pd.DataFrame, date: str | None = None) -> pd.DataFrame:
    """Compatibility signal entrypoint backed by the unified V6 model system."""
    outputs = V6ModelSystem().infer_frame(features, trade_date=date)
    if outputs.empty:
        return pd.DataFrame(columns=["trade_date", "symbol", "alpha", "confidence"])
    return (
        outputs.rename(columns={"alpha_5d": "alpha"})[["trade_date", "symbol", "alpha", "confidence"]]
        .sort_values("symbol")
        .reset_index(drop=True)
    )
