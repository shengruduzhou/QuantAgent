"""Training diagnostics that run after the main experiment finishes.

Two functions live here:

* :func:`compute_factor_ic_decay` — monthly rank-IC matrix for every
  numeric feature in the training dataset against the primary forward-
  return label. The decay over months exposes factors that crowd /
  decay / regime-shift, which is invisible from the aggregate ICIR
  alone.

* :func:`render_ic_decay_heatmap` — write a PNG heatmap (one row per
  factor, one column per calendar month) plus a sidecar JSON. The PNG
  is matplotlib-only so it works on headless CI; if matplotlib is not
  installed we silently fall back to JSON-only.

Both functions are pure: they take a DataFrame, return artefacts, and
do not touch global state.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd


def compute_factor_ic_decay(
    dataset: pd.DataFrame,
    feature_columns: list[str],
    label_column: str,
    *,
    min_monthly_observations: int = 20,
) -> pd.DataFrame:
    """Return a (factor × month) rank-IC matrix.

    Each cell is the cross-sectional Spearman rank correlation between
    the factor value and ``label_column`` across all symbols active that
    month. NaN cells (insufficient observations) are preserved so the
    caller can render gaps faithfully.
    """
    if dataset is None or dataset.empty or not feature_columns:
        return pd.DataFrame()
    if label_column not in dataset.columns:
        raise KeyError(f"label column missing from dataset: {label_column}")
    frame = dataset[["trade_date", "symbol", label_column, *feature_columns]].copy()
    frame["trade_date"] = pd.to_datetime(frame["trade_date"], errors="coerce")
    frame = frame.dropna(subset=["trade_date", "symbol", label_column])
    if frame.empty:
        return pd.DataFrame()
    frame["month"] = frame["trade_date"].dt.to_period("M").dt.to_timestamp()
    rows: list[dict[str, object]] = []
    for month, group in frame.groupby("month", sort=True):
        if len(group) < min_monthly_observations:
            continue
        label = pd.to_numeric(group[label_column], errors="coerce")
        if label.notna().sum() < min_monthly_observations:
            continue
        for factor in feature_columns:
            values = pd.to_numeric(group[factor], errors="coerce")
            mask = values.notna() & label.notna()
            if int(mask.sum()) < min_monthly_observations:
                rows.append({"month": month, "factor": factor, "rank_ic": np.nan})
                continue
            try:
                ic = float(values[mask].corr(label[mask], method="spearman"))
            except Exception:
                ic = np.nan
            rows.append({"month": month, "factor": factor, "rank_ic": ic})
    if not rows:
        return pd.DataFrame()
    long_df = pd.DataFrame(rows)
    wide = long_df.pivot(index="factor", columns="month", values="rank_ic")
    return wide.sort_index(axis=1)


def render_ic_decay_heatmap(decay: pd.DataFrame, output_path: Path | str) -> dict[str, str]:
    """Write PNG + JSON diagnostics for an IC-decay matrix."""
    out = Path(output_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    json_path = out.with_suffix(".json")
    json_payload = {
        "factors": list(decay.index),
        "months": [str(c.date()) if hasattr(c, "date") else str(c) for c in decay.columns],
        "rank_ic": [
            [None if pd.isna(v) else float(v) for v in row]
            for row in decay.to_numpy()
        ],
    }
    json_path.write_text(json.dumps(json_payload, ensure_ascii=False, indent=2), encoding="utf-8")
    png_path = out.with_suffix(".png")
    try:
        import matplotlib
        matplotlib.use("Agg")  # headless
        import matplotlib.pyplot as plt

        fig, ax = plt.subplots(figsize=(max(10, decay.shape[1] / 4), max(6, decay.shape[0] / 5)))
        data = decay.to_numpy()
        vmax = float(np.nanmax(np.abs(data))) if data.size and np.isfinite(data).any() else 0.1
        im = ax.imshow(data, aspect="auto", cmap="RdBu_r", vmin=-vmax, vmax=vmax)
        ax.set_yticks(range(decay.shape[0]))
        ax.set_yticklabels(decay.index, fontsize=6)
        month_labels = [c.strftime("%Y-%m") if hasattr(c, "strftime") else str(c) for c in decay.columns]
        ax.set_xticks(range(decay.shape[1]))
        ax.set_xticklabels(month_labels, rotation=90, fontsize=6)
        ax.set_title(f"Factor monthly rank-IC ({decay.shape[0]} factors × {decay.shape[1]} months)")
        fig.colorbar(im, ax=ax, label="rank IC")
        fig.tight_layout()
        fig.savefig(png_path, dpi=120)
        plt.close(fig)
        return {"json": str(json_path), "png": str(png_path)}
    except Exception:
        # Headless / matplotlib missing — JSON is still complete.
        return {"json": str(json_path)}


__all__ = ["compute_factor_ic_decay", "render_ic_decay_heatmap"]
