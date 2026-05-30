"""Target-weight liveness diagnostics.

The liveness gate answers one narrow question: did the prediction →
candidate → target_weights chain produce non-empty, explainable weights?
It is not a performance validator and does not optimize the portfolio.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pandas as pd


def _read_json(path: str | Path | None) -> dict[str, Any]:
    if path is None:
        return {}
    file_path = Path(path)
    if not file_path.exists():
        return {}
    return json.loads(file_path.read_text(encoding="utf-8"))


def _weights_long(target_weights: pd.DataFrame) -> pd.DataFrame:
    if target_weights is None or target_weights.empty:
        return pd.DataFrame(columns=["trade_date", "symbol", "weight"])
    frame = target_weights.copy()
    if {"trade_date", "symbol", "weight"}.issubset(frame.columns):
        out = frame[["trade_date", "symbol", "weight"]].copy()
    elif "trade_date" in frame.columns:
        out = frame.melt(id_vars=["trade_date"], var_name="symbol", value_name="weight")
    else:
        out = frame.reset_index().rename(columns={frame.index.name or "index": "trade_date"})
        out = out.melt(id_vars=["trade_date"], var_name="symbol", value_name="weight")
    out["trade_date"] = pd.to_datetime(out["trade_date"], errors="coerce")
    out["symbol"] = out["symbol"].astype(str)
    out["weight"] = pd.to_numeric(out["weight"], errors="coerce").fillna(0.0)
    return out.dropna(subset=["trade_date", "symbol"]).reset_index(drop=True)


def _daily_predictions(predictions: pd.DataFrame | None) -> pd.DataFrame:
    if predictions is None or predictions.empty or "trade_date" not in predictions.columns:
        return pd.DataFrame(columns=["trade_date", "prediction_rows", "prediction_symbols"])
    work = predictions.copy()
    work["trade_date"] = pd.to_datetime(work["trade_date"], errors="coerce")
    work = work.dropna(subset=["trade_date"])
    return (
        work.groupby("trade_date")
        .agg(prediction_rows=("symbol", "size"), prediction_symbols=("symbol", "nunique"))
        .reset_index()
    )


def _daily_candidates(diagnostics: dict[str, Any]) -> pd.DataFrame:
    rows = diagnostics.get("daily_selection", []) if isinstance(diagnostics, dict) else []
    if not isinstance(rows, list) or not rows:
        return pd.DataFrame(columns=["trade_date", "eligible_count", "selected_count"])
    frame = pd.DataFrame(rows)
    if "trade_date" not in frame.columns:
        return pd.DataFrame(columns=["trade_date", "eligible_count", "selected_count"])
    frame["trade_date"] = pd.to_datetime(frame["trade_date"], errors="coerce")
    for col in ("eligible_count", "selected_count"):
        if col not in frame.columns:
            frame[col] = 0
        frame[col] = pd.to_numeric(frame[col], errors="coerce").fillna(0).astype(int)
    return frame[["trade_date", "eligible_count", "selected_count"]].dropna(subset=["trade_date"])


def _warning_dates(diagnostics: dict[str, Any], warning_name: str) -> set[pd.Timestamp]:
    rows = diagnostics.get("warnings", []) if isinstance(diagnostics, dict) else []
    dates: set[pd.Timestamp] = set()
    if not isinstance(rows, list):
        return dates
    for row in rows:
        if not isinstance(row, dict) or row.get("warning") != warning_name:
            continue
        date = pd.to_datetime(row.get("trade_date"), errors="coerce")
        if pd.notna(date):
            dates.add(pd.Timestamp(date))
    return dates


def build_target_weights_liveness(
    target_weights: pd.DataFrame,
    *,
    predictions: pd.DataFrame | None = None,
    diagnostics: dict[str, Any] | None = None,
    epsilon: float = 1e-12,
) -> dict[str, Any]:
    diagnostics = diagnostics or {}
    long = _weights_long(target_weights)
    if long.empty:
        daily_gross = pd.DataFrame(columns=["trade_date", "gross_exposure", "nonzero_symbols"])
    else:
        daily_gross = (
            long.assign(abs_weight=long["weight"].abs(), is_nonzero=long["weight"].abs() > epsilon)
            .groupby("trade_date")
            .agg(gross_exposure=("abs_weight", "sum"), nonzero_symbols=("is_nonzero", "sum"))
            .reset_index()
            .sort_values("trade_date")
        )
    pred_daily = _daily_predictions(predictions)
    cand_daily = _daily_candidates(diagnostics)
    has_candidate_trace = not cand_daily.empty
    trace = daily_gross.merge(pred_daily, on="trade_date", how="outer").merge(cand_daily, on="trade_date", how="outer")
    for col in ("gross_exposure", "nonzero_symbols", "prediction_rows", "prediction_symbols", "eligible_count", "selected_count"):
        if col not in trace.columns:
            trace[col] = 0
        trace[col] = pd.to_numeric(trace[col], errors="coerce").fillna(0)
    trace["trade_date"] = pd.to_datetime(trace["trade_date"], errors="coerce")
    trace = trace.dropna(subset=["trade_date"]).sort_values("trade_date").reset_index(drop=True)

    amount_all_missing_dates = _warning_dates(diagnostics, "liquidity_amount_all_missing_cap_disabled")
    amount_partial_missing_dates = _warning_dates(diagnostics, "liquidity_amount_partial_missing_default_name_cap")

    reasons: list[str] = []
    for _, row in trace.iterrows():
        reason = "ok"
        if float(row["gross_exposure"]) <= epsilon:
            if int(row["prediction_rows"]) <= 0:
                reason = "missing_predictions"
            elif not has_candidate_trace:
                reason = "missing_weight_generation_diagnostics"
            elif int(row["selected_count"]) <= 0:
                reason = "no_candidates_selected"
            else:
                date = pd.Timestamp(row["trade_date"])
                if date in amount_all_missing_dates:
                    reason = "liquidity_amount_missing_previously_would_zero_caps"
                elif date in amount_partial_missing_dates:
                    reason = "partial_liquidity_amount_missing_check_caps"
                else:
                    reason = "unknown_all_zero_after_candidates"
        reasons.append(reason)
    trace["liveness_reason"] = reasons

    zero_days = trace[trace["gross_exposure"] <= epsilon].copy()
    unexplained_zero_days = zero_days[zero_days["liveness_reason"] == "unknown_all_zero_after_candidates"].copy()
    summary = {
        "num_days": int(daily_gross["trade_date"].nunique()) if not daily_gross.empty else 0,
        "num_symbols": int(long["symbol"].nunique()) if not long.empty else 0,
        "nonzero_weight_days": int((daily_gross["gross_exposure"] > epsilon).sum()) if not daily_gross.empty else 0,
        "zero_weight_days": int((daily_gross["gross_exposure"] <= epsilon).sum()) if not daily_gross.empty else 0,
        "mean_gross": float(daily_gross["gross_exposure"].mean()) if not daily_gross.empty else 0.0,
        "median_gross": float(daily_gross["gross_exposure"].median()) if not daily_gross.empty else 0.0,
        "max_gross": float(daily_gross["gross_exposure"].max()) if not daily_gross.empty else 0.0,
        "days_with_predictions": int((trace["prediction_rows"] > 0).sum()) if not trace.empty else 0,
        "days_with_candidates": int((trace["selected_count"] > 0).sum()) if not trace.empty else 0,
        "days_killed_by_regime": 0,
        "days_killed_by_dd": 0,
        "days_killed_by_missing_data": int(trace["liveness_reason"].str.contains("missing", regex=False).sum()) if not trace.empty else 0,
        "unexplained_zero_days": int(len(unexplained_zero_days)),
        "all_zero": bool(not daily_gross.empty and (daily_gross["gross_exposure"] <= epsilon).all()),
        "target_weights_liveness": bool(len(unexplained_zero_days) == 0 and (daily_gross["gross_exposure"] > epsilon).any()),
        "status": "passed" if len(unexplained_zero_days) == 0 else "failed",
    }
    if summary["all_zero"] and summary["unexplained_zero_days"] == 0:
        summary["status"] = "failed"
        summary["target_weights_liveness"] = False
        summary["reason"] = "all_zero_but_explained"
    elif summary["unexplained_zero_days"] > 0:
        summary["reason"] = "unexplained_zero_days"
    else:
        summary["reason"] = "nonzero_or_empty_explained"
    return {
        "summary": summary,
        "daily_gross_exposure": daily_gross,
        "zero_weight_days": zero_days,
        "weight_generation_trace": trace,
    }


def render_liveness_markdown(report: dict[str, Any]) -> str:
    summary = report.get("summary", {})
    return "\n".join(
        [
            "# Target Weights Liveness",
            "",
            f"- status: {summary.get('status')}",
            f"- target_weights_liveness: {summary.get('target_weights_liveness')}",
            f"- all_zero: {summary.get('all_zero')}",
            f"- nonzero_weight_days: {summary.get('nonzero_weight_days')} / {summary.get('num_days')}",
            f"- mean_gross: {summary.get('mean_gross')}",
            f"- unexplained_zero_days: {summary.get('unexplained_zero_days')}",
            f"- reason: {summary.get('reason')}",
            "",
        ]
    )


def write_target_weights_liveness(report: dict[str, Any], output_dir: str | Path) -> dict[str, Path]:
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)
    paths = {
        "json": out / "target_weights_liveness.json",
        "markdown": out / "target_weights_liveness.md",
        "daily_gross": out / "daily_gross_exposure.csv",
        "zero_weight_days": out / "zero_weight_days.csv",
        "trace": out / "weight_generation_trace.csv",
    }
    paths["json"].write_text(json.dumps(report.get("summary", {}), ensure_ascii=False, indent=2, sort_keys=True, default=str), encoding="utf-8")
    paths["markdown"].write_text(render_liveness_markdown(report), encoding="utf-8")
    for key, path in (("daily_gross_exposure", paths["daily_gross"]), ("zero_weight_days", paths["zero_weight_days"]), ("weight_generation_trace", paths["trace"])):
        table = report.get(key)
        if isinstance(table, pd.DataFrame):
            table.to_csv(path, index=False)
        else:
            pd.DataFrame().to_csv(path, index=False)
    return paths


def load_diagnostics(path: str | Path | None) -> dict[str, Any]:
    return _read_json(path)


__all__ = [
    "build_target_weights_liveness",
    "load_diagnostics",
    "render_liveness_markdown",
    "write_target_weights_liveness",
]
