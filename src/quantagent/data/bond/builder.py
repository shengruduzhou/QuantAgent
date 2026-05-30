"""BondFlowBuilder — bond yield/credit/flow daily silver dataset.

Schema (one row per trade_date):

* ``trade_date``        — calendar trade day (string-parsed → Timestamp)
* ``available_at``      — when our pipeline can see this row (T+1 close
                          for EOD bond data; T for intraday DR007)
* ``yield_1y``, ``yield_5y``, ``yield_10y`` — treasury yields in %
* ``spread_10y_1y``     — term premium (10y − 1y)
* ``spread_10y_3m``     — recession indicator (10y − 3m)
* ``credit_spread_aa``  — AA-corp yield − 10y treasury, in %
* ``credit_spread_aaa_aa`` — AAA − AA quality spread, in %
* ``dr007``             — interbank 7-day repo rate in %
* ``bond_fund_flow``    — net inflow into bond ETFs/funds, CNY billions
* ``source``            — vendor / endpoint identifier
* ``source_version``    — for reproducibility
* ``fetched_at``        — when the row was pulled

The gate is conservative: a closed gate means downstream feature
joining must treat the bond layer as absent (silently 0/NaN). Never
proceed with partial coverage that would feed the model future data.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timezone
import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


BOND_FLOW_REQUIRED_COLUMNS: tuple[str, ...] = (
    "trade_date",
    "available_at",
    "yield_1y",
    "yield_5y",
    "yield_10y",
    "spread_10y_1y",
    "spread_10y_3m",
    "credit_spread_aa",
    "credit_spread_aaa_aa",
    "dr007",
    "bond_fund_flow",
    "source",
    "source_version",
    "fetched_at",
)

OPTIONAL_INPUT_COLUMNS: tuple[str, ...] = (
    "yield_3m",  # used to compute spread_10y_3m if spread missing
    "yield_aa",  # used to derive credit_spread_aa if missing
    "yield_aaa",
)


# ---------------------------------------------------------------------------
# Config + result
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class BondFlowConfig:
    source: str = "manual_local_import"
    source_version: str = "unknown"
    output_root: str | Path = "runtime/data/v7"
    # Gate thresholds
    min_days: int = 30
    min_field_coverage: float = 0.50  # ≥50% of yield columns present per row
    min_date_continuity: float = 0.95  # weekday gaps ≤ 5%


@dataclass
class BondFlowResult:
    frame: pd.DataFrame
    coverage: dict[str, Any]
    validation: dict[str, Any]
    output_paths: dict[str, str] = field(default_factory=dict)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _coerce_ts(value: Any) -> pd.Timestamp | None:
    if value is None or (isinstance(value, float) and pd.isna(value)):
        return None
    ts = pd.to_datetime(value, errors="coerce", utc=False)
    if pd.isna(ts):
        return None
    if ts.tzinfo is not None:
        ts = ts.tz_localize(None)
    return ts


def _derive_spreads(frame: pd.DataFrame) -> pd.DataFrame:
    out = frame.copy()
    if "spread_10y_1y" not in out.columns and {"yield_10y", "yield_1y"}.issubset(out.columns):
        out["spread_10y_1y"] = out["yield_10y"] - out["yield_1y"]
    if "spread_10y_3m" not in out.columns and {"yield_10y", "yield_3m"}.issubset(out.columns):
        out["spread_10y_3m"] = out["yield_10y"] - out["yield_3m"]
    if "credit_spread_aa" not in out.columns and {"yield_aa", "yield_10y"}.issubset(out.columns):
        out["credit_spread_aa"] = out["yield_aa"] - out["yield_10y"]
    if "credit_spread_aaa_aa" not in out.columns and {"yield_aaa", "yield_aa"}.issubset(out.columns):
        out["credit_spread_aaa_aa"] = out["yield_aaa"] - out["yield_aa"]
    return out


def _fill_missing_required(frame: pd.DataFrame) -> pd.DataFrame:
    out = frame.copy()
    for col in BOND_FLOW_REQUIRED_COLUMNS:
        if col not in out.columns:
            out[col] = np.nan
    return out


# ---------------------------------------------------------------------------
# Core builder
# ---------------------------------------------------------------------------

def build_bond_flows(
    raw: pd.DataFrame,
    *,
    config: BondFlowConfig | None = None,
) -> BondFlowResult:
    cfg = config or BondFlowConfig()
    if raw is None or raw.empty:
        return _empty_result(cfg)

    if "trade_date" not in raw.columns:
        raise ValueError("raw bond frame missing required column: trade_date")

    work = raw.copy()
    work["trade_date"] = work["trade_date"].map(_coerce_ts)
    before = len(work)
    work = work[work["trade_date"].notna()].sort_values("trade_date").reset_index(drop=True)
    rejected_no_date = before - len(work)

    work = _derive_spreads(work)
    work = _fill_missing_required(work)

    # available_at: prefer caller-supplied, else T+1 business day from
    # trade_date (EOD bond data is published next morning).
    fetched_default = pd.Timestamp(datetime.now(timezone.utc).replace(tzinfo=None))
    work["fetched_at"] = work["fetched_at"].map(_coerce_ts).fillna(fetched_default)
    fallback_available = work["trade_date"] + pd.tseries.offsets.BDay(1)
    work["available_at"] = work["available_at"].map(_coerce_ts).fillna(fallback_available)

    work["source"] = work["source"].fillna(cfg.source).astype(str)
    work["source_version"] = work["source_version"].fillna(cfg.source_version).astype(str)

    # Cast numeric columns
    numeric_cols = (
        "yield_1y", "yield_5y", "yield_10y",
        "spread_10y_1y", "spread_10y_3m",
        "credit_spread_aa", "credit_spread_aaa_aa",
        "dr007", "bond_fund_flow",
    )
    for col in numeric_cols:
        work[col] = pd.to_numeric(work[col], errors="coerce")

    # De-dup by trade_date (last write wins)
    before_dedup = len(work)
    work = work.drop_duplicates(subset=["trade_date"], keep="last").reset_index(drop=True)
    dup_removed = before_dedup - len(work)

    # Final canonical column ordering
    out = work[list(BOND_FLOW_REQUIRED_COLUMNS)].copy()
    n = int(len(out))

    field_coverage = (
        out[list(numeric_cols)].notna().sum(axis=1) / max(1, len(numeric_cols))
    )
    mean_field_coverage = float(field_coverage.mean()) if n else 0.0

    if n >= 2:
        # date continuity over weekdays
        dates = out["trade_date"].sort_values()
        expected_weekdays = pd.bdate_range(dates.min(), dates.max())
        date_continuity = float(min(1.0, n / max(1, len(expected_weekdays))))
    else:
        date_continuity = 0.0

    gate_open = (
        n >= cfg.min_days
        and mean_field_coverage >= cfg.min_field_coverage
        and date_continuity >= cfg.min_date_continuity
    )
    reason = "passed" if gate_open else _gate_reason(
        n, mean_field_coverage, date_continuity, cfg
    )

    coverage = {
        "n_days": n,
        "rejected_no_date": int(rejected_no_date),
        "duplicates_removed": int(dup_removed),
        "mean_field_coverage": mean_field_coverage,
        "date_continuity": date_continuity,
        "field_non_null_counts": {
            col: int(out[col].notna().sum()) for col in numeric_cols
        },
        "gate": {
            "bond_flows_usable_for_features": bool(gate_open),
            "reason": reason,
        },
    }
    validation = {"status": "passed", "n": n, "errors": []}
    return BondFlowResult(frame=out, coverage=coverage, validation=validation)


def _empty_result(cfg: BondFlowConfig) -> BondFlowResult:
    return BondFlowResult(
        frame=pd.DataFrame(columns=list(BOND_FLOW_REQUIRED_COLUMNS)),
        coverage={
            "n_days": 0,
            "mean_field_coverage": 0.0,
            "date_continuity": 0.0,
            "gate": {"bond_flows_usable_for_features": False, "reason": "no_rows"},
        },
        validation={"status": "passed", "n": 0, "errors": []},
    )


def _gate_reason(
    n: int,
    field_coverage: float,
    date_continuity: float,
    cfg: BondFlowConfig,
) -> str:
    if n < cfg.min_days:
        return f"too_few_days_{n}_lt_{cfg.min_days}"
    if field_coverage < cfg.min_field_coverage:
        return f"field_coverage_{field_coverage:.3f}_below_{cfg.min_field_coverage:.3f}"
    if date_continuity < cfg.min_date_continuity:
        return f"date_continuity_{date_continuity:.3f}_below_{cfg.min_date_continuity:.3f}"
    return "unknown"


# ---------------------------------------------------------------------------
# Builder with writer
# ---------------------------------------------------------------------------

class BondFlowBuilder:
    def __init__(self, config: BondFlowConfig | None = None) -> None:
        self.config = config or BondFlowConfig()

    def build(self, raw: pd.DataFrame) -> BondFlowResult:
        return build_bond_flows(raw, config=self.config)

    def write(self, result: BondFlowResult) -> BondFlowResult:
        root = Path(self.config.output_root)
        silver_dir = root / "silver" / "bond_flows"
        silver_dir.mkdir(parents=True, exist_ok=True)
        parquet_path = silver_dir / "bond_flows.parquet"
        result.frame.to_parquet(parquet_path, index=False)
        (silver_dir / "coverage_report.json").write_text(
            json.dumps(result.coverage, indent=2, default=str), encoding="utf-8"
        )
        (silver_dir / "validation_report.json").write_text(
            json.dumps(result.validation, indent=2, default=str), encoding="utf-8"
        )
        manifests_dir = root / "manifests"
        manifests_dir.mkdir(parents=True, exist_ok=True)
        manifest_path = manifests_dir / "bond_flows.json"
        manifest_path.write_text(
            json.dumps(
                {
                    "name": "bond_flows",
                    "rows": int(len(result.frame)),
                    "extra": {"coverage_report": result.coverage},
                    "source_version": self.config.source_version,
                },
                indent=2,
                default=str,
            ),
            encoding="utf-8",
        )
        result.output_paths = {
            "bond_flows": str(parquet_path),
            "coverage_report": str(silver_dir / "coverage_report.json"),
            "validation_report": str(silver_dir / "validation_report.json"),
            "manifest": str(manifest_path),
        }
        return result


# ---------------------------------------------------------------------------
# Feature join helper
# ---------------------------------------------------------------------------

def apply_bond_flow_features(
    panel: pd.DataFrame,
    bond_flows: pd.DataFrame,
    *,
    feature_columns: tuple[str, ...] | None = None,
    prefix: str = "bond_",
) -> pd.DataFrame:
    """Attach bond features to an equity training panel via merge_asof.

    Uses ``available_at`` (not ``trade_date``) as the right-key so a
    2020-08-15 equity row gets the most recent bond row whose
    ``available_at <= 2020-08-15``.  Never leaks future-dated bond
    prints into past equity rows.

    Parameters
    ----------
    panel : DataFrame with ``trade_date`` and ``symbol``.
    bond_flows : Output of the builder (silver/bond_flows.parquet).
    feature_columns : Subset of numeric bond columns to attach.
        Defaults to all numeric bond fields.
    prefix : Column prefix in the output (default ``bond_``).
    """
    if panel is None or panel.empty:
        return panel.copy() if panel is not None else pd.DataFrame()
    if bond_flows is None or bond_flows.empty:
        return panel.copy()

    default_features = (
        "yield_1y", "yield_5y", "yield_10y",
        "spread_10y_1y", "spread_10y_3m",
        "credit_spread_aa", "credit_spread_aaa_aa",
        "dr007", "bond_fund_flow",
    )
    cols = tuple(feature_columns) if feature_columns is not None else default_features

    panel_out = panel.copy()
    panel_out["trade_date"] = pd.to_datetime(panel_out["trade_date"])

    flows = bond_flows.copy()
    flows["available_at"] = pd.to_datetime(flows["available_at"])
    flows = flows.dropna(subset=["available_at"]).sort_values("available_at")

    left = panel_out[["trade_date"]].copy()
    left["__orig_index"] = panel_out.index
    left = left.sort_values("trade_date").reset_index(drop=True)
    flows_keep = flows[["available_at", *[c for c in cols if c in flows.columns]]]
    merged = pd.merge_asof(
        left,
        flows_keep,
        left_on="trade_date",
        right_on="available_at",
        direction="backward",
    )

    for col in cols:
        if col in merged.columns:
            target = f"{prefix}{col}"
            series = pd.Series(np.nan, index=panel_out.index, dtype=float)
            series.loc[merged["__orig_index"].values] = (
                merged[col].astype(float).values
            )
            panel_out[target] = series

    return panel_out


# ---------------------------------------------------------------------------
# Manifest-gated helper
# ---------------------------------------------------------------------------

def bond_flows_for_features(
    bond_flows: pd.DataFrame | None,
    manifest_path: str | Path | None,
) -> pd.DataFrame | None:
    if bond_flows is None or len(bond_flows) == 0:
        return None
    if manifest_path is None:
        return None
    p = Path(manifest_path)
    if not p.exists():
        return None
    try:
        payload = json.loads(p.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError):
        return None
    gate = (
        (payload.get("extra") or {})
        .get("coverage_report", {})
        .get("gate", {})
    )
    if not gate.get("bond_flows_usable_for_features"):
        return None
    return bond_flows
