"""Bond, money-market and fiscal-liquidity silver data product.

All rates are normalised to percentage points and all flow quantities to CNY
billions.  The builder fails closed when units are implausible.  It also keeps
publication and ingestion timestamps separate and derives a transparent
``fiscal_liquidity_impulse`` rather than treating bond-fund flow as a proxy for
all public-sector liquidity.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timezone
import json
from pathlib import Path
from typing import Any, Literal

import numpy as np
import pandas as pd


RATE_COLUMNS: tuple[str, ...] = (
    "yield_3m",
    "yield_1y",
    "yield_5y",
    "yield_10y",
    "yield_aa",
    "yield_aaa",
    "dr001",
    "dr007",
    "r007",
    "shibor_3m",
    "cd_1y",
)

FLOW_COLUMNS_CNY_BN: tuple[str, ...] = (
    "bond_fund_flow",
    "omo_injection",
    "omo_maturity",
    "mlf_injection",
    "mlf_maturity",
    "rrr_released",
    "central_gov_bond_issuance",
    "central_gov_bond_maturity",
    "local_general_bond_issuance",
    "local_general_bond_maturity",
    "local_special_bond_issuance",
    "local_special_bond_maturity",
    "policy_bank_bond_issuance",
    "policy_bank_bond_maturity",
    "government_deposit_change",
)

DERIVED_COLUMNS: tuple[str, ...] = (
    "spread_10y_1y",
    "spread_10y_3m",
    "credit_spread_aa",
    "credit_spread_aa_aaa",
    "credit_spread_aaa_aa",  # backward-compatible alias; positive AA-minus-AAA
    "omo_net",
    "mlf_net",
    "central_gov_bond_net",
    "local_general_bond_net",
    "local_special_bond_net",
    "policy_bank_bond_net",
    "monetary_liquidity_impulse",
    "fiscal_net_financing",
    "fiscal_liquidity_impulse",
)

BOND_FLOW_REQUIRED_COLUMNS: tuple[str, ...] = (
    "trade_date",
    "public_available_at",
    "ingested_at",
    "available_at",
    *RATE_COLUMNS,
    *DERIVED_COLUMNS,
    *FLOW_COLUMNS_CNY_BN,
    "rate_unit",
    "flow_unit",
    "source",
    "source_version",
    "fetched_at",
)


@dataclass(frozen=True)
class BondFlowConfig:
    source: str = "manual_local_import"
    source_version: str = "unknown"
    output_root: str | Path = "runtime/data/v7"
    availability_mode: Literal["public", "ingested"] = "public"
    rate_unit: Literal["auto", "percent", "decimal", "bps"] = "auto"
    flow_unit: Literal["cny_bn", "cny_mn", "cny", "auto"] = "auto"
    min_days: int = 30
    min_field_coverage: float = 0.50
    min_date_continuity: float = 0.90
    max_abs_rate_percent: float = 30.0
    max_abs_flow_cny_bn: float = 1_000_000.0


@dataclass
class BondFlowResult:
    frame: pd.DataFrame
    coverage: dict[str, Any]
    validation: dict[str, Any]
    output_paths: dict[str, str] = field(default_factory=dict)


def _coerce_ts(value: Any) -> pd.Timestamp | None:
    if value is None or (isinstance(value, float) and pd.isna(value)):
        return None
    ts = pd.to_datetime(value, errors="coerce", utc=False)
    if pd.isna(ts):
        return None
    if getattr(ts, "tzinfo", None) is not None:
        ts = ts.tz_localize(None)
    return pd.Timestamp(ts)


def _series(frame: pd.DataFrame, name: str, default: float = np.nan) -> pd.Series:
    if name in frame.columns:
        return pd.to_numeric(frame[name], errors="coerce")
    return pd.Series(default, index=frame.index, dtype=float)


def _detect_rate_unit(values: pd.Series) -> str:
    sample = values.dropna().abs()
    if sample.empty:
        return "percent"
    median = float(sample.median())
    q95 = float(sample.quantile(0.95))
    if q95 <= 0.50:
        return "decimal"
    if median >= 50.0 or q95 > 100.0:
        return "bps"
    return "percent"


def _normalise_rate(values: pd.Series, unit: str) -> tuple[pd.Series, str]:
    resolved = _detect_rate_unit(values) if unit == "auto" else unit
    numeric = pd.to_numeric(values, errors="coerce").astype(float)
    if resolved == "decimal":
        numeric = numeric * 100.0
    elif resolved == "bps":
        numeric = numeric / 100.0
    elif resolved != "percent":
        raise ValueError(f"unsupported rate unit: {resolved}")
    return numeric, resolved


def _detect_flow_unit(values: pd.Series) -> str:
    sample = values.dropna().abs()
    if sample.empty:
        return "cny_bn"
    median = float(sample.median())
    # Daily public-sector flows above 1e8 in raw CNY are common.  Values below
    # roughly 1e5 are more likely already in billions or millions.
    if median >= 1e8:
        return "cny"
    if median >= 1e3:
        return "cny_mn"
    return "cny_bn"


def _normalise_flow(values: pd.Series, unit: str) -> tuple[pd.Series, str]:
    resolved = _detect_flow_unit(values) if unit == "auto" else unit
    numeric = pd.to_numeric(values, errors="coerce").astype(float)
    if resolved == "cny":
        numeric = numeric / 1e9
    elif resolved == "cny_mn":
        numeric = numeric / 1e3
    elif resolved != "cny_bn":
        raise ValueError(f"unsupported flow unit: {resolved}")
    return numeric, resolved


def _normalise_units(frame: pd.DataFrame, cfg: BondFlowConfig) -> tuple[pd.DataFrame, dict[str, str]]:
    out = frame.copy()
    detected: dict[str, str] = {}
    for col in RATE_COLUMNS:
        normalised, unit = _normalise_rate(_series(out, col), cfg.rate_unit)
        out[col] = normalised
        detected[f"rate:{col}"] = unit
    for col in FLOW_COLUMNS_CNY_BN:
        normalised, unit = _normalise_flow(_series(out, col, 0.0), cfg.flow_unit)
        out[col] = normalised.fillna(0.0)
        detected[f"flow:{col}"] = unit
    return out, detected


def _derive(frame: pd.DataFrame) -> pd.DataFrame:
    out = frame.copy()
    out["spread_10y_1y"] = out["yield_10y"] - out["yield_1y"]
    out["spread_10y_3m"] = out["yield_10y"] - out["yield_3m"]
    out["credit_spread_aa"] = out["yield_aa"] - out["yield_10y"]
    # Riskier AA minus safer AAA: a widening positive number means stress.
    out["credit_spread_aa_aaa"] = out["yield_aa"] - out["yield_aaa"]
    out["credit_spread_aaa_aa"] = out["credit_spread_aa_aaa"]

    out["omo_net"] = out["omo_injection"] - out["omo_maturity"]
    out["mlf_net"] = out["mlf_injection"] - out["mlf_maturity"]
    out["central_gov_bond_net"] = (
        out["central_gov_bond_issuance"] - out["central_gov_bond_maturity"]
    )
    out["local_general_bond_net"] = (
        out["local_general_bond_issuance"] - out["local_general_bond_maturity"]
    )
    out["local_special_bond_net"] = (
        out["local_special_bond_issuance"] - out["local_special_bond_maturity"]
    )
    out["policy_bank_bond_net"] = (
        out["policy_bank_bond_issuance"] - out["policy_bank_bond_maturity"]
    )
    out["monetary_liquidity_impulse"] = (
        out["omo_net"] + out["mlf_net"] + out["rrr_released"]
    )
    out["fiscal_net_financing"] = (
        out["central_gov_bond_net"]
        + out["local_general_bond_net"]
        + out["local_special_bond_net"]
        + out["policy_bank_bond_net"]
    )
    # A rise in government deposits withdraws cash from the private banking
    # system, therefore it enters with a negative sign.
    out["fiscal_liquidity_impulse"] = (
        out["monetary_liquidity_impulse"]
        + out["fiscal_net_financing"]
        - out["government_deposit_change"]
    )
    return out


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

    work = raw.copy().reset_index(drop=True)
    work["trade_date"] = work["trade_date"].map(_coerce_ts)
    before = len(work)
    work = work[work["trade_date"].notna()].copy()
    rejected_no_date = before - len(work)

    now = pd.Timestamp(datetime.now(timezone.utc).replace(tzinfo=None))
    public_source = work.get(
        "public_available_at",
        work.get("available_at", work["trade_date"] + pd.tseries.offsets.BDay(1)),
    )
    work["public_available_at"] = public_source.map(_coerce_ts)
    work["public_available_at"] = work["public_available_at"].fillna(
        work["trade_date"] + pd.tseries.offsets.BDay(1)
    )
    ingest_source = work.get(
        "ingested_at", work.get("fetched_at", pd.Series(now, index=work.index))
    )
    work["ingested_at"] = ingest_source.map(_coerce_ts).fillna(now)
    work["fetched_at"] = work["ingested_at"]
    if cfg.availability_mode == "public":
        work["available_at"] = work["public_available_at"]
    elif cfg.availability_mode == "ingested":
        work["available_at"] = work[["public_available_at", "ingested_at"]].max(axis=1)
    else:
        raise ValueError(f"unsupported availability_mode: {cfg.availability_mode}")

    work, detected_units = _normalise_units(work, cfg)
    work = _derive(work)
    work["rate_unit"] = "percent"
    work["flow_unit"] = "cny_bn"
    work["source"] = work.get("source", pd.Series(cfg.source, index=work.index)).fillna(cfg.source).astype(str)
    work["source_version"] = work.get(
        "source_version", pd.Series(cfg.source_version, index=work.index)
    ).fillna(cfg.source_version).astype(str)

    before_dedup = len(work)
    work = work.sort_values(["trade_date", "ingested_at"]).drop_duplicates(
        subset=["trade_date"], keep="last"
    )
    duplicates_removed = before_dedup - len(work)
    out = work[list(BOND_FLOW_REQUIRED_COLUMNS)].reset_index(drop=True)

    numeric_features = [
        *RATE_COLUMNS,
        *DERIVED_COLUMNS,
        *FLOW_COLUMNS_CNY_BN,
    ]
    n = int(len(out))
    mean_field_coverage = float(out[numeric_features].notna().mean(axis=1).mean()) if n else 0.0
    if n >= 2:
        expected = pd.bdate_range(out["trade_date"].min(), out["trade_date"].max())
        date_continuity = float(min(1.0, n / max(1, len(expected))))
    else:
        date_continuity = 0.0

    rate_values = out[list(RATE_COLUMNS)].to_numpy(dtype=float)
    max_abs_rate = float(np.nanmax(np.abs(rate_values))) if np.isfinite(rate_values).any() else 0.0
    flow_values = out[list(FLOW_COLUMNS_CNY_BN)].to_numpy(dtype=float)
    max_abs_flow = float(np.nanmax(np.abs(flow_values))) if np.isfinite(flow_values).any() else 0.0
    negative_credit_spread_rows = int((out["credit_spread_aa_aaa"] < -1e-9).sum())
    availability_violations = int((out["available_at"] < out["public_available_at"]).sum())

    errors: list[str] = []
    if max_abs_rate > cfg.max_abs_rate_percent:
        errors.append(f"rate_unit_implausible:max_abs={max_abs_rate:.4f}%")
    if max_abs_flow > cfg.max_abs_flow_cny_bn:
        errors.append(f"flow_unit_implausible:max_abs={max_abs_flow:.4f}bn")
    if negative_credit_spread_rows:
        errors.append(f"negative_aa_minus_aaa_rows={negative_credit_spread_rows}")
    if availability_violations:
        errors.append(f"availability_before_public={availability_violations}")

    gate_open = (
        n >= cfg.min_days
        and mean_field_coverage >= cfg.min_field_coverage
        and date_continuity >= cfg.min_date_continuity
        and not errors
    )
    reason = "passed" if gate_open else _gate_reason(
        n=n,
        field_coverage=mean_field_coverage,
        date_continuity=date_continuity,
        errors=errors,
        cfg=cfg,
    )
    coverage = {
        "n_days": n,
        "availability_mode": cfg.availability_mode,
        "rejected_no_date": int(rejected_no_date),
        "duplicates_removed": int(duplicates_removed),
        "mean_field_coverage": mean_field_coverage,
        "date_continuity": date_continuity,
        "max_abs_rate_percent": max_abs_rate,
        "max_abs_flow_cny_bn": max_abs_flow,
        "detected_input_units": detected_units,
        "field_non_null_counts": {
            col: int(out[col].notna().sum()) for col in numeric_features
        },
        "gate": {
            "bond_flows_usable_for_features": bool(gate_open),
            "reason": reason,
        },
    }
    validation = {
        "status": "passed" if gate_open else "failed",
        "n": n,
        "errors": errors,
    }
    return BondFlowResult(frame=out, coverage=coverage, validation=validation)


def _gate_reason(
    *,
    n: int,
    field_coverage: float,
    date_continuity: float,
    errors: list[str],
    cfg: BondFlowConfig,
) -> str:
    if errors:
        return errors[0]
    if n < cfg.min_days:
        return f"too_few_days_{n}_lt_{cfg.min_days}"
    if field_coverage < cfg.min_field_coverage:
        return f"field_coverage_{field_coverage:.3f}_below_{cfg.min_field_coverage:.3f}"
    if date_continuity < cfg.min_date_continuity:
        return f"date_continuity_{date_continuity:.3f}_below_{cfg.min_date_continuity:.3f}"
    return "unknown"


def _empty_result(cfg: BondFlowConfig) -> BondFlowResult:
    return BondFlowResult(
        frame=pd.DataFrame(columns=list(BOND_FLOW_REQUIRED_COLUMNS)),
        coverage={
            "n_days": 0,
            "availability_mode": cfg.availability_mode,
            "mean_field_coverage": 0.0,
            "date_continuity": 0.0,
            "gate": {
                "bond_flows_usable_for_features": False,
                "reason": "no_rows",
            },
        },
        validation={"status": "failed", "n": 0, "errors": ["no_rows"]},
    )


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
        coverage_path = silver_dir / "coverage_report.json"
        validation_path = silver_dir / "validation_report.json"
        coverage_path.write_text(json.dumps(result.coverage, indent=2, default=str), encoding="utf-8")
        validation_path.write_text(json.dumps(result.validation, indent=2, default=str), encoding="utf-8")
        manifests_dir = root / "manifests"
        manifests_dir.mkdir(parents=True, exist_ok=True)
        manifest_path = manifests_dir / "bond_flows.json"
        manifest_path.write_text(
            json.dumps(
                {
                    "name": "bond_flows",
                    "rows": int(len(result.frame)),
                    "schema_version": 2,
                    "units": {"rates": "percent", "flows": "cny_bn"},
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
            "coverage_report": str(coverage_path),
            "validation_report": str(validation_path),
            "manifest": str(manifest_path),
        }
        return result


def apply_bond_flow_features(
    panel: pd.DataFrame,
    bond_flows: pd.DataFrame,
    *,
    feature_columns: tuple[str, ...] | None = None,
    prefix: str = "bond_",
) -> pd.DataFrame:
    """Attach the latest publicly available liquidity observation by as-of join."""
    if panel is None or panel.empty:
        return panel.copy() if panel is not None else pd.DataFrame()
    if bond_flows is None or bond_flows.empty:
        return panel.copy()
    if "trade_date" not in panel.columns:
        raise ValueError("panel missing trade_date")

    default_features = (
        "yield_1y",
        "yield_5y",
        "yield_10y",
        "spread_10y_1y",
        "credit_spread_aa",
        "credit_spread_aa_aaa",
        "dr007",
        "cd_1y",
        "bond_fund_flow",
        "monetary_liquidity_impulse",
        "fiscal_net_financing",
        "fiscal_liquidity_impulse",
    )
    cols = tuple(feature_columns) if feature_columns is not None else default_features
    missing = [col for col in cols if col not in bond_flows.columns]
    if missing:
        raise ValueError(f"bond flow frame missing requested features: {missing}")

    panel_out = panel.copy()
    panel_out["trade_date"] = pd.to_datetime(panel_out["trade_date"], errors="coerce")
    flows = bond_flows.copy()
    flows["available_at"] = pd.to_datetime(flows["available_at"], errors="coerce")
    flows = flows.dropna(subset=["available_at"]).sort_values("available_at")

    left = panel_out[["trade_date"]].assign(__orig_index=panel_out.index)
    left = left.sort_values("trade_date").reset_index(drop=True)
    merged = pd.merge_asof(
        left,
        flows[["available_at", *cols]],
        left_on="trade_date",
        right_on="available_at",
        direction="backward",
        allow_exact_matches=True,
    )
    for col in cols:
        target = f"{prefix}{col}"
        values = pd.Series(np.nan, index=panel_out.index, dtype=float)
        values.loc[merged["__orig_index"].values] = pd.to_numeric(
            merged[col], errors="coerce"
        ).values
        panel_out[target] = values
    return panel_out


def bond_flows_for_features(
    bond_flows: pd.DataFrame | None,
    manifest_path: str | Path | None,
) -> pd.DataFrame | None:
    if bond_flows is None or bond_flows.empty or manifest_path is None:
        return None
    path = Path(manifest_path)
    if not path.exists():
        return None
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError):
        return None
    gate = ((payload.get("extra") or {}).get("coverage_report") or {}).get("gate") or {}
    if not gate.get("bond_flows_usable_for_features"):
        return None
    units = payload.get("units") or {}
    if units.get("rates") != "percent" or units.get("flows") != "cny_bn":
        return None
    return bond_flows.copy()
