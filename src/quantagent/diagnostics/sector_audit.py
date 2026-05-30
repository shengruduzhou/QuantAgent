"""Null-safe sector / board / ST post-trade diagnostics.

This module is intentionally audit-only. It reads sector/ST manifests and
produces exposure reports, but it never changes target weights. Optimizer
callers must use the guard helpers here before passing optional sector/ST
tables downstream.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
import json
from pathlib import Path
from typing import Any

import pandas as pd

from quantagent.diagnostics.stratified_ic import board_of
from quantagent.universe.filters import derive_market_flags


UNKNOWN = "UNKNOWN"


@dataclass(frozen=True)
class SectorSTGateStatus:
    sector_usable_for_diagnostics: bool = True
    sector_usable_for_optimization: bool = False
    sector_reason: str = "manifest_missing"
    st_usable_for_risk_filter: bool = False
    st_reason: str = "manifest_missing"
    sector_manifest_path: str | None = None
    st_manifest_path: str | None = None

    @property
    def sector_optimization_enabled(self) -> bool:
        return bool(self.sector_usable_for_optimization)

    @property
    def st_risk_filter_enabled(self) -> bool:
        return bool(self.st_usable_for_risk_filter)

    def to_dict(self) -> dict[str, Any]:
        data = asdict(self)
        data["sector_optimization_enabled"] = self.sector_optimization_enabled
        data["st_risk_filter_enabled"] = self.st_risk_filter_enabled
        return data


def _read_json(path: str | Path | None) -> dict[str, Any]:
    if path is None:
        return {}
    file_path = Path(path)
    if not file_path.exists():
        return {}
    return json.loads(file_path.read_text(encoding="utf-8"))


def _gate_from_manifest(payload: dict[str, Any]) -> dict[str, Any]:
    extra = payload.get("extra") if isinstance(payload, dict) else {}
    coverage = extra.get("coverage_report", {}) if isinstance(extra, dict) else {}
    gate = coverage.get("gate", {}) if isinstance(coverage, dict) else {}
    if not isinstance(gate, dict):
        return {}
    return gate


def _sector_unknown_rate_from_manifest(path: str | Path | None) -> float | None:
    payload = _read_json(path)
    extra = payload.get("extra") if isinstance(payload, dict) else {}
    coverage = extra.get("coverage_report", {}) if isinstance(extra, dict) else {}
    if not isinstance(coverage, dict):
        return None
    gate = coverage.get("gate", {})
    observed = gate.get("observed", {}) if isinstance(gate, dict) else {}
    value = observed.get("unknown_rate", coverage.get("unknown_rate")) if isinstance(observed, dict) else coverage.get("unknown_rate")
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def load_sector_st_gate_status(
    sector_manifest: str | Path | None = None,
    st_manifest: str | Path | None = None,
) -> SectorSTGateStatus:
    """Load null-safe optimization/risk-filter flags from manifests."""

    sector_payload = _read_json(sector_manifest)
    st_payload = _read_json(st_manifest)
    sector_gate = _gate_from_manifest(sector_payload)
    st_gate = _gate_from_manifest(st_payload)

    return SectorSTGateStatus(
        sector_usable_for_diagnostics=bool(sector_gate.get("sector_usable_for_diagnostics", True)),
        sector_usable_for_optimization=bool(sector_gate.get("sector_usable_for_optimization", False)),
        sector_reason=str(sector_gate.get("reason", "manifest_missing")),
        st_usable_for_risk_filter=bool(st_gate.get("st_usable_for_risk_filter", False)),
        st_reason=str(st_gate.get("reason", "manifest_missing")),
        sector_manifest_path=str(sector_manifest) if sector_manifest is not None else None,
        st_manifest_path=str(st_manifest) if st_manifest is not None else None,
    )


def sector_map_for_optimization(
    sector_map: pd.DataFrame | None,
    sector_manifest: str | Path | None = None,
) -> pd.DataFrame | None:
    """Return sector_map only when the manifest allows optimization use.

    The current V7 optimizer expects an ``industry`` column. Canonical
    Step 2.3 sector maps use ``sector_level_1``; when the gate is open,
    this helper adds ``industry`` as a compatibility alias without
    mutating the caller's frame.
    """

    status = load_sector_st_gate_status(sector_manifest=sector_manifest)
    if not status.sector_usable_for_optimization:
        return None
    if sector_map is None or sector_map.empty:
        return None
    out = sector_map.copy()
    if "industry" not in out.columns and "sector_level_1" in out.columns:
        out["industry"] = out["sector_level_1"]
    return out


def st_flags_for_risk_filter(
    st_flags: pd.DataFrame | None,
    st_manifest: str | Path | None = None,
) -> pd.DataFrame | None:
    """Return ST flags only when the manifest allows risk-filter use."""

    status = load_sector_st_gate_status(st_manifest=st_manifest)
    if not status.st_usable_for_risk_filter:
        return None
    if st_flags is None or st_flags.empty:
        return None
    return st_flags.copy()


def target_weights_to_long(target_weights: pd.DataFrame) -> pd.DataFrame:
    """Normalize wide or long target weights into long format."""

    if target_weights is None or target_weights.empty:
        return pd.DataFrame(columns=["trade_date", "symbol", "weight"])
    frame = target_weights.copy()
    if {"trade_date", "symbol", "weight"}.issubset(frame.columns):
        out = frame[["trade_date", "symbol", "weight"]].copy()
    elif "trade_date" in frame.columns:
        out = frame.melt(id_vars=["trade_date"], var_name="symbol", value_name="weight")
    else:
        idx_name = frame.index.name or "trade_date"
        out = frame.reset_index().rename(columns={idx_name: "trade_date"})
        out = out.melt(id_vars=["trade_date"], var_name="symbol", value_name="weight")
    out["trade_date"] = pd.to_datetime(out["trade_date"], errors="coerce")
    out["symbol"] = out["symbol"].astype(str)
    out["weight"] = pd.to_numeric(out["weight"], errors="coerce").fillna(0.0)
    return out.dropna(subset=["trade_date", "symbol"]).reset_index(drop=True)


def _attach_sector_asof(weights_long: pd.DataFrame, sector_map: pd.DataFrame | None) -> pd.DataFrame:
    out = weights_long.copy()
    out["sector_level_1"] = UNKNOWN
    out["sector_level_2"] = UNKNOWN
    if sector_map is None or sector_map.empty:
        return out
    required = {"symbol", "available_at", "sector_level_1"}
    if not required.issubset(sector_map.columns):
        return out

    sm = sector_map.copy()
    sm["symbol"] = sm["symbol"].astype(str)
    sm["available_at"] = pd.to_datetime(sm["available_at"], errors="coerce", utc=True).dt.tz_convert(None)
    sm = sm.dropna(subset=["symbol", "available_at"])
    if "sector_level_2" not in sm.columns:
        sm["sector_level_2"] = sm["sector_level_1"]
    status = sm.get("coverage_status", pd.Series("pit_historical", index=sm.index)).astype(str)
    source = sm.get("source", pd.Series("", index=sm.index)).astype(str)
    real_sector = (status != "missing") & (source != "board_proxy")
    sm = sm.loc[real_sector, ["symbol", "available_at", "sector_level_1", "sector_level_2"]]
    if sm.empty:
        return out

    sm = sm.sort_values(["available_at", "symbol"])
    left = out.drop(columns=["sector_level_1", "sector_level_2"]).sort_values(["trade_date", "symbol"])
    merged = pd.merge_asof(
        left,
        sm,
        left_on="trade_date",
        right_on="available_at",
        by="symbol",
        direction="backward",
        allow_exact_matches=True,
    )
    merged["sector_level_1"] = merged["sector_level_1"].fillna(UNKNOWN).astype(str)
    merged["sector_level_2"] = merged["sector_level_2"].fillna(UNKNOWN).astype(str)
    return merged.drop(columns=["available_at"], errors="ignore")


def _attach_st_asof(weights_long: pd.DataFrame, st_flags: pd.DataFrame | None) -> pd.DataFrame:
    out = weights_long.copy()
    out["st_known"] = False
    out["is_st"] = False
    if st_flags is None or st_flags.empty:
        return out
    required = {"symbol", "available_at", "is_st", "st_known"}
    if not required.issubset(st_flags.columns):
        return out

    st = st_flags.copy()
    st["symbol"] = st["symbol"].astype(str)
    st["available_at"] = pd.to_datetime(st["available_at"], errors="coerce", utc=True).dt.tz_convert(None)
    st = st.dropna(subset=["symbol", "available_at"])
    st = st[["symbol", "available_at", "is_st", "st_known"]].sort_values(["available_at", "symbol"])
    if st.empty:
        return out

    left = out.drop(columns=["st_known", "is_st"]).sort_values(["trade_date", "symbol"])
    merged = pd.merge_asof(
        left,
        st,
        left_on="trade_date",
        right_on="available_at",
        by="symbol",
        direction="backward",
        allow_exact_matches=True,
    )
    merged["st_known"] = merged["st_known"].astype("boolean").fillna(False).astype(bool)
    merged["is_st"] = merged["is_st"].astype("boolean").fillna(False).astype(bool)
    return merged.drop(columns=["available_at"], errors="ignore")


def _attach_suspended(weights_long: pd.DataFrame, market_panel: pd.DataFrame | None) -> pd.DataFrame:
    out = weights_long.copy()
    out["is_suspended_inferred"] = False
    if market_panel is None or market_panel.empty:
        return out
    flags = derive_market_flags(market_panel)
    if flags.empty:
        return out
    flags = flags[["trade_date", "symbol", "is_suspended"]].drop_duplicates(["trade_date", "symbol"], keep="last")
    flags["trade_date"] = pd.to_datetime(flags["trade_date"], errors="coerce")
    flags["symbol"] = flags["symbol"].astype(str)
    merged = out.drop(columns=["is_suspended_inferred"]).merge(flags, on=["trade_date", "symbol"], how="left")
    merged["is_suspended_inferred"] = merged["is_suspended"].fillna(False).astype(bool)
    return merged.drop(columns=["is_suspended"], errors="ignore")


def _exposure_table(frame: pd.DataFrame, bucket_col: str) -> pd.DataFrame:
    if frame.empty:
        return pd.DataFrame(columns=["trade_date", bucket_col, "gross_weight", "net_weight", "symbol_count"])
    grouped = (
        frame.groupby(["trade_date", bucket_col], dropna=False)
        .agg(
            gross_weight=("abs_weight", "sum"),
            net_weight=("weight", "sum"),
            symbol_count=("symbol", "nunique"),
        )
        .reset_index()
        .sort_values(["trade_date", bucket_col])
    )
    grouped[bucket_col] = grouped[bucket_col].fillna(UNKNOWN).astype(str)
    return grouped.reset_index(drop=True)


def build_sector_audit(
    target_weights: pd.DataFrame,
    *,
    sector_map: pd.DataFrame | None = None,
    st_flags: pd.DataFrame | None = None,
    market_panel: pd.DataFrame | None = None,
    sector_manifest: str | Path | None = None,
    st_manifest: str | Path | None = None,
) -> dict[str, Any]:
    """Build post-hoc exposure audit tables without changing weights."""

    gate_status = load_sector_st_gate_status(sector_manifest=sector_manifest, st_manifest=st_manifest)
    long = target_weights_to_long(target_weights)
    long["board_proxy"] = long["symbol"].map(board_of)
    long = _attach_sector_asof(long, sector_map)
    long = _attach_st_asof(long, st_flags)
    long = _attach_suspended(long, market_panel)
    long["abs_weight"] = long["weight"].abs()
    active = long[long["abs_weight"] > 0].copy()

    by_board = _exposure_table(active, "board_proxy")
    by_l1 = _exposure_table(active, "sector_level_1")
    by_l2 = _exposure_table(active, "sector_level_2")
    unknown = active[active["sector_level_1"].eq(UNKNOWN)].copy()
    unknown_exposure = _exposure_table(unknown, "sector_level_1").rename(columns={"sector_level_1": "bucket"})

    if active.empty:
        st_audit = pd.DataFrame(columns=["trade_date", "st_bucket", "gross_weight", "net_weight", "symbol_count"])
    else:
        st_work = active.copy()
        st_work["st_bucket"] = "ST_UNKNOWN"
        st_work.loc[st_work["st_known"] & ~st_work["is_st"], "st_bucket"] = "ST_KNOWN_NOT_ST"
        st_work.loc[st_work["st_known"] & st_work["is_st"], "st_bucket"] = "ST_KNOWN_ST"
        st_work.loc[st_work["is_suspended_inferred"], "st_bucket"] = "SUSPENDED_INFERRED"
        st_audit = _exposure_table(st_work, "st_bucket")

    total_gross = float(active["abs_weight"].sum()) if not active.empty else 0.0
    unknown_gross = float(unknown["abs_weight"].sum()) if not unknown.empty else 0.0
    manifest_unknown_rate = _sector_unknown_rate_from_manifest(sector_manifest)
    if total_gross > 0:
        sector_unknown_rate = float(unknown_gross / total_gross)
        sector_unknown_rate_source = "active_exposure"
    else:
        sector_unknown_rate = float(manifest_unknown_rate) if manifest_unknown_rate is not None else 0.0
        sector_unknown_rate_source = "manifest_coverage" if manifest_unknown_rate is not None else "no_active_exposure"
    real_sector_coverage = float(1.0 - sector_unknown_rate) if total_gross > 0 else 0.0
    summary = {
        **gate_status.to_dict(),
        "target_weights_contamination": False,
        "sector_unknown_rate": sector_unknown_rate,
        "sector_unknown_rate_source": sector_unknown_rate_source,
        "real_sector_coverage": real_sector_coverage,
        "board_proxy_available": bool(not long.empty and long["board_proxy"].ne("OTHER").any()),
        "active_rows": int(len(active)),
        "active_symbols": int(active["symbol"].nunique()) if not active.empty else 0,
        "active_dates": int(active["trade_date"].nunique()) if not active.empty else 0,
    }
    return {
        "gate_status": summary,
        "exposure_by_board_proxy": by_board,
        "exposure_by_sector_l1": by_l1,
        "exposure_by_sector_l2": by_l2,
        "unknown_exposure": unknown_exposure,
        "st_risk_audit": st_audit,
    }


def render_sector_audit_markdown(audit: dict[str, Any]) -> str:
    status = audit.get("gate_status", {})
    return "\n".join(
        [
            "# Sector / Board / ST Audit",
            "",
            "## Gate Status",
            f"- sector_usable_for_diagnostics: {status.get('sector_usable_for_diagnostics')}",
            f"- sector_usable_for_optimization: {status.get('sector_usable_for_optimization')}",
            f"- st_usable_for_risk_filter: {status.get('st_usable_for_risk_filter')}",
            f"- target_weights_contamination: {status.get('target_weights_contamination')}",
            f"- sector_unknown_rate: {status.get('sector_unknown_rate')}",
            f"- real_sector_coverage: {status.get('real_sector_coverage')}",
            f"- board_proxy_available: {status.get('board_proxy_available')}",
            "",
            "This report is post-hoc diagnostics only. It must not modify target weights or optimizer inputs.",
            "",
        ]
    )


def write_sector_audit(audit: dict[str, Any], output_dir: str | Path) -> dict[str, Path]:
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    paths = {
        "gate_status": output / "gate_status.json",
        "exposure_by_board_proxy": output / "exposure_by_board_proxy.csv",
        "exposure_by_sector_l1": output / "exposure_by_sector_l1.csv",
        "exposure_by_sector_l2": output / "exposure_by_sector_l2.csv",
        "unknown_exposure": output / "unknown_exposure.csv",
        "st_risk_audit": output / "st_risk_audit.csv",
        "markdown": output / "sector_audit.md",
    }
    paths["gate_status"].write_text(
        json.dumps(audit.get("gate_status", {}), ensure_ascii=False, indent=2, sort_keys=True, default=str),
        encoding="utf-8",
    )
    for key in ("exposure_by_board_proxy", "exposure_by_sector_l1", "exposure_by_sector_l2", "unknown_exposure", "st_risk_audit"):
        table = audit.get(key)
        if isinstance(table, pd.DataFrame):
            table.to_csv(paths[key], index=False)
        else:
            pd.DataFrame().to_csv(paths[key], index=False)
    paths["markdown"].write_text(render_sector_audit_markdown(audit), encoding="utf-8")
    return paths


__all__ = [
    "SectorSTGateStatus",
    "build_sector_audit",
    "load_sector_st_gate_status",
    "render_sector_audit_markdown",
    "sector_map_for_optimization",
    "st_flags_for_risk_filter",
    "target_weights_to_long",
    "write_sector_audit",
]
