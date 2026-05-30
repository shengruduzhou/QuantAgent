"""Stage 6 — v11 feature integration wrapper.

Loads every Stage 2-5 silver data product (respecting each manifest
gate) and attaches features to the training panel in a single pass.
The output panel feeds into the v11 training run (12-fold × 3-seed).

Each attach step is *gate-aware*: when a product's manifest gate is
closed, the step is **silently skipped** rather than feeding stale or
partial data into the model.  Every skip is recorded in the returned
attach log so post-mortem can reconstruct exactly which features were
present at training time.

Why gate-aware silent skip vs hard fail:
* If we hard-failed on a closed gate, an ops glitch (e.g. one missing
  manifest after a fetch script crashed) would block the entire
  training. The v11 design tolerates partial inputs and learns from
  what's available.
* The attach log lets us audit which features the model actually saw,
  so we never lie about "Stage 4 features were used" when the
  policy_events gate was closed.
"""

from __future__ import annotations

from dataclasses import dataclass, field
import json
from pathlib import Path
from typing import Any

import pandas as pd


@dataclass(frozen=True)
class V11IntegrationConfig:
    lake_root: str | Path = "runtime/data/v7"

    @property
    def silver_dir(self) -> Path:
        return Path(self.lake_root) / "silver"

    @property
    def manifests_dir(self) -> Path:
        return Path(self.lake_root) / "manifests"


@dataclass
class AttachLogEntry:
    product: str
    attempted: bool
    attached: bool
    reason: str
    columns_added: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return {
            "product": self.product,
            "attempted": self.attempted,
            "attached": self.attached,
            "reason": self.reason,
            "columns_added": list(self.columns_added),
        }


@dataclass
class V11IntegrationResult:
    panel: pd.DataFrame
    attach_log: list[AttachLogEntry] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return {
            "n_rows": int(len(self.panel)),
            "n_columns": int(len(self.panel.columns)),
            "attach_log": [e.to_dict() for e in self.attach_log],
            "features_attached": [e.product for e in self.attach_log if e.attached],
            "features_skipped": [e.product for e in self.attach_log if not e.attached],
        }


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _read_silver(path: Path) -> pd.DataFrame | None:
    if not path.exists():
        return None
    try:
        return pd.read_parquet(path)
    except (OSError, ValueError):
        return None


def _diff_cols(before: pd.DataFrame, after: pd.DataFrame) -> list[str]:
    return [c for c in after.columns if c not in before.columns]


# ---------------------------------------------------------------------------
# Per-product attach steps
# ---------------------------------------------------------------------------

def _attach_sector_map(panel: pd.DataFrame, cfg: V11IntegrationConfig) -> tuple[pd.DataFrame, AttachLogEntry]:
    product = "sector_map"
    parquet = cfg.silver_dir / "sector_map" / "sector_map.parquet"
    manifest = cfg.manifests_dir / "sector_map.json"
    if not parquet.exists():
        return panel, AttachLogEntry(product, attempted=True, attached=False, reason="silver_missing")
    try:
        from quantagent.diagnostics.sector_audit import sector_map_for_optimization
    except ImportError:
        return panel, AttachLogEntry(product, attempted=True, attached=False, reason="helper_missing")
    silver = _read_silver(parquet)
    if silver is None:
        return panel, AttachLogEntry(product, attempted=True, attached=False, reason="silver_read_failed")
    gated = sector_map_for_optimization(silver, manifest if manifest.exists() else None)
    if gated is None:
        return panel, AttachLogEntry(product, attempted=True, attached=False, reason="gate_closed")
    # Attach sector_level_1 to panel via symbol left-join. If the panel
    # already carries the column (e.g. from an earlier preprocessing
    # step), we still record success but add no new columns.
    if "symbol" not in panel.columns:
        return panel, AttachLogEntry(product, attempted=True, attached=False, reason="panel_missing_symbol")
    if "sector_level_1" in panel.columns:
        return panel, AttachLogEntry(
            product, attempted=True, attached=True, reason="ok_panel_already_has_column",
            columns_added=[],
        )
    before = panel.copy()
    out = panel.merge(
        gated[["symbol", "sector_level_1"]].drop_duplicates("symbol"),
        on="symbol",
        how="left",
    )
    return out, AttachLogEntry(
        product, attempted=True, attached=True, reason="ok",
        columns_added=_diff_cols(before, out),
    )


def _attach_st_flags(panel: pd.DataFrame, cfg: V11IntegrationConfig) -> tuple[pd.DataFrame, AttachLogEntry]:
    product = "st_flags"
    parquet = cfg.silver_dir / "st_flags" / "st_flags.parquet"
    manifest = cfg.manifests_dir / "st_flags.json"
    if not parquet.exists():
        return panel, AttachLogEntry(product, attempted=True, attached=False, reason="silver_missing")
    try:
        from quantagent.diagnostics.sector_audit import st_flags_for_risk_filter
    except ImportError:
        return panel, AttachLogEntry(product, attempted=True, attached=False, reason="helper_missing")
    silver = _read_silver(parquet)
    if silver is None:
        return panel, AttachLogEntry(product, attempted=True, attached=False, reason="silver_read_failed")
    gated = st_flags_for_risk_filter(silver, manifest if manifest.exists() else None)
    if gated is None:
        return panel, AttachLogEntry(product, attempted=True, attached=False, reason="gate_closed")
    if not {"trade_date", "symbol"}.issubset(panel.columns):
        return panel, AttachLogEntry(product, attempted=True, attached=False, reason="panel_keys_missing")
    before = panel.copy()
    # Sort then asof-merge so a 2020 panel row only sees ST flags
    # available_at ≤ trade_date
    left = panel[["trade_date", "symbol"]].assign(__idx=panel.index)
    left["trade_date"] = pd.to_datetime(left["trade_date"])
    left = left.sort_values(["trade_date", "symbol"]).reset_index(drop=True)
    right = gated[["symbol", "available_at", "is_st"]].copy()
    right["available_at"] = pd.to_datetime(right["available_at"])
    right = right.dropna(subset=["available_at"]).sort_values(["available_at", "symbol"])
    merged = pd.merge_asof(
        left,
        right,
        left_on="trade_date",
        right_on="available_at",
        by="symbol",
        direction="backward",
    )
    is_st_series = pd.Series(False, index=panel.index, dtype=bool)
    is_st_series.loc[merged["__idx"].values] = merged["is_st"].fillna(False).astype(bool).values
    out = panel.copy()
    out["is_st"] = is_st_series.values
    return out, AttachLogEntry(
        product, attempted=True, attached=True, reason="ok",
        columns_added=_diff_cols(before, out),
    )


def _attach_sector_pool(panel: pd.DataFrame, cfg: V11IntegrationConfig) -> tuple[pd.DataFrame, AttachLogEntry]:
    product = "sector_pool"
    parquet = cfg.silver_dir / "sector_pool" / "sector_pool.parquet"
    manifest = cfg.manifests_dir / "sector_pool.json"
    if not parquet.exists():
        return panel, AttachLogEntry(product, attempted=True, attached=False, reason="silver_missing")
    try:
        from quantagent.data.sector import sector_pool_for_weight_overlay
    except ImportError:
        return panel, AttachLogEntry(product, attempted=True, attached=False, reason="helper_missing")
    silver = _read_silver(parquet)
    if silver is None:
        return panel, AttachLogEntry(product, attempted=True, attached=False, reason="silver_read_failed")
    gated = sector_pool_for_weight_overlay(silver, manifest if manifest.exists() else None)
    if gated is None:
        return panel, AttachLogEntry(product, attempted=True, attached=False, reason="gate_closed")
    if "sector_level_1" not in panel.columns:
        return panel, AttachLogEntry(product, attempted=True, attached=False, reason="panel_missing_sector")
    before = panel.copy()
    out = panel.merge(
        gated[["sector_level_1", "pool_tier", "overlay_weight"]].rename(
            columns={"overlay_weight": "sector_pool_overlay_weight"}
        ),
        on="sector_level_1",
        how="left",
    )
    return out, AttachLogEntry(
        product, attempted=True, attached=True, reason="ok",
        columns_added=_diff_cols(before, out),
    )


def _attach_fundamental_ranker(panel: pd.DataFrame, cfg: V11IntegrationConfig) -> tuple[pd.DataFrame, AttachLogEntry]:
    product = "fundamental_ranker"
    parquet = cfg.silver_dir / "fundamental_ranker" / "fundamental_ranker.parquet"
    manifest = cfg.manifests_dir / "fundamental_ranker.json"
    if not parquet.exists():
        return panel, AttachLogEntry(product, attempted=True, attached=False, reason="silver_missing")
    try:
        from quantagent.data.fundamental import fundamental_ranker_for_overlay
    except ImportError:
        return panel, AttachLogEntry(product, attempted=True, attached=False, reason="helper_missing")
    silver = _read_silver(parquet)
    if silver is None:
        return panel, AttachLogEntry(product, attempted=True, attached=False, reason="silver_read_failed")
    gated = fundamental_ranker_for_overlay(silver, manifest if manifest.exists() else None)
    if gated is None:
        return panel, AttachLogEntry(product, attempted=True, attached=False, reason="gate_closed")
    if not {"trade_date", "symbol"}.issubset(panel.columns):
        return panel, AttachLogEntry(product, attempted=True, attached=False, reason="panel_keys_missing")
    before = panel.copy()
    left = panel[["trade_date", "symbol"]].assign(__idx=panel.index)
    left["trade_date"] = pd.to_datetime(left["trade_date"])
    left = left.sort_values(["trade_date", "symbol"]).reset_index(drop=True)
    right_cols = ["symbol", "as_of_date", "composite_rank", "valuation_score", "quality_score", "growth_score"]
    available = [c for c in right_cols if c in gated.columns]
    right = gated[available].copy()
    right["as_of_date"] = pd.to_datetime(right["as_of_date"])
    right = right.dropna(subset=["as_of_date"]).sort_values(["as_of_date", "symbol"])
    merged = pd.merge_asof(
        left, right,
        left_on="trade_date", right_on="as_of_date",
        by="symbol", direction="backward",
    )
    out = panel.copy()
    for col in ("composite_rank", "valuation_score", "quality_score", "growth_score"):
        if col in merged.columns:
            target_col = f"fundamental_{col}"
            series = pd.Series(float("nan"), index=panel.index, dtype=float)
            series.loc[merged["__idx"].values] = pd.to_numeric(merged[col], errors="coerce").values
            out[target_col] = series
    return out, AttachLogEntry(
        product, attempted=True, attached=True, reason="ok",
        columns_added=_diff_cols(before, out),
    )


def _attach_policy_events(panel: pd.DataFrame, cfg: V11IntegrationConfig) -> tuple[pd.DataFrame, AttachLogEntry]:
    product = "policy_events"
    parquet = cfg.silver_dir / "policy_events" / "policy_events.parquet"
    manifest = cfg.manifests_dir / "policy_events.json"
    if not parquet.exists():
        return panel, AttachLogEntry(product, attempted=True, attached=False, reason="silver_missing")
    try:
        from quantagent.data.policy import (
            apply_policy_lag_features,
            estimate_policy_lag,
            policy_events_for_features,
        )
    except ImportError:
        return panel, AttachLogEntry(product, attempted=True, attached=False, reason="helper_missing")
    silver = _read_silver(parquet)
    if silver is None:
        return panel, AttachLogEntry(product, attempted=True, attached=False, reason="silver_read_failed")
    gated = policy_events_for_features(silver, manifest if manifest.exists() else None)
    if gated is None:
        return panel, AttachLogEntry(product, attempted=True, attached=False, reason="gate_closed")
    before = panel.copy()
    # Without a sector-return panel we can't run the lag estimator; use a
    # safe default lag of 5 BD for every theme.
    out = apply_policy_lag_features(panel, gated, lag_table=None, default_lag=5)
    return out, AttachLogEntry(
        product, attempted=True, attached=True, reason="ok",
        columns_added=_diff_cols(before, out),
    )


def _attach_bond_flows(panel: pd.DataFrame, cfg: V11IntegrationConfig) -> tuple[pd.DataFrame, AttachLogEntry]:
    product = "bond_flows"
    parquet = cfg.silver_dir / "bond_flows" / "bond_flows.parquet"
    manifest = cfg.manifests_dir / "bond_flows.json"
    if not parquet.exists():
        return panel, AttachLogEntry(product, attempted=True, attached=False, reason="silver_missing")
    try:
        from quantagent.data.bond import apply_bond_flow_features, bond_flows_for_features
    except ImportError:
        return panel, AttachLogEntry(product, attempted=True, attached=False, reason="helper_missing")
    silver = _read_silver(parquet)
    if silver is None:
        return panel, AttachLogEntry(product, attempted=True, attached=False, reason="silver_read_failed")
    gated = bond_flows_for_features(silver, manifest if manifest.exists() else None)
    if gated is None:
        return panel, AttachLogEntry(product, attempted=True, attached=False, reason="gate_closed")
    before = panel.copy()
    out = apply_bond_flow_features(panel, gated)
    return out, AttachLogEntry(
        product, attempted=True, attached=True, reason="ok",
        columns_added=_diff_cols(before, out),
    )


def _attach_state_team(panel: pd.DataFrame, cfg: V11IntegrationConfig) -> tuple[pd.DataFrame, AttachLogEntry]:
    product = "state_team_inference"
    parquet = cfg.silver_dir / "state_team_inference" / "state_team_inference.parquet"
    manifest = cfg.manifests_dir / "state_team_inference.json"
    if not parquet.exists():
        return panel, AttachLogEntry(product, attempted=True, attached=False, reason="silver_missing")
    try:
        from quantagent.data.state_team import (
            apply_state_team_features,
            state_team_inference_for_features,
        )
    except ImportError:
        return panel, AttachLogEntry(product, attempted=True, attached=False, reason="helper_missing")
    silver = _read_silver(parquet)
    if silver is None:
        return panel, AttachLogEntry(product, attempted=True, attached=False, reason="silver_read_failed")
    gated = state_team_inference_for_features(silver, manifest if manifest.exists() else None)
    if gated is None:
        return panel, AttachLogEntry(product, attempted=True, attached=False, reason="gate_closed")
    before = panel.copy()
    out = apply_state_team_features(panel, gated)
    return out, AttachLogEntry(
        product, attempted=True, attached=True, reason="ok",
        columns_added=_diff_cols(before, out),
    )


def _attach_broker_reports(panel: pd.DataFrame, cfg: V11IntegrationConfig) -> tuple[pd.DataFrame, AttachLogEntry]:
    product = "broker_reports"
    parquet = cfg.silver_dir / "broker_reports" / "broker_reports.parquet"
    manifest = cfg.manifests_dir / "broker_reports.json"
    if not parquet.exists():
        return panel, AttachLogEntry(product, attempted=True, attached=False, reason="silver_missing")
    try:
        from quantagent.data.broker import (
            apply_broker_report_features,
            broker_reports_for_features,
        )
    except ImportError:
        return panel, AttachLogEntry(product, attempted=True, attached=False, reason="helper_missing")
    silver = _read_silver(parquet)
    if silver is None:
        return panel, AttachLogEntry(product, attempted=True, attached=False, reason="silver_read_failed")
    gated = broker_reports_for_features(silver, manifest if manifest.exists() else None)
    if gated is None:
        return panel, AttachLogEntry(product, attempted=True, attached=False, reason="gate_closed")
    before = panel.copy()
    out = apply_broker_report_features(panel, gated)
    return out, AttachLogEntry(
        product, attempted=True, attached=True, reason="ok",
        columns_added=_diff_cols(before, out),
    )


# Ordered pipeline — Stage 2 first (sector/ST establish identity columns),
# Stage 4/5 next (which depend on those identities).
PIPELINE_ORDER: tuple[tuple[str, Any], ...] = (
    ("sector_map", _attach_sector_map),
    ("st_flags", _attach_st_flags),
    ("sector_pool", _attach_sector_pool),
    ("fundamental_ranker", _attach_fundamental_ranker),
    ("policy_events", _attach_policy_events),
    ("bond_flows", _attach_bond_flows),
    ("state_team_inference", _attach_state_team),
    ("broker_reports", _attach_broker_reports),
)


def attach_v11_features(
    panel: pd.DataFrame,
    config: V11IntegrationConfig | None = None,
) -> V11IntegrationResult:
    """Walk every Stage 2-5 product and attach features to ``panel``.

    Returns the augmented panel + a per-product attach log. The log
    captures whether each product was attempted, whether it attached,
    why it skipped (gate_closed / silver_missing / etc.), and which
    columns were added.
    """
    cfg = config or V11IntegrationConfig()
    if panel is None or panel.empty:
        return V11IntegrationResult(panel=panel if panel is not None else pd.DataFrame())
    out = panel.copy()
    log: list[AttachLogEntry] = []
    for product, attach_fn in PIPELINE_ORDER:
        out, entry = attach_fn(out, cfg)
        log.append(entry)
    return V11IntegrationResult(panel=out, attach_log=log)


def write_v11_attach_log(
    result: V11IntegrationResult,
    output_dir: str | Path,
) -> Path:
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)
    log_path = out / "v11_attach_log.json"
    log_path.write_text(json.dumps(result.to_dict(), indent=2, default=str), encoding="utf-8")
    return log_path
