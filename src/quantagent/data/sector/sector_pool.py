"""Sector pool builder — IC-driven tiering of sector_level_1 buckets.

Inputs
------
* Per-sector IC table produced by ``quantagent.diagnostics.stratified_ic``
  (one row per ``(bucket, horizon)`` with columns ``ic_mean``, ``ic_std``,
  ``ic_ir``, ``n_dates``, ``n_symbols``).
* Optional sector_map (Stage 2.2 silver/sector_map.parquet) used only to
  report which sectors in the IC table actually have coverage and to
  count how many symbols are tagged to each tier.

Output
------
A silver-layer data product, **not** a portfolio signal:
``silver/sector_pool/sector_pool.parquet`` — one row per sector_level_1,
with ``pool_tier`` in ``{core, watch, short_term, excluded}``, the
underlying IC stats, and the rule that triggered the assignment.

Tiering rules at the chosen reference horizon (default 20d):

* **excluded**: ``ic_mean <= 0`` OR sample-size below
  ``min_dates`` / ``min_symbols``. The model has no demonstrated edge
  in this sector at this horizon, or we don't have enough OOS data to
  judge.
* **core**: ``ic_mean`` ranks in the top ``core_quantile`` (default top
  0.30) AND ``ic_ir >= core_ir_threshold`` (default 0.30). Both raw
  edge and stability are required — high IC with high vol does not
  count as core.
* **short_term**: positive ``ic_mean`` but ``ic_ir`` below
  ``core_ir_threshold`` and/or ``ic_std >= short_term_vol_threshold``.
  The signal exists but is unstable; treat it as tactical with a
  reduced weight rather than a structural overweight.
* **watch**: everything left — positive ic_mean, ic_ir below the core
  cutoff but above ``watch_ir_threshold``. Reasonable signal, doesn't
  earn a structural overweight yet.

This module never emits target weights and never changes optimiser
output. It is consumed in two ways downstream:

1. As a diagnostic report (which sectors does the model actually
   understand?).
2. As an input to a future weight overlay that the caller is
   responsible for wiring; the helper ``sector_pool_for_weight_overlay``
   returns the tier-to-weight scaling table only when a manifest gate
   is open, matching the Stage 2.3 audit-only contract.
"""

from __future__ import annotations

from dataclasses import dataclass, field
import json
from pathlib import Path
from typing import Iterable

import pandas as pd

from quantagent.data.manifest import build_manifest_for_frame, utc_now_iso


SECTOR_POOL_REQUIRED_COLUMNS: tuple[str, ...] = (
    "sector_level_1",
    "pool_tier",
    "horizon",
    "ic_mean",
    "ic_std",
    "ic_ir",
    "n_dates",
    "n_symbols",
    "tier_reason",
    "generated_at",
    "source_version",
)

VALID_POOL_TIERS: tuple[str, ...] = ("core", "watch", "short_term", "excluded")

DEFAULT_TIER_WEIGHTS: dict[str, float] = {
    "core": 1.00,
    "watch": 0.70,
    "short_term": 0.30,
    "excluded": 0.00,
}


@dataclass(frozen=True)
class SectorPoolConfig:
    """Tiering parameters.

    The defaults assume a 20d primary horizon and 12-fold OOS coverage,
    matching the v9/v10 stratified IC report. They are deliberately
    conservative so that "core" status reflects real out-of-sample
    edge rather than top-of-pack noise.
    """

    reference_horizon: int = 20
    min_dates: int = 60
    min_symbols: int = 20
    core_quantile: float = 0.30      # top 30% of ic_mean → eligible for core
    core_ir_threshold: float = 0.30  # required IR floor for core
    watch_ir_threshold: float = 0.10  # below this falls into short_term
    short_term_vol_threshold: float = 0.10  # ic_std above → short_term flavour
    source_version: str = "unknown"
    output_root: str | Path = "runtime/data/v7"


@dataclass(frozen=True)
class SectorPoolResult:
    frame: pd.DataFrame
    coverage: dict[str, object] = field(default_factory=dict)
    validation: dict[str, object] = field(default_factory=dict)
    output_paths: dict[str, str] = field(default_factory=dict)
    tier_distribution: pd.DataFrame = field(default_factory=pd.DataFrame)


def _coerce_ic_table(frame: pd.DataFrame) -> pd.DataFrame:
    """Accept ``compute_stratified_ic.by_axis['sector_level_1']`` shape.

    The IC table from stratified_ic carries the sector name in a column
    called ``bucket``. Callers may also pass a frame that already uses
    ``sector_level_1``; the helper accepts either spelling. Any row
    with a non-string or empty bucket is dropped.
    """
    if frame is None or frame.empty:
        return pd.DataFrame(
            columns=["sector_level_1", "horizon", "ic_mean", "ic_std", "ic_ir", "n_dates", "n_symbols"]
        )
    data = frame.copy()
    if "sector_level_1" not in data.columns and "bucket" in data.columns:
        data = data.rename(columns={"bucket": "sector_level_1"})
    if "sector_level_1" not in data.columns:
        raise ValueError("IC table must include either 'sector_level_1' or 'bucket' column")
    for col, default in (("horizon", 20), ("ic_mean", 0.0), ("ic_std", 0.0), ("ic_ir", 0.0), ("n_dates", 0), ("n_symbols", 0)):
        if col not in data.columns:
            data[col] = default
    data["sector_level_1"] = data["sector_level_1"].astype(str).str.strip()
    data = data[data["sector_level_1"].ne("") & data["sector_level_1"].ne("UNKNOWN")].copy()
    data["horizon"] = pd.to_numeric(data["horizon"], errors="coerce").fillna(0).astype(int)
    data["ic_mean"] = pd.to_numeric(data["ic_mean"], errors="coerce").fillna(0.0)
    data["ic_std"] = pd.to_numeric(data["ic_std"], errors="coerce").fillna(0.0)
    data["ic_ir"] = pd.to_numeric(data["ic_ir"], errors="coerce").fillna(0.0)
    data["n_dates"] = pd.to_numeric(data["n_dates"], errors="coerce").fillna(0).astype(int)
    data["n_symbols"] = pd.to_numeric(data["n_symbols"], errors="coerce").fillna(0).astype(int)
    return data.reset_index(drop=True)


def _select_reference_horizon(ic_table: pd.DataFrame, horizon: int) -> pd.DataFrame:
    """Pick the row for ``horizon`` per sector, or the closest available.

    Walk-forward IC reports do not always carry every horizon (a fold
    can drop one if labels are missing). Falling back to "closest
    horizon" is preferred over silently excluding a sector — the
    reference is still a single H per sector so the tier comparison
    is fair, and the chosen horizon is recorded per row for audit.
    """
    if ic_table.empty:
        return ic_table
    chosen_rows: list[dict[str, object]] = []
    for sector, group in ic_table.groupby("sector_level_1"):
        if group.empty:
            continue
        primary = group[group["horizon"] == horizon]
        if not primary.empty:
            best = primary.sort_values("n_dates", ascending=False).iloc[0]
        else:
            group = group.assign(_distance=(group["horizon"] - horizon).abs())
            best = group.sort_values(["_distance", "n_dates"], ascending=[True, False]).iloc[0]
        chosen_rows.append(
            {
                "sector_level_1": str(sector),
                "horizon": int(best["horizon"]),
                "ic_mean": float(best["ic_mean"]),
                "ic_std": float(best["ic_std"]) if pd.notna(best["ic_std"]) else 0.0,
                "ic_ir": float(best["ic_ir"]) if pd.notna(best["ic_ir"]) else 0.0,
                "n_dates": int(best["n_dates"]),
                "n_symbols": int(best["n_symbols"]),
            }
        )
    return pd.DataFrame(chosen_rows)


def _assign_tier(row: pd.Series, config: SectorPoolConfig, core_cutoff_ic_mean: float) -> tuple[str, str]:
    """Return ``(tier, reason)`` for one sector at the chosen horizon."""
    if row["n_dates"] < config.min_dates or row["n_symbols"] < config.min_symbols:
        return "excluded", "insufficient_sample"
    if row["ic_mean"] <= 0:
        return "excluded", "non_positive_ic"
    is_top_quantile = row["ic_mean"] >= core_cutoff_ic_mean
    if is_top_quantile and row["ic_ir"] >= config.core_ir_threshold:
        return "core", "top_quantile_and_stable"
    if row["ic_ir"] < config.watch_ir_threshold or row["ic_std"] >= config.short_term_vol_threshold:
        return "short_term", "positive_ic_unstable_or_volatile"
    return "watch", "positive_ic_below_core_cutoff"


def build_sector_pool(
    ic_table: pd.DataFrame,
    *,
    config: SectorPoolConfig | None = None,
    generated_at: str | None = None,
) -> SectorPoolResult:
    """Apply tier rules and return the full result bundle."""
    cfg = config or SectorPoolConfig()
    data = _coerce_ic_table(ic_table)
    reference = _select_reference_horizon(data, cfg.reference_horizon)
    if reference.empty:
        empty = pd.DataFrame(columns=SECTOR_POOL_REQUIRED_COLUMNS)
        return SectorPoolResult(
            frame=empty,
            coverage={"total_sectors": 0, "tier_counts": {tier: 0 for tier in VALID_POOL_TIERS}, "status": "empty_input"},
            validation={"status": "empty_input", "row_count": 0},
            tier_distribution=pd.DataFrame(columns=["pool_tier", "sector_count"]),
        )

    eligible = reference[(reference["n_dates"] >= cfg.min_dates) & (reference["n_symbols"] >= cfg.min_symbols)]
    if not eligible.empty:
        core_cutoff = float(eligible["ic_mean"].quantile(1.0 - cfg.core_quantile))
    else:
        core_cutoff = float("inf")  # nothing eligible → no core, everything excluded

    rows: list[dict[str, object]] = []
    generated = generated_at or utc_now_iso()
    for _, row in reference.sort_values("sector_level_1").iterrows():
        tier, reason = _assign_tier(row, cfg, core_cutoff)
        rows.append(
            {
                "sector_level_1": str(row["sector_level_1"]),
                "pool_tier": tier,
                "horizon": int(row["horizon"]),
                "ic_mean": float(round(row["ic_mean"], 6)),
                "ic_std": float(round(row["ic_std"], 6)),
                "ic_ir": float(round(row["ic_ir"], 6)),
                "n_dates": int(row["n_dates"]),
                "n_symbols": int(row["n_symbols"]),
                "tier_reason": reason,
                "generated_at": generated,
                "source_version": cfg.source_version,
            }
        )
    frame = pd.DataFrame(rows, columns=SECTOR_POOL_REQUIRED_COLUMNS)

    tier_distribution = (
        frame.groupby("pool_tier", dropna=False)
        .agg(sector_count=("sector_level_1", "nunique"))
        .reindex(VALID_POOL_TIERS, fill_value=0)
        .reset_index()
    )
    coverage = {
        "total_sectors": int(len(frame)),
        "tier_counts": {row["pool_tier"]: int(row["sector_count"]) for _, row in tier_distribution.iterrows()},
        "core_cutoff_ic_mean": core_cutoff if core_cutoff != float("inf") else None,
        "reference_horizon": int(cfg.reference_horizon),
        "thresholds": {
            "min_dates": int(cfg.min_dates),
            "min_symbols": int(cfg.min_symbols),
            "core_quantile": float(cfg.core_quantile),
            "core_ir_threshold": float(cfg.core_ir_threshold),
            "watch_ir_threshold": float(cfg.watch_ir_threshold),
            "short_term_vol_threshold": float(cfg.short_term_vol_threshold),
        },
        "status": "passed",
    }
    validation = {
        "status": "passed" if not frame["pool_tier"].isna().any() else "failed",
        "row_count": int(len(frame)),
        "missing_columns": [c for c in SECTOR_POOL_REQUIRED_COLUMNS if c not in frame.columns],
        "invalid_tiers": sorted(set(frame["pool_tier"]) - set(VALID_POOL_TIERS)),
    }
    if validation["missing_columns"] or validation["invalid_tiers"]:
        validation["status"] = "failed"
    return SectorPoolResult(
        frame=frame,
        coverage=coverage,
        validation=validation,
        tier_distribution=tier_distribution,
    )


def sector_pool_for_weight_overlay(
    sector_pool: pd.DataFrame | None,
    manifest_path: str | Path | None = None,
    *,
    tier_weights: dict[str, float] | None = None,
) -> pd.DataFrame | None:
    """Return a ``(sector_level_1 → overlay_weight)`` table or ``None``.

    Matches the audit-only contract from
    ``quantagent.diagnostics.sector_audit``: the helper returns ``None``
    unless a manifest is supplied AND its gate is open. Callers must
    treat ``None`` as "no overlay — keep optimiser weights as-is"
    instead of crashing or silently using stale tiers.
    """
    if sector_pool is None or sector_pool.empty:
        return None
    if manifest_path is None:
        return None
    path = Path(manifest_path)
    if not path.exists():
        return None
    payload = json.loads(path.read_text(encoding="utf-8"))
    extra = payload.get("extra", {}) if isinstance(payload, dict) else {}
    coverage = extra.get("coverage_report", {}) if isinstance(extra, dict) else {}
    gate = coverage.get("gate", {}) if isinstance(coverage, dict) else {}
    if not bool(gate.get("sector_pool_usable_for_overlay", False)):
        return None
    weights = {**DEFAULT_TIER_WEIGHTS, **(tier_weights or {})}
    out = sector_pool[["sector_level_1", "pool_tier"]].copy()
    out["overlay_weight"] = out["pool_tier"].map(weights).astype(float)
    return out


class SectorPoolBuilder:
    """Stateful wrapper that materialises the silver/sector_pool artifact.

    Mirrors ``SectorMapBuilder`` so callers can ``.build()`` then
    ``.write()`` and get parquet + coverage_report.json +
    validation_report.json + tier_distribution.csv + manifest in one
    shot. The gate result is also embedded in the manifest under
    ``extra.coverage_report.gate`` for the gate helper to read.
    """

    def __init__(self, config: SectorPoolConfig | None = None) -> None:
        self.config = config or SectorPoolConfig()

    def build(
        self,
        ic_table: pd.DataFrame,
        *,
        generated_at: str | None = None,
    ) -> SectorPoolResult:
        result = build_sector_pool(ic_table, config=self.config, generated_at=generated_at)
        # Populate a gate verdict on the coverage report so downstream
        # consumers can answer "is this pool usable as an overlay?"
        gate = self._gate(result)
        coverage = dict(result.coverage)
        coverage["gate"] = gate
        return SectorPoolResult(
            frame=result.frame,
            coverage=coverage,
            validation=result.validation,
            tier_distribution=result.tier_distribution,
        )

    def _gate(self, result: SectorPoolResult) -> dict[str, object]:
        tier_counts = result.coverage.get("tier_counts", {}) if isinstance(result.coverage, dict) else {}
        core_count = int(tier_counts.get("core", 0))
        excluded_count = int(tier_counts.get("excluded", 0))
        total = int(result.coverage.get("total_sectors", 0))
        reasons: list[str] = []
        if total == 0:
            reasons.append("no_sectors_in_input")
        if core_count == 0:
            reasons.append("no_core_sector")
        if total and excluded_count / max(total, 1) > 0.70:
            reasons.append("excluded_ratio_above_threshold")
        usable = not reasons
        return {
            "sector_pool_usable_for_diagnostics": True,
            "sector_pool_usable_for_overlay": bool(usable),
            "reason": "passed" if usable else ",".join(reasons),
            "core_sector_count": core_count,
            "excluded_sector_count": excluded_count,
            "total_sectors": total,
        }

    def write(self, result: SectorPoolResult) -> SectorPoolResult:
        root = Path(self.config.output_root)
        out_dir = root / "silver" / "sector_pool"
        out_dir.mkdir(parents=True, exist_ok=True)
        parquet_path = out_dir / "sector_pool.parquet"
        coverage_path = out_dir / "coverage_report.json"
        validation_path = out_dir / "validation_report.json"
        distribution_path = out_dir / "tier_distribution.csv"
        result.frame.to_parquet(parquet_path, index=False)
        coverage_path.write_text(json.dumps(result.coverage, ensure_ascii=False, indent=2, default=str), encoding="utf-8")
        validation_path.write_text(json.dumps(result.validation, ensure_ascii=False, indent=2, default=str), encoding="utf-8")
        result.tier_distribution.to_csv(distribution_path, index=False)
        manifest = build_manifest_for_frame(
            dataset_name="sector_pool",
            vendor="local",
            frame=result.frame,
            output_paths=[parquet_path],
            required_columns=SECTOR_POOL_REQUIRED_COLUMNS,
            extra={
                "coverage_report": result.coverage,
                "validation_report": result.validation,
                "tier_distribution": result.tier_distribution.to_dict("records"),
                "policy": (
                    "diagnostic data product — sector_pool_for_weight_overlay is the "
                    "only sanctioned consumer for any weight-level decision, and "
                    "requires sector_pool_usable_for_overlay=True in the manifest gate"
                ),
            },
        )
        manifest_path = root / "manifests" / "sector_pool.json"
        manifest.write(manifest_path)
        paths = {
            "sector_pool": str(parquet_path),
            "coverage_report": str(coverage_path),
            "validation_report": str(validation_path),
            "tier_distribution": str(distribution_path),
            "manifest": str(manifest_path),
        }
        return SectorPoolResult(
            frame=result.frame,
            coverage=result.coverage,
            validation=result.validation,
            tier_distribution=result.tier_distribution,
            output_paths=paths,
        )


__all__ = [
    "DEFAULT_TIER_WEIGHTS",
    "SECTOR_POOL_REQUIRED_COLUMNS",
    "SectorPoolBuilder",
    "SectorPoolConfig",
    "SectorPoolResult",
    "VALID_POOL_TIERS",
    "build_sector_pool",
    "sector_pool_for_weight_overlay",
]
