"""Daily data-layer health check for QuantAgent V7.

Reads manifests from all five data products and emits a consolidated
OK / WARN / FAIL report.  The module is side-effect-free: it writes
nothing unless ``DailyHealthChecker.run()`` is called explicitly.

Exit-code contract (used by systemd OnFailure):
  0  → all products OK
  1  → at least one WARN, none FAIL
  2  → at least one FAIL
"""

from __future__ import annotations

import json
import sys
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

# ---------------------------------------------------------------------------
# Status levels
# ---------------------------------------------------------------------------

OK = "OK"
WARN = "WARN"
FAIL = "FAIL"

_LEVEL_ORDER = {OK: 0, WARN: 1, FAIL: 2}


def _max_status(*statuses: str) -> str:
    return max(statuses, key=lambda s: _LEVEL_ORDER.get(s, 0))


# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class DailyHealthConfig:
    lake_root: str | Path = "runtime/data/v7"
    output_root: str | Path = "runtime/reports/daily_health"

    @property
    def manifests_dir(self) -> Path:
        return Path(self.lake_root) / "manifests"

    @property
    def reports_dir(self) -> Path:
        return Path(self.output_root)


# ---------------------------------------------------------------------------
# Per-product check result
# ---------------------------------------------------------------------------

@dataclass
class ProductHealth:
    product: str
    status: str
    gate_open: bool | None
    gate_key: str
    reason: str
    manifest_path: str
    manifest_found: bool
    details: dict[str, Any] = field(default_factory=dict)


# ---------------------------------------------------------------------------
# Individual product checkers
# ---------------------------------------------------------------------------

def _read_manifest(path: Path) -> dict[str, Any] | None:
    if not path.exists():
        return None
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError):
        return None


def _gate_from_manifest(payload: dict[str, Any], gate_key: str) -> dict[str, Any]:
    extra = payload.get("extra") or {}
    coverage = extra.get("coverage_report") or {}
    return coverage.get("gate") or {}


def _check_market_features(config: DailyHealthConfig) -> ProductHealth:
    """market_features has no manifest gate; check that the parquet exists."""
    product = "market_features"
    parquet = Path(config.lake_root) / "silver" / "market_panel" / "market_features.parquet"
    manifest_path = str(config.manifests_dir / "market_features.json")
    payload = _read_manifest(config.manifests_dir / "market_features.json")

    if payload is not None:
        # Manifest exists → check for a gate key if present
        gate = _gate_from_manifest(payload, "market_features_usable")
        gate_open: bool | None = gate.get("market_features_usable", None)
        if gate_open is False:
            return ProductHealth(
                product=product,
                status=FAIL,
                gate_open=False,
                gate_key="market_features_usable",
                reason=gate.get("reason", "gate_closed"),
                manifest_path=manifest_path,
                manifest_found=True,
            )
        # Manifest present + gate either open or not present in manifest
        if parquet.exists():
            return ProductHealth(
                product=product,
                status=OK,
                gate_open=True,
                gate_key="market_features_usable",
                reason="passed",
                manifest_path=manifest_path,
                manifest_found=True,
            )
        return ProductHealth(
            product=product,
            status=WARN,
            gate_open=None,
            gate_key="market_features_usable",
            reason="manifest_present_but_parquet_missing",
            manifest_path=manifest_path,
            manifest_found=True,
        )

    # No manifest yet — check raw parquet
    if parquet.exists():
        return ProductHealth(
            product=product,
            status=WARN,
            gate_open=None,
            gate_key="market_features_usable",
            reason="manifest_missing_parquet_present",
            manifest_path=manifest_path,
            manifest_found=False,
        )
    return ProductHealth(
        product=product,
        status=FAIL,
        gate_open=None,
        gate_key="market_features_usable",
        reason="manifest_and_parquet_missing",
        manifest_path=manifest_path,
        manifest_found=False,
    )


def _check_gated_product(
    config: DailyHealthConfig,
    product: str,
    gate_key: str,
    silver_subpath: str,
) -> ProductHealth:
    """Generic checker for sector_map, st_flags, sector_pool, fundamental_ranker."""
    manifest_path = config.manifests_dir / f"{product}.json"
    payload = _read_manifest(manifest_path)

    if payload is None:
        parquet = Path(config.lake_root) / "silver" / silver_subpath
        status = WARN if parquet.exists() else FAIL
        reason = "manifest_missing_parquet_present" if parquet.exists() else "manifest_and_parquet_missing"
        return ProductHealth(
            product=product,
            status=status,
            gate_open=None,
            gate_key=gate_key,
            reason=reason,
            manifest_path=str(manifest_path),
            manifest_found=False,
        )

    gate = _gate_from_manifest(payload, gate_key)
    gate_open = gate.get(gate_key)
    reason = gate.get("reason", "unknown")

    if gate_open is True:
        status = OK
    elif gate_open is False:
        status = FAIL
    else:
        # Gate key absent — manifest present but no usable flag
        status = WARN
        reason = "gate_key_absent_in_manifest"

    extra = payload.get("extra") or {}
    coverage = extra.get("coverage_report") or {}

    return ProductHealth(
        product=product,
        status=status,
        gate_open=gate_open,
        gate_key=gate_key,
        reason=reason,
        manifest_path=str(manifest_path),
        manifest_found=True,
        details={k: v for k, v in coverage.items() if k != "gate"},
    )


# ---------------------------------------------------------------------------
# Consolidated report
# ---------------------------------------------------------------------------

@dataclass
class DailyHealthReport:
    generated_at: str
    aggregate_status: str
    products: list[ProductHealth]
    lake_root: str
    output_root: str

    def to_dict(self) -> dict[str, Any]:
        return {
            "generated_at": self.generated_at,
            "aggregate_status": self.aggregate_status,
            "lake_root": self.lake_root,
            "products": [
                {
                    "product": p.product,
                    "status": p.status,
                    "gate_open": p.gate_open,
                    "gate_key": p.gate_key,
                    "reason": p.reason,
                    "manifest_found": p.manifest_found,
                    "manifest_path": p.manifest_path,
                    "details": p.details,
                }
                for p in self.products
            ],
        }

    def to_markdown(self) -> str:
        lines: list[str] = []
        lines.append("# QuantAgent V7 — Daily Data Layer Health")
        lines.append(f"\n**Generated:** {self.generated_at}")
        lines.append(f"**Aggregate status:** {self.aggregate_status}")
        lines.append(f"**Lake root:** `{self.lake_root}`\n")
        lines.append("| Product | Status | Gate | Reason |")
        lines.append("|---|---|---|---|")
        for p in self.products:
            gate_str = "open" if p.gate_open is True else ("closed" if p.gate_open is False else "n/a")
            lines.append(f"| {p.product} | {p.status} | {gate_str} | {p.reason} |")
        lines.append("")
        # Details section for WARN / FAIL entries
        problem_products = [p for p in self.products if p.status != OK]
        if problem_products:
            lines.append("## Issues")
            for p in problem_products:
                lines.append(f"\n### {p.product} — {p.status}")
                lines.append(f"- Gate key: `{p.gate_key}`")
                lines.append(f"- Reason: `{p.reason}`")
                lines.append(f"- Manifest path: `{p.manifest_path}`")
                lines.append(f"- Manifest found: {p.manifest_found}")
                if p.details:
                    lines.append(f"- Coverage details: {json.dumps(p.details)}")
        else:
            lines.append("All data products healthy.")
        return "\n".join(lines)

    @property
    def exit_code(self) -> int:
        return _LEVEL_ORDER.get(self.aggregate_status, 2)


# ---------------------------------------------------------------------------
# Main checker
# ---------------------------------------------------------------------------

PRODUCTS_SPEC: list[tuple[str, str, str]] = [
    # (product_name, gate_key, silver_relative_subpath/file)
    # Note: sector_map and st_flags use unprefixed gate keys to match the
    # actual builder output in data/sector/{sector_mapping,st_history}.py
    ("sector_map", "sector_usable_for_optimization", "sector_map/sector_map.parquet"),
    ("st_flags", "st_usable_for_risk_filter", "st_flags/st_flags.parquet"),
    ("sector_pool", "sector_pool_usable_for_overlay", "sector_pool/sector_pool.parquet"),
    ("fundamental_ranker", "fundamental_ranker_usable_for_overlay", "fundamental_ranker/fundamental_ranker.parquet"),
    ("policy_events", "policy_events_usable_for_features", "policy_events/policy_events.parquet"),
    ("bond_flows", "bond_flows_usable_for_features", "bond_flows/bond_flows.parquet"),
    ("state_team_inference", "state_team_inference_usable_for_features", "state_team_inference/state_team_inference.parquet"),
    ("broker_reports", "broker_reports_usable_for_features", "broker_reports/broker_reports.parquet"),
]


class DailyHealthChecker:
    def __init__(self, config: DailyHealthConfig | None = None) -> None:
        self.config = config or DailyHealthConfig()

    def check(self) -> DailyHealthReport:
        products: list[ProductHealth] = []
        products.append(_check_market_features(self.config))
        for product, gate_key, silver_sub in PRODUCTS_SPEC:
            products.append(
                _check_gated_product(self.config, product, gate_key, silver_sub)
            )
        aggregate = max(
            (_LEVEL_ORDER.get(p.status, 0) for p in products),
            default=0,
        )
        agg_str = [OK, WARN, FAIL][min(aggregate, 2)]
        return DailyHealthReport(
            generated_at=datetime.now(timezone.utc).isoformat(),
            aggregate_status=agg_str,
            products=products,
            lake_root=str(self.config.lake_root),
            output_root=str(self.config.output_root),
        )

    def run(self, *, write: bool = True) -> DailyHealthReport:
        report = self.check()
        if write:
            out = self.config.reports_dir
            out.mkdir(parents=True, exist_ok=True)
            (out / "health_report.json").write_text(
                json.dumps(report.to_dict(), indent=2), encoding="utf-8"
            )
            (out / "health_report.md").write_text(
                report.to_markdown(), encoding="utf-8"
            )
        return report


# ---------------------------------------------------------------------------
# CLI entry point (plain __main__, no typer dependency needed for a health check)
# ---------------------------------------------------------------------------

def main() -> None:
    import argparse

    parser = argparse.ArgumentParser(description="QuantAgent V7 daily data-layer health check")
    parser.add_argument("--lake-root", default="runtime/data/v7", help="Data lake root directory")
    parser.add_argument("--output-root", default="runtime/reports/daily_health", help="Where to write reports")
    parser.add_argument("--no-write", action="store_true", help="Print report but do not write files")
    args = parser.parse_args()

    config = DailyHealthConfig(lake_root=args.lake_root, output_root=args.output_root)
    checker = DailyHealthChecker(config)
    report = checker.run(write=not args.no_write)

    print(report.to_markdown())
    if not args.no_write:
        print(f"\nWrote reports to: {config.reports_dir}")
    sys.exit(report.exit_code)


if __name__ == "__main__":  # pragma: no cover
    main()
