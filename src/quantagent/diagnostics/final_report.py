"""Stage 6 — final report aggregator.

Consolidates every artifact the v7 pipeline produces into a single
markdown + JSON report:

* The 6-stage gate compliance table (Stage 2-5 manifests + Stage 3
  hard gate / 10-term loss / sub-models / Stage 4 policy/bond/state-
  team / Stage 5 broker/credibility/decision-chain/post-mortem).
* The latest sleeve-replay summary aggregated to fold-level.
* The v11 integration attach log (which Stage 2-5 features actually
  fed the model).
* Per-trade post-mortem aggregates if available.

This is a pure-Python read-only aggregator — it never writes to the
silver layer, never re-runs training, never modifies the input data.
Run it after a v11 launch (or at any time) to produce a snapshot.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timezone
import json
from pathlib import Path
from typing import Any


# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class FinalReportConfig:
    lake_root: str | Path = "runtime/data/v7"
    models_root: str | Path = "runtime/models"
    reports_root: str | Path = "runtime/reports"
    output_dir: str | Path = "runtime/reports/final_report"

    # Default replay directory to pull the headline number from
    replay_dir_candidates: tuple[str, ...] = (
        "sleeve_replay_v11",
        "sleeve_replay_v10_stage3",
        "sleeve_replay_v10",
    )

    # v11 attach log (produced by run_v11_ensemble.sh step 1)
    integration_audit_path: str = "v7_alpha_v11/integration_audit/v11_attach_log.json"

    # Post-mortem aggregate
    post_mortem_dir: str = "post_mortem"


# ---------------------------------------------------------------------------
# Stage spec — what to look for under manifests/
# ---------------------------------------------------------------------------

STAGE_DATA_PRODUCTS: tuple[tuple[str, str, str], ...] = (
    # (stage, product, gate_key)
    ("2.2", "sector_map", "sector_usable_for_optimization"),
    ("2.2", "st_flags", "st_usable_for_risk_filter"),
    ("2.x", "sector_pool", "sector_pool_usable_for_overlay"),
    ("2.x", "fundamental_ranker", "fundamental_ranker_usable_for_overlay"),
    ("4.1", "policy_events", "policy_events_usable_for_features"),
    ("4.3", "bond_flows", "bond_flows_usable_for_features"),
    ("4.4", "state_team_inference", "state_team_inference_usable_for_features"),
    ("5.1", "broker_reports", "broker_reports_usable_for_features"),
)


# ---------------------------------------------------------------------------
# Data-product gate extraction
# ---------------------------------------------------------------------------

def _read_manifest(path: Path) -> dict[str, Any] | None:
    if not path.exists():
        return None
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError):
        return None


def collect_data_product_gates(config: FinalReportConfig) -> list[dict[str, Any]]:
    manifests_dir = Path(config.lake_root) / "manifests"
    rows: list[dict[str, Any]] = []
    for stage, product, gate_key in STAGE_DATA_PRODUCTS:
        manifest_path = manifests_dir / f"{product}.json"
        payload = _read_manifest(manifest_path)
        if payload is None:
            rows.append(
                {
                    "stage": stage,
                    "product": product,
                    "gate_key": gate_key,
                    "manifest_found": False,
                    "gate_open": None,
                    "reason": "manifest_missing",
                    "n_rows": 0,
                }
            )
            continue
        gate = (
            (payload.get("extra") or {})
            .get("coverage_report", {})
            .get("gate", {})
        )
        rows.append(
            {
                "stage": stage,
                "product": product,
                "gate_key": gate_key,
                "manifest_found": True,
                "gate_open": bool(gate.get(gate_key, False)),
                "reason": str(gate.get("reason", "unknown")),
                "n_rows": int(payload.get("rows", 0)),
            }
        )
    return rows


# ---------------------------------------------------------------------------
# Replay summary
# ---------------------------------------------------------------------------

def load_latest_replay(config: FinalReportConfig) -> dict[str, Any] | None:
    """Return the most-recent available replay summary (per candidate dir order)."""
    reports_root = Path(config.reports_root)
    for candidate in config.replay_dir_candidates:
        path = reports_root / candidate / "replay_summary.json"
        if path.exists():
            try:
                payload = json.loads(path.read_text(encoding="utf-8"))
            except (json.JSONDecodeError, OSError):
                continue
            per_fold = [f for f in payload.get("per_fold", []) if f.get("status") == "ok"]
            if not per_fold:
                continue
            excess = [float(f.get("excess_ann_%", 0.0)) for f in per_fold]
            dd = [float(f.get("max_DD_%", 0.0)) for f in per_fold]
            return {
                "source_dir": candidate,
                "n_folds": len(per_fold),
                "excess_mean_pct": float(sum(excess) / len(excess)),
                "excess_min_pct": float(min(excess)),
                "excess_max_pct": float(max(excess)),
                "max_dd_worst_pct": float(min(dd)),  # most negative
                "max_dd_mean_pct": float(sum(dd) / len(dd)),
                "per_fold": per_fold,
            }
    return None


# ---------------------------------------------------------------------------
# v11 integration audit
# ---------------------------------------------------------------------------

def load_integration_audit(config: FinalReportConfig) -> dict[str, Any] | None:
    path = Path(config.models_root) / config.integration_audit_path
    return _read_manifest(path)


# ---------------------------------------------------------------------------
# Post-mortem aggregate
# ---------------------------------------------------------------------------

def load_post_mortem_summary(config: FinalReportConfig) -> dict[str, Any] | None:
    path = Path(config.reports_root) / config.post_mortem_dir / "aggregate_summary.json"
    return _read_manifest(path)


# ---------------------------------------------------------------------------
# Stage gate verdicts (the timeline gates per stage)
# ---------------------------------------------------------------------------

@dataclass
class StageVerdict:
    stage: str
    target: str
    actual: str
    status: str   # "pass" / "fail" / "deferred"
    notes: str = ""

    def to_dict(self) -> dict[str, Any]:
        return {
            "stage": self.stage,
            "target": self.target,
            "actual": self.actual,
            "status": self.status,
            "notes": self.notes,
        }


def derive_stage_verdicts(
    replay: dict[str, Any] | None,
    data_gates: list[dict[str, Any]],
) -> list[StageVerdict]:
    """Compute per-stage pass/fail verdicts.

    Stage 1: excess >+12% & DD ≤9%
    Stage 2: excess >+14% & DD ≤9%
    Stage 3: excess >+15% & DD ≤8%
    Stage 4: policy-driven fold +10pp (requires real policy data → deferred)
    Stage 5: excess >+16% & DD ≤8%
    Stage 6: all above + 12 fold × 3 seed v11 run
    """
    verdicts: list[StageVerdict] = []
    if replay is None:
        # No replay → every replay-based verdict is "deferred"
        for stage, target in [
            ("1", "mean excess >+12% & worst-fold DD ≤9%"),
            ("2", "mean excess >+14% & worst-fold DD ≤9%"),
            ("3", "mean excess >+15% & worst-fold DD ≤8%"),
            ("5", "mean excess >+16% & worst-fold DD ≤8%"),
            ("6", "v11 ensemble: mean excess >+15% & worst-fold DD ≤8%"),
        ]:
            verdicts.append(StageVerdict(stage=stage, target=target, actual="no_replay_data", status="deferred"))
    else:
        excess = replay["excess_mean_pct"]
        worst_dd = abs(replay["max_dd_worst_pct"])

        def _verdict(stage: str, ex_min: float, dd_max: float, target: str) -> StageVerdict:
            status = "pass" if (excess > ex_min and worst_dd <= dd_max) else "fail"
            actual = f"mean_excess={excess:.2f}%, worst_fold_DD={worst_dd:.2f}%"
            return StageVerdict(stage=stage, target=target, actual=actual, status=status)

        verdicts.append(_verdict("1", 12.0, 9.0, "excess >+12% & DD ≤9%"))
        verdicts.append(_verdict("2", 14.0, 9.0, "excess >+14% & DD ≤9%"))
        verdicts.append(_verdict("3", 15.0, 8.0, "excess >+15% & DD ≤8%"))
        verdicts.append(_verdict("5", 16.0, 8.0, "excess >+16% & DD ≤8%"))
        verdicts.append(_verdict("6", 15.0, 8.0, "v11 ensemble: excess >+15% & DD ≤8%"))

    # Stage 4 — requires policy data + retrained model
    has_policy = any(g["product"] == "policy_events" and g["gate_open"] for g in data_gates)
    if has_policy:
        verdicts.insert(
            -1,
            StageVerdict(
                stage="4",
                target="policy-driven fold +10pp",
                actual="requires policy-fold split + retrained model",
                status="deferred",
                notes="Code complete; gate verification needs Stage 4 feature retraining.",
            ),
        )
    else:
        verdicts.insert(
            -1,
            StageVerdict(
                stage="4",
                target="policy-driven fold +10pp",
                actual="policy_events gate closed",
                status="deferred",
                notes="Awaiting real policy data ingestion.",
            ),
        )

    return sorted(verdicts, key=lambda v: v.stage)


# ---------------------------------------------------------------------------
# Markdown render
# ---------------------------------------------------------------------------

def _render_markdown(
    *,
    generated_at: str,
    data_gates: list[dict[str, Any]],
    replay: dict[str, Any] | None,
    integration_audit: dict[str, Any] | None,
    post_mortem: dict[str, Any] | None,
    verdicts: list[StageVerdict],
) -> str:
    lines: list[str] = []
    lines.append("# QuantAgent V7 — Final Report")
    lines.append(f"\n**Generated:** {generated_at}\n")

    # Stage verdicts table
    lines.append("## Stage gate verdicts\n")
    lines.append("| Stage | Target | Actual | Status |")
    lines.append("|---|---|---|---|")
    for v in verdicts:
        emoji = {"pass": "PASS", "fail": "FAIL", "deferred": "DEFERRED"}[v.status]
        lines.append(f"| {v.stage} | {v.target} | {v.actual} | {emoji} |")
    lines.append("")

    # Data product gates
    lines.append("## Data layer manifest gates\n")
    lines.append("| Stage | Product | Manifest | Gate | Rows | Reason |")
    lines.append("|---|---|---|---|---|---|")
    for g in data_gates:
        gate_str = "open" if g["gate_open"] is True else "closed" if g["gate_open"] is False else "n/a"
        manifest_str = "yes" if g["manifest_found"] else "no"
        lines.append(
            f"| {g['stage']} | {g['product']} | {manifest_str} | {gate_str} | "
            f"{g['n_rows']} | {g['reason']} |"
        )
    lines.append("")

    # Replay summary
    if replay is not None:
        lines.append("## Latest replay snapshot\n")
        lines.append(f"- Source: `runtime/reports/{replay['source_dir']}/`")
        lines.append(f"- Folds OK: {replay['n_folds']}")
        lines.append(f"- Excess (mean / min / max): "
                     f"{replay['excess_mean_pct']:.2f}% / "
                     f"{replay['excess_min_pct']:.2f}% / "
                     f"{replay['excess_max_pct']:.2f}%")
        lines.append(f"- Max DD (worst / mean): "
                     f"{replay['max_dd_worst_pct']:.2f}% / "
                     f"{replay['max_dd_mean_pct']:.2f}%")
        lines.append("")
        # Per-fold table (top 5 only to keep the report readable)
        lines.append("### Per-fold excerpt (worst 5 folds by excess)\n")
        sorted_folds = sorted(replay["per_fold"], key=lambda f: f.get("excess_ann_%", 0))[:5]
        lines.append("| Fold | excess_ann_% | max_DD_% | sharpe |")
        lines.append("|---|---|---|---|")
        for f in sorted_folds:
            lines.append(
                f"| {f.get('fold')} | {f.get('excess_ann_%')} | "
                f"{f.get('max_DD_%')} | {f.get('sharpe')} |"
            )
        lines.append("")
    else:
        lines.append("## Latest replay snapshot\n_no replay summary found_\n")

    # Integration attach log
    if integration_audit is not None:
        lines.append("## v11 integration attach log\n")
        lines.append(f"- Panel rows: {integration_audit.get('n_rows', 0)}")
        lines.append(
            f"- Features attached: {', '.join(integration_audit.get('features_attached', []) or ['none'])}"
        )
        lines.append(
            f"- Features skipped: {', '.join(integration_audit.get('features_skipped', []) or ['none'])}"
        )
        attach_log = integration_audit.get("attach_log", []) or []
        if attach_log:
            lines.append("\n| Product | Attached | Reason | Columns added |")
            lines.append("|---|---|---|---|")
            for e in attach_log:
                cols = ", ".join(e.get("columns_added", []) or [])
                lines.append(
                    f"| {e.get('product')} | {e.get('attached')} | "
                    f"{e.get('reason')} | {cols or '(none)'} |"
                )
        lines.append("")
    else:
        lines.append("## v11 integration attach log\n_no attach log found_\n")

    # Post-mortem
    if post_mortem is not None:
        lines.append("## Per-trade post-mortem aggregate\n")
        lines.append(f"- Trades: {post_mortem.get('n_trades', 0)}")
        lines.append(f"- Realised win rate: {post_mortem.get('win_rate_realized', 0.0):.3f}")
        lines.append(f"- Excess win rate: {post_mortem.get('win_rate_excess', 0.0):.3f}")
        lines.append(f"- Mean realised P&L: {post_mortem.get('mean_realized_pnl_pct', 0.0):.4f}")
        lines.append(f"- Mean excess P&L: {post_mortem.get('mean_excess_pct', 0.0):.4f}")
        lines.append("")
    else:
        lines.append("## Per-trade post-mortem aggregate\n_no post-mortem aggregate found_\n")

    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Top-level entry
# ---------------------------------------------------------------------------

@dataclass
class FinalReport:
    generated_at: str
    data_gates: list[dict[str, Any]]
    replay: dict[str, Any] | None
    integration_audit: dict[str, Any] | None
    post_mortem: dict[str, Any] | None
    verdicts: list[StageVerdict] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return {
            "generated_at": self.generated_at,
            "data_gates": self.data_gates,
            "replay": self.replay,
            "integration_audit": self.integration_audit,
            "post_mortem": self.post_mortem,
            "stage_verdicts": [v.to_dict() for v in self.verdicts],
        }

    def to_markdown(self) -> str:
        return _render_markdown(
            generated_at=self.generated_at,
            data_gates=self.data_gates,
            replay=self.replay,
            integration_audit=self.integration_audit,
            post_mortem=self.post_mortem,
            verdicts=self.verdicts,
        )


def build_final_report(config: FinalReportConfig | None = None) -> FinalReport:
    cfg = config or FinalReportConfig()
    data_gates = collect_data_product_gates(cfg)
    replay = load_latest_replay(cfg)
    audit = load_integration_audit(cfg)
    pm = load_post_mortem_summary(cfg)
    verdicts = derive_stage_verdicts(replay, data_gates)
    return FinalReport(
        generated_at=datetime.now(timezone.utc).isoformat(),
        data_gates=data_gates,
        replay=replay,
        integration_audit=audit,
        post_mortem=pm,
        verdicts=verdicts,
    )


def write_final_report(
    report: FinalReport,
    output_dir: str | Path,
) -> dict[str, str]:
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)
    md_path = out / "final_report.md"
    json_path = out / "final_report.json"
    md_path.write_text(report.to_markdown(), encoding="utf-8")
    json_path.write_text(
        json.dumps(report.to_dict(), indent=2, default=str), encoding="utf-8"
    )
    return {"markdown": str(md_path), "json": str(json_path)}


def main() -> None:  # pragma: no cover - manual CLI
    import argparse
    parser = argparse.ArgumentParser(description="QuantAgent V7 final report aggregator")
    parser.add_argument("--lake-root", default="runtime/data/v7")
    parser.add_argument("--reports-root", default="runtime/reports")
    parser.add_argument("--models-root", default="runtime/models")
    parser.add_argument("--output-dir", default="runtime/reports/final_report")
    args = parser.parse_args()
    cfg = FinalReportConfig(
        lake_root=args.lake_root,
        reports_root=args.reports_root,
        models_root=args.models_root,
        output_dir=args.output_dir,
    )
    report = build_final_report(cfg)
    paths = write_final_report(report, args.output_dir)
    print(report.to_markdown())
    print(f"\nWrote: {paths['markdown']}")
    print(f"Wrote: {paths['json']}")


if __name__ == "__main__":  # pragma: no cover
    main()
