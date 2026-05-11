from __future__ import annotations

from dataclasses import asdict, dataclass, field
from pathlib import Path

import json


@dataclass(frozen=True)
class V6ValidationReport:
    passed: bool
    metrics: dict[str, float]
    checks: dict[str, bool]
    warnings: tuple[str, ...] = field(default_factory=tuple)

    def write(self, output_dir: str | Path) -> tuple[Path, Path]:
        path = Path(output_dir)
        path.mkdir(parents=True, exist_ok=True)
        json_path = path / "v6_validation_report.json"
        md_path = path / "v6_validation_report.md"
        json_path.write_text(json.dumps(asdict(self), indent=2, sort_keys=True), encoding="utf-8")
        lines = [
            "# V6 验证报告 / Validation Report",
            "",
            f"- passed: `{self.passed}`",
            "",
            "## 指标 / Metrics",
            *[f"- `{key}`: {value:.6f}" for key, value in self.metrics.items()],
            "",
            "## 检查 / Checks",
            *[f"- `{key}`: {value}" for key, value in self.checks.items()],
        ]
        if self.warnings:
            lines.extend(["", "## 警告 / Warnings", *[f"- {item}" for item in self.warnings]])
        md_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
        return md_path, json_path


def build_smoke_validation_report() -> V6ValidationReport:
    checks = {
        "walk_forward": True,
        "purged_cv": True,
        "embargo": True,
        "no_lookahead_assertion": True,
        "factor_leakage_check": True,
        "event_cutoff_check": True,
        "temporal_split": True,
        "ablation_report": True,
    }
    metrics = {
        "oos_ic": 0.03,
        "rank_ic": 0.04,
        "icir": 0.50,
        "turnover_adjusted_return": 0.02,
        "cost_adjusted_return": 0.018,
        "hit_ratio": 0.52,
        "calibration_coverage": 0.90,
        "max_drawdown": -0.08,
        "exposure_stability": 0.88,
    }
    return V6ValidationReport(True, metrics, checks, warnings=("smoke_validation_uses_synthetic_fixture",))

