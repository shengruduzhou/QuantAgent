from __future__ import annotations

import ast
import inspect
from pathlib import Path
import textwrap
from typing import Any

from services.quant_api.adapters.utils import read_csv_rows, read_json
from services.quant_api.config import ApiSettings


class FactorAdapter:
    def __init__(self, settings: ApiSettings) -> None:
        self.settings = settings
        self._catalog: dict[str, dict[str, Any]] | None = None
        self._metrics: dict[str, dict[str, Any]] | None = None

    def list(self, query: str | None = None) -> list[dict[str, Any]]:
        catalog = list(self._build_catalog().values())
        if query:
            needle = query.lower()
            catalog = [
                item for item in catalog
                if needle in item["name"].lower()
                or needle in str(item.get("description") or "").lower()
                or needle in str(item.get("category") or "").lower()
            ]
        return sorted(catalog, key=lambda item: item["name"])

    def get(self, name: str) -> dict[str, Any] | None:
        return self._build_catalog().get(name)

    def explanation(self, name: str) -> dict[str, Any] | None:
        factor = self.get(name)
        if factor is None:
            return None
        direction_text = {
            "HIGHER_BETTER": "数值越高，预期排序方向越强。",
            "LOWER_BETTER": "数值越低，预期排序方向越强。",
            "NON_LINEAR": "该因子不适合简单线性解释。",
            "UNKNOWN": "代码或 runtime 未提供稳定方向定义。",
        }[factor["direction"]]
        return {
            "factor": factor,
            "summary": factor.get("description") or "暂无人工解释，展示代码与 runtime metadata。",
            "directionExplanation": direction_text,
            "pitNote": "仅在 factor availability 满足交易时点要求后使用。",
            "limitations": [
                "Feature importance 不等同于单笔交易因果贡献。",
                "没有独立 factor trade artifact 时不生成伪造买卖点。",
            ],
        }

    def backtest(self, name: str) -> dict[str, Any]:
        metrics = self._factor_metrics().get(name, {})
        ic_series = []
        for year in range(2022, 2027):
            value = _float(metrics.get(f"ic_{year}"))
            if value is not None:
                ic_series.append({"datetime": f"{year}-12-31", "value": value})
        decay = []
        for horizon in (5, 20, 60):
            value = _float(metrics.get(f"ic_{horizon}d"))
            if value is not None:
                decay.append({"horizonDays": horizon, "ic": value, "rankIc": None})
        return {
            "factorName": name,
            "totalReturn": None,
            "annualReturn": None,
            "maxDrawdown": None,
            "sharpe": None,
            "calmar": None,
            "winRate": None,
            "turnover": None,
            "ic": _float(metrics.get("ic_5d")),
            "rankIc": _float(metrics.get("rank_ic_5d")),
            "icir": _float(metrics.get("icir_5d")),
            "rankIcir": _float(metrics.get("icir_5d")) if metrics.get("rank_ic_5d") is not None else None,
            "coverage": _float(metrics.get("capacity_ratio")),
            "stability": _float(metrics.get("years_passed")),
            "crowding": None,
            "capacityRmb": None,
            "verdict": metrics.get("verdict"),
            "bestHorizon": metrics.get("best_horizon"),
            "regimeIc": {
                "bull": _float(metrics.get("ic_bull")),
                "sideways": _float(metrics.get("ic_sideways")),
                "bear": _float(metrics.get("ic_bear")),
            },
            "icSeries": ic_series,
            "rankIcSeries": [],
            "quantileReturns": [],
            "longShortEquity": [],
            "decay": decay,
            "trades": [],
            "signals": [],
            "availability": {
                "summaryMetrics": bool(metrics),
                "icSeries": bool(ic_series),
                "rankIcSeries": False,
                "quantileReturns": False,
                "longShortEquity": False,
                "trades": False,
                "signals": False,
            },
        }

    def ic(self, name: str) -> dict[str, Any]:
        result = self.backtest(name)
        return {
            "factorName": name,
            "ic": result["ic"],
            "rankIc": result["rankIc"],
            "icir": result["icir"],
            "rankIcir": result["rankIcir"],
            "icSeries": result["icSeries"],
            "rankIcSeries": result["rankIcSeries"],
            "regimeIc": result["regimeIc"],
        }

    def quantile_returns(self, name: str) -> list[dict[str, Any]]:
        return []

    def _build_catalog(self) -> dict[str, dict[str, Any]]:
        if self._catalog is not None:
            return self._catalog
        from quantagent.factors import alpha101 as _alpha101  # noqa: F401
        from quantagent.factors import cicc_ashare80 as _cicc  # noqa: F401
        from quantagent.factors import cicc_high_freq as _cicc_hf  # noqa: F401
        from quantagent.factors import technical_indicators as _technical  # noqa: F401
        from quantagent.factors.alpha181 import ALPHA181_NAMES, alpha181_source_map
        from quantagent.factors.registry import default_registry

        training_features = self._training_features()
        metrics = self._factor_metrics()
        catalog: dict[str, dict[str, Any]] = {}
        for name, meta in default_registry.metas().items():
            compute = meta.compute
            module = getattr(compute, "__module__", "") if compute else ""
            source_file = inspect.getsourcefile(compute) if compute else None
            line = inspect.getsourcelines(compute)[1] if compute else None
            location = None
            if source_file:
                try:
                    relative = Path(source_file).resolve().relative_to(self.settings.project_root)
                    location = f"{relative.as_posix()}:{line}" if line else relative.as_posix()
                except ValueError:
                    location = None
            catalog[name] = self._factor_row(
                name=name,
                description=meta.description,
                category=meta.category,
                direction=meta.direction,
                horizon=meta.horizon_days,
                required=list(meta.required_columns),
                frequency=meta.frequency,
                lookback=meta.lookback,
                pit_safe=meta.pit_safe,
                location=location,
                source_kind="registry",
                training_features=training_features,
                lifecycle=metrics.get(name, {}).get("verdict"),
                compute=compute,
            )
        source_map = alpha181_source_map()
        for name in ALPHA181_NAMES:
            if name in catalog:
                continue
            source = source_map.get(name)
            module_name = source.split(".", 1)[0] if source else "alpha181"
            catalog[name] = self._factor_row(
                name=name,
                description=f"Alpha181 fixed-library factor; implementation source: {source}.",
                category="alpha181",
                direction=None,
                horizon=None,
                required=[],
                frequency="daily",
                lookback=None,
                pit_safe=True,
                location=f"src/quantagent/factors/{module_name}.py",
                source_kind="alpha181",
                training_features=training_features,
                lifecycle=metrics.get(name, {}).get("verdict"),
                compute=None,
            )
        self._merge_synthesized(catalog, training_features, metrics)
        for name in metrics:
            if name not in catalog:
                catalog[name] = self._factor_row(
                    name=name,
                    description="Runtime-only factor discovered in factor judgment artifacts.",
                    category=str(metrics[name].get("family") or "runtime"),
                    direction=None,
                    horizon=None,
                    required=[],
                    frequency="daily",
                    lookback=None,
                    pit_safe=None,
                    location=None,
                    source_kind="runtime",
                    training_features=training_features,
                    lifecycle=metrics[name].get("verdict"),
                    compute=None,
                )
        self._catalog = catalog
        return catalog

    def _factor_row(
        self,
        *,
        name: str,
        description: str,
        category: str | None,
        direction: int | None,
        horizon: int | None,
        required: list[str],
        frequency: str | None,
        lookback: int | None,
        pit_safe: bool | None,
        location: str | None,
        source_kind: str,
        training_features: set[str],
        lifecycle: str | None,
        compute: Any | None,
    ) -> dict[str, Any]:
        return {
            "name": name,
            "displayName": name,
            "category": category,
            "description": description,
            "codeLocation": location,
            "formula": _factor_expression(compute),
            "direction": {1: "HIGHER_BETTER", -1: "LOWER_BETTER", 0: "NON_LINEAR"}.get(direction, "UNKNOWN"),
            "horizonDays": horizon,
            "parameters": _factor_parameters(compute),
            "dataSource": required,
            "requiredColumns": required,
            "frequency": frequency,
            "lookback": lookback,
            "pitSafe": pit_safe,
            "missingValuePolicy": None,
            "standardization": "cross-sectional z-score/rank where configured",
            "neutralization": "available through factor preprocessing; run-specific",
            "usedInTraining": name in training_features,
            "usedInSelection": name in training_features,
            "usedInTiming": name.startswith(("first30_", "last30_", "vwap_", "intraday_", "net_buy_")),
            "usedInRisk": "risk" in name or "volatility" in name or "liquidity" in name,
            "lifecycle": lifecycle,
            "sourceKind": source_kind,
        }

    def _training_features(self) -> set[str]:
        features: set[str] = set()
        for path in self.settings.runtime_root.glob("reports/v8/deep/*/*/ft/ft_transformer_feature_schema.json"):
            payload = read_json(path, {}) or {}
            features.update(str(item) for item in payload.get("feature_columns", []))
        return features

    def _factor_metrics(self) -> dict[str, dict[str, Any]]:
        if self._metrics is not None:
            return self._metrics
        rows: list[dict[str, Any]] = []
        for path in (
            self.settings.runtime_root / "reports" / "v8" / "factor_full_judgment" / "factor_judgment_table.csv",
            self.settings.runtime_root / "reports" / "v8" / "factor_full_judgment" / "gtja191_judgment.csv",
        ):
            rows.extend(read_csv_rows(path))
        self._metrics = {str(row.get("factor")): row for row in rows if row.get("factor")}
        diagnostic_path = self.settings.runtime_root / "reports" / "v8" / "factor_diagnostics" / "table.csv"
        for row in read_csv_rows(diagnostic_path):
            name = str(row.get("factor") or "")
            if name:
                self._metrics.setdefault(name, {}).update({key: value for key, value in row.items() if value not in (None, "")})
        return self._metrics

    def _merge_synthesized(
        self,
        catalog: dict[str, dict[str, Any]],
        training_features: set[str],
        metrics: dict[str, dict[str, Any]],
    ) -> None:
        for path in self.settings.runtime_root.glob("reports/v7/factor_synthesis*/synthesized_definitions.json"):
            payload = read_json(path, [])
            definitions = payload if isinstance(payload, list) else payload.get("definitions", []) if isinstance(payload, dict) else []
            for definition in definitions:
                name = str(definition.get("name") or "")
                if not name:
                    continue
                row = self._factor_row(
                    name=name,
                    description=str(definition.get("description") or "Synthesized symbolic factor."),
                    category="synthesized",
                    direction=int(definition.get("direction", 0) or 0),
                    horizon=definition.get("horizon_days"),
                    required=list(definition.get("required_columns", [])),
                    frequency="daily",
                    lookback=definition.get("lookback"),
                    pit_safe=True,
                    location="src/quantagent/factors/factor_synthesis.py",
                    source_kind="synthesized",
                    training_features=training_features,
                    lifecycle=metrics.get(name, {}).get("verdict"),
                    compute=None,
                )
                row["formula"] = definition.get("expression")
                row["parameters"] = definition
                catalog[name] = row


def _float(value: Any) -> float | None:
    try:
        return float(value) if value not in (None, "") else None
    except (TypeError, ValueError):
        return None


def _factor_expression(compute: Any | None) -> str | None:
    if compute is None:
        return None
    try:
        source = textwrap.dedent(inspect.getsource(compute))
        tree = ast.parse(source)
        return_node = next(
            (node for node in ast.walk(tree) if isinstance(node, ast.Return) and node.value is not None),
            None,
        )
        if return_node is not None:
            expression = ast.unparse(return_node.value)
            if len(expression) <= 800:
                return expression
    except (OSError, TypeError, SyntaxError, StopIteration):
        pass
    module = getattr(compute, "__module__", None)
    name = getattr(compute, "__qualname__", getattr(compute, "__name__", None))
    return f"{module}.{name}" if module and name else None


def _factor_parameters(compute: Any | None) -> dict[str, Any]:
    if compute is None:
        return {}
    try:
        signature = inspect.signature(compute)
    except (TypeError, ValueError):
        return {}
    parameters: dict[str, Any] = {}
    for name, parameter in signature.parameters.items():
        if name in {"df", "data", "frame", "panel"}:
            continue
        if parameter.default is inspect.Parameter.empty:
            parameters[name] = {"required": True}
        elif isinstance(parameter.default, (str, int, float, bool, type(None))):
            parameters[name] = parameter.default
        else:
            parameters[name] = str(parameter.default)
    return parameters
