from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path
import re
from typing import Any, Iterable

import polars as pl

from services.quant_api.adapters.utils import read_csv_rows, read_json, require_relative_path
from services.quant_api.config import ApiSettings, project_relative, safe_project_path, stable_id


MODEL_BINARY_SUFFIXES = {".pt", ".pth", ".joblib", ".pkl", ".pickle", ".zip"}
MODEL_METADATA_NAMES = {
    "training_summary.json",
    "verdict.json",
    "strict_eval_2026.json",
    "eval_2026.json",
    "ev_backtest_report.json",
    "ft_transformer_metrics.json",
    "ft_transformer_config.json",
    "ft_transformer_feature_schema.json",
    "run_config.json",
    "metrics.json",
}


class ModelAdapter:
    """Discover and normalize every persisted QuantAgent model family.

    Binary model content is never deserialized. The adapter only reads adjacent
    metadata, evaluation tables, predictions and feature-importance artifacts.
    """

    def __init__(self, settings: ApiSettings) -> None:
        self.settings = settings
        self._models: dict[str, dict[str, Any]] = {}

    def list(self) -> list[dict[str, Any]]:
        self._models = {}
        represented: set[Path] = set()

        self._discover_deep_ft(represented)
        self._discover_registries(represented)
        self._discover_rl_policies(represented)
        self._discover_do_t_models(represented)
        self._discover_generic_binaries(represented)

        models = [dict(record["summary"]) for record in self._models.values()]
        models.sort(key=lambda item: item.pop("sortTime", 0.0), reverse=True)
        return models

    def get(self, model_id: str) -> dict[str, Any] | None:
        record = self._resolve(model_id)
        if record is None:
            return None
        return self.observability(model_id)

    def observability(self, model_id: str) -> dict[str, Any]:
        record = self._require(model_id)
        summary = dict(record["summary"])
        summary.pop("sortTime", None)
        artifacts = self._artifact_rows(record)
        payloads = self._metadata_payloads(record)
        metrics = _normalized_metrics(payloads)
        evaluations = [
            {"name": path.name, "path": project_relative(self.settings, path), "data": payload}
            for path, payload in payloads
            if path.name in {
                "verdict.json",
                "strict_eval_2026.json",
                "eval_2026.json",
                "ev_backtest_report.json",
                "metrics.json",
            }
        ]
        return {
            **summary,
            "metrics": metrics,
            "artifacts": artifacts,
            "evaluations": evaluations,
            "config": self._config_payload(record, payloads),
            "availability": {
                "trainingMetrics": bool(self.training_metrics(model_id)),
                "featureImportance": bool(self.feature_importance(model_id)),
                "predictions": self._prediction_path(record) is not None or record["kind"] == "rl",
                "evaluation": bool(evaluations),
                "checkpoint": any(item["role"] == "checkpoint" for item in artifacts),
                "config": bool(payloads),
            },
            "checkpoint": {
                "contentExposed": False,
                "count": sum(item["role"] == "checkpoint" for item in artifacts),
                "sizeBytes": sum(item["sizeBytes"] for item in artifacts if item["role"] == "checkpoint"),
            },
        }

    def compare(self, model_ids: list[str]) -> dict[str, Any]:
        rows = []
        metric_names: set[str] = set()
        for model_id in model_ids[:6]:
            detail = self.observability(model_id)
            metrics = {
                item["key"]: item["value"]
                for item in detail["metrics"]
                if isinstance(item.get("value"), (int, float))
            }
            metric_names.update(metrics)
            rows.append({
                "id": detail["id"],
                "version": detail.get("version"),
                "modelType": detail.get("modelType"),
                "modelFamily": detail.get("modelFamily"),
                "verdict": detail.get("verdict"),
                "status": detail.get("status"),
                "metrics": metrics,
            })
        preferred = [
            "total_return",
            "annualised_return",
            "annualized_return",
            "strict_annualized",
            "sharpe",
            "annualised_sharpe",
            "strict_sharpe",
            "calmar",
            "win_rate",
            "rank_ic_mean",
            "ICIR",
            "excess_return_after_costs",
            "max_drawdown",
            "strict_maxDD",
            "hit_rate",
            "turnover",
            "mean_daily_turnover_policy",
        ]
        ordered = [name for name in preferred if name in metric_names]
        ordered.extend(sorted(metric_names - set(ordered))[:20])
        return {"models": rows, "metricKeys": ordered}

    def training_metrics(self, model_id: str) -> list[dict[str, Any]]:
        record = self._require(model_id)
        if record["kind"] == "ft":
            metrics = read_json(record["directory"] / "ft_transformer_metrics.json", {}) or {}
            return [
                {
                    "epoch": _int(row.get("epoch")) or index,
                    "loss": _float(row.get("loss")),
                    "validationLoss": _float(row.get("val_loss")),
                    "metrics": {
                        key: float(value)
                        for key, value in row.items()
                        if key not in {"epoch", "loss", "val_loss"} and isinstance(value, (int, float))
                    },
                }
                for index, row in enumerate(metrics.get("training_history", []))
            ]
        if record["kind"] == "rl":
            summary = _first_payload(record["metadata_paths"], "training_summary.json")
            if summary:
                return [{
                    "epoch": 0,
                    "loss": None,
                    "validationLoss": None,
                    "metrics": {
                        "timesteps": float(summary.get("timesteps") or 0),
                        "cudaAvailable": float(bool(summary.get("cuda_available"))),
                    },
                }]
        return []

    def feature_importance(self, model_id: str) -> list[dict[str, Any]]:
        record = self._require(model_id)
        candidates = list(record.get("importance_paths", []))
        directory: Path = record["directory"]
        candidates.extend([
            directory / "feature_importance.csv",
            directory.parent / "feature_importance.csv",
            directory.parent.parent / "feature_importance.csv",
        ])
        seen: set[Path] = set()
        for path in candidates:
            path = path.resolve()
            if path in seen or not path.exists():
                continue
            seen.add(path)
            rows = read_csv_rows(path)
            output = []
            for row in rows:
                feature = str(row.get("feature") or row.get("factor") or row.get("name") or "")
                importance = _float(row.get("importance") or row.get("weight") or row.get("gain"))
                if feature and importance is not None:
                    output.append({"feature": feature, "importance": importance, "method": "persisted"})
            if output:
                return sorted(output, key=lambda item: abs(item["importance"]), reverse=True)
        return []

    def predictions(
        self,
        model_id: str,
        *,
        symbol: str | None = None,
        limit: int = 2_000,
    ) -> list[dict[str, Any]]:
        record = self._require(model_id)
        if record["kind"] == "rl":
            return self._rl_weights(record, symbol=symbol, limit=limit)
        path = self._prediction_path(record)
        if path is None:
            return []
        lazy = pl.scan_parquet(path)
        schema = lazy.collect_schema()
        symbol_column = next((name for name in ("symbol", "instrument", "code") if name in schema), None)
        if symbol and symbol_column:
            lazy = lazy.filter(pl.col(symbol_column) == symbol)
        score_column = next(
            (name for name in ("alpha_score", "prediction", "composite_score", "score", "target_weight") if name in schema),
            None,
        )
        if score_column is None:
            return []
        date_column = next((name for name in ("trade_date", "datetime", "date", "timestamp") if name in schema), None)
        columns = [
            name for name in (
                date_column,
                symbol_column,
                score_column,
                "forward_return",
                "actual_return",
                "rank",
            )
            if name and name in schema
        ]
        lazy = lazy.select(columns)
        if date_column:
            lazy = lazy.sort(date_column, descending=True)
        frame = lazy.head(min(limit, self.settings.max_chart_points)).collect()
        horizon = (record["summary"].get("horizons") or [None])[0]
        return [{
            "datetime": _iso(row.get(date_column)) if date_column else "",
            "symbol": str(row.get(symbol_column) or "") if symbol_column else "",
            "score": _float(row.get(score_column)) or 0.0,
            "horizon": f"{horizon}d" if horizon is not None else record["summary"].get("modelFamily"),
            "actualReturn": _float(row.get("forward_return") or row.get("actual_return")),
            "rank": _int(row.get("rank")),
        } for row in frame.to_dicts()]

    def _discover_deep_ft(self, represented: set[Path]) -> None:
        pattern = "reports/**/ft/ft_transformer_metrics.json"
        for metrics_path in self.settings.runtime_root.glob(pattern):
            directory = metrics_path.parent
            checkpoint = directory / "ft_transformer.pt"
            represented.add(checkpoint.resolve())
            config = read_json(directory / "ft_transformer_config.json", {}) or {}
            feature_schema = read_json(directory / "ft_transformer_feature_schema.json", {}) or {}
            run_config = read_json(directory.parent / "run_config.json", {}) or {}
            metrics = read_json(metrics_path, {}) or {}
            version = directory.parent.parent.name
            horizon_name = directory.parent.name
            row = self._summary(
                source_path=directory,
                model_type=feature_schema.get("architecture") or "ft_transformer",
                family="deep_alpha",
                version=f"{version} · {horizon_name}",
                created_at=_mtime_iso(metrics_path),
                horizons=_horizons(feature_schema.get("horizons", config.get("horizons", []))),
                feature_count=len(feature_schema.get("feature_columns", config.get("feature_columns", []))),
                sample_count=_int(metrics.get("sample_count")),
                device=metrics.get("device") or config.get("device"),
                gpu_name=metrics.get("gpu_name"),
                production_ready=False,
                status="ready",
                verdict=_verdict(metrics),
                source_kind="deep_run",
                capabilities={
                    "trainingMetrics": bool(metrics.get("training_history")),
                    "featureImportance": any(path.exists() for path in (directory / "feature_importance.csv", directory.parent / "feature_importance.csv")),
                    "predictions": (directory.parent / "predictions.parquet").exists(),
                    "evaluation": True,
                },
                sort_time=metrics_path.stat().st_mtime,
                train_start=run_config.get("train_start"),
                train_end=run_config.get("train_end"),
                test_end=run_config.get("test_end"),
            )
            self._store(
                row,
                "ft",
                directory,
                [
                    metrics_path,
                    directory / "ft_transformer_config.json",
                    directory / "ft_transformer_feature_schema.json",
                    directory.parent / "run_config.json",
                    directory.parent / "backtest" / "metrics.json",
                ],
                binary_paths=[checkpoint],
            )

    def _discover_registries(self, represented: set[Path]) -> None:
        for registry_path in self.settings.runtime_root.glob("models/**/registry/*.json"):
            if registry_path.name == "latest.json":
                continue
            payload = read_json(registry_path, {}) or {}
            if not payload.get("model_version"):
                continue
            metadata = payload.get("metadata") or {}
            metrics = payload.get("metrics") or {}
            issues = []
            output_dir = metadata.get("output_dir")
            directory = registry_path.parent
            if output_dir:
                try:
                    resolved = safe_project_path(self.settings, output_dir)
                    if resolved.exists():
                        directory = resolved
                    else:
                        issues.append({
                            "code": "model_output_missing",
                            "message": "registered output directory does not exist",
                            "path": str(output_dir),
                            "recoverable": True,
                        })
                except ValueError:
                    issues.append({
                        "code": "model_output_outside_project",
                        "message": "registered output directory is outside the project",
                        "path": None,
                        "recoverable": True,
                    })
            represented.add(registry_path.resolve())
            row = self._summary(
                source_path=registry_path,
                model_type=metadata.get("model") or "registered_model",
                family="registered_alpha",
                version=payload.get("model_version"),
                created_at=payload.get("created_at") or _mtime_iso(registry_path),
                horizons=_horizons(metadata.get("horizons", [])),
                feature_count=_int(metrics.get("feature_count")),
                sample_count=_int(metrics.get("prediction_rows")),
                device="cuda" if metrics.get("cuda_available") else metadata.get("device"),
                gpu_name=metadata.get("gpu_name"),
                production_ready=metadata.get("production_ready"),
                status="partial" if issues else "ready",
                verdict=_verdict({**metrics, **metadata}),
                source_kind="registry",
                capabilities={
                    "trainingMetrics": False,
                    "featureImportance": False,
                    "predictions": self._registry_prediction_path(payload) is not None,
                    "evaluation": bool(metrics),
                },
                sort_time=registry_path.stat().st_mtime,
                issues=issues,
                feature_version=payload.get("feature_version"),
                train_start=metadata.get("train_start"),
                train_end=metadata.get("train_end"),
                test_end=metadata.get("test_end"),
            )
            self._store(row, "registry", directory, [registry_path], payload=payload)

    def _discover_rl_policies(self, represented: set[Path]) -> None:
        candidates = list(self.settings.runtime_root.glob("models/**/policy.zip"))
        candidates.extend(self.settings.runtime_root.glob("reports/v8/**/policy.zip"))
        for policy_path in sorted({path.resolve() for path in candidates}):
            if policy_path in represented:
                continue
            represented.add(policy_path)
            directory = policy_path.parent
            metadata_paths = _nearby_metadata(directory)
            summary = _first_payload(metadata_paths, "training_summary.json") or {}
            verdict_payload = _first_payload(metadata_paths, "verdict.json")
            strict_eval = _first_payload(metadata_paths, "strict_eval_2026.json")
            evaluation = verdict_payload or strict_eval or _first_payload(metadata_paths, "eval_2026.json") or {}
            config = summary.get("config") or {}
            env = config.get("env") or {}
            version_root = directory.parent if directory.name == "policy" else directory
            row = self._summary(
                source_path=version_root,
                model_type="ppo_policy",
                family="reinforcement_learning",
                version=version_root.name,
                created_at=_mtime_iso(policy_path),
                horizons=[],
                feature_count=None,
                sample_count=_int(summary.get("timesteps") or evaluation.get("timesteps")),
                device=summary.get("device") or config.get("device"),
                gpu_name=summary.get("gpu_name"),
                production_ready=False,
                status="ready",
                verdict=str(evaluation.get("verdict") or summary.get("status") or "research"),
                source_kind="rl_policy",
                capabilities={
                    "trainingMetrics": bool(summary),
                    "featureImportance": False,
                    "predictions": any(path.name == "weights_test.parquet" for path in _nearby_files(version_root)),
                    "evaluation": bool(evaluation),
                },
                sort_time=policy_path.stat().st_mtime,
                issues=[] if evaluation else [{
                    "code": "rl_evaluation_missing",
                    "message": "policy exists but no persisted strict evaluation was found",
                    "recoverable": True,
                }],
            )
            self._store(
                row,
                "rl",
                version_root,
                metadata_paths,
                binary_paths=[policy_path],
                extra={"env": env},
            )

    def _discover_do_t_models(self, represented: set[Path]) -> None:
        for model_path in self.settings.runtime_root.glob("reports/**/do_t_models.joblib"):
            resolved = model_path.resolve()
            if resolved in represented:
                continue
            represented.add(resolved)
            directory = model_path.parent
            report_path = directory / "ev_backtest_report.json"
            report = read_json(report_path, {}) or {}
            diagnostics = report.get("diagnostics") or {}
            row = self._summary(
                source_path=directory,
                model_type="do_t_model_bundle",
                family="intraday_t_plus_one",
                version=directory.name,
                created_at=_mtime_iso(model_path),
                horizons=[],
                feature_count=_int(diagnostics.get("feature_count")),
                sample_count=_int(report.get("n_train_rows") or diagnostics.get("train_rows")),
                device=None,
                gpu_name=None,
                production_ready=False,
                status="ready" if report else "partial",
                verdict=str(report.get("verdict") or "research"),
                source_kind="joblib_bundle",
                capabilities={
                    "trainingMetrics": False,
                    "featureImportance": (directory / "factor_importance.csv").exists(),
                    "predictions": False,
                    "evaluation": bool(report),
                },
                sort_time=model_path.stat().st_mtime,
                issues=[] if report else [{
                    "code": "do_t_evaluation_missing",
                    "message": "Do-T model bundle has no evaluation report",
                    "recoverable": True,
                }],
            )
            self._store(
                row,
                "do_t",
                directory,
                [report_path, directory / "REFACTOR_REPORT.md"],
                binary_paths=[model_path],
            )

    def _discover_generic_binaries(self, represented: set[Path]) -> None:
        roots = [self.settings.runtime_root / "models", self.settings.runtime_root / "reports"]
        runtime_tmp = (self.settings.runtime_root / "tmp").resolve()
        for root in roots:
            if not root.exists():
                continue
            for path in root.rglob("*"):
                if not path.is_file() or path.suffix.lower() not in MODEL_BINARY_SUFFIXES:
                    continue
                resolved = path.resolve()
                if resolved in represented or runtime_tmp in resolved.parents:
                    continue
                represented.add(resolved)
                directory = path.parent
                metadata_paths = _nearby_metadata(directory)
                row = self._summary(
                    source_path=directory,
                    model_type=path.stem,
                    family="generic_artifact",
                    version=directory.name,
                    created_at=_mtime_iso(path),
                    horizons=[],
                    feature_count=None,
                    sample_count=None,
                    device=None,
                    gpu_name=None,
                    production_ready=False,
                    status="partial",
                    verdict="metadata_only",
                    source_kind="binary_artifact",
                    capabilities={
                        "trainingMetrics": False,
                        "featureImportance": False,
                        "predictions": False,
                        "evaluation": bool(metadata_paths),
                    },
                    sort_time=path.stat().st_mtime,
                    issues=[{
                        "code": "generic_model_metadata_only",
                        "message": "binary model is visible, but no specialized adapter is available",
                        "recoverable": True,
                    }],
                )
                self._store(row, "generic", directory, metadata_paths, binary_paths=[path])

    def _summary(
        self,
        *,
        source_path: Path,
        model_type: str,
        family: str,
        version: str | None,
        created_at: str | None,
        horizons: list[int],
        feature_count: int | None,
        sample_count: int | None,
        device: str | None,
        gpu_name: str | None,
        production_ready: bool | None,
        status: str,
        verdict: str,
        source_kind: str,
        capabilities: dict[str, bool],
        sort_time: float,
        issues: list[dict[str, Any]] | None = None,
        feature_version: str | None = None,
        train_start: str | None = None,
        train_end: str | None = None,
        test_end: str | None = None,
    ) -> dict[str, Any]:
        relative = require_relative_path(self.settings, source_path)
        return {
            "id": stable_id("model", relative),
            "modelType": model_type,
            "modelFamily": family,
            "version": version or source_path.name,
            "featureVersion": feature_version,
            "createdAt": created_at,
            "trainStart": train_start,
            "trainEnd": train_end,
            "testEnd": test_end,
            "horizons": horizons,
            "featureCount": feature_count,
            "sampleCount": sample_count,
            "device": device,
            "gpuName": gpu_name,
            "productionReady": production_ready,
            "status": status,
            "path": relative,
            "issues": issues or [],
            "sourceKind": source_kind,
            "verdict": verdict,
            "capabilities": capabilities,
            "sortTime": sort_time,
        }

    def _store(
        self,
        summary: dict[str, Any],
        kind: str,
        directory: Path,
        metadata_paths: Iterable[Path],
        *,
        binary_paths: Iterable[Path] = (),
        payload: dict[str, Any] | None = None,
        extra: dict[str, Any] | None = None,
    ) -> None:
        self._models[summary["id"]] = {
            "kind": kind,
            "directory": directory,
            "summary": summary,
            "metadata_paths": [path for path in metadata_paths if path.exists()],
            "binary_paths": [path for path in binary_paths if path.exists()],
            "payload": payload or {},
            **(extra or {}),
        }

    def _artifact_rows(self, record: dict[str, Any]) -> list[dict[str, Any]]:
        paths: list[Path] = []
        paths.extend(record.get("binary_paths", []))
        paths.extend(record.get("metadata_paths", []))
        prediction = self._prediction_path(record)
        if prediction:
            paths.append(prediction)
        paths.extend(record.get("importance_paths", []))
        seen: set[Path] = set()
        rows = []
        for path in paths:
            path = path.resolve()
            if path in seen or not path.exists():
                continue
            seen.add(path)
            role = "checkpoint" if path.suffix.lower() in MODEL_BINARY_SUFFIXES else _artifact_role(path)
            rows.append({
                "role": role,
                "name": path.name,
                "path": project_relative(self.settings, path),
                "extension": path.suffix.lower(),
                "sizeBytes": path.stat().st_size,
                "modifiedAt": _mtime_iso(path),
                "previewable": role != "checkpoint",
            })
        return sorted(rows, key=lambda item: (item["role"] != "checkpoint", item["name"]))

    def _metadata_payloads(self, record: dict[str, Any]) -> list[tuple[Path, dict[str, Any]]]:
        payloads = []
        for path in record.get("metadata_paths", []):
            if path.suffix.lower() != ".json" or not path.exists():
                continue
            payload = read_json(path, None)
            if isinstance(payload, dict):
                payloads.append((path, payload))
        if record["kind"] == "registry" and record.get("payload"):
            registry_path = next(iter(record.get("metadata_paths", [])), record["directory"])
            if not any(path == registry_path for path, _ in payloads):
                payloads.append((registry_path, record["payload"]))
        return payloads

    def _config_payload(
        self,
        record: dict[str, Any],
        payloads: list[tuple[Path, dict[str, Any]]],
    ) -> dict[str, Any]:
        if record["kind"] == "ft":
            directory: Path = record["directory"]
            return {
                "model": read_json(directory / "ft_transformer_config.json", {}) or {},
                "featureSchema": read_json(directory / "ft_transformer_feature_schema.json", {}) or {},
                "run": read_json(directory.parent / "run_config.json", {}) or {},
            }
        for path, payload in payloads:
            if path.name == "training_summary.json":
                return payload.get("config") or payload
        if record["kind"] == "registry":
            return record.get("payload", {})
        return {}

    def _prediction_path(self, record: dict[str, Any]) -> Path | None:
        if record["kind"] == "ft":
            candidate = record["directory"].parent / "predictions.parquet"
            return candidate if candidate.exists() else None
        if record["kind"] == "registry":
            return self._registry_prediction_path(record.get("payload", {}))
        if record["kind"] == "rl":
            for path in _nearby_files(record["directory"]):
                if path.name == "weights_test.parquet":
                    return path
        return None

    def _registry_prediction_path(self, payload: dict[str, Any]) -> Path | None:
        output_dir = (payload.get("metadata") or {}).get("output_dir")
        if not output_dir:
            return None
        try:
            root = safe_project_path(self.settings, output_dir)
        except ValueError:
            return None
        for candidate in (
            root / "predictions.parquet",
            root.parent / "predictions.parquet",
            root / "prediction.parquet",
        ):
            if candidate.exists():
                return candidate
        return None

    def _rl_weights(self, record: dict[str, Any], *, symbol: str | None, limit: int) -> list[dict[str, Any]]:
        path = self._prediction_path(record)
        if path is None:
            return []
        schema = pl.read_parquet_schema(path)
        date_column = next((name for name in ("trade_date", "datetime", "date") if name in schema), None)
        symbols = [name for name in schema if name != date_column]
        if symbol:
            symbols = [name for name in symbols if name == symbol]
        symbols = symbols[: min(250, len(symbols))]
        if not symbols:
            return []
        rows_per_date = max(1, len(symbols))
        dates = max(1, min(20, limit // rows_per_date + 1))
        columns = ([date_column] if date_column else []) + symbols
        frame = pl.scan_parquet(path).select(columns).tail(dates).collect()
        output = []
        for row in frame.to_dicts():
            for name in symbols:
                value = _float(row.get(name))
                if value is None:
                    continue
                output.append({
                    "datetime": _iso(row.get(date_column)) if date_column else "",
                    "symbol": name,
                    "score": value,
                    "horizon": "policy_weight",
                    "actualReturn": None,
                    "rank": None,
                })
                if len(output) >= limit:
                    return output
        return output

    def _resolve(self, model_id: str) -> dict[str, Any] | None:
        if model_id not in self._models:
            self.list()
        return self._models.get(model_id)

    def _require(self, model_id: str) -> dict[str, Any]:
        record = self._resolve(model_id)
        if record is None:
            raise KeyError(model_id)
        return record


def _nearby_files(directory: Path) -> list[Path]:
    output = []
    for root in (directory, directory.parent):
        if not root.exists():
            continue
        output.extend(path for path in root.iterdir() if path.is_file())
    return output


def _nearby_metadata(directory: Path) -> list[Path]:
    return [
        path
        for path in _nearby_files(directory)
        if path.name in MODEL_METADATA_NAMES or path.suffix.lower() in {".json", ".md"}
    ][:30]


def _first_payload(paths: Iterable[Path], name: str) -> dict[str, Any] | None:
    for path in paths:
        if path.name == name:
            payload = read_json(path, None)
            return payload if isinstance(payload, dict) else None
    return None


def _normalized_metrics(payloads: Iterable[tuple[Path, dict[str, Any]]]) -> list[dict[str, Any]]:
    values: dict[str, dict[str, Any]] = {}
    for path, payload in payloads:
        for key, value in _flatten_numeric(payload):
            if key in values:
                continue
            values[key] = {
                "key": key,
                "label": key.replace("_", " ").replace(".", " · "),
                "value": value,
                "source": path.name,
                "group": _metric_group(key),
                "unit": _metric_unit(key),
            }
    return sorted(values.values(), key=lambda item: (item["group"], item["label"]))


def _flatten_numeric(value: Any, prefix: str = "", depth: int = 0) -> Iterable[tuple[str, float]]:
    if depth > 4:
        return
    if isinstance(value, dict):
        for key, child in value.items():
            if key in {"training_history", "feature_columns", "null_strict_anns_untrained"}:
                continue
            next_prefix = f"{prefix}.{key}" if prefix else str(key)
            yield from _flatten_numeric(child, next_prefix, depth + 1)
    elif isinstance(value, bool):
        yield prefix, float(value)
    elif isinstance(value, (int, float)) and prefix:
        yield prefix, float(value)


def _metric_group(key: str) -> str:
    lower = key.lower()
    tokens = set(re.split(r"[^a-z0-9]+", lower))
    if any(token in lower for token in ("return", "annual", "uplift", "value_add", "pnl", "edge")):
        return "return"
    if any(token in lower for token in ("drawdown", "risk", "adverse", "fail", "cost", "turnover")):
        return "risk"
    if tokens.intersection({"row", "rows", "count", "feature", "features", "timesteps", "fold", "folds"}):
        return "scale"
    if tokens.intersection({"ic", "icir", "rankic"}) or any(
        token in lower for token in ("sharpe", "hit_rate", "brier", "mae", "accuracy")
    ):
        return "quality"
    return "other"


def _metric_unit(key: str) -> str:
    lower = key.lower()
    if any(token in lower for token in ("rate", "return", "drawdown", "uplift", "value_add", "turnover", "maxdd")):
        return "ratio"
    if "bps" in lower:
        return "bps"
    if any(token in lower for token in ("row", "count", "feature", "timesteps", "fold")):
        return "count"
    return "number"


def _artifact_role(path: Path) -> str:
    name = path.name.lower()
    if "prediction" in name or "weights_test" in name:
        return "predictions"
    if "importance" in name:
        return "feature_importance"
    if "config" in name or "schema" in name:
        return "config"
    if "metric" in name or "eval" in name or "verdict" in name or "report" in name:
        return "evaluation"
    return "metadata"


def _verdict(payload: dict[str, Any]) -> str:
    explicit = payload.get("verdict") or payload.get("status")
    if explicit:
        return str(explicit)
    if payload.get("production_ready") is True:
        return "production_ready"
    if payload.get("adverse_regime_passed") in (0, 0.0, False):
        return "research_only"
    return "research"


def _horizons(values: Any) -> list[int]:
    output = []
    for value in values or []:
        parsed = _int(value)
        if parsed is not None:
            output.append(parsed)
    return output


def _mtime_iso(path: Path) -> str:
    return datetime.fromtimestamp(path.stat().st_mtime, tz=timezone.utc).isoformat()


def _float(value: Any) -> float | None:
    try:
        return float(value) if value not in (None, "") else None
    except (TypeError, ValueError):
        return None


def _int(value: Any) -> int | None:
    try:
        return int(value) if value not in (None, "") else None
    except (TypeError, ValueError):
        return None


def _iso(value: Any) -> str:
    return value.isoformat() if hasattr(value, "isoformat") else str(value or "")
