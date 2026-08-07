from __future__ import annotations

from datetime import datetime, timezone
from hashlib import sha256
import json
from pathlib import Path
import re
import shutil
from threading import RLock
from typing import Any
from uuid import uuid4

from quantagent.research.verdict import required_oos_days as _required_oos_days
from services.quant_api.config import ApiSettings, project_relative, safe_project_path
from services.quant_api.schemas.strategy import StrategyDraft
from services.quant_api.services.run_results import RunResultResolver


DECISION_COUNCIL = (
    ("data_quality", "Data Quality", "PIT、覆盖率、重复键与隔离区"),
    ("factor_research", "Factor Research", "因子评审、相关性和失效条件"),
    ("model_validation", "Model Validation", "滚动切分、embargo 与 OOS 证据"),
    ("portfolio", "Portfolio", "目标权重、集中度与换手约束"),
    ("backtest", "Backtest", "A股撮合、成本、涨跌停与 T+1"),
    ("risk", "Risk", "回撤、流动性、kill switch 与否决"),
    ("challenger", "Challenger", "反证、敏感性与替代解释"),
    ("human_gate", "Human Gate", "最终研究启动授权"),
)

HORIZON_COLUMN = re.compile(r"^forward_return_(\d+)d$")

# `run-full-real-training-v7` fixes the per-fold validation window at 20 trading
# days and the web layer does not expose it, so the OOS days a run will produce
# are determined entirely by the fold count chosen here.
OOS_DAYS_PER_FOLD = 20

# Mirrors the CLI defaults for `run-full-real-training-v7`; the web layer does
# not expose either, so the pre-flight projection has to assume them.
_MIN_TRAIN_DAYS = 120
_EMBARGO_DAYS = 5

_SYMBOL_CACHE: dict[tuple[str, float, str], bool] = {}
_PANEL_DAYS_CACHE: dict[tuple[str, float], int | None] = {}


def _panel_trading_days(path: Path) -> int | None:
    """Distinct trading days in a panel. None when it cannot be determined."""
    try:
        key = (str(path), path.stat().st_mtime)
    except OSError:
        return None
    if key in _PANEL_DAYS_CACHE:
        return _PANEL_DAYS_CACHE[key]
    try:
        import pyarrow.compute as compute
        import pyarrow.dataset as parquet_dataset

        dataset = parquet_dataset.dataset(path, format="parquet")
        if "trade_date" not in dataset.schema.names:
            return None
        column = dataset.to_table(columns=["trade_date"]).column("trade_date")
        days = int(compute.count_distinct(column).as_py())
    except Exception:
        return None
    if len(_PANEL_DAYS_CACHE) > 32:
        _PANEL_DAYS_CACHE.clear()
    _PANEL_DAYS_CACHE[key] = days
    return days


def _parquet_horizons(path: Path) -> list[int]:
    import pyarrow.dataset as parquet_dataset

    columns = parquet_dataset.dataset(path, format="parquet").schema.names
    return sorted(
        {
            int(match.group(1))
            for column in columns
            if (match := HORIZON_COLUMN.fullmatch(column))
        }
    )


def _panel_contains_symbol(path: Path, symbol: str) -> bool | None:
    """Is this symbol present in the panel? None when it cannot be determined.

    A benchmark that is not in the panel aborts the run late, after the whole
    training has been paid for, so it is worth one scan of a single column to
    find out before launching.
    """
    try:
        key = (str(path), path.stat().st_mtime, symbol)
    except OSError:
        return None
    if key in _SYMBOL_CACHE:
        return _SYMBOL_CACHE[key]
    try:
        import pyarrow.compute as compute
        import pyarrow.dataset as parquet_dataset

        dataset = parquet_dataset.dataset(path, format="parquet")
        if "symbol" not in dataset.schema.names:
            return None
        table = dataset.to_table(
            columns=["symbol"],
            filter=compute.field("symbol") == symbol,
        )
        found = table.num_rows > 0
    except Exception:
        return None
    if len(_SYMBOL_CACHE) > 64:
        _SYMBOL_CACHE.clear()
    _SYMBOL_CACHE[key] = found
    return found


class StrategyService:
    """Strategies as durable entities: versions, runs, results and deletion.

    Every save used to create a standalone file and the list endpoint returned
    each file as if it were a separate strategy, so three edits of one idea
    looked like three strategies and there was no way to remove any of them.
    A strategy is now one identity with an ordered version history and the runs
    launched from it, and it can be archived or deleted like any other record.
    """

    RUNS_FILENAME = "runs.jsonl"

    def __init__(self, settings: ApiSettings, runs: "RunResultResolver | None" = None) -> None:
        self.settings = settings
        self.root = settings.runtime_root / "strategies"
        self.root.mkdir(parents=True, exist_ok=True)
        self.archive_root = settings.runtime_root / "archives" / "strategies"
        self.results = runs or RunResultResolver(settings)
        self._lock = RLock()

    # ------------------------------------------------------------------
    # identity + versions
    # ------------------------------------------------------------------

    def list(self, *, include_versions: bool = False) -> list[dict[str, Any]]:
        """One entry per strategy, newest first, with its version history."""
        strategies: list[dict[str, Any]] = []
        for directory in sorted(self.root.iterdir() if self.root.exists() else []):
            if not directory.is_dir():
                continue
            versions = self._versions(directory)
            if not versions:
                continue
            latest = versions[0]
            runs = self.runs(directory.name)
            entry = {
                **latest,
                "versionCount": len(versions),
                "firstCreatedAt": versions[-1]["createdAt"],
                "updatedAt": latest["createdAt"],
                "runCount": len(runs),
                "lastRun": runs[0] if runs else None,
            }
            if include_versions:
                entry["versions"] = versions
            strategies.append(entry)
        strategies.sort(key=lambda item: item.get("updatedAt") or "", reverse=True)
        return strategies

    def get(self, strategy_id: str, version: str | None = None) -> dict[str, Any] | None:
        directory = self._strategy_dir(strategy_id)
        if directory is None:
            return None
        versions = self._versions(directory)
        if not versions:
            return None
        selected = (
            next((item for item in versions if item["version"] == version), None)
            if version
            else versions[0]
        )
        if selected is None:
            return None
        return {
            **selected,
            "versions": versions,
            "versionCount": len(versions),
            "runs": self.runs(strategy_id),
        }

    def _strategy_dir(self, strategy_id: str) -> Path | None:
        slug = self._slug(strategy_id)
        if slug != strategy_id.strip().lower():
            return None
        directory = self.root / slug
        return directory if directory.is_dir() else None

    def _versions(self, directory: Path) -> list[dict[str, Any]]:
        versions: list[dict[str, Any]] = []
        for path in directory.glob("*.json"):
            if path.name == self.RUNS_FILENAME:
                continue
            try:
                payload = json.loads(path.read_text(encoding="utf-8"))
            except (OSError, ValueError):
                continue
            if "draft" not in payload or "id" not in payload:
                continue
            versions.append(self._public(payload, path))
        versions.sort(key=lambda item: item["version"], reverse=True)
        return versions

    # ------------------------------------------------------------------
    # runs
    # ------------------------------------------------------------------

    def register_run(
        self,
        *,
        strategy_id: str,
        version: str,
        job_id: str,
        output_dir: str,
        name: str,
    ) -> dict[str, Any]:
        """Record that a job was launched from a specific strategy version.

        Without this link a finished job is an anonymous output directory: the
        parameters that produced it, the hypothesis behind it and the artifacts
        it wrote could not be connected to each other.
        """
        record = {
            "runId": f"run_{uuid4().hex[:12]}",
            "strategyId": strategy_id,
            "strategyVersion": version,
            "strategyName": name,
            "jobId": job_id,
            "outputDir": output_dir,
            "createdAt": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        }
        path = self.root / strategy_id / self.RUNS_FILENAME
        path.parent.mkdir(parents=True, exist_ok=True)
        with self._lock:
            with path.open("a", encoding="utf-8") as handle:
                handle.write(json.dumps(record, ensure_ascii=False) + "\n")
        return record

    def runs(self, strategy_id: str | None = None) -> list[dict[str, Any]]:
        paths = (
            [self.root / strategy_id / self.RUNS_FILENAME]
            if strategy_id
            else sorted(self.root.glob(f"*/{self.RUNS_FILENAME}"))
        )
        runs: list[dict[str, Any]] = []
        for path in paths:
            if not path.exists():
                continue
            for line in path.read_text(encoding="utf-8").splitlines():
                line = line.strip()
                if not line:
                    continue
                try:
                    runs.append(json.loads(line))
                except ValueError:
                    continue
        runs.sort(key=lambda item: item.get("createdAt") or "", reverse=True)
        return runs

    def run(self, run_id: str) -> dict[str, Any] | None:
        return next((item for item in self.runs() if item.get("runId") == run_id), None)

    def run_result(self, run_id: str) -> dict[str, Any] | None:
        record = self.run(run_id)
        if record is None:
            return None
        return {**record, "result": self.results.resolve(record["outputDir"])}

    # Comparable fields, with the direction that counts as better. `None` means
    # there is no universally better direction and the UI must not imply one.
    COMPARISON_METRICS: tuple[tuple[str, str, str, str | None], ...] = (
        ("training.rankIcMean", "Rank IC", "训练", "higher"),
        ("training.icir", "ICIR", "训练", "higher"),
        ("training.hitRate", "命中率", "训练", "higher"),
        ("training.evaluatedDays", "评估交易日", "训练", "higher"),
        ("governance.pbo", "PBO", "过拟合治理", "lower"),
        ("governance.dsrProbability", "DSR 概率", "过拟合治理", "higher"),
        ("governance.spaPValue", "SPA p-value", "过拟合治理", "lower"),
        ("governance.cumulativeTrials", "累计试验", "过拟合治理", "lower"),
        ("backtest.totalReturn", "区间收益", "成本后回测", "higher"),
        ("backtest.maxDrawdown", "最大回撤", "成本后回测", "higher"),
        ("backtest.orderCount", "成交笔数", "成本后回测", None),
        ("backtest.skippedOrderCount", "被约束跳过", "成本后回测", "lower"),
        ("acceptance.passedCount", "通过闸门数", "验收", "higher"),
    )
    COMPARISON_LIMIT = 4

    def compare_runs(self, run_ids: list[str]) -> dict[str, Any]:
        """Align several runs on the same fields so they can be judged together.

        Bounded to four: beyond that the table stops being readable and starts
        inviting the selection of a winner from noise. Every cell is copied from
        a run's own artifacts — a run that never produced a field shows a gap,
        not a zero.
        """
        if not run_ids:
            raise ValueError("compare requires at least one run")
        if len(run_ids) > self.COMPARISON_LIMIT:
            raise ValueError(
                f"compare is bounded to {self.COMPARISON_LIMIT} runs; received {len(run_ids)}"
            )
        known = {item["runId"]: item for item in self.runs()}
        missing = [run_id for run_id in run_ids if run_id not in known]
        if missing:
            raise KeyError(", ".join(missing))

        resolved = [
            {**known[run_id], "result": self.results.resolve(known[run_id]["outputDir"])}
            for run_id in run_ids
        ]

        def read(payload: dict[str, Any], path: str) -> Any:
            node: Any = payload["result"]
            for part in path.split("."):
                if not isinstance(node, dict):
                    return None
                node = node.get(part)
            return node

        metrics = []
        for path, label, group, direction in self.COMPARISON_METRICS:
            values = [read(item, path) for item in resolved]
            if all(value is None for value in values):
                continue
            numeric = [value for value in values if isinstance(value, (int, float))]
            best_index = None
            if direction and len(numeric) > 1 and len(set(numeric)) > 1:
                target = max(numeric) if direction == "higher" else min(numeric)
                best_index = next(
                    (index for index, value in enumerate(values) if value == target),
                    None,
                )
            metrics.append({
                "key": path,
                "label": label,
                "group": group,
                "direction": direction,
                "values": values,
                "bestIndex": best_index,
            })

        gate_names: list[str] = []
        for item in resolved:
            for gate in ((item["result"].get("acceptance") or {}).get("gates") or []):
                if gate["name"] not in gate_names:
                    gate_names.append(gate["name"])
        gates = [
            {
                "name": name,
                "values": [
                    next(
                        (
                            {"passed": gate["passed"], "actual": gate["actual"], "threshold": gate["threshold"]}
                            for gate in ((item["result"].get("acceptance") or {}).get("gates") or [])
                            if gate["name"] == name
                        ),
                        None,
                    )
                    for item in resolved
                ],
            }
            for name in gate_names
        ]

        return {
            "runs": [
                {
                    "runId": item["runId"],
                    "strategyId": item["strategyId"],
                    "strategyName": item["strategyName"],
                    "strategyVersion": item["strategyVersion"],
                    "createdAt": item["createdAt"],
                    "outputDir": item["outputDir"],
                    "outcome": (item["result"].get("conclusion") or {}).get("outcome"),
                    "headline": (item["result"].get("conclusion") or {}).get("headline"),
                    "promotable": bool((item["result"].get("conclusion") or {}).get("promotable")),
                }
                for item in resolved
            ],
            "metrics": metrics,
            "gates": gates,
            "note": (
                "同列数字来自各自运行的产物；空缺表示该运行没有产出该字段，"
                "不代表 0。研究范围、宇宙和评估窗口不同的运行不可直接比较。"
            ),
        }

    # ------------------------------------------------------------------
    # deletion
    # ------------------------------------------------------------------

    def delete(
        self,
        strategy_id: str,
        *,
        version: str | None = None,
        delete_outputs: bool = False,
    ) -> dict[str, Any]:
        """Archive a strategy (or one version) and optionally its run outputs.

        Deleting moves the manifest into ``runtime/archives`` rather than
        unlinking it: a research record that can be silently destroyed is not an
        auditable one. Run outputs are only removed on an explicit request, and
        only from inside the runtime subtree.
        """
        directory = self._strategy_dir(strategy_id)
        if directory is None:
            raise KeyError(strategy_id)
        stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
        destination = self.archive_root / f"{strategy_id}_{stamp}"
        destination.mkdir(parents=True, exist_ok=True)

        moved: list[str] = []
        removed_outputs: list[str] = []
        errors: list[str] = []

        if version:
            source = directory / f"{version}.json"
            if not source.exists():
                raise KeyError(f"{strategy_id}@{version}")
            shutil.move(str(source), str(destination / source.name))
            moved.append(project_relative(self.settings, destination / source.name))
            remaining = self._versions(directory)
            if not remaining:
                # The last version is gone; retire the identity too.
                self._archive_runs(directory, destination, moved)
                shutil.rmtree(directory, ignore_errors=True)
        else:
            if delete_outputs:
                for run in self.runs(strategy_id):
                    error = self._remove_output(run.get("outputDir"))
                    if error:
                        errors.append(error)
                    elif run.get("outputDir"):
                        removed_outputs.append(str(run["outputDir"]))
            for item in sorted(directory.iterdir()):
                shutil.move(str(item), str(destination / item.name))
                moved.append(project_relative(self.settings, destination / item.name))
            directory.rmdir()

        return {
            "id": strategy_id,
            "version": version,
            "archivedTo": project_relative(self.settings, destination),
            "archivedFiles": moved,
            "outputsRemoved": removed_outputs,
            "errors": errors,
        }

    def _archive_runs(self, directory: Path, destination: Path, moved: list[str]) -> None:
        runs_path = directory / self.RUNS_FILENAME
        if runs_path.exists():
            shutil.move(str(runs_path), str(destination / runs_path.name))
            moved.append(project_relative(self.settings, destination / runs_path.name))

    def _remove_output(self, output_dir: str | None) -> str | None:
        if not output_dir:
            return None
        try:
            path = safe_project_path(self.settings, output_dir)
        except ValueError as exc:
            return f"{output_dir}: {exc}"
        runtime = self.settings.runtime_root.resolve()
        resolved = path.resolve()
        if resolved == runtime or runtime not in resolved.parents:
            return f"{output_dir}: refused, outside the bounded runtime subtree"
        if not path.exists():
            return None
        try:
            shutil.rmtree(path) if path.is_dir() else path.unlink()
        except OSError as exc:
            return f"{output_dir}: {exc}"
        return None

    def defaults(self) -> dict[str, Any]:
        groups = {
            "marketPanelPath": (
                "runtime/data/gold/full_universe/adjusted_market_panel.parquet",
                "runtime/data/v7/silver/market_panel/market_panel.parquet",
                "runtime/data/v7/full_universe/full_universe_market_panel.parquet",
                "runtime/data/u0/panel/daily_bars_raw.parquet",
            ),
            "labelsPath": (
                "runtime/data/gold/full_universe/labels.parquet",
                "runtime/data/v7/labels.parquet",
            ),
            "fundamentalsRoot": (
                "runtime/data/v7/silver/fundamentals",
                "runtime/data/u0/pit/fundamentals",
            ),
            "valuationPath": (
                "runtime/data/v7/silver/valuation/valuation.parquet",
                "runtime/data/u0/pit/valuation.parquet",
            ),
            "disclosuresPath": (
                "runtime/data/v7/silver/disclosures/disclosures.parquet",
                "runtime/data/u0/pit/disclosures.parquet",
            ),
            "trainingDatasetPath": (
                "runtime/data/gold/full_universe/training_dataset.parquet",
                "runtime/data/v7/gold/training_dataset/training_dataset.parquet",
            ),
            "sectorMapPath": (
                "runtime/data/v7/silver/sector_map.parquet",
                "runtime/data/u0/pit/sector_map.parquet",
            ),
        }
        selected: dict[str, str | None] = {}
        evidence: list[dict[str, Any]] = []
        options: dict[str, list[dict[str, Any]]] = {}
        for field, candidates in groups.items():
            found: list[tuple[int, float, str]] = []
            field_options: list[dict[str, Any]] = []
            for candidate in candidates:
                path = safe_project_path(self.settings, candidate)
                is_expected_input = path.is_dir() if field == "fundamentalsRoot" else path.is_file()
                available_horizons: list[int] = []
                if is_expected_input and field == "labelsPath":
                    try:
                        available_horizons = _parquet_horizons(path)
                    except Exception:
                        available_horizons = []
                if is_expected_input:
                    stat = path.stat()
                    modified_at = datetime.fromtimestamp(
                        stat.st_mtime,
                        timezone.utc,
                    ).isoformat(timespec="seconds")
                    option = {
                        "field": field,
                        "path": candidate,
                        "exists": True,
                        "isDirectory": path.is_dir(),
                        "sizeBytes": stat.st_size,
                        "modifiedAt": modified_at,
                        "availableHorizons": available_horizons,
                    }
                    evidence.append(option)
                    coverage_score = len(available_horizons) if field == "labelsPath" else 0
                    found.append((coverage_score, stat.st_mtime, candidate))
                else:
                    option = {
                        "field": field,
                        "path": candidate,
                        "exists": False,
                        "isDirectory": field == "fundamentalsRoot",
                        "sizeBytes": 0,
                        "modifiedAt": None,
                        "availableHorizons": [],
                    }
                field_options.append(option)
            found.sort(reverse=True, key=lambda item: (item[0], item[1]))
            selected[field] = found[0][2] if found else None
            options[field] = field_options
        return {
            "selected": selected,
            "options": options,
            "evidence": sorted(
                evidence,
                key=lambda item: item["modifiedAt"] or "",
                reverse=True,
            ),
            "selectionRule": (
                "labels prefer the widest available horizon schema, then newest; "
                "other inputs prefer newest existing canonical path"
            ),
        }

    def validate(self, draft: StrategyDraft) -> dict[str, Any]:
        issues: list[dict[str, Any]] = []

        def add_issue(
            code: str,
            severity: str,
            title: str,
            detail: str,
            *,
            field: str | None = None,
            action: dict[str, str] | None = None,
            evidence: dict[str, Any] | None = None,
        ) -> None:
            issue: dict[str, Any] = {
                "code": code,
                "severity": severity,
                "title": title,
                "detail": detail,
            }
            if field:
                issue["field"] = field
            if action:
                issue["action"] = action
            if evidence:
                issue["evidence"] = evidence
            issues.append(issue)

        inputs = {
            "marketPanelPath": draft.market_panel_path,
            "labelsPath": draft.labels_path,
            "sectorMapPath": draft.sector_map_path,
            "trainingDatasetPath": draft.training_dataset_path,
            "synthesizedFactorsPath": draft.synthesized_factors_path,
            "fundamentalsRoot": draft.fundamentals_root,
            "valuationPath": draft.valuation_path,
            "disclosuresPath": draft.disclosures_path,
            "minutePanelPath": draft.minute_panel_path,
            "universeSymbolsFile": draft.universe_symbols_file,
        }
        resolved_inputs: dict[str, str] = {}
        resolved_paths: dict[str, Path] = {}
        required_paths = {"marketPanelPath", "labelsPath"}
        if draft.do_t_mode in {"intraday", "both"}:
            required_paths.add("minutePanelPath")
        for key, value in inputs.items():
            if not value:
                continue
            try:
                path = safe_project_path(self.settings, value)
                resolved_inputs[key] = project_relative(self.settings, path)
                resolved_paths[key] = path
                if not path.exists():
                    required = key in required_paths
                    fundamental_input = key in {
                        "fundamentalsRoot",
                        "trainingDatasetPath",
                    } and "fundamental" in draft.stock_selection_modes
                    if required or not fundamental_input:
                        add_issue(
                            f"{key}_missing",
                            "blocking" if required else "warning",
                            "必需输入不存在" if required else "可选输入不可用",
                            (
                                f"{key}: input path does not exist"
                                if required
                                else (
                                    f"{key}: optional input path does not exist; "
                                    "dependent candidate will fail closed."
                                )
                            ),
                            field=key,
                            action={
                                "kind": "choose_input",
                                "label": "选择可用输入",
                            },
                        )
            except ValueError as exc:
                add_issue(
                    f"{key}_unsafe",
                    "blocking",
                    "输入路径不安全",
                    f"{key}: {exc}",
                    field=key,
                )

        requested_horizons = sorted(
            {int(value) for value in draft.horizons.split(",")}
        )
        available_horizons: list[int] = []
        labels_path = resolved_paths.get("labelsPath")
        if labels_path is not None and labels_path.exists():
            try:
                available_horizons = _parquet_horizons(labels_path)
            except Exception as exc:
                add_issue(
                    "labels_unreadable",
                    "blocking",
                    "Labels 无法读取",
                    (
                        "labelsPath: unreadable Parquet schema; rebuild the label artifact "
                        f"({type(exc).__name__})"
                    ),
                    field="labelsPath",
                    action={"kind": "rebuild_labels", "label": "重建 Labels"},
                )
            else:
                missing_horizons = sorted(
                    set(requested_horizons) - set(available_horizons)
                )
                if missing_horizons:
                    missing_columns = [
                        f"forward_return_{horizon}d"
                        for horizon in missing_horizons
                    ]
                    add_issue(
                        "missing_horizon_columns",
                        "blocking",
                        "Labels 缺少研究周期",
                        (
                            "labelsPath: missing requested horizon columns "
                            f"{', '.join(missing_columns)}; rebuild labels or remove "
                            "those Horizons"
                        ),
                        field="horizons",
                        action={
                            "kind": "resolve_horizons",
                            "label": "选择修复方式",
                        },
                        evidence={
                            "requestedHorizons": requested_horizons,
                            "availableHorizons": available_horizons,
                            "missingHorizons": missing_horizons,
                        },
                    )
        try:
            output = safe_project_path(self.settings, draft.output_dir)
            runtime = self.settings.runtime_root.resolve()
            if output != runtime and runtime not in output.parents:
                add_issue(
                    "output_outside_runtime",
                    "blocking",
                    "输出目录越界",
                    "outputDir must remain inside runtime",
                    field="outputDir",
                )
        except ValueError as exc:
            add_issue(
                "output_path_unsafe",
                "blocking",
                "输出目录不安全",
                f"outputDir: {exc}",
                field="outputDir",
            )

        if not draft.human_approved:
            add_issue(
                "human_gate_pending",
                "warning",
                "等待 Human Gate",
                "Human Gate 尚未授权；可保存草稿和验证，但不能启动。",
                field="humanApproved",
                action={"kind": "arm_human_gate", "label": "人工确认后授权"},
            )
        if draft.model == "ft_transformer" and not draft.require_gpu:
            add_issue(
                "gpu_not_enforced",
                "warning",
                "GPU 契约未锁定",
                "FT-Transformer 未强制 GPU；运行时将 fail closed。",
                field="requireGpu",
            )

        fundamentals_path = resolved_paths.get("fundamentalsRoot")
        training_path = resolved_paths.get("trainingDatasetPath")
        fundamentals_available = bool(
            (fundamentals_path and fundamentals_path.is_dir())
            or (training_path and training_path.is_file())
        )
        if (
            "fundamental" in draft.stock_selection_modes
            and not fundamentals_available
        ):
            has_baseline = "none" in draft.stock_selection_modes
            add_issue(
                "fundamentals_missing",
                "warning" if has_baseline else "blocking",
                "基本面候选缺少 PIT 输入",
                (
                    "基本面选股候选缺少 PIT 财务输入；该候选将 fail closed，"
                    + (
                        "无筛选基线仍可比较。"
                        if has_baseline
                        else "且没有无筛选基线可回退。"
                    )
                ),
                field="fundamentalsRoot",
                action={
                    "kind": "resolve_fundamentals",
                    "label": "选择目录或关闭基本面候选",
                },
            )

        if len(set(draft.top_k_candidates)) > 1:
            add_issue(
                "pareto_top_k_protocol",
                "info",
                "Top-K 使用同成本候选赛",
                "Top-K 将逐候选执行同一成本后回测，再从 Pareto 前沿按研究偏好选冠军。",
            )
        if (
            draft.fundamental_selection_mode == "auto"
            and "fundamental" in draft.stock_selection_modes
        ):
            candidate_count = len(set(draft.top_k_candidates)) * (
                (1 if "none" in draft.stock_selection_modes else 0)
                + len(set(draft.fundamental_threshold_candidates))
                * len(set(draft.fundamental_blend_candidates))
            )
            add_issue(
                "bounded_early_oos_search",
                "info",
                "参数只在早期 OOS 学习",
                (
                    "基本面权重、入围分位和 Top-K 将只在早期滚动 OOS 上搜索 "
                    f"{candidate_count} 组；参数冻结后，最终 holdout 只验收一次。"
                ),
                evidence={
                    "candidateCount": candidate_count,
                    "candidateLimit": draft.selection_max_candidates,
                },
            )

        blend_details = {
            "adaptive_oos": (
                "以早期 OOS 横截面 RankIC/稳定性学习非负权重；"
                "按各周期清除 forward-label 重叠区，并在最终 holdout 前冻结。"
            ),
            "balanced": "以 20D/60D 中周期为主的固定审计权重。",
            "short_tactical": "以 1D/5D 短周期为主，强调战术响应。",
            "long_fundamental": "以 20D/60D/120D 中长周期为主，强调基本面兑现。",
            "primary_only": "仅使用主周期，作为最易解释的消融基线。",
        }
        add_issue(
            "horizon_blend_policy",
            "info",
            "周期融合已声明",
            blend_details[draft.horizon_blend_method],
            evidence={"method": draft.horizon_blend_method},
        )

        add_issue(
            "overfit_gates",
            "info",
            "过拟合闸门",
            (
                f"PBO ≤ {draft.max_pbo:.2f}、"
                f"DSR probability ≥ {draft.min_dsr_probability:.2f}、"
                f"SPA p-value ≤ {draft.max_spa_pvalue:.2f}；任一失败即阻止晋级。"
            ),
        )
        if not draft.benchmark_symbol:
            add_issue(
                "benchmark_missing",
                "blocking" if draft.objective_weights.excess_return > 0 else "warning",
                "缺少超额收益基准",
                "未指定 benchmarkSymbol；无法验证最大超额目标。",
                field="benchmarkSymbol",
                action={"kind": "clear_excess_objective", "label": "把超额权重改为 0"},
            )
        else:
            # Measured against the actual panel, not assumed. A full-universe
            # stock panel normally holds no index symbol, and the run aborts on
            # this at ~60% progress — after the entire training has been paid for.
            panel = resolved_paths.get("marketPanelPath")
            present = (
                _panel_contains_symbol(panel, draft.benchmark_symbol)
                if panel is not None and panel.is_file()
                else None
            )
            if present is False:
                add_issue(
                    "benchmark_absent_from_panel",
                    "blocking",
                    "基准标的不在行情面板内",
                    (
                        f"benchmarkSymbol={draft.benchmark_symbol} 在所选行情面板中不存在。"
                        "运行会在组合阶段中止，而训练成本已经付出。"
                        "个股全宇宙面板通常不含指数标的。"
                    ),
                    field="benchmarkSymbol",
                    action={"kind": "resolve_benchmark", "label": "改用面板内标的或清空基准"},
                    evidence={
                        "benchmarkSymbol": draft.benchmark_symbol,
                        "marketPanelPath": resolved_inputs.get("marketPanelPath"),
                    },
                )

        # The nested selection protocol needs more OOS days than a short
        # walk-forward produces. This is pure arithmetic and knowable now; left
        # unchecked it aborts the run after training completes.
        projected_oos_days = draft.n_splits * OOS_DAYS_PER_FOLD
        # Uses the run's own definition, which reserves the day that NAV
        # differencing consumes before governance ever sees the segment.
        required_oos_days = _required_oos_days(
            draft.selection_min_oos_days, draft.selection_min_holdout_days
        )
        if projected_oos_days < required_oos_days:
            minimum_splits = -(-required_oos_days // OOS_DAYS_PER_FOLD)  # ceil
            add_issue(
                "insufficient_projected_oos_days",
                "blocking",
                "样本外交易日不足以执行选择与 holdout 协议",
                (
                    f"{draft.n_splits} 折 × {OOS_DAYS_PER_FOLD} 天 = {projected_oos_days} 个"
                    f"样本外交易日，少于协议所需的 {required_oos_days} 天"
                    f"（选择 {draft.selection_min_oos_days} + holdout "
                    f"{draft.selection_min_holdout_days}）。"
                    f"至少需要 {minimum_splits} 折。"
                ),
                field="nSplits",
                action={"kind": "resolve_oos_days", "label": f"提高到 {minimum_splits} 折"},
                evidence={
                    "nSplits": draft.n_splits,
                    "oosDaysPerFold": OOS_DAYS_PER_FOLD,
                    "projectedOosDays": projected_oos_days,
                    "requiredOosDays": required_oos_days,
                    "minimumSplits": minimum_splits,
                },
            )
        else:
            # The fold count above is what the request asks for; whether the
            # panel's date span can seat that many folds is a separate question,
            # and answering it needs the data.
            self._check_panel_supports_fold_budget(
                draft, resolved_paths.get("labelsPath"), required_oos_days, add_issue
            )
        if draft.do_t_mode in {"daily_swing", "both"}:
            add_issue(
                "daily_swing_contract",
                "info",
                "T+1 日线能力边界",
                "日线波段使用 ATR timing gate 与持有期软锁，不冒充盘中成交能力。",
            )
        if draft.universe_scope == "pilot":
            add_issue(
                "pilot_universe_scope",
                "warning",
                "试点宇宙：结论不可外推到全宇宙",
                (
                    "本次只在指定的试点股票集合上运行。它用于在投入数小时算力前"
                    "验证配置与链路；任何 IC、回撤或超额结论都只对该子集成立，"
                    "不得当作全宇宙结果引用。"
                ),
                field="universeScope",
                evidence={
                    "universeSymbolsFile": draft.universe_symbols_file,
                    "inlineSymbolCount": len(
                        [item for item in (draft.universe_symbols or "").split(",") if item.strip()]
                    ),
                },
            )
        add_issue(
            "research_only",
            "info",
            "研究目标不是收益承诺",
            "优化权重只表达研究偏好；验收只读取真实 OOS 与成本后回测产物。",
        )
        add_issue(
            "public_principles_only",
            "info",
            "原创且可审计的系统边界",
            (
                "架构吸收公开可验证的机构工程原则，并由本项目独立实现；"
                "不声称访问或复制任何机构未公开的私有系统。"
            ),
        )

        errors = [
            issue["detail"]
            for issue in issues
            if issue["severity"] == "blocking"
        ]
        warnings = [
            issue["detail"]
            for issue in issues
            if issue["severity"] == "warning"
        ]
        role_issue_codes = {
            "data_quality": {
                "marketPanelPath_missing",
                "labelsPath_missing",
                "labels_unreadable",
                "missing_horizon_columns",
                "fundamentals_missing",
            },
            "factor_research": {
                "fundamentals_missing",
                "horizon_blend_policy",
            },
            "model_validation": {
                "missing_horizon_columns",
                "gpu_not_enforced",
                "overfit_gates",
                "insufficient_projected_oos_days",
            },
            "portfolio": {
                "pareto_top_k_protocol",
                "bounded_early_oos_search",
                "benchmark_missing",
                "benchmark_absent_from_panel",
                "insufficient_projected_oos_days",
            },
            "backtest": {
                "minutePanelPath_missing",
                "daily_swing_contract",
                "benchmark_missing",
                "benchmark_absent_from_panel",
            },
            "risk": {
                "overfit_gates",
                "benchmark_missing",
                "research_only",
            },
            "challenger": {
                "bounded_early_oos_search",
                "overfit_gates",
                "public_principles_only",
            },
            "human_gate": {"human_gate_pending"},
        }
        next_actions = {
            "data_quality": "核对 PIT、覆盖率与 Labels schema",
            "factor_research": "检查融合权重和因子冗余证据",
            "model_validation": "复核滚动切分、embargo 与 holdout 隔离",
            "portfolio": "复核 Pareto 候选、集中度和换手",
            "backtest": "复核 A 股撮合、成本、涨跌停与 T+1",
            "risk": "复核回撤、压力场景与 kill switch",
            "challenger": "提交反证、敏感性和替代解释",
            "human_gate": "人工核对研究范围后决定是否授权",
        }
        decision_council: list[dict[str, Any]] = []
        for role_id, label, responsibility in DECISION_COUNCIL:
            relevant = [
                issue
                for issue in issues
                if issue["code"] in role_issue_codes[role_id]
            ]
            blocking = [
                issue for issue in relevant if issue["severity"] == "blocking"
            ]
            if role_id == "human_gate":
                status = "approved" if draft.human_approved else "waiting"
            else:
                status = "blocked" if blocking else "ready"
            primary_issue = blocking[0] if blocking else next(
                (
                    issue
                    for issue in relevant
                    if issue["severity"] == "warning"
                ),
                None,
            )
            decision_council.append(
                {
                    "id": role_id,
                    "label": label,
                    "responsibility": responsibility,
                    "status": status,
                    "veto": role_id
                    in {
                        "data_quality",
                        "model_validation",
                        "risk",
                        "challenger",
                        "human_gate",
                    },
                    "finding": (
                        primary_issue["title"]
                        if primary_issue
                        else "当前契约无角色级阻塞"
                    ),
                    "nextAction": (
                        primary_issue.get("action", {}).get("label")
                        if primary_issue
                        else next_actions[role_id]
                    ) or next_actions[role_id],
                    "issueCount": len(relevant),
                    "issueCodes": [issue["code"] for issue in relevant],
                }
            )

        return {
            "valid": not errors,
            "errors": errors,
            "warnings": warnings,
            "issues": issues,
            "requestedHorizons": requested_horizons,
            "availableHorizons": available_horizons,
            "resolvedInputs": resolved_inputs,
            "launch": {
                "jobType": "strategy-pipeline",
                "commandId": "run-full-real-training-v7",
                "parameters": self.launch_parameters(draft),
                "armed": draft.human_approved and not errors,
            },
            "decisionCouncil": decision_council,
        }

    def _check_panel_supports_fold_budget(
        self,
        draft: StrategyDraft,
        labels_path: Path | None,
        required_oos_days: int,
        add_issue,
    ) -> None:
        """Can this panel's date span actually seat the requested folds?

        ``n_splits x 20`` is what the operator asked for. Whether the data can
        deliver it depends on the panel's trading-day count minus the training
        window, the embargo and the label purge. The run itself re-checks this
        exactly once the dataset is built; this is the cheap early warning so an
        operator is not told to raise ``nSplits`` to a number the panel cannot
        support either.
        """
        if labels_path is None or not labels_path.is_file():
            return
        try:
            from quantagent.training.splitters import WalkForwardSplitConfig, plan_walk_forward
        except Exception:
            return
        panel_days = _panel_trading_days(labels_path)
        if panel_days is None:
            return

        horizons = [int(value) for value in str(draft.horizons).split(",") if value.strip().isdigit()]
        max_horizon = max(horizons) if horizons else int(draft.primary_horizon)
        # Feature warm-up consumes leading dates before any row is complete. The
        # exact cost depends on which factors survive screening, so this is a
        # deliberate under-estimate: it only fires when the shortfall is certain.
        optimistic_usable_days = panel_days - max_horizon
        cfg = WalkForwardSplitConfig(
            n_splits=draft.n_splits,
            valid_size_days=OOS_DAYS_PER_FOLD,
            min_train_days=_MIN_TRAIN_DAYS,
            embargo_days=_EMBARGO_DAYS,
            purge_days=max_horizon,
            mode=draft.split_mode,
        )
        plan = plan_walk_forward(optimistic_usable_days, cfg)
        if plan.oos_days >= required_oos_days:
            return
        add_issue(
            "panel_span_cannot_seat_folds",
            "blocking",
            "面板日期跨度不足以支撑所需折数",
            (
                f"标签面板有 {panel_days} 个交易日；扣除 {_MIN_TRAIN_DAYS} 天训练窗口、"
                f"{_EMBARGO_DAYS} 天 embargo 与 {max_horizon} 天标签 purge 后，"
                f"最多只能容纳 {plan.achievable_splits} 折 = {plan.oos_days} 个样本外交易日，"
                f"少于协议所需的 {required_oos_days} 天。"
                "提高 nSplits 无法解决，需要更长的数据跨度或更短的标签期限。"
            ),
            field="labelsPath",
            evidence={
                "panelTradingDays": panel_days,
                "maxHorizon": max_horizon,
                "achievableSplits": plan.achievable_splits,
                "achievableOosDays": plan.oos_days,
                "requiredOosDays": required_oos_days,
                "note": "warm-up 未计入，实际可用交易日只会更少",
            },
        )

    def save(self, draft: StrategyDraft) -> dict[str, Any]:
        validation = self.validate(draft)
        strategy_id = draft.id or self._slug(draft.name)
        strategy_dir = self.root / strategy_id
        strategy_dir.mkdir(parents=True, exist_ok=True)
        now = datetime.now(timezone.utc)
        version = now.strftime("%Y%m%dT%H%M%S%fZ")
        payload = {
            "schemaVersion": "quantagent.strategy.v1",
            "id": strategy_id,
            "version": version,
            "createdAt": now.isoformat(timespec="seconds"),
            "trustClass": "research_only",
            "draft": draft.model_dump(by_alias=True),
            "validation": validation,
        }
        content = json.dumps(payload, ensure_ascii=False, indent=2)
        payload["contentHash"] = sha256(content.encode("utf-8")).hexdigest()
        path = strategy_dir / f"{version}.json"
        path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
        return self._public(payload, path)

    @staticmethod
    def launch_parameters(draft: StrategyDraft) -> dict[str, Any]:
        parameters: dict[str, Any] = {
            "market_panel_path": draft.market_panel_path,
            "labels_path": draft.labels_path,
            "output_dir": draft.output_dir,
            "factor_library": draft.factor_library,
            "model": draft.model,
            "horizons": draft.horizons,
            "primary_horizon": draft.primary_horizon,
            "horizon_blend_method": draft.horizon_blend_method,
            "split_mode": draft.split_mode,
            "n_splits": draft.n_splits,
            "require_gpu": draft.require_gpu,
            "top_k": draft.top_k,
            "top_k_candidates": draft.top_k_candidates,
            "stock_selection_modes": draft.stock_selection_modes,
            "fundamental_selection_mode": draft.fundamental_selection_mode,
            "fundamental_selection_threshold": draft.fundamental_selection_threshold,
            "fundamental_blend_weight": draft.fundamental_blend_weight,
            "fundamental_threshold_candidates": draft.fundamental_threshold_candidates,
            "fundamental_blend_candidates": draft.fundamental_blend_candidates,
            "selection_max_candidates": draft.selection_max_candidates,
            "selection_min_oos_days": draft.selection_min_oos_days,
            "selection_min_holdout_days": draft.selection_min_holdout_days,
            "max_pbo": draft.max_pbo,
            "min_dsr_probability": draft.min_dsr_probability,
            "max_spa_pvalue": draft.max_spa_pvalue,
            "factor_screening_mode": draft.factor_screening_mode,
            "do_t_mode": draft.do_t_mode,
            "max_weight_per_name": draft.max_weight_per_name,
            "max_sector_weight": draft.max_sector_weight,
            "max_turnover": draft.max_turnover,
            "objective": draft.objective,
            "weighting": draft.weighting,
            "initial_cash": draft.initial_cash,
            "objective_excess_weight": draft.objective_weights.excess_return,
            "objective_annual_weight": draft.objective_weights.annual_return,
            "objective_drawdown_weight": draft.objective_weights.drawdown_control,
            "acceptance_max_drawdown": draft.risk_limits.max_drawdown,
            "acceptance_min_sharpe": draft.risk_limits.min_sharpe,
        }
        if draft.universe_scope == "pilot":
            if draft.universe_symbols:
                parameters["symbols"] = draft.universe_symbols
            if draft.universe_symbols_file:
                parameters["symbols_file"] = draft.universe_symbols_file
        optional = {
            "sector_map_path": draft.sector_map_path,
            "training_dataset_path": draft.training_dataset_path,
            "synthesized_factors_path": draft.synthesized_factors_path,
            "benchmark_symbol": draft.benchmark_symbol,
            "fundamentals_root": draft.fundamentals_root,
            "valuation_path": draft.valuation_path,
            "disclosures_path": draft.disclosures_path,
            "minute_panel_path": draft.minute_panel_path,
        }
        parameters.update({key: value for key, value in optional.items() if value})
        return parameters

    def _public(self, payload: dict[str, Any], path: Path) -> dict[str, Any]:
        return {
            "id": payload["id"],
            "version": payload["version"],
            "name": payload["draft"]["name"],
            "createdAt": payload["createdAt"],
            "trustClass": payload["trustClass"],
            "contentHash": payload.get("contentHash"),
            "path": project_relative(self.settings, path),
            "valid": bool(payload.get("validation", {}).get("valid")),
            "humanApproved": bool(payload["draft"].get("humanApproved")),
            "draft": payload["draft"],
        }

    @staticmethod
    def _slug(value: str) -> str:
        slug = re.sub(r"[^a-zA-Z0-9_.-]+", "-", value.strip()).strip("-").lower()
        return (slug or "strategy")[:48]
