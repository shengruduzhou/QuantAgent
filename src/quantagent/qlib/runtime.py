"""Lazy, governed runtime bridge to upstream ``pyqlib``."""

from __future__ import annotations

from dataclasses import dataclass
import importlib
import json
from pathlib import Path
import re
from typing import Any

import yaml

from quantagent.qlib.catalog import QLIB_MIN_VERSION, QLIB_SUPPORTED_SERIES


class QlibUnavailable(RuntimeError):
    """Raised when the optional Qlib runtime cannot satisfy the integration contract."""


def _version_tuple(value: str) -> tuple[int, int, int]:
    parts = [int(item) for item in re.findall(r"\d+", str(value))[:3]]
    while len(parts) < 3:
        parts.append(0)
    return tuple(parts[:3])  # type: ignore[return-value]


@dataclass
class QlibRuntime:
    """Access Qlib without making it a core QuantAgent import dependency.

    QuantAgent owns governance (PIT, split discipline, final-holdout policy and
    promotion gates).  Qlib owns its model/data/workflow/record/backtest/online
    implementations.  This bridge keeps the boundary explicit.
    """

    provider_uri: str | None = None
    region: str = "cn"
    allow_untested_version: bool = False
    _initialized: bool = False

    def _module(self, name: str):
        try:
            return importlib.import_module(name)
        except Exception as exc:  # pragma: no cover - depends on optional runtime
            raise QlibUnavailable(
                f"cannot import {name!r}; install QuantAgent research extras "
                f"(pyqlib>={QLIB_MIN_VERSION},<0.10)"
            ) from exc

    def qlib(self):
        module = self._module("qlib")
        version = str(getattr(module, "__version__", "0"))
        current = _version_tuple(version)
        minimum = _version_tuple(QLIB_MIN_VERSION)
        if not self.allow_untested_version:
            if current < minimum or current[:2] != minimum[:2]:
                raise QlibUnavailable(
                    f"unsupported pyqlib version {version!r}; QuantAgent currently "
                    f"tests {QLIB_SUPPORTED_SERIES} with minimum {QLIB_MIN_VERSION}. "
                    "Use allow_untested_version only for an explicit compatibility audit."
                )
        return module

    def initialize(self, *, provider_uri: str | None = None, region: str | None = None, **kwargs: Any) -> None:
        qlib = self.qlib()
        resolved_uri = provider_uri or self.provider_uri
        resolved_region = region or self.region
        if not resolved_uri:
            raise QlibUnavailable("Qlib provider_uri is required")
        qlib.init(provider_uri=str(Path(resolved_uri).expanduser()), region=resolved_region, **kwargs)
        self.provider_uri = str(Path(resolved_uri).expanduser())
        self.region = resolved_region
        self._initialized = True

    def ensure_initialized(self) -> None:
        if not self._initialized:
            self.initialize()

    def data_api(self):
        self.ensure_initialized()
        return self._module("qlib.data").D

    def calendar(self, *, start_time: str | None = None, end_time: str | None = None, freq: str = "day"):
        return self.data_api().calendar(start_time=start_time, end_time=end_time, freq=freq)

    def instruments(self, market: str = "all", **kwargs: Any):
        return self.data_api().instruments(market, **kwargs)

    def features(
        self,
        instruments: Any,
        fields: list[str] | tuple[str, ...],
        *,
        start_time: str | None = None,
        end_time: str | None = None,
        freq: str = "day",
        **kwargs: Any,
    ):
        return self.data_api().features(
            instruments,
            list(fields),
            start_time=start_time,
            end_time=end_time,
            freq=freq,
            **kwargs,
        )

    def instantiate(self, config: dict[str, Any], **kwargs: Any):
        """Instantiate any upstream Qlib component from its normal config dict."""
        utils = self._module("qlib.utils")
        return utils.init_instance_by_config(config, **kwargs)

    def recorder_api(self):
        """Return Qlib's experiment/Recorder facade ``R``."""
        self.qlib()
        return self._module("qlib.workflow").R

    def evaluate_module(self):
        """Return Qlib's analysis helpers (risk_analysis, indicator analysis, etc.)."""
        self.qlib()
        return self._module("qlib.contrib.evaluate")

    def task_train(
        self,
        task_config: dict[str, Any],
        *,
        experiment_name: str,
        recorder_name: str | None = None,
    ):
        self.ensure_initialized()
        trainer = self._module("qlib.model.trainer")
        return trainer.task_train(
            task_config,
            experiment_name=experiment_name,
            recorder_name=recorder_name,
        )

    def train_tasks(
        self,
        tasks: list[dict[str, Any]] | dict[str, Any],
        *,
        experiment_name: str,
        mode: str = "recorder",
        task_pool: str | None = None,
        **kwargs: Any,
    ):
        """Run Qlib TrainerR, DelayTrainerR or TrainerRM.

        ``task_manager``/TrainerRM is intentionally opt-in because upstream
        TaskManager normally requires MongoDB and has a materially different
        operational footprint.
        """
        self.ensure_initialized()
        trainer = self._module("qlib.model.trainer")
        normalized = [tasks] if isinstance(tasks, dict) else list(tasks)
        if mode == "recorder":
            obj = trainer.TrainerR(experiment_name=experiment_name, **kwargs)
        elif mode == "delay":
            obj = trainer.DelayTrainerR(experiment_name=experiment_name, **kwargs)
        elif mode == "task_manager":
            obj = trainer.TrainerRM(
                experiment_name=experiment_name,
                task_pool=task_pool or experiment_name,
                **kwargs,
            )
        else:
            raise ValueError("mode must be recorder, delay, or task_manager")
        return obj(normalized)

    def build_online_manager(self, config: dict[str, Any]):
        """Instantiate an upstream OnlineManager/OnlineStrategy graph from config."""
        self.ensure_initialized()
        return self.instantiate(config)

    def run_workflow_config(
        self,
        config: dict[str, Any] | str | Path,
        *,
        experiment_name: str | None = None,
        recorder_name: str | None = None,
    ):
        """Execute the qrun research core from YAML/dict through upstream task_train.

        The method deliberately supports the stable qrun contract (``qlib_init``
        + ``task``) instead of importing Qlib's CLI internals.
        """
        payload = _load_workflow_payload(config)
        init_cfg = dict(payload.get("qlib_init") or {})
        provider_uri = init_cfg.pop("provider_uri", None) or self.provider_uri
        region = init_cfg.pop("region", None) or self.region
        self.initialize(provider_uri=provider_uri, region=region, **init_cfg)
        task = payload.get("task")
        if not isinstance(task, dict):
            raise ValueError("Qlib workflow config requires a mapping at 'task'")
        exp = (
            experiment_name
            or payload.get("experiment_name")
            or payload.get("experiment")
            or "quantagent-qlib"
        )
        return self.task_train(task, experiment_name=str(exp), recorder_name=recorder_name)

    def health_check(self, *, probe_components: bool = True) -> dict[str, object]:
        try:
            qlib = self.qlib()
        except QlibUnavailable as exc:
            return {"status": "unavailable", "reason": str(exc)}
        result: dict[str, object] = {
            "status": "passed",
            "version": str(getattr(qlib, "__version__", "unknown")),
            "tested_series": QLIB_SUPPORTED_SERIES,
            "minimum_version": QLIB_MIN_VERSION,
            "provider_uri": self.provider_uri,
            "initialized": self._initialized,
        }
        if probe_components:
            probes = {
                "data": "qlib.data",
                "dataset": "qlib.data.dataset",
                "model": "qlib.model",
                "strategy_backtest": "qlib.backtest",
                "workflow": "qlib.workflow",
                "recorder": "qlib.workflow.recorder",
                "report": "qlib.contrib.evaluate",
                "online": "qlib.workflow.online",
                "task_management": "qlib.workflow.task.manage",
                "meta": "qlib.model.meta",
                "reinforcement_learning": "qlib.rl",
                "highfreq": "qlib.contrib.strategy",
                "serialization": "qlib.utils.serial",
            }
            statuses: dict[str, str] = {}
            for key, module_name in probes.items():
                try:
                    importlib.import_module(module_name)
                except Exception as exc:  # pragma: no cover - optional dependency
                    statuses[key] = f"failed:{type(exc).__name__}"
                else:
                    statuses[key] = "passed"
            result["components"] = statuses
            if any(value != "passed" for value in statuses.values()):
                result["status"] = "warning"
        return result


def _load_workflow_payload(config: dict[str, Any] | str | Path) -> dict[str, Any]:
    if isinstance(config, dict):
        return dict(config)
    path = Path(config).expanduser()
    if not path.exists():
        raise FileNotFoundError(path)
    with path.open("r", encoding="utf-8") as handle:
        if path.suffix.lower() == ".json":
            payload = json.load(handle)
        else:
            payload = yaml.safe_load(handle)
    if not isinstance(payload, dict):
        raise ValueError("Qlib workflow config must decode to a mapping")
    return payload
