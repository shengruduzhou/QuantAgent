"""Qlib task builders with QuantAgent split and leakage guardrails."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import pandas as pd

from quantagent.data.v7_auto_range import to_qlib_instrument


@dataclass(frozen=True)
class QlibSegments:
    train: tuple[str, str]
    valid: tuple[str, str]
    test: tuple[str, str]

    def validate(self, *, minimum_gap_days: int = 0) -> None:
        parsed = {
            "train": tuple(pd.Timestamp(value) for value in self.train),
            "valid": tuple(pd.Timestamp(value) for value in self.valid),
            "test": tuple(pd.Timestamp(value) for value in self.test),
        }
        for name, (start, end) in parsed.items():
            if pd.isna(start) or pd.isna(end) or start > end:
                raise ValueError(f"invalid Qlib {name} segment: {getattr(self, name)}")
        if parsed["train"][1] >= parsed["valid"][0]:
            raise ValueError("Qlib train/valid segments overlap or touch")
        if parsed["valid"][1] >= parsed["test"][0]:
            raise ValueError("Qlib valid/test segments overlap or touch")
        if minimum_gap_days < 0:
            raise ValueError("minimum_gap_days must be >= 0")
        if minimum_gap_days:
            tv_gap = (parsed["valid"][0] - parsed["train"][1]).days - 1
            vt_gap = (parsed["test"][0] - parsed["valid"][1]).days - 1
            if tv_gap < minimum_gap_days or vt_gap < minimum_gap_days:
                raise ValueError(
                    "Qlib segment gap is smaller than the governed purge/embargo "
                    f"requirement ({minimum_gap_days} calendar days)"
                )

    def to_dict(self) -> dict[str, list[str]]:
        self.validate()
        return {key: list(value) for key, value in asdict(self).items()}


def _govern_processor_config(
    processor: dict[str, Any] | str,
    *,
    train_start: str,
    train_end: str,
) -> dict[str, Any] | str:
    """Force any explicit Qlib fit window to remain inside the train segment."""
    if isinstance(processor, str):
        return processor
    result = dict(processor)
    kwargs = dict(result.get("kwargs") or {})
    fit_start = kwargs.get("fit_start_time")
    fit_end = kwargs.get("fit_end_time")
    if (fit_start is None) ^ (fit_end is None):
        raise ValueError("Qlib processor fit_start_time/fit_end_time must be supplied together")
    if fit_start is not None:
        if pd.Timestamp(fit_start) < pd.Timestamp(train_start):
            raise ValueError("Qlib processor fit_start_time precedes the train segment")
        if pd.Timestamp(fit_end) > pd.Timestamp(train_end):
            raise ValueError("Qlib processor fit_end_time exceeds the train segment")
    result["kwargs"] = kwargs
    return result


def build_static_parquet_task(
    *,
    parquet_path: str | Path,
    model_config: dict[str, Any],
    segments: QlibSegments,
    benchmark_symbol: str,
    topk: int = 50,
    n_drop: int = 5,
    account: float = 100_000_000.0,
    limit_threshold: float = 0.095,
    deal_price: str = "close",
    open_cost: float = 0.0005,
    close_cost: float = 0.0015,
    min_cost: float = 5.0,
    infer_processors: list[dict[str, Any] | str] | None = None,
    learn_processors: list[dict[str, Any] | str] | None = None,
    minimum_gap_days: int = 0,
    include_signal_analysis: bool = True,
) -> dict[str, Any]:
    """Build a qrun-compatible task using QuantAgent's PIT Parquet artifact."""
    path = Path(parquet_path).expanduser()
    if path.suffix.lower() != ".parquet":
        raise ValueError("parquet_path must end in .parquet")
    if not model_config or not isinstance(model_config, dict):
        raise ValueError("model_config must be a non-empty Qlib config mapping")
    if not str(benchmark_symbol).strip():
        raise ValueError("benchmark_symbol is required; QuantAgent never guesses a benchmark")
    if topk <= 0 or n_drop < 0 or n_drop > topk:
        raise ValueError("require topk > 0 and 0 <= n_drop <= topk")
    segments.validate(minimum_gap_days=minimum_gap_days)

    train_start, train_end = segments.train
    infer = [
        _govern_processor_config(item, train_start=train_start, train_end=train_end)
        for item in (infer_processors or [])
    ]
    learn = [
        _govern_processor_config(item, train_start=train_start, train_end=train_end)
        for item in (learn_processors or [])
    ]

    handler = {
        "class": "DataHandlerLP",
        "module_path": "qlib.data.dataset.handler",
        "kwargs": {
            "data_loader": {
                "class": "StaticDataLoader",
                "module_path": "qlib.data.dataset.loader",
                "kwargs": {"config": str(path)},
            },
            "infer_processors": infer,
            "learn_processors": learn,
        },
    }
    dataset = {
        "class": "DatasetH",
        "module_path": "qlib.data.dataset",
        "kwargs": {
            "handler": handler,
            "segments": segments.to_dict(),
        },
    }

    qlib_benchmark = to_qlib_instrument(benchmark_symbol)
    port_analysis = {
        "strategy": {
            "class": "TopkDropoutStrategy",
            "module_path": "qlib.contrib.strategy",
            "kwargs": {
                "signal": "<PRED>",
                "topk": int(topk),
                "n_drop": int(n_drop),
            },
        },
        "backtest": {
            "start_time": segments.test[0],
            "end_time": segments.test[1],
            "account": float(account),
            "benchmark": qlib_benchmark,
            "exchange_kwargs": {
                "limit_threshold": float(limit_threshold),
                "deal_price": deal_price,
                "open_cost": float(open_cost),
                "close_cost": float(close_cost),
                "min_cost": float(min_cost),
            },
        },
    }

    records: list[dict[str, Any]] = [
        {
            "class": "SignalRecord",
            "module_path": "qlib.workflow.record_temp",
            "kwargs": {"model": "<MODEL>", "dataset": "<DATASET>"},
        }
    ]
    if include_signal_analysis:
        records.append(
            {
                "class": "SigAnaRecord",
                "module_path": "qlib.workflow.record_temp",
                "kwargs": {"ana_long_short": False, "ann_scaler": 252},
            }
        )
    records.append(
        {
            "class": "PortAnaRecord",
            "module_path": "qlib.workflow.record_temp",
            "kwargs": {"config": port_analysis},
        }
    )
    return {
        "model": dict(model_config),
        "dataset": dataset,
        "record": records,
    }


def build_workflow_payload(
    *,
    provider_uri: str | Path,
    task: dict[str, Any],
    region: str = "cn",
    experiment_name: str = "quantagent-qlib",
) -> dict[str, Any]:
    if not task:
        raise ValueError("task is required")
    return {
        "qlib_init": {
            "provider_uri": str(Path(provider_uri).expanduser()),
            "region": region,
        },
        "experiment_name": experiment_name,
        "task": task,
    }
