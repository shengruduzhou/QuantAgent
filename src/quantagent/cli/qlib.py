"""CLI surface for the governed Qlib integration."""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
import typer
import yaml

from quantagent.cli._utils import app
from quantagent.qlib.docs_audit import audit_live_documentation, build_coverage_audit
from quantagent.qlib.parquet import write_qlib_static_parquet
from quantagent.qlib.runtime import QlibRuntime
from quantagent.qlib.workflow import QlibSegments, build_static_parquet_task, build_workflow_payload


def _parse_csv(raw: str) -> tuple[str, ...]:
    return tuple(dict.fromkeys(item.strip() for item in raw.split(",") if item.strip()))


def _read_frame(path: Path) -> pd.DataFrame:
    suffix = path.suffix.lower()
    if suffix == ".parquet":
        return pd.read_parquet(path)
    if suffix == ".csv":
        return pd.read_csv(path)
    raise typer.BadParameter("input must be .parquet or .csv")


def _write_json(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2, default=str), encoding="utf-8")


@app.command("qlib-capabilities")
def qlib_capabilities(output: str = typer.Option("", help="Optional JSON output path.")) -> None:
    payload = build_coverage_audit()
    if output:
        _write_json(Path(output).expanduser(), payload)
    typer.echo(json.dumps(payload, ensure_ascii=False, indent=2, default=str))


@app.command("audit-qlib-coverage")
def audit_qlib_coverage(
    output: str = typer.Option("data/qlib/coverage.json"),
    live_docs: bool = typer.Option(False, "--live-docs/--no-live-docs"),
    allow_network: bool = typer.Option(False, "--allow-network/--no-allow-network"),
) -> None:
    payload: dict[str, object] = {"static": build_coverage_audit()}
    if live_docs:
        if not allow_network:
            raise typer.BadParameter("--live-docs requires explicit --allow-network")
        payload["live"] = audit_live_documentation()
        if payload["live"].get("status") != "passed":  # type: ignore[union-attr]
            _write_json(Path(output).expanduser(), payload)
            raise typer.Exit(code=2)
    _write_json(Path(output).expanduser(), payload)
    typer.echo(json.dumps(payload, ensure_ascii=False, indent=2, default=str))


@app.command("prepare-qlib-parquet")
def prepare_qlib_parquet(
    input_path: str = typer.Option(..., "--input"),
    output_path: str = typer.Option(..., "--output"),
    features: str = typer.Option(..., help="Comma-separated feature columns."),
    labels: str = typer.Option("", help="Comma-separated label columns."),
    symbol_column: str = typer.Option("symbol"),
    time_column: str = typer.Option("available_at"),
    manifest_output: str = typer.Option(""),
) -> None:
    source = _read_frame(Path(input_path).expanduser())
    manifest = write_qlib_static_parquet(
        source,
        Path(output_path).expanduser(),
        feature_columns=_parse_csv(features),
        label_columns=_parse_csv(labels),
        symbol_column=symbol_column,
        time_column=time_column,
    )
    payload = manifest.to_dict()
    if manifest_output:
        _write_json(Path(manifest_output).expanduser(), payload)
    typer.echo(json.dumps(payload, ensure_ascii=False, indent=2, default=str))


@app.command("build-qlib-task")
def build_qlib_task(
    parquet_path: str = typer.Option(...),
    model_config: str = typer.Option(..., help="YAML/JSON file containing a Qlib model config."),
    benchmark_symbol: str = typer.Option(...),
    train_start: str = typer.Option(...),
    train_end: str = typer.Option(...),
    valid_start: str = typer.Option(...),
    valid_end: str = typer.Option(...),
    test_start: str = typer.Option(...),
    test_end: str = typer.Option(...),
    output: str = typer.Option("data/qlib/workflow.yaml"),
    minimum_gap_days: int = typer.Option(0),
    provider_uri: str = typer.Option(""),
    experiment_name: str = typer.Option("quantagent-qlib"),
) -> None:
    model_path = Path(model_config).expanduser()
    raw = model_path.read_text(encoding="utf-8")
    model = json.loads(raw) if model_path.suffix.lower() == ".json" else yaml.safe_load(raw)
    if not isinstance(model, dict):
        raise typer.BadParameter("model config must decode to a mapping")
    segments = QlibSegments(
        train=(train_start, train_end),
        valid=(valid_start, valid_end),
        test=(test_start, test_end),
    )
    task = build_static_parquet_task(
        parquet_path=parquet_path,
        model_config=model,
        segments=segments,
        benchmark_symbol=benchmark_symbol,
        minimum_gap_days=minimum_gap_days,
    )
    payload: dict[str, object]
    if provider_uri:
        payload = build_workflow_payload(
            provider_uri=provider_uri,
            task=task,
            experiment_name=experiment_name,
        )
    else:
        payload = {"experiment_name": experiment_name, "task": task}
    output_path = Path(output).expanduser()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(yaml.safe_dump(payload, allow_unicode=True, sort_keys=False), encoding="utf-8")
    typer.echo(str(output_path))


@app.command("run-qlib-workflow")
def run_qlib_workflow(
    config: str = typer.Option(...),
    provider_uri: str = typer.Option(""),
    region: str = typer.Option("cn"),
    experiment_name: str = typer.Option(""),
    recorder_name: str = typer.Option(""),
    allow_untested_version: bool = typer.Option(False),
) -> None:
    runtime = QlibRuntime(
        provider_uri=provider_uri or None,
        region=region,
        allow_untested_version=allow_untested_version,
    )
    recorder = runtime.run_workflow_config(
        Path(config).expanduser(),
        experiment_name=experiment_name or None,
        recorder_name=recorder_name or None,
    )
    info = getattr(recorder, "info", {})
    typer.echo(json.dumps({"status": "passed", "recorder": info}, ensure_ascii=False, indent=2, default=str))


@app.command("qlib-runtime-check")
def qlib_runtime_check(
    provider_uri: str = typer.Option(""),
    region: str = typer.Option("cn"),
    allow_untested_version: bool = typer.Option(False),
) -> None:
    runtime = QlibRuntime(
        provider_uri=provider_uri or None,
        region=region,
        allow_untested_version=allow_untested_version,
    )
    typer.echo(json.dumps(runtime.health_check(), ensure_ascii=False, indent=2, default=str))
