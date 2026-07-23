from __future__ import annotations

from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
import json
from pathlib import Path
import re
import subprocess
import sys
from threading import RLock, Thread
from typing import Any, Callable, Iterator
from uuid import uuid4

from services.quant_api.config import ApiSettings, project_relative, safe_project_path
from services.quant_api.events import EventBroker


@dataclass
class JobRecord:
    id: str
    type: str
    status: str
    commandId: str
    createdAt: str
    startedAt: str | None = None
    finishedAt: str | None = None
    progress: float | None = None
    message: str | None = None
    outputPaths: list[str] = field(default_factory=list)
    error: str | None = None
    logPath: str | None = None


COMMANDS: dict[str, dict[str, Any]] = {
    "fetch-tickflow-daily": {
        "type": "data",
        "entrypoint": "scripts/fetch_tickflow_daily_klines.py",
        "required": {"start_date", "end_date", "output", "allow_network"},
        "required_any": (("symbols", "symbols_file"),),
        "allowed": {"symbols", "symbols_file", "start_date", "end_date", "batch_size", "output", "allow_network"},
        "path_inputs": {"symbols_file"},
        "path_outputs": {"output"},
        "control": {"allow_network"},
    },
    "fetch-tickflow-minute": {
        "type": "data",
        "entrypoint": "scripts/fetch_tickflow_minute_history.py",
        "required": {"start", "end", "allow_network"},
        "required_any": (("symbols", "symbols_file", "holdings_csv"),),
        "allowed": {"symbols", "symbols_file", "holdings_csv", "start", "end", "sleep", "limit", "allow_network"},
        "path_inputs": {"symbols_file", "holdings_csv"},
        "path_outputs": set(),
        "fixed_outputs": ("runtime/data/v7/silver/minute_bars",),
        "control": {"allow_network"},
    },
    "record-tickflow-depth": {
        "type": "data",
        "entrypoint": "scripts/collect_tickflow_depth.py",
        "required": {"allow_network"},
        "required_any": (("symbols", "symbols_file", "book_csv"),),
        "allowed": {"symbols", "symbols_file", "book_csv", "loop_seconds", "max_iterations", "sleep", "allow_network"},
        "path_inputs": {"symbols_file", "book_csv"},
        "path_outputs": set(),
        "fixed_outputs": ("runtime/data/v7/silver/depth_snapshots",),
        "control": {"allow_network"},
    },
    "record-tickflow-quotes": {
        "type": "data",
        "entrypoint": "scripts/record_tickflow_quotes.py",
        "required": {"allow_network"},
        "required_any": (("symbols", "symbols_file"),),
        "allowed": {"symbols", "symbols_file", "loop_seconds", "max_iterations", "allow_network"},
        "path_inputs": {"symbols_file"},
        "path_outputs": set(),
        "fixed_outputs": ("runtime/data/v7/silver/tick_snapshots",),
        "control": {"allow_network"},
    },
    "data-manager-transfer": {
        "type": "data",
        "entrypoint": "scripts/data_manager_transfer.py",
        "required": {"operation", "source", "output"},
        "allowed": {"operation", "source", "output", "date_column", "symbol_column", "start_date", "end_date", "symbols"},
        "path_inputs": {"source"},
        "path_outputs": {"output"},
        "control": set(),
        "choices": {"operation": {"import", "export"}},
    },
    "run-strict-a-share-backtest-v8": {
        "type": "backtest",
        "required": {"target_weights_path", "market_panel_path", "output_dir"},
        "allowed": {
            "target_weights_path", "market_panel_path", "sector_map_path",
            "factor_weights_path", "output_dir", "slippage_bps", "initial_cash",
        },
        "path_inputs": {"target_weights_path", "market_panel_path", "sector_map_path", "factor_weights_path"},
        "path_outputs": {"output_dir"},
    },
    "train-v8-deep": {
        "type": "train",
        "required": {"dataset_path", "silver_panel_path", "output_dir"},
        "allowed": {
            "horizon_class", "dataset_path", "silver_panel_path", "symbols", "symbols_file",
            "train_start", "train_end", "test_end", "embargo_days", "top_k", "max_epochs",
            "batch_size", "d_token", "n_blocks", "n_heads", "dates_per_step",
            "train_micro_batch", "cross_sectional_norm", "label_norm", "feature_policy",
            "attention_dropout", "ffn_dropout", "weight_decay", "early_stopping_patience",
            "learning_rate", "regime_filter", "regime_min_rows", "require_gpu", "output_dir",
        },
        "path_inputs": {"dataset_path", "silver_panel_path", "symbols_file"},
        "path_outputs": {"output_dir"},
    },
    "synthesize-factors-v7": {
        "type": "factor-discovery",
        "required": {"market_panel_path", "output_dir"},
        "allowed": {
            "market_panel_path", "labels_path", "output_dir", "rd_agent",
            "label_column", "rounds", "factors_per_round", "population",
            "generations", "top_k", "max_depth", "validation_fraction",
            "min_validation_rank_ic", "fitness_sample_dates",
            "fitness_sample_symbols", "seed", "warm_start_fraction",
            "icir_weight", "reference_columns", "max_reference_correlation",
            "max_sota_correlation", "use_llm", "allow_network", "llm_model",
            "llm_start_round", "llm_candidates_per_round",
            "rag_escalation_round", "llm_timeout_seconds", "memory_path",
            "train_end", "exclude_st", "min_validation_icir",
        },
        "path_inputs": {"market_panel_path", "labels_path", "memory_path"},
        "path_outputs": {"output_dir"},
        "control": {"allow_network"},
        "conditional_controls": {"allow_network": "use_llm"},
    },
    "predict-alpha-v7": {
        "type": "infer",
        "required": {"model_dir", "feature_dataset", "output"},
        "allowed": {"model_dir", "feature_dataset", "output", "primary_horizon"},
        "path_inputs": {"model_dir", "feature_dataset"},
        "path_outputs": {"output"},
    },
    "build-akshare-market-panel-v7": {
        "type": "data",
        "required": {"start_date", "end_date", "output", "allow_network"},
        "required_any": (("symbols", "symbols_file"),),
        "allowed": {
            "symbols", "symbols_file", "start_date", "end_date", "output_root", "output",
            "allow_network", "adjust", "provider_uri_for_range", "as_of_date",
        },
        "path_inputs": {"symbols_file", "provider_uri_for_range"},
        "path_outputs": {"output_root", "output"},
    },
    "build-market-panel-v7": {
        "type": "data",
        "required": {"provider_uri", "start_date", "end_date", "output_root"},
        "required_any": (("symbols", "symbols_file", "universe"),),
        "allowed": {
            "provider_uri", "start_date", "end_date", "output_root", "symbols",
            "symbols_file", "universe", "region", "require_optional_flags",
        },
        "path_inputs": {"provider_uri", "symbols_file"},
        "path_outputs": {"output_root"},
    },
    "build-fundamentals-v7": {
        "type": "data",
        "required": {"start_date", "end_date", "provider", "fundamentals_root", "allow_network"},
        "required_any": (("symbols", "symbols_file"),),
        "allowed": {
            "symbols", "symbols_file", "start_date", "end_date", "provider",
            "fundamentals_root", "allow_network", "token_env",
        },
        "path_inputs": {"symbols_file"},
        "path_outputs": {"fundamentals_root"},
    },
    # ---- H-031 governed operational / full-universe data commands -----------
    # All are parameterless or take a single bounded control; none exposes a
    # free-form shell field. Outputs are fixed Runtime paths; none reads or
    # reports candidate performance.
    "validate-shadow-days": {
        "type": "governance",
        "entrypoint": "scripts/shadow_day_registry.py",
        "required": set(),
        "allowed": {"quiet"},
        "path_inputs": set(),
        "path_outputs": set(),
        "fixed_outputs": (
            "runtime/paper/fresh_blind/shadow_day_registry.json",
            "runtime/paper/fresh_blind/shadow_accumulating_status.json",
        ),
        "control": set(),
    },
    "certify-s4-batch-replay": {
        "type": "governance",
        "entrypoint": "scripts/s4_batch_replay.py",
        "required": set(),
        "allowed": {"cutoff"},
        "path_inputs": set(),
        "path_outputs": set(),
        "fixed_outputs": ("runtime/reports/h030/s4_readiness_certificate.json",),
        "control": set(),
    },
    "build-u0-security-master": {
        "type": "data",
        "entrypoint": "scripts/u0_build_security_master.py",
        "required": set(),
        "allowed": set(),
        "path_inputs": set(),
        "path_outputs": set(),
        "fixed_outputs": (
            "runtime/data/u0/historical_security_master.parquet",
            "runtime/data/u0/pit_field_availability.json",
        ),
        "control": set(),
    },
    "report-u0-provider-coverage": {
        "type": "data",
        "entrypoint": "scripts/u0_provider_coverage.py",
        "required": set(),
        "allowed": set(),
        "path_inputs": set(),
        "path_outputs": set(),
        "fixed_outputs": (
            "runtime/data/u0/provider_coverage_matrix.parquet",
            "runtime/data/u0/provider_coverage_matrix.csv",
        ),
        "control": set(),
    },
    "assemble-u0-full-universe": {
        "type": "data",
        "entrypoint": "scripts/u0_full_universe_backfill.py",
        "fixed_args": ("assemble",),
        "required": set(),
        "allowed": set(),
        "path_inputs": set(),
        "path_outputs": set(),
        "fixed_outputs": ("runtime/data/v7/full_universe/full_universe_market_panel.parquet",),
        "control": set(),
    },
    "audit-u0-full-universe": {
        "type": "data",
        "entrypoint": "scripts/u0_audit.py",
        "required": set(),
        "allowed": set(),
        "path_inputs": set(),
        "path_outputs": set(),
        "fixed_outputs": ("runtime/data/u0/full_universe_readiness_certificate.json",),
        "control": set(),
    },
    "backfill-u0-market-panel": {
        "type": "data",
        "entrypoint": "scripts/u0_full_universe_backfill.py",
        "fixed_args": ("fetch",),
        "required": {"allow_network"},
        "allowed": {"max_minutes", "allow_network"},
        "path_inputs": set(),
        "path_outputs": set(),
        "fixed_outputs": ("runtime/data/v7/full_universe/_staging",),
        "control": {"allow_network"},
    },
}


class JobManager:
    def __init__(
        self,
        settings: ApiSettings,
        events: EventBroker | None = None,
        on_success: Callable[[], None] | None = None,
    ) -> None:
        self.settings = settings
        self.events = events
        self.on_success = on_success
        self.state_path = settings.jobs_root / "jobs.json"
        self._lock = RLock()
        self._jobs: dict[str, JobRecord] = {}
        self._processes: dict[str, subprocess.Popen] = {}
        self._cancel_requested: set[str] = set()
        self._load()

    def submit(self, job_type: str, command_id: str, parameters: dict[str, Any]) -> dict[str, Any]:
        spec, normalized = self._validate(job_type, command_id, parameters)
        job_id = f"job_{uuid4().hex[:16]}"
        log_path = self.settings.jobs_root / f"{job_id}.log"
        record = JobRecord(
            id=job_id,
            type=job_type,
            status="queued",
            commandId=command_id,
            createdAt=_now(),
            message="queued",
            logPath=project_relative(self.settings, log_path),
        )
        with self._lock:
            self._jobs[job_id] = record
            self._persist()
        self._emit(record)
        Thread(target=self._run, args=(job_id, command_id, normalized, spec, log_path), daemon=True).start()
        return self._public(record)

    def validate(self, job_type: str, command_id: str, parameters: dict[str, Any]) -> dict[str, Any]:
        spec, _ = self._validate(job_type, command_id, parameters)
        outputs = [
            str(parameters[key])
            for key in spec["path_outputs"]
            if parameters.get(key) not in (None, "")
        ]
        outputs.extend(spec.get("fixed_outputs", ()))
        warnings = ["GPU availability is checked by the training process"] if parameters.get("require_gpu") else []
        if command_id == "synthesize-factors-v7":
            warnings.append("Factor discovery writes research candidates only; registration and training remain separate human-gated steps")
            if parameters.get("use_llm"):
                warnings.append("LLM network execution is armed for this research job")
        return {
            "valid": True,
            "type": job_type,
            "commandId": command_id,
            "entrypoint": spec.get("entrypoint") or "quantagent.cli",
            "outputPaths": outputs,
            "warnings": warnings,
        }

    def _validate(
        self,
        job_type: str,
        command_id: str,
        parameters: dict[str, Any],
    ) -> tuple[dict[str, Any], dict[str, Any]]:
        spec = COMMANDS.get(command_id)
        if spec is None or spec["type"] != job_type:
            raise ValueError(f"command {command_id!r} is not allowed for {job_type}")
        unknown = set(parameters) - set(spec["allowed"])
        if unknown:
            raise ValueError(f"unsupported parameters: {sorted(unknown)}")
        missing = {
            key for key in spec.get("required", set())
            if parameters.get(key) in (None, "", [])
        }
        if missing:
            raise ValueError(f"missing required parameters: {sorted(missing)}")
        for key, choices in spec.get("choices", {}).items():
            if parameters.get(key) not in choices:
                raise ValueError(f"{key} must be one of {sorted(choices)}")
        conditional_controls = spec.get("conditional_controls", {})
        for control_key in spec.get("control", set()):
            trigger_key = conditional_controls.get(control_key)
            required = trigger_key is None or parameters.get(trigger_key) is True
            if required and parameters.get(control_key) is not True:
                raise ValueError(f"{control_key} must be explicitly confirmed")
        for group in spec.get("required_any", ()):
            if not any(parameters.get(key) not in (None, "", []) for key in group):
                raise ValueError(f"one of {sorted(group)} is required")
        normalized = self._normalize_parameters(spec, parameters)
        return spec, normalized

    def list(self) -> list[dict[str, Any]]:
        with self._lock:
            return [
                self._public(record)
                for record in sorted(self._jobs.values(), key=lambda item: item.createdAt, reverse=True)
            ]

    def get(self, job_id: str) -> dict[str, Any] | None:
        with self._lock:
            record = self._jobs.get(job_id)
            return self._public(record) if record else None


    def cancel(self, job_id: str) -> dict[str, Any]:
        terminal = {"succeeded", "failed", "cancelled"}
        with self._lock:
            record = self._jobs.get(job_id)
            if record is None:
                raise KeyError(job_id)
            if record.status in terminal:
                raise ValueError(f"job {job_id} already finished with status {record.status}")
            self._cancel_requested.add(job_id)
            process = self._processes.get(job_id)
            if process is None:
                record.status = "cancelled"
                record.message = "cancelled before process start"
                record.finishedAt = _now()
            else:
                record.status = "cancelling"
                record.message = "termination requested"
            self._persist()
            public = self._public(record)
        self._emit(record)
        if process is not None:
            try:
                process.terminate()
            except OSError:
                pass
        return public

    def logs(self, job_id: str, limit: int = 500) -> list[str]:
        with self._lock:
            record = self._jobs.get(job_id)
        if record is None or not record.logPath:
            return []
        path = safe_project_path(self.settings, record.logPath)
        if not path.exists():
            return []
        from services.quant_api.runtime_indexer.parsers import parse_log

        return list(parse_log(path, limit).get("data") or [])

    def stream(self, job_id: str) -> Iterator[str]:
        position = 0
        pending = ""
        path = self.settings.jobs_root / f"{job_id}.log"
        while True:
            record = self.get(job_id)
            if record is None:
                yield f"event: error\ndata: {json.dumps({'message': 'job not found'})}\n\n"
                return
            if path.exists():
                with path.open("r", encoding="utf-8", errors="replace") as handle:
                    handle.seek(position)
                    chunk = handle.read()
                    position = handle.tell()
                text = pending + chunk
                lines = text.splitlines(keepends=True)
                pending = ""
                if lines and not lines[-1].endswith(("\n", "\r")):
                    pending = lines.pop()
                for line in lines:
                    yield f"event: log\ndata: {json.dumps({'line': line.rstrip()}, ensure_ascii=False)}\n\n"
            terminal = record["status"] in {"succeeded", "failed", "cancelled"}
            if terminal and pending:
                yield f"event: log\ndata: {json.dumps({'line': pending}, ensure_ascii=False)}\n\n"
                pending = ""
            yield f"event: status\ndata: {json.dumps(record, ensure_ascii=False)}\n\n"
            if terminal:
                return
            import time

            time.sleep(1.0)

    def _run(
        self,
        job_id: str,
        command_id: str,
        parameters: dict[str, Any],
        spec: dict[str, Any],
        log_path: Path,
    ) -> None:
        with self._lock:
            if job_id in self._cancel_requested or self._jobs[job_id].status == "cancelled":
                return
        self._update(job_id, status="running", startedAt=_now(), progress=0.0, message="running")
        entrypoint = spec.get("entrypoint")
        command = (
            [sys.executable, str(self.settings.project_root / entrypoint)]
            if entrypoint
            else [sys.executable, "-m", "quantagent.cli", command_id]
        )
        # positional subcommands (e.g. `u0_full_universe_backfill.py fetch`) are
        # fixed by the allowlist, never taken from user input.
        command.extend(spec.get("fixed_args", ()))
        for key, value in parameters.items():
            if value is None:
                continue
            option = f"--{key.replace('_', '-')}"
            if isinstance(value, bool):
                if key in {"require_gpu", "label_norm"}:
                    command.append(option if value else f"--no-{key.replace('_', '-')}")
                elif value:
                    command.append(option)
                continue
            if isinstance(value, list):
                value = ",".join(str(item) for item in value)
            command.extend([option, str(value)])
        log_path.parent.mkdir(parents=True, exist_ok=True)
        try:
            with log_path.open("w", encoding="utf-8") as handle:
                handle.write(f"$ {' '.join(command)}\n")
                handle.flush()
                process = subprocess.Popen(
                    command,
                    cwd=self.settings.project_root,
                    stdout=subprocess.PIPE,
                    stderr=subprocess.STDOUT,
                    text=True,
                    bufsize=1,
                )
                with self._lock:
                    self._processes[job_id] = process
                    cancel_requested = job_id in self._cancel_requested
                if cancel_requested:
                    process.terminate()
                assert process.stdout is not None
                for line in process.stdout:
                    handle.write(line)
                    handle.flush()
                    progress = _progress_from_line(line)
                    if progress is not None:
                        self._update(
                            job_id,
                            progress=progress,
                            message=line.strip()[:240] or "running",
                        )
                code = process.wait()
            with self._lock:
                self._processes.pop(job_id, None)
                cancelled = job_id in self._cancel_requested
            outputs = [
                project_relative(self.settings, value)
                for key, value in parameters.items()
                if key in spec["path_outputs"] and value is not None
            ]
            outputs.extend(spec.get("fixed_outputs", ()))
            if cancelled:
                self._update(
                    job_id, status="cancelled", finishedAt=_now(),
                    message="cancelled", outputPaths=outputs,
                )
            elif code == 0:
                if self.on_success is not None:
                    try:
                        self.on_success()
                    except Exception:
                        pass
                self._update(
                    job_id, status="succeeded", finishedAt=_now(), progress=1.0,
                    message="completed; runtime catalog invalidated", outputPaths=outputs,
                )
            else:
                self._update(
                    job_id, status="failed", finishedAt=_now(), message=f"exit code {code}",
                    error=f"command exited with code {code}",
                )
        except OSError as exc:
            with self._lock:
                self._processes.pop(job_id, None)
                cancelled = job_id in self._cancel_requested
            if cancelled:
                self._update(job_id, status="cancelled", finishedAt=_now(), message="cancelled")
            else:
                self._update(
                    job_id, status="failed", finishedAt=_now(), message="failed to start",
                    error=str(exc),
                )

    def _normalize_parameters(self, spec: dict[str, Any], parameters: dict[str, Any]) -> dict[str, Any]:
        normalized: dict[str, Any] = {}
        for key, value in parameters.items():
            if value is None:
                normalized[key] = None
                continue
            if isinstance(value, str) and ("\x00" in value or "\n" in value or "\r" in value):
                raise ValueError(f"invalid control character in {key}")
            if key in spec["path_inputs"] | spec["path_outputs"]:
                path = safe_project_path(self.settings, str(value))
                if key in spec["path_inputs"] and not path.exists():
                    raise ValueError(f"input path does not exist: {key}")
                if key in spec["path_outputs"]:
                    runtime = self.settings.runtime_root.resolve()
                    if path != runtime and runtime not in path.parents:
                        raise ValueError(f"output path must be inside runtime: {key}")
                normalized[key] = str(path)
            else:
                if isinstance(value, str) and not re.fullmatch(r"[\w.,:+/ -]*", value):
                    raise ValueError(f"unsupported characters in {key}")
                normalized[key] = value
        return normalized

    def _update(self, job_id: str, **changes: Any) -> None:
        with self._lock:
            record = self._jobs[job_id]
            for key, value in changes.items():
                setattr(record, key, value)
            self._persist()
        self._emit(record)

    def _emit(self, record: JobRecord) -> None:
        if self.events is None:
            return
        self.events.publish(
            topic=f"jobs:{record.id}",
            event_type="job.status",
            payload={"job": self._public(record)},
            source="quant_api.jobs",
            correlation_id=record.id,
        )

    def _load(self) -> None:
        if not self.state_path.exists():
            return
        try:
            payload = json.loads(self.state_path.read_text(encoding="utf-8"))
            self._jobs = {item["id"]: JobRecord(**item) for item in payload}
            for record in self._jobs.values():
                if record.status in {"queued", "running", "cancelling"}:
                    record.status = "failed"
                    record.finishedAt = _now()
                    record.error = "API process restarted before job completed"
        except (OSError, ValueError, TypeError):
            self._jobs = {}

    def _persist(self) -> None:
        self.state_path.parent.mkdir(parents=True, exist_ok=True)
        temp = self.state_path.with_suffix(".tmp")
        temp.write_text(
            json.dumps([asdict(record) for record in self._jobs.values()], ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
        temp.replace(self.state_path)

    @staticmethod
    def _public(record: JobRecord) -> dict[str, Any]:
        data = asdict(record)
        data.pop("logPath", None)
        return data


def _now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def _progress_from_line(line: str) -> float | None:
    """Parse provider progress without requiring a provider-specific protocol."""
    match = re.search(r"\[(\d+)\s*/\s*(\d+)\]", line)
    if match and int(match.group(2)) > 0:
        return min(0.99, int(match.group(1)) / int(match.group(2)))
    try:
        payload = json.loads(line)
    except (TypeError, ValueError):
        return None
    value = payload.get("progress")
    if isinstance(value, (int, float)):
        return min(0.99, max(0.0, float(value)))
    for current_key, total_key in (
        ("batch", "total_batches"),
        ("iteration", "total_iterations"),
        ("rows_written", "total_rows"),
    ):
        current, total = payload.get(current_key), payload.get(total_key)
        if isinstance(current, (int, float)) and isinstance(total, (int, float)) and total > 0:
            return min(0.99, max(0.0, float(current) / float(total)))
    return None
