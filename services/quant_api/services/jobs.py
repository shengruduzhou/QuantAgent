from __future__ import annotations

from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
import json
from pathlib import Path
import re
import subprocess
import sys
from threading import RLock, Thread
from typing import Any, Iterator
from uuid import uuid4

from services.quant_api.config import ApiSettings, project_relative, safe_project_path


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
    "predict-alpha-v7": {
        "type": "infer",
        "required": {"model_dir", "feature_dataset", "output"},
        "allowed": {"model_dir", "feature_dataset", "output", "primary_horizon"},
        "path_inputs": {"model_dir", "feature_dataset"},
        "path_outputs": {"output"},
    },
}


class JobManager:
    def __init__(self, settings: ApiSettings) -> None:
        self.settings = settings
        self.state_path = settings.jobs_root / "jobs.json"
        self._lock = RLock()
        self._jobs: dict[str, JobRecord] = {}
        self._load()

    def submit(self, job_type: str, command_id: str, parameters: dict[str, Any]) -> dict[str, Any]:
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
        normalized = self._normalize_parameters(spec, parameters)
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
        Thread(target=self._run, args=(job_id, command_id, normalized, spec, log_path), daemon=True).start()
        return self._public(record)

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
        self._update(job_id, status="running", startedAt=_now(), message="running")
        command = [sys.executable, "-m", "quantagent.cli", command_id]
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
                    stdout=handle,
                    stderr=subprocess.STDOUT,
                    text=True,
                )
                code = process.wait()
            outputs = [
                project_relative(self.settings, value)
                for key, value in parameters.items()
                if key in spec["path_outputs"] and value is not None
            ]
            if code == 0:
                self._update(
                    job_id, status="succeeded", finishedAt=_now(), progress=1.0,
                    message="completed", outputPaths=outputs,
                )
            else:
                self._update(
                    job_id, status="failed", finishedAt=_now(), message=f"exit code {code}",
                    error=f"command exited with code {code}",
                )
        except OSError as exc:
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
                normalized[key] = project_relative(self.settings, path)
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

    def _load(self) -> None:
        if not self.state_path.exists():
            return
        try:
            payload = json.loads(self.state_path.read_text(encoding="utf-8"))
            self._jobs = {item["id"]: JobRecord(**item) for item in payload}
            for record in self._jobs.values():
                if record.status in {"queued", "running"}:
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
