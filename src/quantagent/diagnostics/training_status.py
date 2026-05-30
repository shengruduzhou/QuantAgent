"""V10 training status reconciliation utilities."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
import json
from pathlib import Path
import subprocess
from typing import Any


DEFAULT_HORIZONS: tuple[int, ...] = (5, 20, 60, 120, 126)


@dataclass(frozen=True)
class V10StatusConfig:
    base_output: Path = Path("runtime/models/v7_alpha_full_universe_nosynth_v10")
    log_dir: Path = Path("runtime/logs")
    seeds: tuple[int, ...] = (1729, 4096, 8191)
    expected_folds: int = 12
    horizons: tuple[int, ...] = DEFAULT_HORIZONS


def _active_process_lines() -> list[str]:
    try:
        completed = subprocess.run(
            ["ps", "-eo", "pid,etime,pcpu,pmem,cmd"],
            check=False,
            capture_output=True,
            text=True,
        )
    except Exception:
        return []
    if completed.returncode != 0:
        return []
    lines = []
    for line in completed.stdout.splitlines():
        text = line.lower()
        if "run_full_universe_train.py" in text or "run_v10_ensemble.sh" in text:
            if "grep" not in text:
                lines.append(line.strip())
    return lines


def _file_mtime_iso(path: Path) -> str | None:
    if not path.exists():
        return None
    return datetime.fromtimestamp(path.stat().st_mtime).astimezone().isoformat(timespec="seconds")


def _fold_complete(fold_dir: Path, horizons: tuple[int, ...]) -> bool:
    if not fold_dir.exists():
        return False
    for horizon in horizons:
        if not (fold_dir / f"fold_{horizon:03d}d_oos_predictions.parquet").exists():
            return False
        if not (fold_dir / f"fold_{horizon:03d}d_strategy_metrics.json").exists():
            return False
    return (fold_dir / "ft_transformer_metrics.json").exists()


def _log_error_flags(log_path: Path) -> dict[str, bool]:
    if not log_path.exists():
        return {"has_traceback": False, "has_oom": False, "has_cublas": False, "has_killed": False}
    text = log_path.read_text(encoding="utf-8", errors="ignore").lower()
    return {
        "has_traceback": "traceback" in text,
        "has_oom": "out of memory" in text or "oom" in text,
        "has_cublas": "cublas" in text,
        "has_killed": "killed" in text,
    }


def scan_v10_training_status(config: V10StatusConfig | None = None) -> dict[str, Any]:
    cfg = config or V10StatusConfig()
    active_lines = _active_process_lines()
    seeds_payload: dict[str, Any] = {}
    any_active = bool(active_lines)
    for seed in cfg.seeds:
        out_dir = Path(f"{cfg.base_output}_seed{seed}")
        walk = out_dir / "walk_forward"
        marker = out_dir / "_seed_completed.txt"
        completed_folds: list[str] = []
        missing_folds: list[str] = []
        for idx in range(int(cfg.expected_folds)):
            name = f"fold_{idx:03d}"
            if _fold_complete(walk / name, cfg.horizons):
                completed_folds.append(name)
            else:
                missing_folds.append(name)
        log_path = cfg.log_dir / f"v10_seed{seed}.log"
        seed_active = any(str(seed) in line or str(out_dir) in line for line in active_lines)
        errors = _log_error_flags(log_path)
        if marker.exists() or len(completed_folds) == cfg.expected_folds:
            status = "completed"
            exit_reason = "completed_marker" if marker.exists() else "all_folds_present"
        elif seed_active:
            status = "running"
            exit_reason = "active_process"
        elif errors["has_traceback"] or errors["has_oom"] or errors["has_cublas"] or errors["has_killed"]:
            status = "error_or_killed"
            exit_reason = ",".join(key for key, value in errors.items() if value)
        elif completed_folds:
            status = "interrupted_or_stopped"
            exit_reason = "unknown"
        elif out_dir.exists() or log_path.exists():
            status = "started_no_complete_folds"
            exit_reason = "unknown"
        else:
            status = "not_started"
            exit_reason = "not_started"
        seeds_payload[str(seed)] = {
            "status": status,
            "output_dir": str(out_dir),
            "log_path": str(log_path),
            "completed_folds": completed_folds,
            "completed_fold_count": int(len(completed_folds)),
            "missing_folds": missing_folds,
            "missing_fold_count": int(len(missing_folds)),
            "aggregate_ready": bool(len(completed_folds) == cfg.expected_folds),
            "active_process": bool(seed_active),
            "last_log_timestamp": _file_mtime_iso(log_path),
            "exit_reason": exit_reason,
            "resume_required": bool(status in {"interrupted_or_stopped", "error_or_killed", "started_no_complete_folds"}),
            "resume_from": missing_folds[0] if missing_folds else None,
            "error_flags": errors,
        }
    aggregate_ready = all(item["aggregate_ready"] for item in seeds_payload.values())
    return {
        "generated_at": datetime.now().astimezone().isoformat(timespec="seconds"),
        "expected_folds": int(cfg.expected_folds),
        "horizons": list(cfg.horizons),
        "active_training_process": any_active,
        "active_process_lines": active_lines,
        "aggregate_ready": bool(aggregate_ready),
        "seeds": seeds_payload,
    }


def render_training_status_markdown(status: dict[str, Any]) -> str:
    lines = [
        "# V10 Training Status",
        "",
        f"- generated_at: {status.get('generated_at')}",
        f"- active_training_process: {status.get('active_training_process')}",
        f"- aggregate_ready: {status.get('aggregate_ready')}",
        "",
        "## Seeds",
    ]
    for seed, item in status.get("seeds", {}).items():
        lines.extend(
            [
                f"### seed {seed}",
                f"- status: {item.get('status')}",
                f"- completed_folds: {item.get('completed_fold_count')} / {status.get('expected_folds')}",
                f"- missing_folds: {item.get('missing_folds')}",
                f"- active_process: {item.get('active_process')}",
                f"- last_log_timestamp: {item.get('last_log_timestamp')}",
                f"- exit_reason: {item.get('exit_reason')}",
                f"- resume_required: {item.get('resume_required')}",
                f"- resume_from: {item.get('resume_from')}",
                "",
            ]
        )
    return "\n".join(lines)


def write_training_status(status: dict[str, Any], output_dir: str | Path) -> dict[str, Path]:
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)
    json_path = out / "training_status.json"
    md_path = out / "training_status.md"
    json_path.write_text(json.dumps(status, ensure_ascii=False, indent=2, sort_keys=True, default=str), encoding="utf-8")
    md_path.write_text(render_training_status_markdown(status), encoding="utf-8")
    return {"json": json_path, "markdown": md_path}


def build_resume_command(seed: int, *, base_output: str = "runtime/models/v7_alpha_full_universe_nosynth_v10") -> list[str]:
    out = f"{base_output}_seed{int(seed)}"
    return [
        "QA_TRAINING_DATASET=runtime/data/v7/gold/training_dataset/training_dataset_alpha181_full_nosynth.parquet",
        f"QA_TRAINING_OUTPUT={out}",
        "QA_MIN_SYNTH_FEATURES=0",
        "QA_N_SPLITS=12",
        f"QA_FT_SEED={int(seed)}",
        "QA_SKIP_FINAL_FIT=1",
        "AI_quant_venv/bin/python -u scripts/run_full_universe_train.py",
    ]


__all__ = [
    "V10StatusConfig",
    "build_resume_command",
    "render_training_status_markdown",
    "scan_v10_training_status",
    "write_training_status",
]
