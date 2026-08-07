"""Supervisor process that outlives the API and records a real exit status.

Without this, a job's exit code lives only in the memory of the API process
that spawned it. Restart the API mid-training and that code is gone: the job
could only be marked "failed — API restarted", even when the training went on
to finish successfully seconds later.

This runner sits between the API and the actual command. It writes the worker
PID as soon as the worker exists and the true exit code when it ends, both to a
status file on disk. A restarted API reads that file and reports what actually
happened instead of guessing.

It is deliberately tiny and takes no user-controlled arguments of its own: the
command to run is passed after ``--`` exactly as the job layer assembled it,
so this adds no new injection surface.
"""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
from typing import Any


def _now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def _write_status(path: Path, payload: dict[str, Any]) -> None:
    """Atomically publish the status so a reader never sees a half-written file."""
    temp = path.with_suffix(".tmp")
    try:
        temp.write_text(json.dumps(payload, ensure_ascii=False), encoding="utf-8")
        temp.replace(path)
    except OSError:
        pass


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="QuantAgent job supervisor")
    parser.add_argument("--status-file", required=True)
    parser.add_argument("--job-id", required=True)
    parser.add_argument("command", nargs=argparse.REMAINDER)
    arguments = parser.parse_args(argv)

    command = list(arguments.command)
    if command and command[0] == "--":
        command = command[1:]
    if not command:
        print("job runner received an empty command", file=sys.stderr, flush=True)
        return 2

    status_path = Path(arguments.status_file)
    status_path.parent.mkdir(parents=True, exist_ok=True)

    try:
        worker = subprocess.Popen(command, stdout=sys.stdout, stderr=subprocess.STDOUT)
    except OSError as exc:
        _write_status(status_path, {
            "jobId": arguments.job_id,
            "state": "start_failed",
            "error": str(exc),
            "finishedAt": _now(),
            "exitCode": 127,
        })
        print(f"failed to start command: {exc}", file=sys.stderr, flush=True)
        return 127

    _write_status(status_path, {
        "jobId": arguments.job_id,
        "state": "running",
        "workerPid": worker.pid,
        "runnerPid": os.getpid(),
        "startedAt": _now(),
    })

    def forward(signal_number: int, _frame: Any) -> None:
        # Cancel targets the runner; the worker must receive it too, otherwise
        # the runner exits and leaves an orphaned training process behind.
        try:
            worker.send_signal(signal_number)
        except OSError:
            pass

    for forwarded in (signal.SIGTERM, signal.SIGINT):
        try:
            signal.signal(forwarded, forward)
        except (OSError, ValueError):
            pass

    exit_code = worker.wait()
    _write_status(status_path, {
        "jobId": arguments.job_id,
        "state": "exited",
        "workerPid": worker.pid,
        "runnerPid": os.getpid(),
        "exitCode": int(exit_code),
        "finishedAt": _now(),
    })
    return int(exit_code) if exit_code >= 0 else 128 + (-int(exit_code))


if __name__ == "__main__":
    raise SystemExit(main())
