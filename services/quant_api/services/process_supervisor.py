"""Identify, re-adopt and measure job processes across API restarts.

Two facts drive this module.

First, a PID alone cannot prove a process is still *our* process: PIDs are
reused. Linux exposes each process's start time in ``/proc/<pid>/stat`` field
22, measured in clock ticks since boot, and the ``(pid, start_ticks)`` pair is
unique for the lifetime of a boot. Persisting that pair is what lets the API
restart and honestly answer "is that training still running?" instead of
declaring every in-flight job dead.

Second, a re-adopted process is not a child of the new API process, so
``wait()`` is unavailable and there is no exit code to collect. We watch it with
``kill(pid, 0)`` and derive the outcome from the artifacts and log it left
behind. That is less information than a real exit status, and the job record
says so rather than inventing one.
"""

from __future__ import annotations

from dataclasses import dataclass
import os
from pathlib import Path
import subprocess
from typing import Any


def process_start_ticks(pid: int) -> int | None:
    """Return the process start time in clock ticks since boot, or None."""
    try:
        stat = Path(f"/proc/{pid}/stat").read_text(encoding="utf-8", errors="replace")
    except OSError:
        return None
    # The comm field may contain spaces and parentheses, so split after the
    # final ')' rather than on whitespace from the beginning.
    close = stat.rfind(")")
    if close < 0:
        return None
    fields = stat[close + 2:].split()
    # After comm and state, field 22 of the original stat line is index 19 here.
    if len(fields) < 20:
        return None
    try:
        return int(fields[19])
    except ValueError:
        return None


def process_is_alive(pid: int | None, start_ticks: int | None) -> bool:
    """True only when this PID is still running *and* is the same process."""
    if not pid:
        return False
    try:
        os.kill(pid, 0)
    except (OSError, ProcessLookupError, PermissionError):
        return False
    if start_ticks is None:
        # Without an identity anchor we refuse to claim it is ours.
        return False
    return process_start_ticks(pid) == start_ticks


@dataclass(frozen=True)
class ResourceSample:
    cpu_percent: float | None
    rss_bytes: int | None
    gpu_memory_mib: float | None
    threads: int | None
    children: int
    sampled_at: str

    def to_dict(self) -> dict[str, Any]:
        return {
            "cpuPercent": self.cpu_percent,
            "rssBytes": self.rss_bytes,
            "gpuMemoryMiB": self.gpu_memory_mib,
            "threads": self.threads,
            "childProcesses": self.children,
            "sampledAt": self.sampled_at,
        }


def gpu_memory_by_pid() -> dict[int, float]:
    """Per-process GPU memory in MiB; empty when no NVIDIA runtime is present."""
    try:
        result = subprocess.run(
            [
                "nvidia-smi",
                "--query-compute-apps=pid,used_memory",
                "--format=csv,noheader,nounits",
            ],
            capture_output=True,
            text=True,
            timeout=3,
            check=False,
        )
    except (FileNotFoundError, OSError, subprocess.SubprocessError):
        return {}
    if result.returncode != 0:
        return {}
    usage: dict[int, float] = {}
    for line in result.stdout.splitlines():
        parts = [item.strip() for item in line.split(",")]
        if len(parts) != 2:
            continue
        try:
            usage[int(parts[0])] = float(parts[1])
        except ValueError:
            continue
    return usage


# psutil computes CPU percent as the delta since the previous call *on the same
# Process object*. Building a fresh object every sample therefore reports 0.0
# forever, which reads as "the job is doing nothing".
_PROCESS_CACHE: dict[int, Any] = {}


def _tracked_process(psutil_module: Any, pid: int) -> Any:
    cached = _PROCESS_CACHE.get(pid)
    if cached is not None:
        return cached
    process = psutil_module.Process(pid)
    process.cpu_percent(interval=None)  # prime the delta baseline
    if len(_PROCESS_CACHE) > 256:
        _PROCESS_CACHE.clear()
    _PROCESS_CACHE[pid] = process
    return process


def sample_process(
    pid: int,
    *,
    gpu_usage: dict[int, float] | None = None,
    now: str,
) -> ResourceSample | None:
    """Aggregate CPU/RSS/GPU across a job process and its descendants."""
    try:
        import psutil
    except ImportError:
        return None
    try:
        root = _tracked_process(psutil, pid)
        family = [root, *[_tracked_process(psutil, child.pid) for child in root.children(recursive=True)]]
    except Exception:  # psutil raises several process-gone variants
        _PROCESS_CACHE.pop(pid, None)
        return None

    gpu_usage = gpu_memory_by_pid() if gpu_usage is None else gpu_usage
    cpu = 0.0
    rss = 0
    threads = 0
    gpu = 0.0
    seen = 0
    for process in family:
        try:
            with process.oneshot():
                cpu += float(process.cpu_percent(interval=None))
                rss += int(process.memory_info().rss)
                threads += int(process.num_threads())
            gpu += gpu_usage.get(process.pid, 0.0)
            seen += 1
        except Exception:
            continue
    if not seen:
        return None
    return ResourceSample(
        cpu_percent=round(cpu, 1),
        rss_bytes=rss,
        gpu_memory_mib=round(gpu, 1) if gpu else None,
        threads=threads,
        children=max(0, seen - 1),
        sampled_at=now,
    )


def terminate_tree(pid: int, *, was_paused: bool, timeout: float = 5.0) -> None:
    """Stop a job process and every worker it spawned.

    A suspended process ignores SIGTERM until it runs again, so continue it
    first. Children are signalled too: killing only the parent leaves data
    loaders and CUDA workers holding memory the operator believes they freed.
    """
    import signal as signal_module

    # Every managed job starts a new session. On POSIX, signalling that process
    # group reaches both the durable supervisor and its worker tree. Traversing
    # downward from workerPid alone misses the supervisor parent and can leave
    # the job stuck in "cancelling" after a paused worker is terminated.
    if os.name != "nt":
        try:
            process_group = os.getpgid(pid)
            if process_group != os.getpgrp():
                if was_paused:
                    os.killpg(process_group, signal_module.SIGCONT)
                os.killpg(process_group, signal_module.SIGTERM)
        except OSError:
            pass

    try:
        import psutil
    except ImportError:
        psutil = None  # type: ignore[assignment]

    if psutil is None:
        try:
            if was_paused:
                os.kill(pid, signal_module.SIGCONT)
            os.kill(pid, signal_module.SIGTERM)
        except OSError:
            pass
        return

    try:
        root = psutil.Process(pid)
    except Exception:
        return
    family = [root]
    try:
        family.extend(root.children(recursive=True))
    except Exception:
        pass
    for process in family:
        try:
            if was_paused:
                process.send_signal(signal_module.SIGCONT)
            process.terminate()
        except Exception:
            continue
    try:
        _, alive = psutil.wait_procs(family, timeout=timeout)
    except Exception:
        return
    for process in alive:
        try:
            process.kill()
        except Exception:
            continue


def signal_tree(pid: int, sig: int) -> None:
    """Send a signal to a job process and its descendants."""
    if os.name != "nt":
        try:
            process_group = os.getpgid(pid)
            if process_group != os.getpgrp():
                os.killpg(process_group, sig)
                return
        except OSError:
            pass
    try:
        import psutil

        root = psutil.Process(pid)
        family = [root, *root.children(recursive=True)]
    except Exception:
        os.kill(pid, sig)
        return
    for process in family:
        try:
            process.send_signal(sig)
        except Exception:
            continue


__all__ = [
    "ResourceSample",
    "gpu_memory_by_pid",
    "process_is_alive",
    "process_start_ticks",
    "sample_process",
    "signal_tree",
    "terminate_tree",
]
