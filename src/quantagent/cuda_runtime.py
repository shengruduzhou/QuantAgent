"""CUDA runtime helpers for training entrypoints.

The helpers are intentionally lightweight and safe to import before
PyTorch. They make GPU visibility failures auditable without silently
falling back to CPU when a caller required CUDA.
"""

from __future__ import annotations

import os
from pathlib import Path
import shutil
import subprocess
from typing import Any


def configure_cuda_environment(*, default_visible_devices: str | None = None) -> None:
    """Set CUDA-related process defaults before importing torch.

    ``PYTORCH_NVML_BASED_CUDA_CHECK=1`` avoids poisoning later CUDA
    initialization in forked/subprocess launchers. ``CUDA_VISIBLE_DEVICES``
    is only filled when a caller explicitly supplies a default; existing
    scheduler/container values are preserved.
    """

    os.environ.setdefault("PYTORCH_NVML_BASED_CUDA_CHECK", "1")
    os.environ.setdefault("CUDA_DEVICE_ORDER", "PCI_BUS_ID")
    if default_visible_devices is not None and os.environ.get("CUDA_VISIBLE_DEVICES") in {None, ""}:
        os.environ["CUDA_VISIBLE_DEVICES"] = default_visible_devices


def cuda_runtime_probe(torch_module: Any | None = None) -> dict[str, object]:
    """Return an actionable CUDA visibility report without requiring CUDA."""

    probe: dict[str, object] = {
        "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
        "cuda_device_order": os.environ.get("CUDA_DEVICE_ORDER"),
        "pytorch_nvml_based_cuda_check": os.environ.get("PYTORCH_NVML_BASED_CUDA_CHECK"),
    }
    if torch_module is not None:
        probe["torch_version"] = getattr(torch_module, "__version__", None)
        version = getattr(torch_module, "version", None)
        probe["torch_cuda_version"] = getattr(version, "cuda", None)
        cuda = getattr(torch_module, "cuda", None)
        if cuda is not None:
            try:
                probe["torch_cuda_available"] = bool(cuda.is_available())
            except Exception as exc:
                probe["torch_cuda_available_error"] = str(exc)
            try:
                probe["torch_cuda_device_count"] = int(cuda.device_count())
            except Exception as exc:
                probe["torch_cuda_device_count_error"] = str(exc)
            try:
                if bool(probe.get("torch_cuda_available")):
                    probe["torch_gpu_name"] = cuda.get_device_name(0)
            except Exception as exc:
                probe["torch_gpu_name_error"] = str(exc)
    nvidia_smi = shutil.which("nvidia-smi")
    probe["nvidia_smi_path"] = nvidia_smi
    if nvidia_smi:
        try:
            completed = subprocess.run(
                [
                    nvidia_smi,
                    "--query-gpu=index,name,driver_version,memory.total",
                    "--format=csv,noheader",
                ],
                check=False,
                capture_output=True,
                text=True,
                timeout=5,
            )
            probe["nvidia_smi_returncode"] = completed.returncode
            output = completed.stdout.strip()
            error = completed.stderr.strip()
            if output:
                lines = output.splitlines()
                probe["nvidia_smi_output"] = lines[:8]
                first = lines[0].split(",")
                if len(first) >= 2:
                    probe["nvidia_smi_gpu_name"] = first[1].strip()
            if error:
                probe["nvidia_smi_error"] = error.splitlines()[:8]
        except Exception as exc:
            probe["nvidia_smi_error"] = str(exc)
    return probe


def format_cuda_diagnostic(probe: dict[str, object]) -> str:
    details = [
        f"CUDA_VISIBLE_DEVICES={probe.get('cuda_visible_devices')!r}",
        f"CUDA_DEVICE_ORDER={probe.get('cuda_device_order')!r}",
        f"torch={probe.get('torch_version')}",
        f"torch_cuda={probe.get('torch_cuda_version')}",
        f"device_count={probe.get('torch_cuda_device_count')}",
        f"nvidia_smi_returncode={probe.get('nvidia_smi_returncode')}",
    ]
    if probe.get("nvidia_smi_output"):
        details.append(f"nvidia_smi={probe['nvidia_smi_output']}")
    if probe.get("nvidia_smi_error"):
        details.append(f"nvidia_smi_error={probe['nvidia_smi_error']}")
    return "CUDA diagnostic: " + "; ".join(details)


def write_cuda_diagnostic(path: str | os.PathLike[str], torch_module: Any | None = None) -> Path:
    """Write a JSON CUDA diagnostic file and return its path."""

    import json

    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(
        json.dumps(cuda_runtime_probe(torch_module), ensure_ascii=False, indent=2, sort_keys=True),
        encoding="utf-8",
    )
    return output


__all__ = [
    "configure_cuda_environment",
    "cuda_runtime_probe",
    "format_cuda_diagnostic",
    "write_cuda_diagnostic",
]
