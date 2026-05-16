"""Tests for the unified ``E:\\Project\\QuantAgent\\runtime`` storage layout."""

from __future__ import annotations

import os
import platform
from pathlib import Path

import pytest

from quantagent.config.paths import (
    DEFAULT_QUANT_HOME_ENV,
    QuantPaths,
    quant_paths,
    resolve_quant_home,
)


def test_resolve_quant_home_prefers_explicit_override(tmp_path, monkeypatch):
    monkeypatch.delenv(DEFAULT_QUANT_HOME_ENV, raising=False)
    home = resolve_quant_home(tmp_path)
    assert home == tmp_path


def test_resolve_quant_home_reads_env(tmp_path, monkeypatch):
    monkeypatch.setenv(DEFAULT_QUANT_HOME_ENV, str(tmp_path))
    home = resolve_quant_home()
    assert home == tmp_path


def test_windows_default_quant_home_is_repo_runtime(monkeypatch):
    monkeypatch.delenv(DEFAULT_QUANT_HOME_ENV, raising=False)
    if platform.system() == "Windows":
        assert resolve_quant_home().name == "runtime"
        assert resolve_quant_home().parent.name == "QuantAgent"


def test_quant_paths_layout_contains_required_directories(tmp_path, monkeypatch):
    monkeypatch.setenv(DEFAULT_QUANT_HOME_ENV, str(tmp_path))
    monkeypatch.delenv("QUANTAGENT_DATA_ROOT", raising=False)
    layout = quant_paths()
    expected = {
        "home",
        "data_root",
        "raw",
        "silver",
        "gold",
        "models",
        "predictions",
        "target_weights",
        "reports",
        "logs",
    }
    assert expected.issubset(set(layout.as_dict().keys()))
    assert layout.data_root.parent == tmp_path


def test_quant_paths_data_root_override(tmp_path, monkeypatch):
    monkeypatch.setenv(DEFAULT_QUANT_HOME_ENV, str(tmp_path / "home"))
    monkeypatch.setenv("QUANTAGENT_DATA_ROOT", str(tmp_path / "elsewhere"))
    layout = quant_paths()
    assert layout.data_root == tmp_path / "elsewhere"
    assert layout.models == tmp_path / "home" / "models"


def test_quant_paths_ensure_creates_all_directories(tmp_path, monkeypatch):
    monkeypatch.setenv(DEFAULT_QUANT_HOME_ENV, str(tmp_path))
    monkeypatch.delenv("QUANTAGENT_DATA_ROOT", raising=False)
    layout = quant_paths().ensure()
    for value in layout.as_dict().values():
        assert os.path.isdir(value)


def test_v7_cli_defaults_do_not_use_repo_local_paths(tmp_path, monkeypatch):
    monkeypatch.setenv(DEFAULT_QUANT_HOME_ENV, str(tmp_path))
    monkeypatch.delenv("QUANTAGENT_DATA_ROOT", raising=False)

    from quantagent.cli._utils import (
        default_artifact_root,
        default_predictions_root,
        default_reports_root,
        default_target_weights_root,
        default_v7_lake_root,
    )
    from quantagent.training.ft_transformer_trainer import FTTransformerTrainerConfig
    from quantagent.training.model_registry import ModelRegistry
    from quantagent.training.v7_deep_trainer import V7DeepAlphaTrainerConfig
    from quantagent.training.v7_experiment import V7TrainingConfig

    defaults = [
        default_v7_lake_root(),
        default_artifact_root(),
        default_predictions_root(),
        default_target_weights_root(),
        default_reports_root(),
        Path(V7TrainingConfig().output_dir),
        Path(V7DeepAlphaTrainerConfig().output_dir),
        Path(FTTransformerTrainerConfig().output_dir),
        ModelRegistry().root,
    ]
    for path in defaults:
        assert path.is_absolute()
        assert str(path).startswith(str(tmp_path))
        assert not str(path).startswith("data")
        assert not str(path).startswith("artifacts")
        assert not str(path).startswith("reports")
