"""Tests for the unified ``E:\\AI量化`` storage layout."""

from __future__ import annotations

import os

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
