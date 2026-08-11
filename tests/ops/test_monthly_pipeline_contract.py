from __future__ import annotations

from pathlib import Path
import subprocess


def _script() -> Path:
    return Path(__file__).resolve().parents[2] / "scripts" / "run_monthly_pipeline.sh"


def test_monthly_pipeline_shell_is_syntactically_valid() -> None:
    result = subprocess.run(
        ["bash", "-n", str(_script())],
        capture_output=True,
        text=True,
        check=False,
    )
    assert result.returncode == 0, result.stderr


def test_monthly_pipeline_has_no_dated_demo_prediction_fallback() -> None:
    text = _script().read_text(encoding="utf-8")

    assert "demo_preds_20260605" not in text
    assert "QUANTAGENT_MONTHLY_PREDICTIONS_PATH" in text
    assert 'PREDICTIONS_PATH="${1:-${QUANTAGENT_MONTHLY_PREDICTIONS_PATH:-}}"' in text
    assert 'if [[ -z "$PREDICTIONS_PATH" ]]' in text
    assert 'if [[ ! -f "$PREDICTIONS_PATH" ]]' in text


def test_all_monthly_symbol_and_pool_consumers_share_one_prediction_artifact() -> None:
    text = _script().read_text(encoding="utf-8")

    assert 'fetch_broker_reports.py --symbols-from "$PREDICTIONS_PATH"' in text
    assert 'fetch_news_sentiment.py --symbols-from "$PREDICTIONS_PATH"' in text
    assert '--predictions-path "$PREDICTIONS_PATH"' in text
    assert "predictions_sha256" in text
    assert "quantagent.monthly-research-input.v1" in text
    assert 'implicit_demo_fallback": False' in text
