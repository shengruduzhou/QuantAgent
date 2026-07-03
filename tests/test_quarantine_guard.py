"""P4 quarantine guard unit tests (no market data required)."""
from __future__ import annotations

import pandas as pd
import pytest

from quantagent.backtest.quarantine import (
    QuarantineWindow,
    check_window,
    clamp_panel_window,
    load_windows,
    violation_message,
)

W = [QuarantineWindow(start=pd.Timestamp("2025-09-01"), end=pd.Timestamp("2026-05-18"),
                      reason="burned holdout", evidence="HOLDOUT_CONTAMINATION_AUDIT.md")]


class TestCheckWindow:
    def test_fully_inside(self):
        assert check_window("2025-10-01", "2025-12-31", W) is not None

    def test_overlap_from_left(self):
        assert check_window("2025-08-01", "2025-09-15", W) is not None

    def test_overlap_from_right(self):
        assert check_window("2026-05-01", "2026-07-01", W) is not None

    def test_containing(self):
        assert check_window("2025-01-01", "2026-12-31", W) is not None

    def test_disjoint_before(self):
        assert check_window("2024-08-28", "2025-08-31", W) is None

    def test_disjoint_after(self):
        assert check_window("2026-05-19", "2026-12-31", W) is None

    def test_boundary_start_equals_quarantine_start(self):
        assert check_window("2025-09-01", "2025-09-01", W) is not None

    def test_boundary_end_equals_quarantine_end(self):
        assert check_window("2026-05-18", None, W) is not None

    def test_open_ended_clean_start_after(self):
        assert check_window("2026-05-19", None, W) is None

    def test_open_ended_from_validation(self):
        # end=None from a pre-quarantine start must hit the window
        assert check_window("2024-08-28", None, W) is not None


class TestClampPanelWindow:
    def test_right_buffer_clamped(self):
        # eval end 2025-08-31 → +10d buffer would reach 2025-09-10
        ps, pe = clamp_panel_window(pd.Timestamp("2024-08-18"), pd.Timestamp("2025-09-10"), W)
        assert pe == pd.Timestamp("2025-08-31")
        assert ps == pd.Timestamp("2024-08-18")

    def test_left_buffer_clamped(self):
        # eval start 2026-05-25 → -10d buffer lands inside quarantine
        ps, pe = clamp_panel_window(pd.Timestamp("2026-05-15"), pd.Timestamp("2026-07-10"), W)
        assert ps == pd.Timestamp("2026-05-19")
        assert pe == pd.Timestamp("2026-07-10")

    def test_no_clamp_needed(self):
        ps, pe = clamp_panel_window(pd.Timestamp("2024-01-01"), pd.Timestamp("2024-12-31"), W)
        assert (ps, pe) == (pd.Timestamp("2024-01-01"), pd.Timestamp("2024-12-31"))

    def test_open_end_untouched(self):
        ps, pe = clamp_panel_window(pd.Timestamp("2026-05-19"), None, W)
        assert ps == pd.Timestamp("2026-05-19") and pe is None


class TestMessagesAndConfig:
    def test_message_content(self):
        msg = violation_message("2025-09-01", "2026-05-15", W[0])
        assert "QUARANTINE VIOLATION" in msg
        assert "HOLDOUT_CONTAMINATION_AUDIT.md" in msg
        assert "--allow-quarantined" in msg

    def test_config_file_loads_burned_window(self):
        windows, log_path = load_windows()
        assert any(w.start == pd.Timestamp("2025-09-01") and w.end == pd.Timestamp("2026-05-18")
                   for w in windows)
        assert log_path.endswith(".jsonl")

    def test_builtin_fallback_on_missing_config(self, tmp_path):
        windows, _ = load_windows(tmp_path / "does_not_exist.json")
        assert any(w.start == pd.Timestamp("2025-09-01") for w in windows)


class TestEvaluatorIntegration:
    def test_evaluate_fails_closed_before_reading_data(self):
        import sys
        from pathlib import Path
        sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
        import baseline_protocol as bp
        from quantagent.backtest.quarantine import QuarantineViolation
        # nonexistent predictions path: guard must fire BEFORE any file read
        with pytest.raises(QuarantineViolation):
            bp.evaluate("/nonexistent/preds.parquet", top_k=10, start="2025-09-01",
                        end="2025-09-30", slippage_bps=8.0,
                        variants=["C_flags_eligible_delay1"])

    def test_evaluate_rejects_blank_override_reason(self):
        import sys
        from pathlib import Path
        sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
        import baseline_protocol as bp
        from quantagent.backtest.quarantine import QuarantineViolation
        with pytest.raises(QuarantineViolation):
            bp.evaluate("/nonexistent/preds.parquet", top_k=10, start="2025-09-01",
                        end="2025-09-30", slippage_bps=8.0,
                        variants=["C_flags_eligible_delay1"], allow_quarantined="   ")
