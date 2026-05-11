from quantagent.services.v6_pipeline_service import build_features_v6, build_portfolio_v6, infer_v6, run_paper_trade_v6, validate_v6


def test_v6_pipeline_end_to_end_runs(tmp_path):
    cfg = {
        "data": {"provider": "mock", "start_date": "2026-01-02", "end_date": "2026-03-31", "cache_dir": str(tmp_path / "cache")},
        "market": {"benchmark": "000300.SH", "universe": "CSI300"},
        "portfolio": {"max_name_weight": 0.05, "max_sector_weight": 0.30, "max_turnover": 0.30},
        "execution": {"audit_log_dir": str(tmp_path / "logs"), "initial_cash": 1000000, "dry_run": True},
        "reporting": {"output_dir": str(tmp_path / "reports")},
    }
    features = build_features_v6(cfg)
    outputs = infer_v6(cfg, feature_frame=features.frame)
    portfolio = build_portfolio_v6(cfg, feature_frame=features.frame)
    paper = run_paper_trade_v6(cfg, target_weights=portfolio["target_weights"], feature_frame=features.frame)
    validation = validate_v6(cfg)
    assert not features.frame.empty
    assert not outputs.empty
    assert "target_weights" in portfolio
    assert portfolio["next_feature_gate_weights"]
    assert "reconciliation" in paper
    assert validation["passed"] is True
