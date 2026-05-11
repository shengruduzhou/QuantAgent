from quantagent.services.v6_pipeline_service import build_features_v6, infer_v6


def test_v6_inference_consumes_conformal_interval():
    cfg = {"data": {"provider": "mock", "start_date": "2026-01-02", "end_date": "2026-03-31"}, "market": {"benchmark": "000300.SH"}}
    features = build_features_v6(cfg).frame
    result = infer_v6(cfg, feature_frame=features)
    assert {"q_low", "q_high", "conformal_confidence", "factor_gate"}.issubset(result.columns)
    assert (result["q_high"] >= result["q_low"]).all()
    assert result["conformal_confidence"].between(0.0, 1.0).all()

