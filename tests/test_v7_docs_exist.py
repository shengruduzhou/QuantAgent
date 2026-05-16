from pathlib import Path


def test_v7_docs_and_config_exist_and_cover_required_boundaries():
    docs = [
        Path("README.md"),
        Path("AGENTS.md"),
        Path("docs/V7_系统架构与Agent接口.md"),
        Path("docs/V7_证据摄取与交易规则.md"),
        Path("docs/V7_realdata_training_pipeline.md"),
        Path("docs/V7_PIT_data_contract.md"),
        Path("docs/V7_training_dataset_schema.md"),
        Path("docs/V7_live_readiness_gates.md"),
        Path("configs/v7.default.yaml"),
    ]
    for path in docs:
        assert path.exists(), path
        text = path.read_text(encoding="utf-8")
        assert len(text) > 200, path

    combined = "\n".join(path.read_text(encoding="utf-8") for path in docs[:-1])
    for term in [
        "EvidenceRecord",
        "Point-in-Time",
        "target_weights",
        "OrderManager",
        "VirtualBroker",
        "T+1",
        "Risk Gate",
        "Audit",
        "Theme Discovery",
        "Financial Fraud Risk",
    ]:
        assert term in combined, term
