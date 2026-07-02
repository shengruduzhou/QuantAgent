from __future__ import annotations

import json

from services.quant_api.schemas.models import BacktestSummary, Factor, ModelSummary, Trade
from services.quant_api.adapters.models import _metric_group
from services.quant_api.services.container import ServiceContainer


def test_backtest_adapter_maps_trade_and_missing_fields(quant_ui_settings) -> None:
    services = ServiceContainer.create(quant_ui_settings)
    backtests = services.backtests.list()
    assert len(backtests) == 1
    backtest_id = backtests[0]["id"]

    trades = services.backtests.trades(backtest_id, page=1, page_size=10)["items"]
    assert [item["action"] for item in trades] == ["BUY", "SELL"]
    assert trades[0]["quantity"] == 100
    assert trades[0]["amount"] == 1008
    assert trades[0]["positionAfter"] == 100
    assert trades[0]["cashAfter"] is None
    assert trades[1]["pnl"] == 76
    assert trades[1]["name"] == "平安银行"
    Trade.model_validate(trades[0])
    BacktestSummary.model_validate(backtests[0])


def test_stock_kline_and_replay_use_real_fixture_fields(quant_ui_settings) -> None:
    services = ServiceContainer.create(quant_ui_settings)
    backtest_id = services.backtests.list()[0]["id"]

    kline = services.backtests.kline("000001.SZ")
    assert len(kline["bars"]) == 2
    assert kline["bars"][0]["open"] == 10.0
    assert kline["bars"][1]["volume"] == 1_200_000

    replay = services.backtests.stock_replay(backtest_id, "000001.SZ")
    assert replay["availability"]["bars"] is True
    assert replay["availability"]["trades"] is True
    assert replay["summary"]["realizedPnl"] == 76


def test_factor_explanation_and_empty_independent_trades(quant_ui_settings) -> None:
    services = ServiceContainer.create(quant_ui_settings)
    factor = services.factors.get("alpha001")
    assert factor is not None
    assert factor["sourceKind"] in {"registry", "alpha181"}
    assert factor["formula"]
    Factor.model_validate(factor)
    assert services.factors.explanation("alpha001")["factor"]["name"] == "alpha001"
    assert services.factors.backtest("alpha001")["ic"] == 0.02
    assert services.factors.backtest("alpha001")["trades"] == []


def test_selection_model_and_do_t_adapters(quant_ui_settings) -> None:
    services = ServiceContainer.create(quant_ui_settings)

    selection = services.selections.list()[0]
    ranking = services.selections.ranking(selection["id"])
    assert ranking[0]["symbol"] == "000001.SZ"
    assert ranking[0]["noOrdersGenerated"] is True
    chain = services.selections.decision_chain(selection["id"], "000001.SZ")
    assert chain["traceType"] == "score_pipeline"

    models = services.models.list()
    assert {model["modelFamily"] for model in models} >= {
        "deep_alpha",
        "reinforcement_learning",
        "intraday_t_plus_one",
    }
    model = next(item for item in models if item["modelFamily"] == "deep_alpha")
    ModelSummary.model_validate(model)
    assert services.models.training_metrics(model["id"])[0]["validationLoss"] == 0.2
    assert services.models.predictions(model["id"], symbol="000001.SZ")
    model_observability = services.models.observability(model["id"])
    assert any(item["name"] == "metrics.json" for item in model_observability["evaluations"])
    assert any(item["key"] == "total_return" for item in model_observability["metrics"])
    assert any(item["name"] == "ft_transformer.pt" for item in model_observability["artifacts"])
    assert model_observability["checkpoint"]["sizeBytes"] > 0

    rl_model = next(item for item in models if item["modelFamily"] == "reinforcement_learning")
    rl_observability = services.models.observability(rl_model["id"])
    assert rl_observability["availability"]["predictions"] is True
    assert services.models.predictions(rl_model["id"])[0]["symbol"] == "000001.SZ"

    source = next(item for item in services.do_t.list_sources() if item["name"] == "intraday_dot_factor_combo_fixture")
    analysis = services.do_t.analyze(source["id"], "000001.SZ")
    assert analysis["summary"]["pairCount"] == 1
    assert analysis["pairs"][0]["quantity"] == 100
    assert analysis["pairs"][0]["buyPrice"] == 10.5


def test_empty_runtime_adapters_degrade_gracefully(empty_quant_ui_settings) -> None:
    services = ServiceContainer.create(empty_quant_ui_settings)
    assert services.backtests.list() == []
    assert services.models.list() == []
    assert services.selections.list() == []
    assert services.do_t.list_sources() == []
    assert services.risk.overview()["maxDrawdown"] is None


def test_research_event_table_is_not_mapped_to_fake_trades(quant_ui_settings) -> None:
    directory = quant_ui_settings.runtime_root / "reports" / "research_event_fixture"
    directory.mkdir(parents=True)
    (directory / "summary.json").write_text(
        json.dumps({"window": "2026-01-01..2026-01-02", "executed_legs": 0}),
        encoding="utf-8",
    )
    (directory / "trades.csv").write_text(
        "trade_date,symbol,weight,executed,uplift\n"
        "2026-01-02,000001.SZ,0.02,false,0.0\n",
        encoding="utf-8",
    )

    services = ServiceContainer.create(quant_ui_settings)
    run = next(item for item in services.backtests.list() if item["name"] == "research_event_fixture")
    trades = services.backtests.trades(run["id"])

    assert run["capabilities"]["researchEvents"] is True
    assert run["capabilities"]["trades"] is False
    assert trades["items"] == []
    assert trades["sourceSchema"] == "research_event_table"
    assert trades["issues"][0]["code"] == "unsupported_trade_schema"


def test_risk_rules_use_code_defaults(quant_ui_settings) -> None:
    services = ServiceContainer.create(quant_ui_settings)
    rules = {item["id"]: item for item in services.risk.rules()}

    assert rules["max_name_weight"]["threshold"] == 0.05
    assert rules["max_drawdown"]["threshold"] == 0.15
    assert rules["max_daily_loss"]["threshold"] == 0.03


def test_runtime_cleanup_requires_confirmation_and_writes_audit(quant_ui_settings) -> None:
    runtime = quant_ui_settings.runtime_root
    registry = runtime / "models" / "v7_alpha" / "registry"
    registry.mkdir(parents=True)
    external_test_output = quant_ui_settings.project_root.parent / "pytest-model-output"
    (registry / "test_model.json").write_text(
        json.dumps({
            "model_version": "test_model",
            "metadata": {"output_dir": str(external_test_output)},
        }),
        encoding="utf-8",
    )
    smoke = runtime / "reports" / "fixture_smoke"
    smoke.mkdir(parents=True)
    (smoke / "result.json").write_text("{}", encoding="utf-8")

    services = ServiceContainer.create(quant_ui_settings)
    analysis = services.cleanup.analyze()
    safe = [item for item in analysis["candidates"] if item["safeDefault"]]
    assert {item["category"] for item in safe} >= {"invalid_test_registry", "smoke_reports"}

    try:
        services.cleanup.execute([safe[0]["id"]], "WRONG")
    except ValueError as exc:
        assert "confirmation" in str(exc)
    else:
        raise AssertionError("cleanup accepted an invalid confirmation")

    result = services.cleanup.execute([item["id"] for item in safe], "DELETE")
    assert result["errors"] == []
    assert result["freedBytes"] > 0
    assert not registry.exists()
    assert not smoke.exists()
    assert (quant_ui_settings.project_root / result["auditPath"]).exists()


def test_model_metric_group_does_not_treat_diagnostics_as_ic() -> None:
    assert _metric_group("diagnostics.total_rows") == "scale"
    assert _metric_group("diagnostics.feature_count") == "scale"
    assert _metric_group("metrics.rank_ic_mean") == "quality"
    assert _metric_group("metrics.max_drawdown") == "risk"


def test_runtime_cleanup_does_not_mark_new_qa_captures_as_stale(quant_ui_settings) -> None:
    cleanup_dir = quant_ui_settings.runtime_root / "reports" / "quant_ui" / "cleanup"
    cleanup_dir.mkdir(parents=True)
    (cleanup_dir / "cleanup_20260620T000000Z.json").write_text("{}", encoding="utf-8")
    qa_dir = quant_ui_settings.runtime_root / "reports" / "quant_ui" / "qa"
    qa_dir.mkdir(parents=True)
    (qa_dir / "current-model-lab.png").write_bytes(b"current")

    services = ServiceContainer.create(quant_ui_settings)
    categories = {item["category"] for item in services.cleanup.analyze()["candidates"]}

    assert "stale_ui_captures" not in categories
