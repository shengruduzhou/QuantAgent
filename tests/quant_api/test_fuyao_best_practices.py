from services.quant_api.services.fuyao_best_practices import BEST_PRACTICES, DATA_GROUPS, best_practice_payload


def test_all_six_data_product_groups_are_registered() -> None:
    ids = {group["id"] for group in DATA_GROUPS}
    assert ids == {"market", "financial", "fund", "special", "calendar", "artifact"}


def test_all_sixteen_best_practices_are_registered_once() -> None:
    assert len(BEST_PRACTICES) == 16
    assert [item.id for item in BEST_PRACTICES] == [f"{index:02d}" for index in range(1, 17)]
    assert len({item.slug for item in BEST_PRACTICES}) == 16


def test_backtest_examples_lock_t_plus_one_execution() -> None:
    for identifier in {"13", "14", "15"}:
        item = next(item for item in BEST_PRACTICES if item.id == identifier)
        assert any("T+1" in line for line in item.contract)


def test_cashflow_and_financial_contracts_are_pit_aware() -> None:
    financial = next(item for item in BEST_PRACTICES if item.id == "02")
    cashflow = next(item for item in BEST_PRACTICES if item.id == "10")
    assert any("report_date_ms" in line for line in financial.contract)
    assert any("披露日" in line for line in cashflow.contract)


def test_dragon_tiger_topology_blocks_fake_intraday_semantics() -> None:
    item = next(item for item in BEST_PRACTICES if item.id == "16")
    assert any("不得伪装" in line for line in item.boundaries)
    assert any("range_days=1" in line for line in item.contract)


def test_output_contract_is_fail_closed_and_browser_secret_free() -> None:
    payload = best_practice_payload()
    output = payload["outputContract"]
    assert output["offlineHtml"] is True
    assert output["browserApiKey"] is False
    assert output["unavailableDataPolicy"] == "no synthetic fallback"
