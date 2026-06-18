from __future__ import annotations


def test_quality_gate_passes_complete_hot_money_report():
    from quantagent.agents.quality_gate import AgentReportQualityGate

    report = """
资金面分析：

| 项目 | 结论 |
|---|---|
| 北向资金 | 沪股通 + 深股通合计净流入 12 亿 |
| 主力资金流 | 主力资金流与大单净额为正 |
| 题材归因 | concept 板块和 reason tags 匹配 |

成交量、换手率、龙虎榜席位和行业轮动均已核对，结论仅作为研究 evidence。
""" * 3

    result = AgentReportQualityGate(min_report_length=80).evaluate_report("hot_money", report)
    assert result.grade == "A"
    assert result.confidence > 0.9
    assert not result.missing_requirements


def test_quality_gate_flags_short_or_failure_dominated_report():
    from quantagent.agents.quality_gate import AgentReportQualityGate

    gate = AgentReportQualityGate(min_report_length=120)
    short = gate.evaluate_report("policy", "无法获取 policy 数据")
    assert short.grade == "D"
    assert short.failure_markers


def test_quality_gate_bundle_reports_blocking_agents():
    from quantagent.agents.quality_gate import AgentReportQualityGate

    good = """
| 指标 | 数据 |
|---|---|
| PE/PB/市值 | 已覆盖 |
| 营收利润 growth | 已覆盖 |
| 现金流 ROE quality | 已覆盖 |
""" * 4
    bad = ""

    bundle = AgentReportQualityGate(min_report_length=80).evaluate_bundle(
        {"fundamentals": good, "lockup": bad}
    )

    assert bundle.overall_grade == "F"
    assert bundle.blocking_agents == ("lockup",)
    assert 0.0 < bundle.data_confidence < 1.0
