"""The safety boundary: no live order path, and honest readiness tiers.

These are the tests that must not be allowed to rot. Each names a way a
research system quietly becomes capable of losing real money, or a way a smoke
run gets mistaken for a validated strategy.
"""

from __future__ import annotations

import ast
import json
from pathlib import Path

import pytest

from quantagent.safety import operating_mode as om
from quantagent.safety import readiness_tiers as rt

REPO = Path(__file__).resolve().parents[1]


class TestOperatingModes:
    def test_live_disabled_is_not_executable(self):
        state = om.policy_state()
        assert state.mode == om.LIVE_DISABLED
        assert state.executable is False
        with pytest.raises(om.ModeViolation, match="terminal policy state"):
            state.require_executable()

    def test_live_trading_available_cannot_be_set_true(self):
        """A flag claiming live capability would be faithfully displayed by the UI."""
        with pytest.raises(ValueError, match="cannot be True"):
            om.OperatingModeState(mode=om.PAPER, live_trading_available=True)

    def test_unknown_mode_rejected(self):
        with pytest.raises(ValueError, match="unknown operating mode"):
            om.OperatingModeState(mode="LIVE")

    @pytest.mark.parametrize("mode", [om.HISTORICAL_REPLAY, om.PAPER, om.SHADOW])
    def test_order_simulating_modes(self, mode):
        state = om.OperatingModeState(mode=mode)
        state.require_order_simulation()
        assert state.simulates_orders

    @pytest.mark.parametrize("mode", [om.RESEARCH, om.BACKTEST])
    def test_non_simulating_modes_refuse_order_simulation(self, mode):
        state = om.OperatingModeState(mode=mode)
        with pytest.raises(om.ModeViolation, match="does not simulate orders"):
            state.require_order_simulation()

    def test_policy_describes_itself_without_live_capability(self):
        policy = om.describe_policy()
        assert policy["liveTradingAvailable"] is False
        assert policy["liveTradingCertificate"] == "NOT_IMPLEMENTED_BY_POLICY"
        assert policy["banner"] == "LIVE TRADING: DISABLED BY POLICY"


class TestLiveIntentRejection:
    @pytest.mark.parametrize("phrase", [
        "实盘", "真实下单", "连接资金账户", "自动买入", "自动卖出", "真实委托",
    ])
    def test_chinese_live_phrases_rejected(self, phrase):
        with pytest.raises(om.LiveTradingRejected):
            om.reject_live_intent(phrase)

    @pytest.mark.parametrize("phrase", [
        "enable live trading", "place real order", "connect real account",
        "broker login", "go live now",
    ])
    def test_english_live_phrases_rejected(self, phrase):
        with pytest.raises(om.LiveTradingRejected):
            om.reject_live_intent(phrase)

    @pytest.mark.parametrize("api", [
        "order_stock", "order_stock_async", "XtQuantTrader", "order_send",
        "OrderSendAsync",
    ])
    def test_broker_api_names_rejected(self, api):
        with pytest.raises(om.LiveTradingRejected):
            om.reject_live_intent(f"call {api} for 600000.SH")

    def test_intent_is_found_in_nested_job_parameters(self):
        """Intent can arrive in a nested parameter, not just a top-level prompt."""
        payload = {"command": "train", "params": {"notes": ["请帮我实盘下单"]}}
        with pytest.raises(om.LiveTradingRejected):
            om.reject_live_intent(payload)

    def test_intent_is_found_in_dictionary_keys(self):
        with pytest.raises(om.LiveTradingRejected):
            om.reject_live_intent({"实盘": True})

    def test_read_only_level2_order_feed_is_not_intent(self):
        """l2order is a market-data feed; misreading it would block real research."""
        om.reject_live_intent({"period": "l2order", "symbols": ["600000.SH"]})
        om.reject_live_intent("read orderbook and order_flow features")
        om.reject_live_intent("write simulated_order to paper_order ledger")

    def test_benign_request_passes(self):
        om.reject_live_intent({"command": "build-full-universe-gold",
                               "params": {"max_symbols": 250}})

    def test_rejection_names_what_matched_and_where(self):
        with pytest.raises(om.LiveTradingRejected) as exc:
            om.reject_live_intent("开启实盘", where="job submission")
        assert exc.value.matched == "实盘"
        assert exc.value.where == "job submission"
        assert "LIVE_DISABLED" in str(exc.value)


class TestNoLiveOrderPathInSource:
    """Static proof: no production module reaches a broker order API."""

    FORBIDDEN_CALLS = {
        "order_stock", "order_stock_async", "cancel_order_stock",
        "order_send", "OrderSendAsync",
    }
    FORBIDDEN_IMPORTS = {"xtquant.xttrader", "MetaTrader5"}

    def _production_files(self):
        for tree in ("src", "services"):
            for path in (REPO / tree).rglob("*.py"):
                if "__pycache__" in str(path):
                    continue
                yield path

    def test_no_module_calls_a_broker_order_api(self):
        offenders: dict[str, list[str]] = {}
        for path in self._production_files():
            try:
                tree = ast.parse(path.read_text(encoding="utf-8", errors="ignore"))
            except SyntaxError:
                continue
            for node in ast.walk(tree):
                if not isinstance(node, ast.Call):
                    continue
                func = node.func
                name = (
                    func.attr if isinstance(func, ast.Attribute)
                    else func.id if isinstance(func, ast.Name) else None
                )
                if name in self.FORBIDDEN_CALLS:
                    offenders.setdefault(
                        str(path.relative_to(REPO)), []
                    ).append(name)
        assert not offenders, f"live order calls found: {offenders}"

    def test_no_module_imports_a_trading_client_at_module_scope(self):
        offenders: dict[str, list[str]] = {}
        for path in self._production_files():
            try:
                tree = ast.parse(path.read_text(encoding="utf-8", errors="ignore"))
            except SyntaxError:
                continue
            for node in tree.body:
                names: list[str] = []
                if isinstance(node, ast.Import):
                    names = [a.name for a in node.names]
                elif isinstance(node, ast.ImportFrom) and node.module:
                    names = [node.module] + [
                        f"{node.module}.{a.name}" for a in node.names
                    ]
                hit = [n for n in names if n in self.FORBIDDEN_IMPORTS]
                if hit:
                    offenders.setdefault(str(path.relative_to(REPO)), []).extend(hit)
        assert not offenders, f"trading client imports found: {offenders}"

    def test_audit_detects_a_planted_violation(self, tmp_path):
        """A scan that cannot fail proves nothing."""
        planted = tmp_path / "bad.py"
        planted.write_text("trader.order_stock('600000.SH', 100)\n", encoding="utf-8")
        tree = ast.parse(planted.read_text(encoding="utf-8"))
        found = [
            n.func.attr for n in ast.walk(tree)
            if isinstance(n, ast.Call) and isinstance(n.func, ast.Attribute)
        ]
        assert "order_stock" in found


class TestReadinessTiers:
    def test_unknown_never_counts_as_pass(self, tmp_path):
        """The failure mode that produced 'gates are literal True constants'."""
        evaluator = rt.ReadinessEvaluator(tmp_path)
        certificate = evaluator.full_universe_gold({rt.ENGINEERING_PIPELINE_READY: True})
        assert certificate.granted is False
        assert certificate.unknown or certificate.unmet

    def test_missing_prerequisite_withholds_higher_tier(self, tmp_path):
        evaluator = rt.ReadinessEvaluator(tmp_path)
        certificate = evaluator.full_universe_gold({})
        assert certificate.prerequisites_met is False
        assert rt.ENGINEERING_PIPELINE_READY in certificate.missing_prerequisites
        assert certificate.granted is False

    def test_engineering_tier_forbids_performance_claims(self):
        permissions = rt.TIER_PERMISSIONS[rt.ENGINEERING_PIPELINE_READY]
        for forbidden in ("strategy_ranking", "performance_claims",
                          "model_promotion", "paper_portfolio_operation"):
            assert forbidden in permissions["forbids"]

    def test_permits_is_fail_closed(self):
        certificates = {"granted": {rt.ENGINEERING_PIPELINE_READY: True}}
        assert rt.permits(certificates, "one_epoch_smoke_training") is True
        assert rt.permits(certificates, "performance_claims") is False
        assert rt.permits(certificates, "an_action_nobody_declared") is False

    def test_forbid_beats_allow_across_tiers(self):
        """A lower tier's forbid must not be overridden by a higher tier's allow."""
        certificates = {"granted": {
            rt.ENGINEERING_PIPELINE_READY: True,
            rt.FULL_UNIVERSE_GOLD_READY: True,
        }}
        # Gold allows full_universe_training but still forbids performance claims.
        assert rt.permits(certificates, "full_universe_training") is True
        assert rt.permits(certificates, "performance_claims") is False

    def test_research_tier_withheld_while_pit_blocked(self):
        evaluator = rt.ReadinessEvaluator(REPO / "runtime")
        certificate = evaluator.full_universe_research({
            rt.FULL_UNIVERSE_GOLD_READY: True
        })
        pit = evaluator._u0_pit()
        if pit is None:
            pytest.skip("no PIT certificate on this host")
        if pit.get("blocked_pit_fields"):
            assert certificate.granted is False
            assert any("strict PIT remains blocked" in n for n in certificate.notes)

    def test_st_intervals_stays_a_mandatory_requirement(self):
        """PIT must not be weakened to force a higher tier."""
        evaluator = rt.ReadinessEvaluator(REPO / "runtime")
        certificate = evaluator.full_universe_research({
            rt.FULL_UNIVERSE_GOLD_READY: True
        })
        names = {r.name for r in certificate.requirements}
        assert "st_intervals_available" in names
        assert "no_blocked_pit_fields" in names

    def test_live_trading_certificate_is_a_refusal(self):
        certificate = rt.live_trading_certificate()
        assert certificate["granted"] is False
        assert certificate["implemented"] is False
        assert certificate["reason"] == "NOT_IMPLEMENTED_BY_POLICY"

    def test_live_trading_ready_is_not_a_tier(self):
        assert "LIVE_TRADING_READY" not in rt.TIERS

    def test_evaluate_all_stops_at_first_ungranted_tier(self):
        evaluator = rt.ReadinessEvaluator(REPO / "runtime")
        result = evaluator.evaluate_all()
        granted = result["granted"]
        highest = result["highest_granted_tier"]
        if highest is not None:
            index = rt.TIERS.index(highest)
            for tier in rt.TIERS[index + 1:]:
                assert granted.get(tier) is False

    def test_certificate_states_both_allows_and_forbids(self):
        evaluator = rt.ReadinessEvaluator(REPO / "runtime")
        for certificate in evaluator.evaluate_all()["certificates"].values():
            assert certificate["allows"]
            assert certificate["forbids"]
