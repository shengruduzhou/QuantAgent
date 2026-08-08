from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from quantagent.execution.broker_base import OrderIntent, OrderSide
from quantagent.risk.kill_switch import KillSwitch
from quantagent.risk.portfolio_risk import PortfolioRiskSnapshot, portfolio_fingerprint
from quantagent.risk.risk_limits import V6RiskLimits


@dataclass(frozen=True)
class RiskGateResult:
    passed: bool
    rejected_symbols: dict[str, str] = field(default_factory=dict)
    violations: tuple[str, ...] = ()
    unknowns: tuple[str, ...] = ()
    status: str = "pass"
    checked_weights: pd.Series | None = None


class RiskGate:
    def __init__(self, limits: V6RiskLimits | None = None, kill_switch: KillSwitch | None = None) -> None:
        self.limits = limits or V6RiskLimits()
        self.kill_switch = kill_switch or KillSwitch()

    @staticmethod
    def _final_result(
        *,
        rejected: dict[str, str],
        violations: list[str],
        unknowns: list[str],
        checked_weights: pd.Series | None = None,
    ) -> RiskGateResult:
        violations = list(dict.fromkeys(violations))
        unknowns = list(dict.fromkeys(unknowns))
        if violations or rejected:
            status = "blocked"
        elif unknowns:
            status = "unknown"
        else:
            status = "pass"
        return RiskGateResult(
            passed=status == "pass",
            rejected_symbols=rejected,
            violations=tuple(violations),
            unknowns=tuple(unknowns),
            status=status,
            checked_weights=checked_weights,
        )

    def check_target_weights(
        self,
        target_weights: pd.Series,
        current_weights: pd.Series | None = None,
        market_state: pd.DataFrame | None = None,
        sector: pd.Series | None = None,
        data_quality_score: float = 1.0,
        model_drift_score: float = 0.0,
        conformal_width: pd.Series | None = None,
        risk_snapshot: PortfolioRiskSnapshot | None = None,
        *,
        production_mode: bool = False,
    ) -> RiskGateResult:
        numeric = pd.to_numeric(target_weights, errors="coerce").astype(float)
        rejected: dict[str, str] = {}
        violations: list[str] = []
        unknowns: list[str] = []

        def evidence_problem(code: str) -> None:
            if production_mode:
                violations.append(code)
            else:
                unknowns.append(code)

        invalid_mask = numeric.isna() | ~np.isfinite(numeric)
        for symbol in numeric.index[invalid_mask]:
            rejected[str(symbol)] = "non_finite_target_weight"
        weights = numeric.replace([np.inf, -np.inf], np.nan).fillna(0.0).copy()

        if self.kill_switch.triggered:
            violations.append("kill_switch_triggered")
        if data_quality_score < self.limits.min_data_quality_score:
            violations.append("data_quality_below_threshold")
        if model_drift_score > self.limits.max_model_drift_score:
            violations.append("model_drift_above_threshold")

        gross_exposure = float(weights.abs().sum())
        if gross_exposure > float(self.limits.max_leverage) + 1e-12:
            violations.append("max_leverage")

        oversized = weights[weights.abs() > self.limits.max_name_weight]
        for symbol in oversized.index:
            rejected[str(symbol)] = "max_name_weight"
        weights = weights.clip(upper=self.limits.max_name_weight, lower=-self.limits.max_name_weight)

        if risk_snapshot is None:
            if production_mode:
                violations.append("portfolio_risk_snapshot_missing")
        else:
            if risk_snapshot.target_fingerprint != portfolio_fingerprint(target_weights):
                violations.append("portfolio_risk_snapshot_target_mismatch")

            beta_ready = True
            if risk_snapshot.beta_coverage < self.limits.min_beta_coverage:
                evidence_problem("beta_coverage_below_threshold")
                beta_ready = False
            if risk_snapshot.beta_pit_safe is not True:
                evidence_problem("beta_pit_evidence_missing")
                beta_ready = False
            if risk_snapshot.beta_freshness_days is None:
                evidence_problem("beta_freshness_missing")
                beta_ready = False
            elif risk_snapshot.beta_freshness_days > self.limits.max_risk_evidence_age_days:
                evidence_problem("beta_evidence_stale")
                beta_ready = False
            if risk_snapshot.beta_exposure is None or not np.isfinite(risk_snapshot.beta_exposure):
                evidence_problem("beta_exposure_missing")
                beta_ready = False
            if beta_ready and abs(float(risk_snapshot.beta_exposure)) > self.limits.beta_exposure_limit:
                violations.append("beta_exposure_limit")

            sector_ready = True
            if risk_snapshot.sector_coverage < self.limits.min_sector_coverage:
                evidence_problem("sector_coverage_below_threshold")
                sector_ready = False
            if risk_snapshot.sector_pit_safe is not True:
                evidence_problem("sector_pit_evidence_missing")
                sector_ready = False
            if risk_snapshot.sector_freshness_days is None:
                evidence_problem("sector_freshness_missing")
                sector_ready = False
            elif risk_snapshot.sector_freshness_days > self.limits.max_risk_evidence_age_days:
                evidence_problem("sector_evidence_stale")
                sector_ready = False
            if gross_exposure > 1e-15 and not risk_snapshot.sector_exposures:
                evidence_problem("sector_exposure_missing")
                sector_ready = False
            if sector_ready:
                for sector_name, value in risk_snapshot.sector_exposures.items():
                    if abs(float(value)) > self.limits.max_sector_weight:
                        violations.append(f"max_sector_weight:{sector_name}")

            style_limits = dict(self.limits.style_exposure_limits)
            style_factors = set(risk_snapshot.style_exposures) | set(style_limits)
            if style_factors:
                style_metadata_ready = True
                if risk_snapshot.style_pit_safe is not True:
                    evidence_problem("style_pit_evidence_missing")
                    style_metadata_ready = False
                if risk_snapshot.style_freshness_days is None:
                    evidence_problem("style_freshness_missing")
                    style_metadata_ready = False
                elif risk_snapshot.style_freshness_days > self.limits.max_risk_evidence_age_days:
                    evidence_problem("style_evidence_stale")
                    style_metadata_ready = False
                for factor in sorted(style_factors):
                    if factor not in risk_snapshot.style_exposures:
                        evidence_problem(f"style_exposure_missing:{factor}")
                        continue
                    coverage = float(risk_snapshot.style_coverage.get(factor, 0.0))
                    if coverage < self.limits.min_style_coverage:
                        evidence_problem(f"style_coverage_below_threshold:{factor}")
                        continue
                    if style_metadata_ready and factor in style_limits:
                        if abs(float(risk_snapshot.style_exposures[factor])) > float(style_limits[factor]):
                            violations.append(f"style_exposure_limit:{factor}")

            if self.limits.tracking_error_limit is not None:
                if risk_snapshot.tracking_overlap < self.limits.min_tracking_overlap:
                    evidence_problem("tracking_error_overlap_insufficient")
                elif risk_snapshot.tracking_error is None or not np.isfinite(risk_snapshot.tracking_error):
                    evidence_problem("tracking_error_missing")
                elif float(risk_snapshot.tracking_error) > float(self.limits.tracking_error_limit):
                    violations.append("tracking_error_limit")
                if not risk_snapshot.benchmark_symbol:
                    evidence_problem("tracking_error_benchmark_missing")
                if not risk_snapshot.tracking_frequency:
                    evidence_problem("tracking_error_frequency_missing")

        # Backward-compatible research diagnostic when callers provide a sector
        # series but do not yet build PortfolioRiskSnapshot. Production mode
        # requires the snapshot above and therefore cannot rely on this fallback.
        if risk_snapshot is None and sector is not None:
            sector_weights = weights.groupby(sector.reindex(weights.index).fillna("unknown")).sum().abs()
            for sector_name, value in sector_weights.items():
                if value > self.limits.max_sector_weight:
                    violations.append(f"max_sector_weight:{sector_name}")

        if current_weights is not None:
            current = pd.to_numeric(current_weights, errors="coerce").reindex(weights.index).fillna(0.0)
            turnover = float((weights - current).abs().sum())
            if turnover > self.limits.max_turnover:
                violations.append("max_turnover")

        relevant_symbols = set(weights.index[weights.abs() > 1e-12])
        if current_weights is not None:
            current_for_relevance = pd.to_numeric(current_weights, errors="coerce").fillna(0.0)
            relevant_symbols.update(current_for_relevance.index[current_for_relevance.abs() > 1e-12])

        if conformal_width is None:
            if production_mode and relevant_symbols:
                violations.append("conformal_evidence_missing")
        else:
            wide = pd.to_numeric(conformal_width, errors="coerce").reindex(sorted(relevant_symbols))
            for symbol in wide.index[wide.isna() | ~np.isfinite(wide)]:
                evidence_problem(f"conformal_evidence_missing:{symbol}")
            valid_wide = wide[wide.notna() & np.isfinite(wide)]
            for symbol in valid_wide[valid_wide > self.limits.conformal_uncertainty_threshold].index:
                rejected[str(symbol)] = "conformal_uncertainty"
                if symbol in weights.index:
                    weights.loc[symbol] = 0.0

        if market_state is None or market_state.empty:
            if production_mode and relevant_symbols:
                violations.append("market_state_missing")
        else:
            if "symbol" not in market_state.columns:
                violations.append("market_state_symbol_column_missing")
                state = pd.DataFrame()
            else:
                state = market_state.set_index("symbol")
                duplicate_symbols = set(state.index[state.index.duplicated(keep=False)])
                for symbol in sorted(duplicate_symbols & relevant_symbols):
                    violations.append(f"market_state_duplicate:{symbol}")
                for symbol in sorted(relevant_symbols):
                    if symbol not in state.index:
                        evidence_problem(f"market_state_missing:{symbol}")
            if not state.empty:
                for symbol in weights.index:
                    if symbol not in state.index or symbol in duplicate_symbols:
                        continue
                    row = state.loc[symbol]
                    if bool(row.get("is_suspended", row.get("suspended", False))):
                        rejected[str(symbol)] = "suspended"
                        weights.loc[symbol] = 0.0
                    if self.limits.no_trade_st and bool(row.get("is_st", False)):
                        rejected[str(symbol)] = "st"
                        weights.loc[symbol] = 0.0
                    if self.limits.no_buy_limit_up and bool(row.get("is_limit_up", False)) and weights.loc[symbol] > 0:
                        rejected[str(symbol)] = "limit_up_no_buy"
                        weights.loc[symbol] = 0.0
                    if (
                        self.limits.no_sell_limit_down
                        and bool(row.get("is_limit_down", False))
                        and current_weights is not None
                    ):
                        current = pd.to_numeric(current_weights, errors="coerce").reindex(weights.index).fillna(0.0)
                        if weights.loc[symbol] < current.loc[symbol]:
                            rejected[str(symbol)] = "limit_down_no_sell"
                            weights.loc[symbol] = current.loc[symbol]

        return self._final_result(
            rejected=rejected,
            violations=violations,
            unknowns=unknowns,
            checked_weights=weights,
        )

    def check_order_intents(
        self,
        intents: list[OrderIntent],
        market_state: pd.DataFrame | None = None,
        cash_available: float = float("inf"),
    ) -> RiskGateResult:
        rejected: dict[str, str] = {}
        violations: list[str] = []
        if self.kill_switch.triggered:
            violations.append("kill_switch_triggered")
        if len(intents) > self.limits.max_orders_per_day:
            violations.append("max_orders_per_day")
        state = market_state.set_index("symbol") if market_state is not None and not market_state.empty else pd.DataFrame()
        buy_value = 0.0
        for intent in intents:
            value = float(intent.quantity) * float(intent.reference_price)
            if value > self.limits.max_order_value:
                rejected[intent.intent_id] = "max_order_value"
            if intent.quantity % self.limits.min_lot_size != 0:
                rejected[intent.intent_id] = "min_lot_size"
            if intent.side == OrderSide.BUY:
                buy_value += value
            if not state.empty and intent.symbol in state.index:
                row = state.loc[intent.symbol]
                if bool(row.get("is_suspended", row.get("suspended", False))):
                    rejected[intent.intent_id] = "suspended"
                if self.limits.no_buy_limit_up and intent.side == OrderSide.BUY and bool(row.get("is_limit_up", False)):
                    rejected[intent.intent_id] = "limit_up_no_buy"
                if self.limits.no_sell_limit_down and intent.side == OrderSide.SELL and bool(row.get("is_limit_down", False)):
                    rejected[intent.intent_id] = "limit_down_no_sell"
        if buy_value > cash_available:
            violations.append("cash_constraint")
        return self._final_result(
            rejected=rejected,
            violations=violations,
            unknowns=[],
        )
