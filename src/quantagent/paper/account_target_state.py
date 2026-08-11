"""Canonical paper-account state for next target construction.

A desired target is not a position.  Paper/shadow portfolio construction must
start from the shares and cash that were actually filled into the canonical
ledger, otherwise rejected/partial orders make the next turnover/risk decision
compare against fiction.

This module provides the narrow boundary needed by the daily signal loop:

* replay the append-only canonical economic ledger;
* mark actually held shares at the exact signal-session raw close;
* fail closed when a held symbol has no exact/raw mark or an unresolved order;
* bind the resulting current weights to the canonical ledger head/count;
* reconcile one desired target against the *union* of desired and currently
  held symbols, so exits count toward the turnover budget.

It deliberately does not place orders and does not certify live pre-trade risk.
"""
from __future__ import annotations

from dataclasses import dataclass
from hashlib import sha256
import json
from typing import Mapping

import numpy as np
import pandas as pd

from quantagent.data.intraday_sessions import (
    ExecutionPriceProvenanceError,
    assert_raw_execution_prices,
)
from quantagent.domain.ledger import CanonicalLedger
from quantagent.paper.recovery import RecoveryRefused, recover_from_canonical
from quantagent.portfolio.v7_target_weights import V7TargetWeightsResult


class PaperAccountStateRefused(RuntimeError):
    """Canonical/current account evidence is insufficient for target freezing."""


@dataclass(frozen=True)
class PaperAccountTargetState:
    as_of_date: str
    current_weights: pd.Series
    quantities: pd.Series
    cash: float
    nav: float
    canonical_records: int
    canonical_head_hash: str
    account_state_sha256: str

    def evidence(self) -> dict[str, object]:
        return {
            "schema": "quantagent.paper.account_target_state.v1",
            "as_of_date": self.as_of_date,
            "cash": float(self.cash),
            "nav": float(self.nav),
            "current_stock_gross": float(self.current_weights.abs().sum()),
            "held_symbols": int((self.quantities.abs() > 1e-12).sum()),
            "canonical_records": int(self.canonical_records),
            "canonical_head_hash": self.canonical_head_hash,
            "account_state_sha256": self.account_state_sha256,
            "source_of_truth": "canonical_ledger_replay_plus_exact_raw_signal_close",
            "production_pretrade_risk_certified": False,
        }


def _stable_state_hash(
    *,
    as_of_date: str,
    cash: float,
    nav: float,
    quantities: pd.Series,
    weights: pd.Series,
    canonical_records: int,
    canonical_head_hash: str,
) -> str:
    symbols = sorted(set(quantities.index.astype(str)) | set(weights.index.astype(str)))
    payload = {
        "as_of_date": as_of_date,
        "cash": round(float(cash), 12),
        "nav": round(float(nav), 12),
        "positions": [
            {
                "symbol": symbol,
                "quantity": round(float(quantities.get(symbol, 0.0)), 12),
                "weight": round(float(weights.get(symbol, 0.0)), 12),
            }
            for symbol in symbols
        ],
        "canonical_records": int(canonical_records),
        "canonical_head_hash": str(canonical_head_hash),
    }
    canonical = json.dumps(payload, sort_keys=True, separators=(",", ":"), ensure_ascii=False)
    return sha256(canonical.encode("utf-8")).hexdigest()


def recover_paper_account_target_state(
    *,
    canonical_ledger_path: str,
    market_panel: pd.DataFrame,
    as_of_date: str,
    portfolio_id: str,
    initial_cash: float,
) -> PaperAccountTargetState:
    """Replay actual fills and value held shares at the exact signal close.

    An empty but valid canonical ledger is a known cash-only first-run account.
    Once shares are held, every mark used to turn quantities into weights must be
    an exact ``as_of_date`` raw/unadjusted market row.  Adjusted research prices
    would change economic weights and are therefore refused.
    """

    as_of = pd.Timestamp(as_of_date).date().isoformat()
    ledger = CanonicalLedger(canonical_ledger_path)
    verification = ledger.verify()
    if not verification.get("valid"):
        raise PaperAccountStateRefused(
            f"canonical ledger verification failed at {verification.get('brokenAt')}"
        )
    if verification.get("writeFailure"):
        raise PaperAccountStateRefused(
            f"canonical ledger is not writable: {verification.get('writeFailure')}"
        )
    try:
        recovered = recover_from_canonical(
            canonical_ledger_path,
            portfolio_id=portfolio_id,
            initial_cash=float(initial_cash),
            as_of_session=as_of,
        )
    except RecoveryRefused as exc:
        raise PaperAccountStateRefused(str(exc)) from exc

    open_orders = recovered.open_orders()
    if open_orders:
        raise PaperAccountStateRefused(
            "canonical account has unresolved open orders; reconcile execution before freezing a new target"
        )

    quantities = pd.Series(
        {
            str(symbol): float(position.total)
            for symbol, position in recovered.portfolio.positions.items()
            if not position.is_flat and float(position.total) > 1e-12
        },
        dtype=float,
    )
    if quantities.empty:
        nav = float(recovered.portfolio.cash)
        if not np.isfinite(nav) or nav <= 0:
            raise PaperAccountStateRefused("cash-only canonical account has invalid NAV")
        weights = pd.Series(dtype=float)
    else:
        required = {"trade_date", "symbol", "close"}
        missing = sorted(required - set(market_panel.columns))
        if missing:
            raise PaperAccountStateRefused(f"market panel missing account mark columns: {missing}")
        marks = market_panel.copy()
        marks["trade_date"] = pd.to_datetime(marks["trade_date"], errors="coerce").dt.normalize()
        marks["symbol"] = marks["symbol"].astype(str)
        day = marks[
            (marks["trade_date"] == pd.Timestamp(as_of))
            & marks["symbol"].isin(quantities.index.astype(str))
        ].copy()
        if day.duplicated(["trade_date", "symbol"]).any():
            raise PaperAccountStateRefused("duplicate account mark rows on signal session")
        missing_symbols = sorted(set(quantities.index.astype(str)) - set(day["symbol"].astype(str)))
        if missing_symbols:
            raise PaperAccountStateRefused(
                f"exact signal-session marks missing for held symbols: {missing_symbols[:10]}"
            )
        try:
            assert_raw_execution_prices(day)
        except ExecutionPriceProvenanceError as exc:
            raise PaperAccountStateRefused(f"account mark price provenance failed: {exc}") from exc
        prices = pd.to_numeric(day.set_index("symbol")["close"], errors="coerce").reindex(quantities.index)
        if prices.isna().any() or not np.isfinite(prices.to_numpy(dtype=float)).all() or bool((prices <= 0).any()):
            raise PaperAccountStateRefused("account mark prices contain missing/non-positive values")
        market_values = quantities * prices.astype(float)
        nav = float(recovered.portfolio.cash + market_values.sum())
        if not np.isfinite(nav) or nav <= 0:
            raise PaperAccountStateRefused("canonical account marked NAV is invalid")
        weights = (market_values / nav).astype(float)

    records = int(verification.get("records", len(ledger)))
    head = str(verification.get("headHash", ledger.head_hash))
    state_hash = _stable_state_hash(
        as_of_date=as_of,
        cash=float(recovered.portfolio.cash),
        nav=nav,
        quantities=quantities,
        weights=weights,
        canonical_records=records,
        canonical_head_hash=head,
    )
    return PaperAccountTargetState(
        as_of_date=as_of,
        current_weights=weights.sort_index(),
        quantities=quantities.sort_index(),
        cash=float(recovered.portfolio.cash),
        nav=nav,
        canonical_records=records,
        canonical_head_hash=head,
        account_state_sha256=state_hash,
    )


def _target_series(frame: pd.DataFrame) -> tuple[pd.Timestamp, pd.Series]:
    if frame is None or frame.empty:
        raise ValueError("target frame is empty")
    if "trade_date" not in frame.columns:
        raise ValueError("target frame must include trade_date")
    if len(frame) != 1:
        raise ValueError(
            "account-aware paper reconciliation accepts exactly one signal-date target; historical panels use their own walk-forward state"
        )
    date = pd.Timestamp(frame.iloc[0]["trade_date"])
    values = pd.to_numeric(frame.drop(columns=["trade_date"]).iloc[0], errors="coerce").fillna(0.0)
    values.index = values.index.astype(str)
    return date, values.astype(float)


def reconcile_target_to_canonical_account(
    result: V7TargetWeightsResult,
    *,
    account_state: PaperAccountTargetState,
    max_turnover: float,
) -> V7TargetWeightsResult:
    """Freeze one desired target against actual current weights.

    ``max_turnover`` keeps the repository's existing L1 weight-churn convention
    (``sum(abs(target-current))``).  Crucially the calculation is performed on
    the union of current and desired symbols, so selling a dropped holding is no
    longer free in the turnover budget.
    """

    if result.target_weights is None or result.target_weights.empty:
        return result
    date, desired = _target_series(result.target_weights)
    current = pd.to_numeric(account_state.current_weights, errors="coerce").dropna().astype(float)
    current.index = current.index.astype(str)
    symbols = sorted(set(desired.index) | set(current.index))
    desired_u = desired.reindex(symbols).fillna(0.0)
    current_u = current.reindex(symbols).fillna(0.0)
    delta = desired_u - current_u
    requested_l1 = float(delta.abs().sum())
    cap = float(max_turnover)
    if cap > 0 and requested_l1 > cap + 1e-12:
        scale = cap / max(requested_l1, 1e-12)
        frozen = current_u + scale * delta
    else:
        scale = 1.0
        frozen = desired_u
    frozen = frozen.where(frozen.abs() > 1e-12, 0.0)
    applied_l1 = float((frozen - current_u).abs().sum())
    if cap > 0 and applied_l1 > cap + 1e-9:
        raise PaperAccountStateRefused(
            f"turnover reconciliation breached cap: {applied_l1:.12f} > {cap:.12f}"
        )

    frame = pd.DataFrame([{ "trade_date": date, **{s: float(frozen[s]) for s in symbols} }])
    diagnostics = dict(result.diagnostics or {})
    diagnostics["canonical_account_reconciliation"] = {
        **account_state.evidence(),
        "requested_l1_weight_churn": requested_l1,
        "applied_l1_weight_churn": applied_l1,
        "max_turnover_l1_weight_churn": cap,
        "turnover_scale": float(scale),
        "desired_stock_gross": float(desired_u.abs().sum()),
        "current_stock_gross": float(current_u.abs().sum()),
        "frozen_stock_gross": float(frozen.abs().sum()),
        "union_symbol_count": len(symbols),
        "dropped_current_symbols_counted": int(
            sum(1 for symbol in current.index if abs(float(current.get(symbol, 0.0))) > 1e-12 and abs(float(desired.get(symbol, 0.0))) <= 1e-12)
        ),
    }
    return V7TargetWeightsResult(target_weights=frame, diagnostics=diagnostics)


__all__ = [
    "PaperAccountStateRefused",
    "PaperAccountTargetState",
    "recover_paper_account_target_state",
    "reconcile_target_to_canonical_account",
]
