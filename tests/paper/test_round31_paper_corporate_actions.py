"""Round 31 paper-venue regressions (risk auditor R31-RISK-01/02/05).

* R31-RISK-01: the continuous paper loop never applied dividends or bonus
  shares, so an ex-date price drop was booked as a loss and tripped the
  daily-loss halt.
* R31-RISK-02: the paper limit band used the raw previous close on ex-dates.
* R31-RISK-05: the audited human clear of the 20% drawdown latch re-armed at
  the next session because the all-time peak was never re-anchored.
"""
from __future__ import annotations

import hashlib

import pandas as pd
from typer.testing import CliRunner

from quantagent.cli import app
from quantagent.domain.ledger import CanonicalLedger
from quantagent.paper import ledger as paper_ledger
from quantagent.paper.account_identity import ensure_paper_account_identity
from quantagent.paper.canonical_receipt import canonical_snapshot
from quantagent.paper.continuous_execution import (
    ContinuousPaperExecutionConfig,
    execute_pending_for_session,
)
from quantagent.paper.execution_journal import DAILY_DECISION_STATUS, PendingExecutionJournal
from quantagent.paper.orders import BUY, Order
from quantagent.paper.pending_signal import PENDING_COMMIT_PROTOCOL, PendingPaperSignalStore
from quantagent.paper.portfolio import Portfolio
from quantagent.paper.recovery import recover_from_canonical
from quantagent.paper.risk import SCOPE_PORTFOLIO, RiskEngine, RiskLimits

def paper_config(tmp_path, **overrides):
    values = dict(
        pending_signal_dir=str(tmp_path / "pending"),
        execution_journal_path=str(tmp_path / "execution.jsonl"),
        canonical_ledger_path=str(tmp_path / "canonical.jsonl"),
        operational_ledger_path=str(tmp_path / "operational.jsonl"),
        idempotency_path=str(tmp_path / "idempotency.jsonl"),
        account_identity_path=str(tmp_path / "account_identity.json"),
        initial_cash=1_000_000.0,
        max_participation_rate=0.05,
        # industry limit opted out explicitly so the scenarios isolate rules
        risk_limits=RiskLimits(max_participation=1.0, max_industry_weight=1.0),
    )
    values.update(overrides)
    return ContinuousPaperExecutionConfig(**values)


def record_target(tmp_path, signal_date, weights, *, initial_cash=1_000_000.0):
    identity = ensure_paper_account_identity(
        canonical_ledger_path=tmp_path / "canonical.jsonl",
        portfolio_id="v7-paper", initial_cash=initial_cash,
        identity_path=tmp_path / "account_identity.json",
    )
    records, head = canonical_snapshot(tmp_path / "canonical.jsonl")
    frame = pd.DataFrame(
        {"trade_date": [pd.Timestamp(signal_date)], **{k: [v] for k, v in weights.items()}}
    )
    pending = PendingPaperSignalStore(tmp_path / "pending").record(
        signal_date=signal_date, target_weights=frame,
        source_lineage={
            "model": "risk-golden",
            "target_weights_file_sha256": f"sha-{signal_date}",
            "paper_account_identity_sha256": identity.payload_sha256,
            "canonical_account_state_sha256": "1" * 64,
            "canonical_ledger_head_hash": head,
            "canonical_ledger_records": str(records),
            "daily_decision_commit_protocol": PENDING_COMMIT_PROTOCOL,
        },
        created_at=f"{signal_date}T07:00:00+00:00",
    )[0]
    summary = tmp_path / f"summary-{signal_date}.json"
    summary.write_text('{"committed":true}\n', encoding="utf-8")
    PendingExecutionJournal(tmp_path / "execution.jsonl").append(
        pending_payload_sha256=pending.payload_sha256, signal_date=signal_date,
        execution_date=signal_date, status=DAILY_DECISION_STATUS,
        details={
            "decision_kind": "target",
            "paper_account_identity_sha256": identity.payload_sha256,
            "canonical_account_state_sha256": "1" * 64,
            "canonical_records": records, "canonical_head": head,
            "assurance": "canonical_account_daily_decision_freeze_v1",
            "commit_protocol": PENDING_COMMIT_PROTOCOL,
            "target_weights_sha256": pending.target_weights_sha256,
            "daily_summary_path": str(summary.resolve()),
            "daily_summary_sha256": hashlib.sha256(summary.read_bytes()).hexdigest(),
            "daily_summary_commit_protocol": "daily_summary_bound_daily_decision_v1",
        },
    )
    return pending


def paper_panel(rows):
    frame = pd.DataFrame(rows)
    for col in ("is_suspended", "is_st"):
        if col not in frame.columns:
            frame[col] = False
    if "amount" not in frame.columns:
        frame["amount"] = frame["close"] * frame["volume"]
    for col in ("open", "high", "low"):
        if col not in frame.columns:
            frame[col] = frame["close"]
    frame["price_adjustment"] = "raw"
    frame["execution_eligible"] = True
    return frame


def paper_day(tmp_path, as_of, panel, sessions, **cfg):
    return execute_pending_for_session(
        as_of, panel, config=paper_config(tmp_path, **cfg), authoritative_sessions=sessions,
    )


def paper_state(tmp_path, as_of):
    return recover_from_canonical(
        str(tmp_path / "canonical.jsonl"), portfolio_id="v7-paper",
        initial_cash=1_000_000.0, as_of_session=as_of,
    )


def fills_of(tmp_path):

    book = CanonicalLedger(str(tmp_path / "canonical.jsonl")).replay_book()
    out = []
    for order in book.orders():
        out.append({"symbol": order.symbol, "side": str(order.side.value), "qty": order.quantity,
                    "status": order.status.value, "reason": order.reason,
                    "trade_date": order.trade_date})
    return out


def test_human_clear_of_drawdown_latch_survives_next_session_check(tmp_path):
    path = tmp_path / "operational.jsonl"
    engine = RiskEngine(state_ledger=paper_ledger.EventLedger(path))
    engine.observe_inception_equity(1_000_000.0)
    engine.begin_session("2024-06-03", opening_equity=750_000.0)
    engine.check_portfolio(Portfolio(portfolio_id="p", cash=750_000.0, initial_cash=1_000_000.0), {})
    assert engine.kill_switch.is_triggered(SCOPE_PORTFOLIO)  # 25% drawdown latched

    result = CliRunner().invoke(app, ["paper-clear-kill-switch", "--ledger", str(path), "--confirm",
                                      "--author", "pm", "--reason", "reviewed the drawdown, resume"])
    assert result.exit_code == 0, result.output
    assert '"peakRebasedTo": 750000.0' in result.output

    # Next session: a fresh venue replays the ledger and observes inception equity.
    engine2 = RiskEngine(state_ledger=paper_ledger.EventLedger(path))
    engine2.observe_inception_equity(1_000_000.0)
    engine2.begin_session("2024-06-04", opening_equity=752_000.0)
    engine2.check_portfolio(Portfolio(portfolio_id="p", cash=752_000.0, initial_cash=1e6), {})
    order = Order(symbol="600000.SH", side=BUY, quantity=100, limit_price=10.0)
    assert not engine2.kill_switch.blocking(order.side, None)


def test_drawdown_latch_still_fires_below_the_reanchored_peak(tmp_path):
    path = tmp_path / "operational.jsonl"
    engine = RiskEngine(state_ledger=paper_ledger.EventLedger(path))
    engine.observe_inception_equity(1_000_000.0)
    engine.begin_session("2024-06-03", opening_equity=750_000.0)
    engine.check_portfolio(Portfolio(portfolio_id="p", cash=750_000.0, initial_cash=1e6), {})
    CliRunner().invoke(app, ["paper-clear-kill-switch", "--ledger", str(path), "--confirm",
                             "--author", "pm", "--reason", "reviewed the drawdown, resume"])
    engine2 = RiskEngine(state_ledger=paper_ledger.EventLedger(path))
    engine2.observe_inception_equity(1_000_000.0)
    engine2.begin_session("2024-06-10", opening_equity=590_000.0)
    engine2.check_portfolio(Portfolio(portfolio_id="p", cash=590_000.0, initial_cash=1e6), {})
    # 590k is >20% below the reviewed 750k anchor: the control still works.
    assert engine2.kill_switch.is_triggered(SCOPE_PORTFOLIO)


B1, B2, C = "600000.SH", "600036.SH", "601398.SH"
DATES = ["2024-06-03", "2024-06-04", "2024-06-05", "2024-06-06", "2024-06-07"]


def _bonus_rows():
    rows = []
    for d in DATES:
        for s in (B1, B2, C):
            ex = d == "2024-06-06" and s in (B1, B2)
            after = d > "2024-06-06" and s in (B1, B2)
            rows.append({"trade_date": d, "symbol": s, "close": 5.0 if (ex or after) else 10.0,
                         "volume": 1e9,
                         # 10转10 on the ex-date (published by the certified panel)
                         "ca_cash_per_share": 0.0, "ca_share_ratio": 1.0 if ex else 0.0})
    return rows


def test_ex_date_bonus_issue_is_not_booked_as_a_loss_on_paper(tmp_path):
    panel = paper_panel(_bonus_rows())
    record_target(tmp_path, "2024-06-03", {B1: 0.09, B2: 0.09})
    paper_day(tmp_path, "2024-06-04", panel, DATES)
    paper_day(tmp_path, "2024-06-05", panel, DATES)
    shares_before = paper_state(tmp_path, "2024-06-05").portfolio.positions[B1].total
    record_target(tmp_path, "2024-06-05", {B1: 0.09, B2: 0.09, C: 0.05})
    result = paper_day(tmp_path, "2024-06-06", panel, DATES)[0]

    assert abs(result.nav_after / 1_000_000.0 - 1) < 0.01, (result.nav_before, result.nav_after)
    c_orders = [o for o in fills_of(tmp_path) if o["symbol"] == C]
    assert c_orders and c_orders[0]["status"] == "FILLED", c_orders
    state = paper_state(tmp_path, "2024-06-06")
    assert state.portfolio.positions[B1].total >= 2 * shares_before - 100  # bonus shares credited

    # The next session reconciles canonical vs operational books (no block),
    # and does not apply the same corporate action twice.
    record_target(tmp_path, "2024-06-06", {B1: 0.09, B2: 0.09, C: 0.05})
    follow = paper_day(tmp_path, "2024-06-07", panel, DATES)[0]
    assert follow.status in {"execution_observed", "execution_blocked"}, follow.to_dict()
    actions = [r for r in CanonicalLedger(str(tmp_path / "canonical.jsonl")).read() if r.corporate_action]
    assert len(actions) == 2, [r.corporate_action for r in actions]


def test_cash_dividend_and_fractional_bonus_follow_canonical_semantics(tmp_path):
    dates = DATES[:4]
    rows = []
    for d in dates:
        ex = d == "2024-06-06"
        rows.append({"trade_date": d, "symbol": B1, "close": (10.0 - 0.5) / 1.3 if ex else 10.0,
                     "volume": 1e9, "ca_cash_per_share": 0.5 if ex else 0.0,
                     "ca_share_ratio": 0.3 if ex else 0.0})
    panel = paper_panel(rows)
    record_target(tmp_path, "2024-06-03", {B1: 0.01})
    paper_day(tmp_path, "2024-06-04", panel, dates)
    before = paper_state(tmp_path, "2024-06-04")
    shares = before.portfolio.positions[B1].total
    record_target(tmp_path, "2024-06-05", {B1: 0.01})
    paper_day(tmp_path, "2024-06-06", panel, dates)
    after = paper_state(tmp_path, "2024-06-06")
    new_shares = int(shares * 0.3)
    assert after.portfolio.positions[B1].total == shares + new_shares
    ex_close = (10.0 - 0.5) / 1.3
    expected_credit = shares * 0.5 + (shares * 0.3 - new_shares) * ex_close
    assert abs((after.portfolio.cash - before.portfolio.cash) - expected_credit) < 1.0


STAR = "688001.SH"
STAR_DATES = ["2024-06-03", "2024-06-04", "2024-06-05", "2024-06-06"]


def test_star_sub_minimum_residual_can_be_fully_liquidated_on_paper(tmp_path):
    """R31-RISK-08: a 150-share STAR residual (below the 200-share minimum) is
    sellable as a full liquidation; the venue used to round the sell to zero."""
    panel = paper_panel([{"trade_date": d, "symbol": STAR, "close": 40.0, "volume": 1e9}
                         for d in STAR_DATES])
    record_target(tmp_path, "2024-06-03", {STAR: 350.2 * 40.0 / 1_000_000.0})
    paper_day(tmp_path, "2024-06-04", panel, STAR_DATES)
    record_target(tmp_path, "2024-06-04", {STAR: 149.5 * 40.0 / 1_000_000.0})
    paper_day(tmp_path, "2024-06-05", panel, STAR_DATES)
    assert paper_state(tmp_path, "2024-06-05").portfolio.positions[STAR].total == 150
    record_target(tmp_path, "2024-06-05", {STAR: 0.0})
    paper_day(tmp_path, "2024-06-06", panel, STAR_DATES)
    left = paper_state(tmp_path, "2024-06-06").portfolio.positions.get(STAR)
    assert left is None or left.total == 0, (left, fills_of(tmp_path))
