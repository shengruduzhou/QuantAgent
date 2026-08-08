from __future__ import annotations

from types import SimpleNamespace

import pytest

from quantagent.execution.broker_base import Order, OrderSide, OrderStatus, OrderType
from quantagent.execution.qmt_gateway import QMTConfig, QMTGateway


class FakeConst:
    STOCK_BUY = 23
    STOCK_SELL = 24
    FIX_PRICE = 11
    MARKET_PEER_PRICE_FIRST = 5

    ORDER_UNREPORTED = 100
    ORDER_WAIT_REPORTING = 101
    ORDER_REPORTED = 102
    ORDER_PART_SUCC = 103
    ORDER_SUCCEEDED = 104
    ORDER_PART_CANCEL = 105
    ORDER_CANCELED = 106
    ORDER_JUNK = 107


class FakeClient:
    def __init__(self) -> None:
        self.account_id = "ACC001"
        self.submit_calls: list[tuple] = []
        self.cancel_calls: list[tuple] = []
        self.orders: list[object] = []
        self.trades: list[object] = []
        self.positions: list[object] = []
        self.stopped = False
        self.next_order_id = 7001

    def query_stock_asset(self, account):
        return SimpleNamespace(account_id=self.account_id, total_asset=1_250_000.0)

    def query_stock_orders(self, account, cancelable_only=False):
        return list(self.orders)

    def query_stock_trades(self, account):
        return list(self.trades)

    def query_stock_positions(self, account):
        return list(self.positions)

    def order_stock(self, *args):
        self.submit_calls.append(args)
        return self.next_order_id

    def cancel_order_stock(self, *args):
        self.cancel_calls.append(args)
        return 0

    def stop(self):
        self.stopped = True


def _gateway(tmp_path, *, require_risk_approval: bool = True) -> tuple[QMTGateway, FakeClient, object]:
    client = FakeClient()
    account = SimpleNamespace(account_id="ACC001")
    gateway = QMTGateway(
        QMTConfig(
            account_id="ACC001",
            mini_qmt_path="C:/QMT/userdata_mini",
            session_id=123,
            dry_run=False,
            live_trading_enabled=True,
            require_preflight=True,
            require_risk_approval=require_risk_approval,
            audit_log_dir=str(tmp_path),
            identity_map_path=str(tmp_path / "qmt_identity_map.jsonl"),
        )
    )
    gateway._bind_live_client(client, account, FakeConst)
    return gateway, client, account


def _approved_order(client_order_id: str = "cid-001") -> Order:
    return Order(
        client_order_id=client_order_id,
        symbol="600000.SH",
        side=OrderSide.BUY,
        quantity=100,
        order_type=OrderType.LIMIT,
        price=10.25,
        strategy_version="prod-candidate",
        risk_check_result="approved",
    )


def test_live_submit_requires_preflight_and_risk_approval(tmp_path) -> None:
    gateway, client, _ = _gateway(tmp_path)

    with pytest.raises(RuntimeError, match="preflight"):
        gateway.submit(_approved_order())

    report = gateway.preflight()
    assert report["ok"] is True

    unchecked = Order(
        client_order_id="cid-unchecked",
        symbol="600000.SH",
        side=OrderSide.BUY,
        quantity=100,
        order_type=OrderType.LIMIT,
        price=10.0,
        risk_check_result="not_checked",
    )
    with pytest.raises(RuntimeError, match="risk_check_result"):
        gateway.submit(unchecked)
    assert client.submit_calls == []


def test_live_submit_is_idempotent_and_embeds_bounded_recovery_identity(tmp_path) -> None:
    gateway, client, _ = _gateway(tmp_path)
    assert gateway.preflight()["ok"] is True

    order = _approved_order("600000.SH-buy-1234567890")
    first = gateway.submit(order)
    second = gateway.submit(order)

    assert first == second
    assert first.status == OrderStatus.SUBMITTED
    assert first.broker_order_id == "7001"
    assert len(client.submit_calls) == 1
    call = client.submit_calls[0]
    assert call[1] == "600000.SH"
    assert call[2] == FakeConst.STOCK_BUY
    assert call[4] == FakeConst.FIX_PRICE
    assert call[5] == pytest.approx(10.25)
    remark = call[7]
    assert remark.startswith("qa:")
    assert len(remark) <= 24
    assert remark != f"qa:{order.client_order_id}"
    assert gateway._encode_remark(order.client_order_id) == remark


def test_cancel_request_is_not_falsely_reported_as_cancelled(tmp_path) -> None:
    gateway, client, _ = _gateway(tmp_path)
    gateway.preflight()
    gateway.submit(_approved_order())
    remark = gateway._encode_remark("cid-001")

    requested = gateway.cancel("cid-001")
    assert requested.status == OrderStatus.SUBMITTED
    assert requested.last_message == "cancel_requested"
    assert len(client.cancel_calls) == 1

    client.orders = [
        SimpleNamespace(
            order_id=7001,
            order_remark=remark,
            order_status=FakeConst.ORDER_CANCELED,
            traded_volume=0,
            traded_price=0.0,
            price=10.25,
        )
    ]
    confirmed = gateway.query_order("cid-001")
    assert confirmed.status == OrderStatus.CANCELLED


def test_startup_preflight_recovers_owned_broker_orders_after_restart(tmp_path) -> None:
    gateway, client, account = _gateway(tmp_path)
    remark = gateway._encode_remark("restored-client-id")
    client.orders = [
        SimpleNamespace(
            order_id=8123,
            order_remark=remark,
            order_status=FakeConst.ORDER_PART_SUCC,
            traded_volume=40,
            traded_price=9.91,
            price=10.00,
        ),
        SimpleNamespace(
            order_id=9000,
            order_remark="manual-order",
            order_status=FakeConst.ORDER_REPORTED,
            traded_volume=0,
            price=8.0,
        ),
    ]

    report = gateway.preflight()
    assert report["ok"] is True
    recovered = gateway.query_order("restored-client-id")
    assert recovered.broker_order_id == "8123"
    assert recovered.status == OrderStatus.PARTIAL
    assert recovered.filled_quantity == 40
    assert recovered.avg_price == pytest.approx(9.91)
    assert len(gateway.query_orders()) == 1

    restarted = QMTGateway(gateway.config)
    restarted._bind_live_client(client, account, FakeConst)
    assert restarted.preflight()["ok"] is True
    assert restarted.query_order("restored-client-id").broker_order_id == "8123"


def test_preflight_fails_if_quantagent_owned_remark_cannot_be_resolved(tmp_path) -> None:
    gateway, client, _ = _gateway(tmp_path)
    client.orders = [
        SimpleNamespace(
            order_id=9999,
            order_remark="qa:0123456789abcdef0123",
            order_status=FakeConst.ORDER_REPORTED,
            traded_volume=0,
            traded_price=0.0,
            price=10.0,
        )
    ]

    report = gateway.preflight()
    assert report["ok"] is False
    assert report["unresolved_owned_remarks"] == ["qa:0123456789abcdef0123"]
    assert any("owned_order_identity_unresolved" in error for error in report["errors"])


def test_manual_broker_order_does_not_fail_identity_preflight(tmp_path) -> None:
    gateway, client, _ = _gateway(tmp_path)
    client.orders = [
        SimpleNamespace(
            order_id=9998,
            order_remark="manual-order",
            order_status=FakeConst.ORDER_REPORTED,
            traded_volume=0,
            traded_price=0.0,
            price=10.0,
        )
    ]
    report = gateway.preflight()
    assert report["ok"] is True
    assert report["orders_synced"] == 0


def test_live_queries_normalise_account_position_and_trade_state(tmp_path) -> None:
    gateway, client, _ = _gateway(tmp_path)
    remark = gateway._encode_remark("cid-001")
    client.positions = [
        SimpleNamespace(
            stock_code="600000.SH",
            volume=1000,
            can_use_volume=700,
            open_price=9.50,
        )
    ]
    client.orders = [
        SimpleNamespace(
            order_id=7001,
            order_remark=remark,
            order_status=FakeConst.ORDER_SUCCEEDED,
            traded_volume=100,
            traded_price=10.25,
            price=10.25,
        )
    ]
    client.trades = [
        SimpleNamespace(
            order_id=7001,
            stock_code="600000.SH",
            order_type=FakeConst.STOCK_BUY,
            traded_volume=100,
            traded_price=10.25,
            traded_time="2026-08-08 10:01:02",
        )
    ]

    assert gateway.preflight()["ok"] is True
    assert gateway.query_account_value() == pytest.approx(1_250_000.0)
    position = gateway.query_positions()[0]
    assert position.available_shares == 700
    assert position.frozen_shares == 300
    assert position.avg_cost == pytest.approx(9.50)

    fill = gateway.query_trades()[0]
    assert fill.client_order_id == "cid-001"
    assert fill.side == OrderSide.BUY
    assert fill.fill_quantity == 100
    assert fill.fill_price == pytest.approx(10.25)

    health = gateway.health()
    assert health["ok"] is True
    assert health["account_value"] == pytest.approx(1_250_000.0)


def test_remark_prefix_that_cannot_fit_is_rejected(tmp_path) -> None:
    with pytest.raises(ValueError, match="too long"):
        QMTGateway(
            QMTConfig(
                # Pure ASCII alphanumeric text passes the character-class gate
                # and reaches the independent 24-character length check.
                order_remark_prefix="prefixlong",
                audit_log_dir=str(tmp_path),
                identity_map_path=str(tmp_path / "identity.jsonl"),
            )
        )
