from quantagent.execution.broker_base import Order, OrderSide, OrderStatus, OrderType
from quantagent.execution.virtual_broker import VirtualBroker


def test_virtual_broker_fills_without_real_account(tmp_path):
    broker = VirtualBroker(initial_cash=100000, audit_log_dir=tmp_path)
    broker.set_market_state([{"symbol": "600000.SH", "volume": 100000, "is_suspended": False, "is_limit_up": False, "is_limit_down": False}])
    order = Order("order1", "600000.SH", OrderSide.BUY, 100, OrderType.LIMIT, price=10.0)
    state = broker.submit(order)
    assert state.status == OrderStatus.FILLED
    assert broker.query_positions()[0].symbol == "600000.SH"
    assert (tmp_path / "virtual_broker_audit.jsonl").exists()


def test_virtual_broker_rejects_limit_up_buy(tmp_path):
    broker = VirtualBroker(initial_cash=100000, audit_log_dir=tmp_path)
    broker.set_market_state([{"symbol": "600000.SH", "volume": 100000, "is_suspended": False, "is_limit_up": True, "is_limit_down": False}])
    order = Order("order2", "600000.SH", OrderSide.BUY, 100, OrderType.LIMIT, price=10.0)
    assert broker.submit(order).status == OrderStatus.REJECTED

