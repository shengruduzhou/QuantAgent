from quantagent.portfolio.allocator import SleeveAllocator
from quantagent.portfolio.sleeve import SleeveType


def test_cash_buffer_increases_with_volatility_and_drawdown():
    allocator = SleeveAllocator()
    calm = allocator.allocate(1_000_000, "main_trend", 0.0, 0.1, 1.0, 1.0, 1.0, 0.1)
    stressed = allocator.allocate(1_000_000, "crash", -0.12, 0.8, 1.0, 1.0, 1.0, 0.8)
    assert stressed.as_dict()[SleeveType.CASH_BUFFER.value] > calm.as_dict()[SleeveType.CASH_BUFFER.value]


def test_short_event_sleeve_shrinks_when_quality_deteriorates():
    allocator = SleeveAllocator()
    strong = allocator.allocate(1_000_000, "main_trend", 0.0, 0.1, 1.0, 1.0, 1.0, 0.1)
    weak = allocator.allocate(1_000_000, "main_trend", 0.0, 0.1, 0.1, 1.0, 1.0, 0.1)
    assert weak.as_dict()[SleeveType.SHORT_EVENT.value] < strong.as_dict()[SleeveType.SHORT_EVENT.value]

