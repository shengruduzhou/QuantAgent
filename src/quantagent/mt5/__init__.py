"""MetaTrader 5 integration: custom-symbol bridge and MQL5 laboratory support.

MT5 is a workstation here, not a data vendor. This package exports canonical
A-share events into custom symbols for charting and Strategy Tester work, and
never reads market data back out of the terminal into the authoritative journal.

Nothing here imports ``MetaTrader5`` at module scope, so the package remains
importable on hosts with no terminal -- which, per
``runtime/data/capabilities/mt5/terminal.json``, includes this one.
"""

from quantagent.mt5 import custom_symbol_bridge  # noqa: F401

__all__ = ["custom_symbol_bridge"]
