"""Compatibility facade for canonical mainland A-share market rules.

The rule facts live in :mod:`quantagent.market_rules.ashare`, a dependency-light
module that can be shared by data, quant-math, backtest, paper and execution
without package-level import cycles. Existing ``quantagent.backtest.ashare_rules``
imports remain supported through this facade.
"""

from quantagent.market_rules.ashare import *  # noqa: F401,F403
