"""QuantAgent V7 CLI entry point.

The CLI is composed of small per-area submodules that all register their
commands on a single Typer ``app``. Submodules cover:

* ``v7_data`` — Qlib / AkShare / labels / training-dataset commands.
* ``v7_train`` — alpha training, evaluation, and real-data orchestration.
* ``v7_backtest`` — walk-forward backtest and paper-trade replays.
* ``v7_readiness`` — live-readiness gate reporting.
* ``v7_research`` — legacy factor/research utilities preserved for tests.

Keeping the surface modular makes the CLI cleaner to navigate and lets
new commands land without touching unrelated code.
"""

from __future__ import annotations

from quantagent.cli._utils import app

# Side-effect imports register Typer commands onto ``app``.
from quantagent.cli import v7_data  # noqa: F401
from quantagent.cli import v7_train  # noqa: F401
from quantagent.cli import v7_backtest  # noqa: F401
from quantagent.cli import v7_readiness  # noqa: F401
from quantagent.cli import v7_research  # noqa: F401
from quantagent.cli import v7_storage  # noqa: F401
from quantagent.cli import v7_optimize  # noqa: F401
from quantagent.cli import v7_evidence  # noqa: F401
from quantagent.cli import v7_sector  # noqa: F401
from quantagent.cli import v7_liveness  # noqa: F401
from quantagent.cli import v7_policy  # noqa: F401
from quantagent.cli import v7_bond  # noqa: F401
from quantagent.cli import paper  # noqa: F401
from quantagent.cli import v8  # noqa: F401
from quantagent.cli import v8_verify  # noqa: F401
from quantagent.cli import v8_deep  # noqa: F401
from quantagent.cli import v8_gated  # noqa: F401
from quantagent.cli import v8_portfolio  # noqa: F401
from quantagent.cli import v8_intraday  # noqa: F401
from quantagent.cli.v7_research import (  # re-exports for legacy entry points
    build_factors_entry,
    build_flow_features_entry,
    build_sector_rotation_entry,
    evaluate_factors_entry,
    generate_factor_report_entry,
    generate_valuation_report_entry,
)


__all__ = [
    "app",
    "build_factors_entry",
    "build_flow_features_entry",
    "build_sector_rotation_entry",
    "evaluate_factors_entry",
    "generate_factor_report_entry",
    "generate_valuation_report_entry",
]


if __name__ == "__main__":  # pragma: no cover - script entry
    app()
