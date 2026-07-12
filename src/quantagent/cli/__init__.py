"""Governed QuantAgent CLI entry point.

The default command surface contains data, training, trusted evaluation,
evidence, paper trading, governance and storage operations. Historical v7/v8
search and one-shot experiment commands are disabled by default.

Set ``QUANTAGENT_ENABLE_LEGACY_CLI=1`` only for explicit reproduction work.
"""

from __future__ import annotations

import os

from quantagent.cli._utils import app

from quantagent.cli import governance  # noqa: F401,E402
from quantagent.cli import paper  # noqa: F401,E402
from quantagent.cli import v7_backtest  # noqa: F401,E402
from quantagent.cli import v7_bond  # noqa: F401,E402
from quantagent.cli import v7_data  # noqa: F401,E402
from quantagent.cli import v7_evidence  # noqa: F401,E402
from quantagent.cli import v7_liveness  # noqa: F401,E402
from quantagent.cli import v7_policy  # noqa: F401,E402
from quantagent.cli import v7_readiness  # noqa: F401,E402
from quantagent.cli import v7_sector  # noqa: F401,E402
from quantagent.cli import v7_storage  # noqa: F401,E402
from quantagent.cli import v7_train  # noqa: F401,E402
from quantagent.cli import v8_deep  # noqa: F401,E402
from quantagent.cli import v8_verify  # noqa: F401,E402


def _legacy_enabled() -> bool:
    return os.getenv("QUANTAGENT_ENABLE_LEGACY_CLI", "0").strip().lower() in {
        "1", "true", "yes", "on"
    }


LEGACY_CLI_ENABLED = _legacy_enabled()
if LEGACY_CLI_ENABLED:
    from quantagent.cli import v7_optimize  # noqa: F401,E402
    from quantagent.cli import v7_research  # noqa: F401,E402
    from quantagent.cli import v8  # noqa: F401,E402
    from quantagent.cli import v8_gated  # noqa: F401,E402
    from quantagent.cli import v8_intraday  # noqa: F401,E402
    from quantagent.cli import v8_portfolio  # noqa: F401,E402


__all__ = ["LEGACY_CLI_ENABLED", "app"]


if __name__ == "__main__":  # pragma: no cover
    app()
