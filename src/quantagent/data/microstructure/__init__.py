"""A-share tick / Level-2 microstructure data layer.

Provider-neutral by construction: the canonical contracts in
:mod:`~quantagent.data.microstructure.contracts` describe A-share exchange
events, not any one vendor's payload. Adapters translate into them; nothing in
this package imports a vendor SDK at module scope, so the layer stays importable
on a host where no market-data client is installed.

``contracts``   canonical event families, data-class taxonomy, session phases
``store``       append-only partitioned raw event journal
``integrity``   forensic checks producing PASS/WARN/FAIL/NOT_RUN verdicts
``reconcile``   tick-to-daily reconciliation against the verified U0 panel
``capability``  entitlement-aware provider capability matrix
``fidelity``    which simulator level a dataset actually licenses
"""

from quantagent.data.microstructure import (  # noqa: F401
    capability,
    contracts,
    fidelity,
    integrity,
    reconcile,
    store,
)

__all__ = ["capability", "contracts", "fidelity", "integrity", "reconcile", "store"]
