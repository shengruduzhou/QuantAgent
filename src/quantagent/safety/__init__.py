"""Safety boundary: operating modes, live-order rejection, readiness tiers.

Two independent guarantees live here.

``operating_mode`` proves the system cannot transmit a real order. LIVE_DISABLED
is a terminal policy state rather than a missing feature, so the guarantee is
assertable instead of merely unimplemented.

``readiness_tiers`` replaces a single vague "ready" flag with four certificates
that each state what they allow *and* what they forbid, so a smoke run can never
be mistaken for a licence to quote performance or run a portfolio.
"""

from quantagent.safety import operating_mode, readiness_tiers  # noqa: F401

__all__ = ["operating_mode", "readiness_tiers"]
