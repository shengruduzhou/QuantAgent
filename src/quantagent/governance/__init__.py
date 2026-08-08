"""Multi-agent research governance: roles, evidence envelopes, hard vetoes.

The problem this package solves is not "how do several models cooperate" but
"how does a system refuse to fool itself". Its commitments include evidence-
backed approvals, hard vetoes, fail-closed missing checks, and an isolated
production-audit board where a finding author cannot approve its own work.

``envelopes``                 per-agent decision envelope and validity rules
``agents``                    role definitions and veto authority
``protocol``                  sequencing and outcome resolution
``audit``                     hash-chained append-only log
``isolated_production_audit`` cross-review gate before the main repair role
"""

from quantagent.governance import agents, audit, envelopes, isolated_production_audit, protocol  # noqa: F401

__all__ = ["agents", "audit", "envelopes", "protocol", "isolated_production_audit"]
