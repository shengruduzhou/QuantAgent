"""Multi-agent research governance: roles, evidence envelopes, hard vetoes.

The problem this package solves is not "how do several models cooperate" but
"how does a system refuse to fool itself". Its three commitments:

* an approval must cite the artifacts it read, by hash;
* an agent guarding a class of harm can stop a decision outright, and no
  quantity of confident approvals overturns it;
* an absent check never reads as a passed check.

``envelopes``  the per-agent decision envelope and its validity rules
``agents``     role definitions: scope, tools, authority, failure behaviour
``protocol``   sequencing and outcome resolution
``audit``      hash-chained append-only log, persisted outside Git
"""

from quantagent.governance import agents, audit, envelopes, protocol  # noqa: F401

__all__ = ["agents", "audit", "envelopes", "protocol"]
