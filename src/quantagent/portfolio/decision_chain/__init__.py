"""Stage 5.3 — 14-step decision chain.

A candidate trade flows through 14 ordered gates. The first failing
gate short-circuits the chain; the trace records every gate evaluated
plus its reason, so post-trade audit can reconstruct exactly why a
candidate was admitted or rejected.

Public surface:

* :class:`DecisionChainConfig` — thresholds + which gates to enable.
* :class:`DecisionContext` — the inputs available at evaluation time
  (alpha, market panel, sector pool, hard gate, regime, fundamentals,
  policy, broker consensus, current holdings).
* :class:`Candidate` — a single (date, symbol) we're considering.
* :func:`run_decision_chain` — evaluates one Candidate.
* :func:`run_decision_chain_batch` — evaluates many Candidates and
  returns a long-form DataFrame of traces.
"""

from quantagent.portfolio.decision_chain.chain import (
    GATE_ORDER,
    Candidate,
    DecisionChainConfig,
    DecisionContext,
    DecisionTrace,
    GateResult,
    run_decision_chain,
    run_decision_chain_batch,
    traces_to_frame,
)

__all__ = [
    "GATE_ORDER",
    "Candidate",
    "DecisionChainConfig",
    "DecisionContext",
    "DecisionTrace",
    "GateResult",
    "run_decision_chain",
    "run_decision_chain_batch",
    "traces_to_frame",
]
